"""
stages/stage3_ltr.py — LambdaMART Learning-to-Rank (< 10ms target).

Why LambdaMART over pointwise XGBoost (Section 6.1 of spec):
  Pointwise regression optimises MSE of per-candidate scores in isolation.
  LambdaMART is a listwise/pairwise hybrid that directly optimises NDCG via
  the λ-gradient trick — it is *aware of relative ordering* within a query's
  candidate list.

Label scheme (Section 6.2):
  3 = hired / offer extended
  2 = reached interview stage
  1 = profile saved / shortlisted
  0 = viewed but skipped
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np

from config import get_settings
from features.feature_extractor import extract_features
from models.schemas import (
    CandidateProfile,
    JobDescription,
    LTRFeatureVector,
    RerankedCandidate,
)

log = logging.getLogger(__name__)
settings = get_settings()


# ── Training pipeline ─────────────────────────────────────────────────────────

def build_training_data(
    samples: List[Tuple[JobDescription, CandidateProfile, int]],
    bm25_scores: Dict[str, float],
    dense_scores: Dict[str, float],
    rrf_scores: Dict[str, float],
    ce_scores: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw (JD, candidate, label) triples into LTR training arrays.

    Args:
        samples:      List of (jd, candidate, relevance_grade) triples.
        *_scores:     Pre-computed retrieval scores keyed by candidate_id.

    Returns:
        X:      Feature matrix (n_samples, n_features)
        y:      Relevance grades (n_samples,)
        groups: Samples-per-query array required by LightGBM ranker.
    """
    X_rows, y_rows, group_sizes = [], [], []

    # Group by JD (each JD is one "query" in LTR terminology)
    jd_groups: Dict[str, List] = {}
    for jd, candidate, label in samples:
        jd_groups.setdefault(jd.jd_id, []).append((jd, candidate, label))

    for jd_id, group in jd_groups.items():
        group_sizes.append(len(group))
        for jd, candidate, label in group:
            cid = candidate.candidate_id
            fv = extract_features(
                candidate=candidate,
                jd=jd,
                bm25_score=bm25_scores.get(cid, 0.0),
                dense_cosine_sim=dense_scores.get(cid, 0.0),
                rrf_score=rrf_scores.get(cid, 0.0),
                cross_encoder_score=ce_scores.get(cid, 0.0),
            )
            X_rows.append(fv.to_numpy())
            y_rows.append(label)

    return (
        np.array(X_rows, dtype=float),
        np.array(y_rows, dtype=int),
        np.array(group_sizes, dtype=int),
    )


def train_ltr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    groups_val: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> lgb.LGBMRanker:
    """
    Train a LambdaMART model (LightGBM Ranker) and optionally save it.

    Hyperparameters are chosen to match production scale (Section 6 of spec).
    """
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10, 20],
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )

    fit_kwargs = dict(
        X=X_train,
        y=y_train,
        group=groups_train,
        feature_name=LTRFeatureVector.feature_names(),
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    if X_val is not None and y_val is not None and groups_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [groups_val]
        fit_kwargs["eval_at"] = [5, 10, 20]

    model.fit(**fit_kwargs)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        log.info("LTR model saved → %s", save_path)

    return model


# ── Inference ─────────────────────────────────────────────────────────────────

class LTRRanker:
    """
    Wraps a trained LGBMRanker for production inference.

    The model is loaded lazily from disk the first time it is needed.
    A cold-start fallback (score weighting) handles the case where
    no trained model exists yet (Section 13.3 of spec).
    """

    def __init__(self) -> None:
        self._model: Optional[lgb.LGBMRanker] = None

    def _load_or_fallback(self) -> Optional[lgb.LGBMRanker]:
        if self._model is not None:
            return self._model
        path = settings.LTR_MODEL_PATH
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info("LTR model loaded from %s", path)
        else:
            log.warning("No LTR model found at %s — using cold-start fallback", path)
        return self._model

    def _cold_start_score(self, fv: LTRFeatureVector) -> float:
        """
        Cold-start rule: weight CE score + skill overlap 70%, engagement 30%.
        Used when no trained model is available (Section 13.3).
        """
        return (
            0.45 * fv.cross_encoder_score
            + 0.25 * fv.skill_overlap_ratio
            + 0.15 * fv.recruiter_response_rate
            + 0.10 * fv.archetype_score
            + 0.05 * fv.github_activity_score
        )

    def rank(
        self,
        jd: JobDescription,
        candidates: List[RerankedCandidate],
        top_k: int | None = None,
        bm25_scores: Optional[Dict[str, float]] = None,
        dense_scores: Optional[Dict[str, float]] = None,
        rrf_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[CandidateProfile, float, LTRFeatureVector]]:
        """
        Apply LambdaMART to produce the final ranking.

        Returns:
            List of (candidate, final_score, feature_vector) sorted by score desc.
        """
        top_k = top_k or settings.TOP_K_LTR
        model = self._load_or_fallback()

        feature_vectors: List[LTRFeatureVector] = []
        profiles: List[CandidateProfile] = []

        for rc in candidates:
            cid = rc.candidate.candidate_id
            fv = extract_features(
                candidate=rc.candidate,
                jd=jd,
                bm25_score=(bm25_scores or {}).get(cid, 0.0),
                dense_cosine_sim=(dense_scores or {}).get(cid, 0.0),
                rrf_score=(rrf_scores or {}).get(cid, rc.rrf_score),
                cross_encoder_score=rc.cross_encoder_score,
            )
            feature_vectors.append(fv)
            profiles.append(rc.candidate)

        if model is not None:
            X = np.array([fv.to_numpy() for fv in feature_vectors])
            # LightGBM ranker predict returns raw scores
            raw_scores = model.predict(X)
            # Min-max normalise to [0, 1]
            lo, hi = raw_scores.min(), raw_scores.max()
            span = hi - lo if hi > lo else 1.0
            final_scores = ((raw_scores - lo) / span).tolist()
        else:
            final_scores = [self._cold_start_score(fv) for fv in feature_vectors]

        # Sort descending
        ranked = sorted(
            zip(profiles, final_scores, feature_vectors),
            key=lambda x: x[1],
            reverse=True,
        )

        log.info(
            "Stage 3 — JD=%s: %d candidates → top %d (score range %.3f–%.3f)",
            jd.jd_id,
            len(candidates),
            top_k,
            ranked[top_k - 1][1] if len(ranked) >= top_k else 0,
            ranked[0][1] if ranked else 0,
        )
        return ranked[:top_k]

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Return SHAP-based feature importance for monitoring / auditing."""
        model = self._load_or_fallback()
        if model is None:
            return None
        names = LTRFeatureVector.feature_names()
        importance = model.feature_importances_
        return dict(sorted(
            zip(names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ))