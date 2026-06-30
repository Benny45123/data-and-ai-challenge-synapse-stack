"""
stages/stage2_reranking.py — Cross-Encoder Re-ranking (< 60ms target).

Why cross-encoders vs bi-encoders (Section 5 of spec):
  Bi-encoders encode JD and profile independently → interaction is a dot product.
  Cross-encoders encode the PAIR jointly → full attention across both texts.
  This captures whether specific JD skills are evidenced *in the right context*
  inside the profile, not just whether the words appear.

Cost-tradeoff: ~100× slower per pair. Stage 1 narrows to 500 first.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

from config import get_settings
from models.schemas import JobDescription, RerankedCandidate, RetrievedCandidate

log = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _load_cross_encoder() -> CrossEncoder:
    log.info("Loading cross-encoder: %s", settings.CROSS_ENCODER_MODEL)
    return CrossEncoder(settings.CROSS_ENCODER_MODEL, max_length=settings.CE_MAX_TOKENS)


def _build_pair_text(jd: JobDescription, profile_text: str) -> tuple[str, str]:
    """
    Construct the (query, document) pair fed into the cross-encoder.

    Input is capped at CE_MAX_TOKENS:
      ~200 tokens: JD summary (title + first 2 sentences of description)
      ~312 tokens: candidate profile summary + top-3 career descriptions
    """
    jd_sentences = jd.description.split(". ")
    jd_summary = ". ".join(jd_sentences[:3])
    query = f"{jd.title}. {jd_summary}. Required: {', '.join(jd.required_skills)}"

    # Truncate profile text to fit the token budget
    doc = profile_text[:1500]   # rough char-to-token approximation

    return query, doc


class CrossEncoderReranker:
    """Stateless cross-encoder re-ranker. Loads the model lazily on first call."""

    def rerank(
        self,
        jd: JobDescription,
        candidates: List[RetrievedCandidate],
        top_k: int | None = None,
    ) -> List[RerankedCandidate]:
        """
        Score all (JD, candidate) pairs and return the top-k by CE score.

        Args:
            jd:         Job description.
            candidates: Output of Stage 1 (up to 500 candidates).
            top_k:      How many to forward to Stage 3 (default: TOP_K_RERANKING).

        Returns:
            List of RerankedCandidate sorted by cross_encoder_score, descending.
        """
        top_k = top_k or settings.TOP_K_RERANKING

        if not candidates:
            return []

        ce = _load_cross_encoder()

        # Build all pairs
        pairs: List[tuple[str, str]] = [
            _build_pair_text(jd, rc.candidate.profile_text)
            for rc in candidates
        ]

        # Batch inference — GPU-batched for throughput
        raw_scores: np.ndarray = ce.predict(
            pairs,
            batch_size=settings.CROSS_ENCODER_BATCH_SIZE,
            show_progress_bar=False,
        )

        # Sigmoid to [0, 1]
        ce_scores = 1.0 / (1.0 + np.exp(-raw_scores))

        # Sort and slice
        indexed = sorted(
            zip(candidates, ce_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked: List[RerankedCandidate] = [
            RerankedCandidate(
                candidate=rc.candidate,
                rrf_score=rc.rrf_score,
                cross_encoder_score=float(score),
            )
            for rc, score in indexed[:top_k]
        ]

        log.info(
            "Stage 2 — JD=%s: scored %d pairs → keeping top %d (CE range %.3f–%.3f)",
            jd.jd_id,
            len(candidates),
            len(reranked),
            reranked[-1].cross_encoder_score if reranked else 0,
            reranked[0].cross_encoder_score if reranked else 0,
        )
        return reranked