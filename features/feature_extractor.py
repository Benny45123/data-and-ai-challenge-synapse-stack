"""
features/feature_extractor.py — Compute all LTR feature groups.

Each group maps to a section of the architecture spec:
  Group A — Semantic relevance  (from Stages 1 & 2)
  Group B — Engagement signals  (platform logs)
  Group C — Career trajectory   (extracted from career_history)
  Group D — Trust signals       (platform verification)

The extractor is stateless: every method is a pure function of its inputs.
Redis caching is applied at the pipeline layer, not here.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Set

from models.schemas import CandidateProfile, JobDescription, LTRFeatureVector

log = logging.getLogger(__name__)

# ── Shipper keyword signals (Section 3.3 of spec) ─────────────────────────────
_SHIPPER_SIGNALS = frozenset({
    "deployed", "launched", "production", "shipped", "migrated",
    "scaled", "served", "qps", "endpoint", "latency", "throughput",
    "containerised", "containerized", "k8s", "kubernetes", "sla",
})
_RESEARCHER_SIGNALS = frozenset({
    "explored", "investigated", "proposed", "authored paper",
    "researched", "ablation", "hypothesised", "submitted to",
})


def _tokenise(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


# ── Skill overlap (ontology-aware, simplified) ────────────────────────────────

# A minimal equivalence map — in production this is a full O*NET graph lookup.
_SKILL_SYNONYMS: Dict[str, str] = {
    "pytorch": "pytorch", "torch": "pytorch",
    "tensorflow": "tensorflow", "tf": "tensorflow",
    "hugging face": "huggingface", "huggingface": "huggingface",
    "transformers": "huggingface",
    "faiss": "faiss", "qdrant": "vector_db", "pinecone": "vector_db",
    "weaviate": "vector_db", "milvus": "vector_db",
    "bm25": "sparse_retrieval", "elasticsearch": "sparse_retrieval",
    "ndcg": "ranking_metrics", "mrr": "ranking_metrics",
    "lambdamart": "ltr", "lightgbm": "ltr", "xgboost": "gbm",
    "python": "python", "rust": "rust", "golang": "go", "go": "go",
    "aws": "cloud", "gcp": "cloud", "azure": "cloud",
}

def _normalise_skill(skill: str) -> str:
    return _SKILL_SYNONYMS.get(skill.lower().strip(), skill.lower().strip())


def compute_skill_overlap(
    candidate_skills: List[str],
    jd_required: List[str],
    jd_preferred: List[str],
) -> tuple[float, float]:
    """
    Returns:
        (overlap_ratio, depth_score)

        overlap_ratio: fraction of JD required skills covered (ontology-aware).
        depth_score:   average specificity weight of matched skills (0–1).
    """
    all_jd = set(_normalise_skill(s) for s in jd_required + jd_preferred)
    required_jd = set(_normalise_skill(s) for s in jd_required)
    candidate_norm = set(_normalise_skill(s) for s in candidate_skills)

    if not required_jd:
        return 0.0, 0.0

    matched_required = required_jd & candidate_norm
    overlap_ratio = len(matched_required) / len(required_jd)

    # Depth: specific skills (vector_db, ltr) score higher than generic (python)
    _depth_weights = {
        "vector_db": 0.9, "ltr": 0.9, "sparse_retrieval": 0.85,
        "ranking_metrics": 0.9, "pytorch": 0.8, "huggingface": 0.8,
        "faiss": 0.85, "python": 0.4, "cloud": 0.5,
    }
    depth = (
        sum(_depth_weights.get(s, 0.6) for s in matched_required) / len(required_jd)
        if matched_required else 0.0
    )
    return round(overlap_ratio, 4), round(depth, 4)


# ── Archetype (shipper) score ─────────────────────────────────────────────────

def compute_archetype_score(career_history_text: str) -> float:
    """
    Lightweight rule-based shipper score.

    In production this is replaced by the fine-tuned DistilBERT classifier
    (Section 3.3). This version is used as a fallback when the classifier
    model is not loaded.

    Returns: float in [0, 1] where 1.0 = pure shipper.
    """
    tokens = _tokenise(career_history_text)
    shipper_hits = len(tokens & _SHIPPER_SIGNALS)
    researcher_hits = len(tokens & _RESEARCHER_SIGNALS)
    total = shipper_hits + researcher_hits
    if total == 0:
        return 0.5  # neutral prior
    return round(shipper_hits / total, 4)


# ── Career trajectory features (Group C) ─────────────────────────────────────

def compute_tenure_stats(
    career_history: list,
) -> tuple[float, float, int, float]:
    """
    Returns:
        (relevant_yoe, avg_tenure_months, gap_months, career_velocity)
    """
    if not career_history:
        return 0.0, 0.0, 0, 0.5

    tenures = []
    ends = []

    for entry in career_history:
        try:
            start = datetime.strptime(entry.start_date[:7], "%Y-%m")
        except (ValueError, AttributeError):
            continue

        if entry.is_current or not entry.end_date:
            end = datetime.utcnow()
        else:
            try:
                end = datetime.strptime(entry.end_date[:7], "%Y-%m")
            except ValueError:
                end = datetime.utcnow()

        months = max(0, (end.year - start.year) * 12 + (end.month - start.month))
        tenures.append(months)
        ends.append(end)

    avg_tenure = sum(tenures) / len(tenures) if tenures else 0.0

    # Gap detection
    sorted_ends = sorted(ends)
    gap_months = 0
    for i in range(1, len(sorted_ends) - 1):
        gap = (sorted_ends[i + 1].year - sorted_ends[i].year) * 12 + (
            sorted_ends[i + 1].month - sorted_ends[i].month
        )
        if gap > 3:   # gaps > 3 months count
            gap_months += gap - 3

    # Career velocity: normalised rate of role progression (proxy)
    role_count = len(career_history)
    total_yoe = sum(tenures) / 12.0
    velocity = min(1.0, role_count / max(total_yoe, 1.0) * 0.5)

    # Relevant YoE = total for now; in production filtered by ontology match
    relevant_yoe = total_yoe * 0.7   # conservative assumption

    return (
        round(relevant_yoe, 2),
        round(avg_tenure, 2),
        gap_months,
        round(velocity, 4),
    )


def count_production_systems(career_history: list) -> int:
    """
    Count evidence of production system ownership in career descriptions.
    Proxy: sentences containing shipper keywords in past-tense context.
    """
    count = 0
    for entry in career_history:
        text = (entry.description or "").lower()
        if any(sig in text for sig in _SHIPPER_SIGNALS):
            count += 1
    return count


# ── Master extractor ──────────────────────────────────────────────────────────

def extract_features(
    candidate: CandidateProfile,
    jd: JobDescription,
    bm25_score: float = 0.0,
    dense_cosine_sim: float = 0.0,
    rrf_score: float = 0.0,
    cross_encoder_score: float = 0.0,
) -> LTRFeatureVector:
    """
    Build the complete LTR feature vector for a (JD, candidate) pair.

    Semantic scores (A) come from the earlier pipeline stages.
    Behavioural / career / trust signals (B, C, D) are computed here.
    """
    # ── Group A ──────────────────────────────────────────────────────────────
    career_text = " ".join(
        e.description for e in candidate.career_history if e.description
    )
    archetype_score = compute_archetype_score(career_text)

    overlap_ratio, depth_score = compute_skill_overlap(
        candidate.skills,
        jd.required_skills,
        jd.preferred_skills,
    )

    # ── Group C ──────────────────────────────────────────────────────────────
    relevant_yoe, avg_tenure, gap_months, velocity = compute_tenure_stats(
        candidate.career_history
    )
    production_count = count_production_systems(candidate.career_history)

    # ── Log-transform response time (lower is better) ─────────────────────────
    log_response_time = math.log1p(candidate.avg_response_time_hours)

    return LTRFeatureVector(
        # Group A
        bm25_score=bm25_score,
        dense_cosine_sim=dense_cosine_sim,
        rrf_score=rrf_score,
        cross_encoder_score=cross_encoder_score,
        skill_overlap_ratio=overlap_ratio,
        skill_depth_score=depth_score,
        archetype_score=archetype_score,
        # Group B
        recruiter_response_rate=candidate.recruiter_response_rate,
        avg_response_time_hours=log_response_time,
        interview_completion_rate=candidate.interview_completion_rate,
        days_since_last_active=candidate.days_since_last_active,
        profile_completeness_score=candidate.profile_completeness_score,
        github_activity_score=candidate.github_activity_score,
        assessment_score_percentile=candidate.assessment_score_percentile,
        # Group C
        total_yoe=candidate.total_yoe,
        relevant_yoe=relevant_yoe,
        avg_tenure_months=avg_tenure,
        career_velocity=velocity,
        production_system_count=production_count,
        gap_months=gap_months,
        # Group D
        verified_email=candidate.verified_email,
        verified_phone=candidate.verified_phone,
        linkedin_connected=candidate.linkedin_connected,
        profile_age_days=candidate.profile_age_days,
    )