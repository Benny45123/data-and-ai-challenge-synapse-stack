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


# ─────────────────────────────────────────────────────────────────────────────
# Real hackathon schema handler
# Maps the actual candidates.jsonl schema → LTRFeatureVector
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_from_raw(
    candidate: dict,
    bm25_score: float = 0.0,
    dense_cosine_sim: float = 0.0,
    rrf_score: float = 0.0,
    cross_encoder_score: float = 0.0,
    tfidf_score: float = 0.0,
) -> "LTRFeatureVector":
    """
    Extract LTR features from the raw hackathon candidate dict.

    The real schema uses:
      candidate["profile"]          → headline, summary, years_of_experience, etc.
      candidate["career_history"]   → list of role dicts with description
      candidate["skills"]           → list of {name, proficiency, endorsements, duration_months}
      candidate["redrob_signals"]   → the 23 behavioral signals
      candidate["education"]        → list of education dicts

    This replaces the simplified CandidateProfile schema for batch ranking.
    """
    import math
    from datetime import date, datetime

    TODAY = date(2026, 6, 30)

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills_list = candidate.get("skills", [])
    education = candidate.get("education", [])

    # ── Career trajectory ──────────────────────────────────────────────────
    career_desc = " ".join(e.get("description", "") for e in career)
    relevant_yoe, avg_tenure, gap_months, velocity = compute_tenure_stats_raw(career, TODAY)
    production_count = count_production_systems(
        [type("E", (), {"description": e.get("description", "")})() for e in career]
    )

    # ── Archetype (shipper) score from descriptions ────────────────────────
    archetype_score = compute_archetype_score(career_desc)

    # ── Skill overlap using real skills list ───────────────────────────────
    skill_names = [s.get("name", "") for s in skills_list]
    from features.description_scorer import _PROD_RE
    prod_hits = sum(1 for r in _PROD_RE if r.search(career_desc))
    skill_overlap_ratio = min(1.0, prod_hits / 10.0)
    skill_depth_score = min(1.0, prod_hits / 12.0)

    # ── Behavioral signals (Group B) ───────────────────────────────────────
    rr = float(signals.get("recruiter_response_rate", 0.5))
    response_time = float(signals.get("avg_response_time_hours", 24.0))
    log_response_time = math.log1p(response_time)
    icr = float(signals.get("interview_completion_rate", 0.8))

    try:
        last_active = date.fromisoformat(signals.get("last_active_date", "2025-01-01"))
        days_inactive = (TODAY - last_active).days
    except ValueError:
        days_inactive = 180

    profile_completeness = float(signals.get("profile_completeness_score", 70)) / 100.0

    gh = float(signals.get("github_activity_score", -1))
    github_score = (gh / 100.0) if gh >= 0 else 0.0

    # Skill assessment: average score across all completed assessments
    assessments = signals.get("skill_assessment_scores", {})
    assessment_pct = (
        sum(assessments.values()) / len(assessments) / 100.0
        if assessments else 0.5
    )

    # ── Profile age ───────────────────────────────────────────────────────
    try:
        signup = date.fromisoformat(signals.get("signup_date", "2024-01-01"))
        profile_age_days = (TODAY - signup).days
    except ValueError:
        profile_age_days = 365

    # ── Trust signals (Group D) ───────────────────────────────────────────
    verified_email = bool(signals.get("verified_email", False))
    verified_phone = bool(signals.get("verified_phone", False))
    linkedin = bool(signals.get("linkedin_connected", False))

    # ── Use tfidf_score as best proxy for semantic relevance ──────────────
    # It replaces cross_encoder_score when CE model isn't loaded
    effective_ce = cross_encoder_score if cross_encoder_score > 0 else tfidf_score

    return LTRFeatureVector(
        # Group A
        bm25_score=bm25_score,
        dense_cosine_sim=dense_cosine_sim,
        rrf_score=rrf_score,
        cross_encoder_score=effective_ce,
        skill_overlap_ratio=skill_overlap_ratio,
        skill_depth_score=skill_depth_score,
        archetype_score=archetype_score,
        # Group B
        recruiter_response_rate=rr,
        avg_response_time_hours=log_response_time,
        interview_completion_rate=icr,
        days_since_last_active=days_inactive,
        profile_completeness_score=profile_completeness,
        github_activity_score=github_score,
        assessment_score_percentile=assessment_pct,
        # Group C
        total_yoe=float(profile.get("years_of_experience", 0)),
        relevant_yoe=relevant_yoe,
        avg_tenure_months=avg_tenure,
        career_velocity=velocity,
        production_system_count=production_count,
        gap_months=gap_months,
        # Group D
        verified_email=verified_email,
        verified_phone=verified_phone,
        linkedin_connected=linkedin,
        profile_age_days=profile_age_days,
    )


def compute_tenure_stats_raw(career: list, today: "date") -> tuple:
    """
    Same as compute_tenure_stats() but operates on raw career_history dicts
    (real schema) rather than CareerEntry Pydantic objects.
    """
    from datetime import date as _date

    if not career:
        return 0.0, 0.0, 0, 0.5

    tenures = []
    ends = []

    for entry in career:
        try:
            start = _date.fromisoformat(entry.get("start_date", "2020-01-01"))
        except ValueError:
            continue

        raw_end = entry.get("end_date")
        if not raw_end or entry.get("is_current", False):
            end = today
        else:
            try:
                end = _date.fromisoformat(raw_end)
            except ValueError:
                end = today

        months = max(0, (end.year - start.year) * 12 + (end.month - start.month))
        tenures.append(months)
        ends.append(end)

    avg_tenure = sum(tenures) / len(tenures) if tenures else 0.0
    sorted_ends = sorted(ends)
    gap_months = 0
    for i in range(len(sorted_ends) - 1):
        gap = (
            (sorted_ends[i + 1].year - sorted_ends[i].year) * 12 +
            (sorted_ends[i + 1].month - sorted_ends[i].month)
        )
        if gap > 3:
            gap_months += gap - 3

    role_count = len(career)
    total_yoe = sum(tenures) / 12.0
    velocity = min(1.0, role_count / max(total_yoe, 1.0) * 0.5)
    relevant_yoe = total_yoe * 0.7

    return round(relevant_yoe, 2), round(avg_tenure, 2), gap_months, round(velocity, 4)
