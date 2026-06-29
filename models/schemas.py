"""
models/schemas.py — Typed data contracts for every layer of the pipeline.

Schemas are intentionally verbose: every field that flows through the pipeline
is declared here so mypy / FastAPI validation catches mismatches at the boundary.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Candidate / JD input schemas
# ─────────────────────────────────────────────────────────────────────────────

class CareerEntry(BaseModel):
    company: str
    title: str
    start_date: str                     # "YYYY-MM" or "YYYY"
    end_date: Optional[str] = None      # None if current
    description: str = ""
    is_current: bool = False


class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    email: Optional[str] = None

    # ── Structured career data ────────────────────────────────────────────────
    skills: List[str] = Field(default_factory=list)
    career_history: List[CareerEntry] = Field(default_factory=list)
    education: Optional[str] = None
    total_yoe: float = 0.0
    github_url: Optional[str] = None
    location: Optional[str] = None
    open_to_remote: bool = True

    # ── Platform engagement signals (Group B) ─────────────────────────────────
    recruiter_response_rate: float = Field(0.5, ge=0.0, le=1.0)
    avg_response_time_hours: float = Field(24.0, ge=0.0)
    interview_completion_rate: float = Field(0.8, ge=0.0, le=1.0)
    days_since_last_active: int = Field(30, ge=0)
    profile_completeness_score: float = Field(0.7, ge=0.0, le=1.0)
    github_activity_score: float = Field(0.5, ge=0.0, le=1.0)
    assessment_score_percentile: float = Field(0.5, ge=0.0, le=1.0)

    # ── Trust signals (Group D) ───────────────────────────────────────────────
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False
    duplicate_flag: bool = False
    profile_age_days: int = 365

    @property
    def profile_text(self) -> str:
        """Concatenated free-text for embedding / BM25 indexing."""
        parts = [self.education or ""]
        parts += [f"{e.title} at {e.company}: {e.description}" for e in self.career_history]
        parts += self.skills
        return " ".join(filter(None, parts))


class JobDescription(BaseModel):
    jd_id: str
    title: str
    description: str
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_yoe: float = 0.0
    max_yoe: float = 99.0
    location: Optional[str] = None
    remote_ok: bool = True
    archetype: Optional[str] = None     # "shipper" | "researcher" | "hybrid"
    target_company_size: Optional[str] = None  # "startup" | "mid" | "enterprise"

    @property
    def full_text(self) -> str:
        skills = ", ".join(self.required_skills + self.preferred_skills)
        return f"{self.title}. {self.description}. Required skills: {skills}"


# ─────────────────────────────────────────────────────────────────────────────
# Internal inter-stage types
# ─────────────────────────────────────────────────────────────────────────────

class RetrievedCandidate(BaseModel):
    """Passed from Stage 1 → Stage 2."""
    candidate: CandidateProfile
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0


class RerankedCandidate(BaseModel):
    """Passed from Stage 2 → Stage 3."""
    candidate: CandidateProfile
    rrf_score: float = 0.0
    cross_encoder_score: float = 0.0


class LTRFeatureVector(BaseModel):
    """Full feature vector fed into LambdaMART."""
    # Group A — Semantic relevance
    bm25_score: float = 0.0
    dense_cosine_sim: float = 0.0
    rrf_score: float = 0.0
    cross_encoder_score: float = 0.0
    skill_overlap_ratio: float = 0.0
    skill_depth_score: float = 0.0
    archetype_score: float = 0.5

    # Group B — Engagement
    recruiter_response_rate: float = 0.5
    avg_response_time_hours: float = 24.0
    interview_completion_rate: float = 0.8
    days_since_last_active: int = 30
    profile_completeness_score: float = 0.7
    github_activity_score: float = 0.5
    assessment_score_percentile: float = 0.5

    # Group C — Career trajectory
    total_yoe: float = 0.0
    relevant_yoe: float = 0.0
    avg_tenure_months: float = 24.0
    career_velocity: float = 0.5
    production_system_count: int = 0
    gap_months: int = 0

    # Group D — Trust
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False
    profile_age_days: int = 365

    def to_numpy(self):
        import numpy as np
        return np.array([
            self.bm25_score, self.dense_cosine_sim, self.rrf_score,
            self.cross_encoder_score, self.skill_overlap_ratio,
            self.skill_depth_score, self.archetype_score,
            self.recruiter_response_rate, self.avg_response_time_hours,
            self.interview_completion_rate, self.days_since_last_active,
            self.profile_completeness_score, self.github_activity_score,
            self.assessment_score_percentile,
            self.total_yoe, self.relevant_yoe, self.avg_tenure_months,
            self.career_velocity, float(self.production_system_count),
            float(self.gap_months),
            float(self.verified_email), float(self.verified_phone),
            float(self.linkedin_connected), float(self.profile_age_days),
        ], dtype=float)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "bm25_score", "dense_cosine_sim", "rrf_score",
            "cross_encoder_score", "skill_overlap_ratio",
            "skill_depth_score", "archetype_score",
            "recruiter_response_rate", "avg_response_time_hours",
            "interview_completion_rate", "days_since_last_active",
            "profile_completeness_score", "github_activity_score",
            "assessment_score_percentile",
            "total_yoe", "relevant_yoe", "avg_tenure_months",
            "career_velocity", "production_system_count",
            "gap_months",
            "verified_email", "verified_phone",
            "linkedin_connected", "profile_age_days",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# API request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class RankingRequest(BaseModel):
    jd: JobDescription
    # If provided, rank this pool; otherwise query Pinecone index.
    candidate_pool: Optional[List[CandidateProfile]] = None
    top_k: int = Field(25, ge=1, le=100)
    include_explanations: bool = False
    hard_filters: Optional[Dict[str, Any]] = None


class CandidateResult(BaseModel):
    rank: int
    candidate_id: str
    name: str
    final_score: float = Field(..., ge=0.0, le=1.0)
    stage_scores: Dict[str, float] = Field(default_factory=dict)
    top_features: Optional[Dict[str, float]] = None
    fit_summary: Optional[str] = None
    explanation_confidence: Optional[float] = None
    flags: List[str] = Field(default_factory=list)  # fairness / quality flags


class RankingResponse(BaseModel):
    jd_id: str
    candidates: List[CandidateResult]
    latency_ms: float
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    total_considered: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Feedback / label ingestion
# ─────────────────────────────────────────────────────────────────────────────

class RecruiterAction(BaseModel):
    """Relevance grade mapping (Section 6.2 of spec)."""
    HIRED = 3
    INTERVIEWED = 2
    SAVED = 1
    SKIPPED = 0
    DISMISSED = -1

class FeedbackEvent(BaseModel):
    jd_id: str
    candidate_id: str
    action: str     # "saved" | "interviewed" | "hired" | "skipped" | "dismissed"
    recruiter_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rank_at_time: Optional[int] = None

    @property
    def relevance_grade(self) -> int:
        mapping = {
            "hired": 3, "interviewed": 2,
            "saved": 1, "skipped": 0, "dismissed": -1,
        }
        return mapping.get(self.action, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Upsert schema (for indexing candidates into Pinecone)
# ─────────────────────────────────────────────────────────────────────────────

class UpsertRequest(BaseModel):
    candidates: List[CandidateProfile]

class UpsertResponse(BaseModel):
    indexed: int
    failed: int
    errors: List[str] = Field(default_factory=list)