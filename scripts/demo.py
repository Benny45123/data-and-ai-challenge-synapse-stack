"""
scripts/demo.py — End-to-end smoke test using synthetic candidate data.

Demonstrates the full 4-stage pipeline without requiring live Pinecone or
Anthropic credentials. Use --live to enable real API calls.

Usage:
    python scripts/demo.py              # offline mode (mock Pinecone)
    python scripts/demo.py --live       # live APIs (needs .env)
"""
from __future__ import annotations

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schemas import (
    CareerEntry,
    CandidateProfile,
    JobDescription,
    RankingRequest,
)

# ── Synthetic JD ──────────────────────────────────────────────────────────────

SENIOR_AI_ENGINEER_JD = JobDescription(
    jd_id="jd-001",
    title="Senior AI Engineer — Search & Ranking",
    description=(
        "We are building a production embedding and retrieval platform serving "
        "10M+ queries per day. You will own the ranking pipeline end-to-end: "
        "from hybrid retrieval (BM25 + dense) through learning-to-rank. "
        "You must have shipped embedding systems to production, not just prototypes. "
        "Familiarity with NDCG, MRR, and LambdaMART is required."
    ),
    required_skills=["Python", "PyTorch", "vector database", "BM25", "LambdaMART", "NDCG"],
    preferred_skills=["Qdrant", "Pinecone", "FAISS", "Hugging Face", "LightGBM"],
    min_yoe=5.0,
    max_yoe=12.0,
    remote_ok=True,
    archetype="shipper",
)

# ── Synthetic candidate pool ───────────────────────────────────────────────────

CANDIDATES = [
    # Strong shipper — should rank #1
    CandidateProfile(
        candidate_id="c-001",
        name="Priya Sharma",
        skills=["Python", "PyTorch", "Qdrant", "FAISS", "BM25", "LambdaMART", "NDCG", "LightGBM"],
        career_history=[
            CareerEntry(
                company="VectorSearch Inc.",
                title="Senior ML Engineer",
                start_date="2021-03",
                end_date="2024-06",
                description=(
                    "Deployed hybrid BM25 + dense retrieval pipeline to production serving "
                    "5K QPS on AWS SageMaker. Migrated from pure BM25 to FAISS with +14% NDCG@10. "
                    "Implemented LambdaMART LTR model achieving MRR of 0.62 on A/B test."
                ),
            ),
            CareerEntry(
                company="SearchStart",
                title="ML Engineer",
                start_date="2018-06",
                end_date="2021-02",
                description="Built and shipped text embedding service using Hugging Face transformers. "
                            "Production latency < 50ms P99.",
                is_current=False,
            ),
        ],
        education="MS Computer Science, Carnegie Mellon University",
        total_yoe=7.5,
        recruiter_response_rate=0.92,
        avg_response_time_hours=4.0,
        interview_completion_rate=0.95,
        days_since_last_active=5,
        profile_completeness_score=0.95,
        github_activity_score=0.88,
        assessment_score_percentile=0.91,
        verified_email=True,
        linkedin_connected=True,
        open_to_remote=True,
    ),

    # Strong researcher — good skills but weaker shipper signal
    CandidateProfile(
        candidate_id="c-002",
        name="Marcus Chen",
        skills=["Python", "PyTorch", "FAISS", "NDCG", "BM25", "Transformer models"],
        career_history=[
            CareerEntry(
                company="DeepLearn Labs",
                title="Research Scientist",
                start_date="2020-01",
                description=(
                    "Investigated novel dense retrieval architectures. Authored paper on "
                    "bi-encoder training. Explored contrastive learning for passage retrieval. "
                    "Proposed improvements to NDCG-based evaluation frameworks."
                ),
                is_current=True,
            ),
            CareerEntry(
                company="State University",
                title="PhD Researcher",
                start_date="2016-09",
                end_date="2019-12",
                description="PhD thesis on information retrieval. Researched BM25 extensions.",
            ),
        ],
        education="PhD Computer Science, MIT",
        total_yoe=8.0,
        recruiter_response_rate=0.55,
        avg_response_time_hours=72.0,
        interview_completion_rate=0.70,
        days_since_last_active=45,
        profile_completeness_score=0.80,
        github_activity_score=0.60,
        assessment_score_percentile=0.88,
        verified_email=True,
        open_to_remote=True,
    ),

    # Partial skill match — 5 years experience, good engagement
    CandidateProfile(
        candidate_id="c-003",
        name="Ade Okonkwo",
        skills=["Python", "Elasticsearch", "BM25", "scikit-learn", "LightGBM"],
        career_history=[
            CareerEntry(
                company="E-Commerce Platform",
                title="Search Engineer",
                start_date="2020-03",
                description=(
                    "Deployed Elasticsearch-based product search serving 2K QPS. "
                    "Launched LightGBM ranking model for product results, +8% CTR. "
                    "Migrated legacy keyword search to hybrid retrieval."
                ),
                is_current=True,
            ),
            CareerEntry(
                company="Data Agency",
                title="Data Scientist",
                start_date="2018-07",
                end_date="2020-02",
                description="Built ML models for marketing analytics.",
            ),
        ],
        education="BSc Computer Science",
        total_yoe=6.0,
        recruiter_response_rate=0.85,
        avg_response_time_hours=8.0,
        interview_completion_rate=0.90,
        days_since_last_active=12,
        profile_completeness_score=0.88,
        github_activity_score=0.72,
        assessment_score_percentile=0.78,
        verified_email=True,
        linkedin_connected=True,
        open_to_remote=True,
    ),

    # Under-qualified — 2 years experience, mostly LangChain
    CandidateProfile(
        candidate_id="c-004",
        name="Jordan Lee",
        skills=["Python", "LangChain", "OpenAI API", "Pinecone"],
        career_history=[
            CareerEntry(
                company="AI Startup",
                title="ML Engineer",
                start_date="2023-01",
                description=(
                    "Built RAG chatbot using LangChain and Pinecone. "
                    "Explored vector embeddings for document retrieval."
                ),
                is_current=True,
            ),
        ],
        education="BS Computer Science",
        total_yoe=1.5,
        recruiter_response_rate=0.78,
        avg_response_time_hours=12.0,
        interview_completion_rate=0.85,
        days_since_last_active=3,
        profile_completeness_score=0.65,
        github_activity_score=0.45,
        assessment_score_percentile=0.50,
        open_to_remote=True,
    ),
]


async def run_demo(live: bool = False):
    """Run the pipeline offline (no real API calls) using mock components."""
    print("\n" + "═" * 60)
    print("  REDROB — 4-Stage Ranking Pipeline Demo")
    print("═" * 60)

    if not live:
        print("\n[OFFLINE MODE] Mocking Pinecone and Anthropic API calls.\n")
        _patch_for_offline()

    from pipeline import RankingPipeline

    pipeline = RankingPipeline()

    print(f"Indexing {len(CANDIDATES)} synthetic candidates…")
    pipeline.index_candidates(CANDIDATES)

    request = RankingRequest(
        jd=SENIOR_AI_ENGINEER_JD,
        candidate_pool=CANDIDATES,
        top_k=4,
        include_explanations=live,   # only in live mode
    )

    print(f"\nRanking for JD: '{SENIOR_AI_ENGINEER_JD.title}'")
    print("─" * 60)

    response = await pipeline.rank(request)

    print(f"\n{'Rank':<5} {'Name':<20} {'Score':<8} {'CE':<8} {'Arch':<8} {'Skill':<8}")
    print("─" * 60)
    for c in response.candidates:
        ce = c.stage_scores.get("cross_encoder", 0)
        arch = c.stage_scores.get("archetype", 0)
        skill = c.stage_scores.get("skill_overlap", 0)
        print(
            f"#{c.rank:<4} {c.name:<20} {c.final_score:<8.4f} "
            f"{ce:<8.4f} {arch:<8.4f} {skill:<8.4f}"
        )
        if c.fit_summary:
            print(f"       → {c.fit_summary[:100]}…")

    print(f"\nTotal sync latency: {response.latency_ms:.1f}ms")
    print("Stage breakdown:")
    for stage, ms in response.stage_latencies_ms.items():
        print(f"  {stage}: {ms:.1f}ms")

    print(f"\nTotal candidates considered: {response.total_considered}")
    print("═" * 60 + "\n")


def _patch_for_offline():
    """Replace API calls with lightweight mocks for offline testing."""
    import unittest.mock as mock
    import numpy as np

    # Mock Pinecone
    import utils.pinecone_client as pc_module
    pc_module.upsert_candidates = lambda *a, **kw: (len(a[0]), 0)
    pc_module.dense_query = lambda query_vector, top_k=500, **kw: [
        (c.candidate_id, float(np.random.uniform(0.5, 0.95)))
        for c in CANDIDATES
    ]
    pc_module.index_stats = lambda **kw: {"total_vector_count": len(CANDIDATES)}

    # Mock LLM
    import stages.stage4_explainability as exp_module
    async def _mock_summary(candidate, jd, fv=None):
        verified = exp_module.extract_verified_context(candidate, jd, fv)
        skills = ", ".join(verified["confirmed_required_skills"][:3]) or "limited"
        missing = ", ".join(verified["missing_required_skills"][:2]) or "none"
        return {
            "candidate_id": candidate.candidate_id,
            "summary": (
                f"{candidate.name} confirms {skills} with production evidence. "
                f"Gap: {missing} not explicitly stated in profile."
            ),
            "confidence": 0.85,
            "flags": [],
            "verified_context": verified,
        }
    exp_module.generate_fit_summary = _mock_summary


if __name__ == "__main__":
    live = "--live" in sys.argv
    asyncio.run(run_demo(live=live))