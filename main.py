"""
main.py — FastAPI application entry-point.

Endpoints:
  POST /rank                   Core ranking pipeline
  POST /admin/upsert           Index new / updated candidate profiles
  POST /feedback               Ingest recruiter action labels
  GET  /health                 Health + index stats
  GET  /admin/feature-importance  LTR feature importance (for auditing)

Lifespan:
  The pipeline (models + indexes) is initialised once at startup and shared
  across all requests via FastAPI dependency injection.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from config import get_settings
from models.schemas import (
    FeedbackEvent,
    RankingRequest,
    RankingResponse,
    UpsertRequest,
    UpsertResponse,
)
from pipeline import RankingPipeline
from utils.embeddings import embed
from utils.pinecone_client import index_stats, upsert_candidates

# ── Logging ───────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()
settings = get_settings()


# ── Application lifespan (startup / shutdown) ─────────────────────────────────

_pipeline: RankingPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise all heavy resources once at startup.

    Models are loaded here so the first request doesn't pay the cold-start
    penalty. In production, this also warms up the Redis connection pool.
    """
    global _pipeline

    log.info("event", message="Starting Redrob ranking service…")
    _pipeline = RankingPipeline()

    # Pre-load embedding and cross-encoder models (avoids cold-start latency)
    from utils.embeddings import _load_model as _load_dense
    from stages.stage2_reranking import _load_cross_encoder
    _load_dense()
    _load_cross_encoder()

    log.info("event", message="Pipeline ready.")
    yield

    # Shutdown
    log.info("event", message="Shutting down.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Redrob — Intelligent Candidate Ranking API",
    version="1.0.0",
    description=(
        "Four-stage ranking pipeline: "
        "Hybrid Retrieval → Cross-Encoder → LambdaMART → LLM Explainability"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics on /metrics
Instrumentator().instrument(app).expose(app)


# ── Dependency ────────────────────────────────────────────────────────────────

def get_pipeline() -> RankingPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not yet initialised.",
        )
    return _pipeline


Pipeline = Annotated[RankingPipeline, Depends(get_pipeline)]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/rank",
    response_model=RankingResponse,
    summary="Rank candidates for a job description",
    tags=["Ranking"],
)
async def rank_candidates(
    request: RankingRequest,
    pipeline: Pipeline,
) -> RankingResponse:
    """
    Execute the four-stage ranking cascade for a job description.

    - **Stage 1**: Hybrid BM25 + cosine-similarity retrieval with RRF fusion (< 20ms)
    - **Stage 2**: Cross-encoder re-ranking of top 500 → top 100 (< 120ms)
    - **Stage 3**: LambdaMART LTR signal fusion → final top 25 (< 5ms)
    - **Stage 4**: LLM fit summaries with fact verification (async, < 500ms)

    Set `include_explanations=true` to receive fit summaries (adds latency).
    """
    try:
        return await pipeline.rank(request)
    except Exception as exc:
        log.error("event", message="Ranking failed", error=str(exc), jd_id=request.jd.jd_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ranking pipeline error: {exc}",
        )


@app.post(
    "/admin/upsert",
    response_model=UpsertResponse,
    summary="Index candidate profiles into Pinecone + BM25",
    tags=["Admin"],
)
async def upsert_profiles(
    request: UpsertRequest,
    pipeline: Pipeline,
) -> UpsertResponse:
    """
    Embed and index new or updated candidate profiles.

    Both the Pinecone dense index and the in-memory BM25 index are updated.
    Kafka consumers trigger this endpoint on `profile.updated` events in production.
    """
    candidates = request.candidates
    errors: list[str] = []

    try:
        # Build embeddings for all profiles
        texts = [c.profile_text for c in candidates]
        embeddings = embed(texts, normalise=True)

        ids = [c.candidate_id for c in candidates]
        metadata = [
            {
                "name": c.name,
                "total_yoe": c.total_yoe,
                "open_to_remote": c.open_to_remote,
                "duplicate_flag": c.duplicate_flag,
                "location": c.location or "",
            }
            for c in candidates
        ]

        upserted, failed = upsert_candidates(ids, embeddings, metadata)

        # Update BM25 index and local profile store
        for candidate in candidates:
            pipeline.retriever.add_candidate(candidate)

        return UpsertResponse(indexed=upserted, failed=failed, errors=errors)

    except Exception as exc:
        log.error("event", message="Upsert failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest recruiter action as a training label",
    tags=["Feedback"],
)
async def ingest_feedback(event: FeedbackEvent):
    """
    Accept a recruiter action event and store it as a relevance label.

    Labels flow: recruiter action → Kafka → Flink aggregator → Postgres label store
    → nightly Airflow retraining pipeline.

    In this implementation, events are logged and would be forwarded to Kafka
    in a production deployment.
    """
    log.info(
        "event",
        message="Feedback received",
        jd_id=event.jd_id,
        candidate_id=event.candidate_id,
        action=event.action,
        relevance_grade=event.relevance_grade,
        rank_at_time=event.rank_at_time,
    )
    # Production: publish to Kafka topic "recruiter.signals"
    # kafka_producer.send("recruiter.signals", event.model_dump())
    return {"status": "accepted", "relevance_grade": event.relevance_grade}


@app.get(
    "/admin/feature-importance",
    summary="LTR feature importance for fairness auditing",
    tags=["Admin"],
)
def get_feature_importance(pipeline: Pipeline):
    """
    Return SHAP-based feature importance from the trained LambdaMART model.

    Used for:
    - Auditing proxy discrimination (Section 12.3)
    - Identifying high-importance / low-quality features (data collection priority)
    - Product transparency reporting
    """
    importance = pipeline.feature_importance()
    if importance is None:
        return {"status": "no_model", "message": "No trained LTR model loaded."}
    return {"feature_importance": importance}


@app.get(
    "/health",
    summary="Service health and index statistics",
    tags=["Ops"],
)
def health():
    """
    Returns:
    - Service status
    - Pinecone index vector count
    - BM25 index document count
    """
    try:
        pc_stats = index_stats()
    except Exception as exc:
        pc_stats = {"error": str(exc)}

    bm25_size = _pipeline.retriever._bm25.size if _pipeline else 0

    return {
        "status": "ok",
        "pinecone": pc_stats,
        "bm25_index_size": bm25_size,
        "model": {
            "dense": settings.DENSE_MODEL_NAME,
            "cross_encoder": settings.CROSS_ENCODER_MODEL,
            "ltr_model_loaded": (
                _pipeline.ltr._model is not None if _pipeline else False
            ),
        },
    }


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,                  # GPU model must stay in one process
        reload=os.getenv("ENV") == "development",
    )