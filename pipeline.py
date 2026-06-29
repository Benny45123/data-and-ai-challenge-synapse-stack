"""
pipeline.py — Orchestrates the four-stage ranking cascade.

Timing budget (P99 targets):
  Stage 1 Hybrid Retrieval  < 20ms
  Stage 2 Cross-Encoder     < 120ms
  Stage 3 LambdaMART        < 5ms
  Stage 4 LLM (async)       < 1000ms  (never on the critical path)
  ──────────────────────────────────
  Total sync path           < 190ms

The pipeline is a single class that holds stateful components (indexes,
model handles) and is instantiated once at application startup via
FastAPI's lifespan context manager.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config import get_settings
from models.schemas import (
    CandidateProfile,
    CandidateResult,
    JobDescription,
    RankingRequest,
    RankingResponse,
)
from stages.stage1_retrieval import HybridRetriever
from stages.stage2_reranking import CrossEncoderReranker
from stages.stage3_ltr import LTRRanker
from stages.stage4_explainability import batch_generate_summaries

log = logging.getLogger(__name__)
settings = get_settings()


class RankingPipeline:
    """
    End-to-end four-stage candidate ranking pipeline.

    Usage:
        pipeline = RankingPipeline()
        pipeline.index_candidates(all_profiles)     # once at startup
        response = await pipeline.rank(request)     # per API call
    """

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.ltr = LTRRanker()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_candidates(self, candidates: List[CandidateProfile]) -> None:
        """
        Populate BM25 and Pinecone indexes. Called once at startup.
        """
        log.info("Indexing %d candidates into hybrid index…", len(candidates))
        self.retriever.index_candidates(candidates)
        log.info("Indexing complete.")

    # ── Main ranking entry-point ──────────────────────────────────────────────

    async def rank(self, request: RankingRequest) -> RankingResponse:
        """
        Execute the four-stage pipeline and return a ranked response.

        If `request.candidate_pool` is provided, those candidates are used
        directly (bypassing Pinecone — useful for unit tests and small pools).
        Otherwise, candidates are retrieved from the live Pinecone index.
        """
        t_total_start = time.perf_counter()
        stage_latencies: Dict[str, float] = {}
        jd = request.jd

        # ── If an inline pool was provided, index it on-the-fly ───────────────
        if request.candidate_pool:
            self.retriever.index_candidates(request.candidate_pool)
            total_candidates = len(request.candidate_pool)
        else:
            total_candidates = 0  # unknown from Pinecone

        # ════════════════════════════════════════════════════════════════════
        # STAGE 1 — Hybrid Retrieval
        # ════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        stage1_results = self.retriever.retrieve(
            jd=jd,
            top_k=settings.TOP_K_RETRIEVAL,
            hard_filters=request.hard_filters,
        )
        stage_latencies["stage1_retrieval_ms"] = _ms(t0)

        if not stage1_results:
            log.warning("Stage 1 returned 0 candidates for JD=%s", jd.jd_id)
            return _empty_response(jd.jd_id, stage_latencies, _ms(t_total_start))

        # Preserve retrieval scores for LTR feature building
        bm25_scores = {r.candidate.candidate_id: r.bm25_score for r in stage1_results}
        dense_scores = {r.candidate.candidate_id: r.dense_score for r in stage1_results}
        rrf_scores = {r.candidate.candidate_id: r.rrf_score for r in stage1_results}

        # ════════════════════════════════════════════════════════════════════
        # STAGE 2 — Cross-Encoder Re-ranking
        # ════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        stage2_results = self.reranker.rerank(
            jd=jd,
            candidates=stage1_results,
            top_k=settings.TOP_K_RERANKING,
        )
        stage_latencies["stage2_reranking_ms"] = _ms(t0)

        # ════════════════════════════════════════════════════════════════════
        # STAGE 3 — LambdaMART LTR
        # ════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        stage3_results = self.ltr.rank(
            jd=jd,
            candidates=stage2_results,
            top_k=request.top_k,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            rrf_scores=rrf_scores,
        )
        stage_latencies["stage3_ltr_ms"] = _ms(t0)

        # ════════════════════════════════════════════════════════════════════
        # STAGE 4 — LLM Explainability (async, does not block response)
        # ════════════════════════════════════════════════════════════════════
        summaries: Dict[str, Any] = {}
        if request.include_explanations and stage3_results:
            t0 = time.perf_counter()
            summaries = await batch_generate_summaries(
                ranked_candidates=[(p, fv) for p, _, fv in stage3_results],
                jd=jd,
                max_candidates=min(10, len(stage3_results)),
            )
            stage_latencies["stage4_explanation_ms"] = _ms(t0)

        # ── Assemble final response ───────────────────────────────────────────
        candidate_results: List[CandidateResult] = []
        for rank, (profile, final_score, fv) in enumerate(stage3_results, start=1):
            cid = profile.candidate_id
            explanation = summaries.get(cid, {})

            candidate_results.append(
                CandidateResult(
                    rank=rank,
                    candidate_id=cid,
                    name=profile.name,
                    final_score=round(float(final_score), 4),
                    stage_scores={
                        "bm25": round(bm25_scores.get(cid, 0.0), 4),
                        "dense_cosine": round(dense_scores.get(cid, 0.0), 4),
                        "rrf": round(rrf_scores.get(cid, 0.0), 4),
                        "cross_encoder": round(
                            next(
                                (r.cross_encoder_score for r in stage2_results
                                 if r.candidate.candidate_id == cid),
                                0.0,
                            ),
                            4,
                        ),
                        "archetype": round(fv.archetype_score, 4),
                        "skill_overlap": round(fv.skill_overlap_ratio, 4),
                    },
                    top_features={
                        "cross_encoder_score": fv.cross_encoder_score,
                        "skill_overlap_ratio": fv.skill_overlap_ratio,
                        "archetype_score": fv.archetype_score,
                        "recruiter_response_rate": fv.recruiter_response_rate,
                        "relevant_yoe": fv.relevant_yoe,
                    },
                    fit_summary=explanation.get("summary"),
                    explanation_confidence=explanation.get("confidence"),
                    flags=explanation.get("flags", []),
                )
            )

        total_latency = _ms(t_total_start)
        if total_latency > settings.MAX_SYNC_LATENCY_MS:
            log.warning(
                "SLO breach — sync latency %.1fms > %dms for JD=%s",
                total_latency, settings.MAX_SYNC_LATENCY_MS, jd.jd_id,
            )

        return RankingResponse(
            jd_id=jd.jd_id,
            candidates=candidate_results,
            latency_ms=round(total_latency, 2),
            stage_latencies_ms={k: round(v, 2) for k, v in stage_latencies.items()},
            total_considered=total_candidates or len(stage1_results),
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Expose LTR feature importance for monitoring dashboards."""
        return self.ltr.feature_importance()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms(t_start: float) -> float:
    return (time.perf_counter() - t_start) * 1000


def _empty_response(
    jd_id: str,
    stage_latencies: Dict[str, float],
    latency_ms: float,
) -> RankingResponse:
    return RankingResponse(
        jd_id=jd_id,
        candidates=[],
        latency_ms=latency_ms,
        stage_latencies_ms=stage_latencies,
        total_considered=0,
    )