"""
stages/stage1_retrieval.py — Hybrid Retrieval (< 20ms target).

Architecture:
    JD text → [BM25 sparse retrieval] ──┐
                                         ├── RRF fusion → Top 500 candidates
    JD text → [Pinecone cosine dense] ──┘

Design decisions (Section 4 of spec):
  - BM25 enforces hard keyword presence for exact technical skills (FAISS, NDCG).
  - Dense retrieval generalises over synonyms and paraphrases.
  - RRF is scale-invariant: no learned weights, robust to BM25 outliers.
  - Pinecone hard filters run server-side before ANN — no post-filtering latency.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from config import get_settings
from models.schemas import CandidateProfile, JobDescription, RetrievedCandidate
from utils.embeddings import embed_jd_facets
from utils.pinecone_client import dense_query
from utils.rrf import fuse_bm25_and_dense
from utils.sparse_encoder import BM25Index

log = logging.getLogger(__name__)
settings = get_settings()


class HybridRetriever:
    """
    Manages both the in-memory BM25 index and the Pinecone dense index.

    Lifecycle:
        1. Call `index_candidates()` once (or on update) to populate both indexes.
        2. Call `retrieve()` per ranking request.
    """

    def __init__(self) -> None:
        self._bm25 = BM25Index()
        # In-memory lookup: candidate_id → CandidateProfile (for the pipeline)
        self._profile_store: Dict[str, CandidateProfile] = {}

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_candidates(self, candidates: List[CandidateProfile]) -> None:
        """
        Populate BM25 index and Pinecone.

        Pinecone upsert is handled separately (see /admin/upsert endpoint),
        so this method only builds the BM25 side and the local profile store.
        Call this on startup when loading a candidate pool from the database.
        """
        import numpy as np
        from utils.embeddings import embed
        from utils.pinecone_client import upsert_candidates

        texts = {c.candidate_id: c.profile_text for c in candidates}
        self._bm25.build(texts)
        self._profile_store = {c.candidate_id: c for c in candidates}

        # Batch-embed and upsert to Pinecone
        ids = [c.candidate_id for c in candidates]
        embeddings = embed([c.profile_text for c in candidates])
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
        log.info("Indexed %d candidates (%d failed)", upserted, failed)

    def add_candidate(self, candidate: CandidateProfile) -> None:
        """Incremental single-candidate index update."""
        from utils.embeddings import embed
        from utils.pinecone_client import upsert_candidates

        self._bm25.add(candidate.candidate_id, candidate.profile_text)
        self._profile_store[candidate.candidate_id] = candidate

        emb = embed(candidate.profile_text)
        upsert_candidates(
            [candidate.candidate_id],
            emb,
            [{"name": candidate.name, "total_yoe": candidate.total_yoe,
              "open_to_remote": candidate.open_to_remote,
              "duplicate_flag": candidate.duplicate_flag,
              "location": candidate.location or ""}],
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        jd: JobDescription,
        top_k: int | None = None,
        hard_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedCandidate]:
        """
        Run hybrid retrieval for a JD.

        Args:
            jd:           The job description to search against.
            top_k:        How many candidates to return (default: settings.TOP_K_RETRIEVAL).
            hard_filters: Pinecone metadata filter dict applied server-side.

        Returns:
            List of RetrievedCandidate sorted by RRF score, descending.
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # ── Build Pinecone filter (hard constraints, Section 4.3) ─────────────
        pinecone_filter = self._build_pinecone_filter(jd, hard_filters)

        # ── BM25 sparse retrieval ─────────────────────────────────────────────
        bm25_results: List[Tuple[str, float]] = self._bm25.query(
            jd.full_text, top_k=top_k
        )
        bm25_ids = [cid for cid, _ in bm25_results]
        bm25_score_map = {cid: score for cid, score in bm25_results}

        # ── Dense retrieval from Pinecone ─────────────────────────────────────
        query_vector = embed_jd_facets(jd.full_text, jd.required_skills)
        dense_results: List[Tuple[str, float]] = dense_query(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=pinecone_filter,
        )
        dense_ids = [cid for cid, _ in dense_results]
        dense_score_map = {cid: score for cid, score in dense_results}

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = fuse_bm25_and_dense(bm25_ids, dense_ids, k=settings.RRF_K)

        # Build lookup maps for rank positions
        bm25_rank_map = {cid: r for r, (cid, _) in enumerate(bm25_results, 1)}
        dense_rank_map = {cid: r for r, (cid, _) in enumerate(dense_results, 1)}

        # ── Assemble output ───────────────────────────────────────────────────
        retrieved: List[RetrievedCandidate] = []
        for candidate_id, rrf_score in fused[:top_k]:
            profile = self._profile_store.get(candidate_id)
            if profile is None:
                # Profile not in local store — skip (stale Pinecone entry)
                continue
            if profile.duplicate_flag:
                continue

            retrieved.append(
                RetrievedCandidate(
                    candidate=profile,
                    bm25_rank=bm25_rank_map.get(candidate_id),
                    dense_rank=dense_rank_map.get(candidate_id),
                    bm25_score=bm25_score_map.get(candidate_id, 0.0),
                    dense_score=dense_score_map.get(candidate_id, 0.0),
                    rrf_score=rrf_score,
                )
            )

        log.info(
            "Stage 1 — JD=%s: BM25=%d, Dense=%d, Fused=%d → returning %d",
            jd.jd_id, len(bm25_ids), len(dense_ids), len(fused), len(retrieved),
        )
        return retrieved

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_pinecone_filter(
        jd: JobDescription,
        extra_filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Translate hard constraints into a Pinecone metadata filter.

        These are applied server-side (payload filtering) before ANN scoring,
        so they add no retrieval latency.
        """
        must: List[Dict] = [
            {"duplicate_flag": {"$eq": False}},
        ]
        if jd.min_yoe > 0 or jd.max_yoe < 99:
            # Small buffer around the stated band (±2 years)
            must.append({
                "total_yoe": {
                    "$gte": max(0, jd.min_yoe - 1),
                    "$lte": jd.max_yoe + 2,
                }
            })
        if jd.remote_ok is False and jd.location:
            must.append({"location": {"$eq": jd.location}})

        if extra_filters:
            must.append(extra_filters)

        return {"$and": must} if len(must) > 1 else must[0] if must else None