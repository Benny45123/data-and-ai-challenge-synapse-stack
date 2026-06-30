"""
utils/pinecone_client.py — Pinecone dense retrieval and index management.

Index metric: cosine (L2-normalised vectors, so dot-product ≡ cosine).
The client exposes upsert and query. Every other stage interacts through
this module, never touching the Pinecone SDK directly.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_index():
    """Lazily initialise the Pinecone index (one connection per process)."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    existing = {idx.name for idx in pc.list_indexes()}
    if settings.PINECONE_INDEX_NAME not in existing:
        log.info("Creating Pinecone index '%s'", settings.PINECONE_INDEX_NAME)
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIM,
            metric="cosine",                # cosine similarity
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.PINECONE_REGION,
            ),
        )
    return pc.Index(settings.PINECONE_INDEX_NAME)


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_candidates(
    candidate_ids: List[str],
    embeddings: np.ndarray,                 # shape (n, EMBEDDING_DIM)
    metadata_list: List[Dict[str, Any]],
    batch_size: int = 100,
    namespace: str = "candidates",
) -> Tuple[int, int]:
    """
    Upsert candidate embeddings into Pinecone.

    Returns:
        (upserted_count, failed_count)
    """
    index = _get_index()
    upserted, failed = 0, 0

    vectors = [
        {
            "id": cid,
            "values": emb.tolist(),
            "metadata": meta,
        }
        for cid, emb, meta in zip(candidate_ids, embeddings, metadata_list)
    ]

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            upserted += len(batch)
        except Exception as exc:
            log.error("Pinecone upsert batch failed: %s", exc)
            failed += len(batch)

    return upserted, failed


# ── Query ─────────────────────────────────────────────────────────────────────

def dense_query(
    query_vector: np.ndarray,               # shape (EMBEDDING_DIM,)
    top_k: int = 500,
    namespace: str = "candidates",
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, float]]:
    """
    Run a cosine-similarity nearest-neighbour query against the Pinecone index.

    Args:
        query_vector: L2-normalised query embedding.
        top_k:        Number of results to retrieve.
        filter_dict:  Pinecone metadata filter (applied server-side, no latency cost).

    Returns:
        List of (candidate_id, cosine_score) sorted descending by score.
    """
    index = _get_index()

    kwargs: Dict[str, Any] = {
        "vector": query_vector.tolist(),
        "top_k": top_k,
        "namespace": namespace,
        "include_metadata": True,
    }
    if filter_dict:
        kwargs["filter"] = filter_dict

    response = index.query(**kwargs)

    return [(match.id, match.score) for match in response.matches]


def delete_candidates(
    candidate_ids: List[str],
    namespace: str = "candidates",
) -> None:
    """Remove stale or duplicate profiles from the index."""
    index = _get_index()
    index.delete(ids=candidate_ids, namespace=namespace)
    log.info("Deleted %d candidates from Pinecone", len(candidate_ids))


def index_stats(namespace: str = "candidates") -> Dict[str, Any]:
    """Return index statistics for monitoring dashboards."""
    return _get_index().describe_index_stats().to_dict()