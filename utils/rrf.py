"""
utils/rrf.py — Reciprocal Rank Fusion.

RRF is score-scale-invariant and requires no learned fusion weights, making it
the correct choice for blending BM25 and dense retrieval scores whose raw
magnitudes are not comparable.

    RRF(d) = Σ_r  1 / (k + rank_r(d))    k = 60 (standard)

Reference: Cormack, Clarke & Buettcher (2009).
"""
from __future__ import annotations
from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Fuse an arbitrary number of ranked ID lists via RRF.

    Args:
        ranked_lists: Each inner list is a ranking of candidate IDs, most
                      relevant first.
        k:            RRF constant. k=60 is the empirically validated default.

    Returns:
        Sorted list of (candidate_id, rrf_score) descending by score.
    """
    scores: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, candidate_id in enumerate(ranking, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_bm25_and_dense(
    bm25_ids: List[str],
    dense_ids: List[str],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Convenience wrapper for the common two-list case."""
    return reciprocal_rank_fusion([bm25_ids, dense_ids], k=k)