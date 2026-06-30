"""
utils/sparse_encoder.py — BM25 sparse retrieval.

In production this would be backed by Elasticsearch. For local development
and testing, we maintain an in-memory BM25 index via rank-bm25.

The BM25Index class is designed so that the production swap to Elasticsearch
requires only a new implementation of `BM25Index` behind the same interface.
"""
from __future__ import annotations

import logging
import re
import string
from typing import Dict, List, Tuple

import nltk
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# Download required NLTK data once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

_STOPWORDS = set(nltk.corpus.stopwords.words("english"))
_PUNCT = str.maketrans("", "", string.punctuation)


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = text.lower().translate(_PUNCT).split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """
    In-memory BM25 index over candidate profile texts.

    Thread-safety: build() and query() are read-only after construction.
    Re-building triggers a full corpus scan — call only on data updates.
    """

    def __init__(self) -> None:
        self._corpus_ids: List[str] = []
        self._bm25: BM25Okapi | None = None

    # ── Build / update ────────────────────────────────────────────────────────

    def build(self, documents: Dict[str, str]) -> None:
        """
        (Re)build the BM25 index from a {candidate_id: profile_text} dict.

        Args:
            documents: Mapping from candidate ID to concatenated profile text.
        """
        self._corpus_ids = list(documents.keys())
        tokenised_corpus = [_tokenise(text) for text in documents.values()]
        self._bm25 = BM25Okapi(tokenised_corpus)
        log.info("BM25 index built — %d documents", len(self._corpus_ids))

    def add(self, candidate_id: str, profile_text: str) -> None:
        """
        Incremental add — rebuilds from existing corpus + new document.

        For production scale, use Elasticsearch's bulk-index API instead.
        """
        if self._bm25 is None:
            self.build({candidate_id: profile_text})
            return
        # rank-bm25 does not support incremental updates; we rebuild.
        # In production, this would be an ES index call.
        existing = {
            cid: " ".join(self._bm25.corpus[i])
            for i, cid in enumerate(self._corpus_ids)
        }
        existing[candidate_id] = profile_text
        self.build(existing)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 500) -> List[Tuple[str, float]]:
        """
        Return the top-k candidate IDs ranked by BM25 score.

        Args:
            query_text: JD text (title + description + skills).
            top_k:      Maximum candidates to return.

        Returns:
            List of (candidate_id, bm25_score) sorted descending.
        """
        if self._bm25 is None or not self._corpus_ids:
            log.warning("BM25 index is empty — returning no results")
            return []

        tokens = _tokenise(query_text)
        scores = self._bm25.get_scores(tokens)

        indexed = sorted(
            zip(self._corpus_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return indexed[:top_k]

    @property
    def size(self) -> int:
        return len(self._corpus_ids)