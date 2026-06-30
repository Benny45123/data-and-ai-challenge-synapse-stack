"""
utils/embeddings.py — Dense embedding generation.

Wraps sentence-transformers so the rest of the pipeline never imports
the library directly. Supports batched encoding and L2 normalisation
(so dot-product ≈ cosine similarity when querying Pinecone).

Production note: swap DENSE_MODEL_NAME to "BAAI/bge-m3" for the full
sparse+dense+multi-vector model. The interface here is identical.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    log.info("Loading dense embedding model: %s", settings.DENSE_MODEL_NAME)
    return SentenceTransformer(settings.DENSE_MODEL_NAME)


def embed(texts: Union[str, List[str]], normalise: bool = True) -> np.ndarray:
    """
    Encode one or more texts into L2-normalised dense vectors.

    Args:
        texts:     Single string or list of strings.
        normalise: If True, L2-normalise so dot-product == cosine similarity.

    Returns:
        Float32 numpy array of shape (n, EMBEDDING_DIM).
    """
    if isinstance(texts, str):
        texts = [texts]

    model = _load_model()
    vectors: np.ndarray = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if normalise:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)   # avoid div-by-zero
        vectors = vectors / norms

    return vectors.astype(np.float32)


def embed_jd_facets(jd_text: str, required_skills: List[str]) -> np.ndarray:
    """
    Embed a JD as three facets (role, requirements, skills) and average them.

    This prevents the candidate from being penalised for not matching every
    sentence of a multi-topic JD with a single pooled vector.
    """
    sentences = jd_text.split(". ")
    mid = max(1, len(sentences) // 2)

    role_text = " ".join(sentences[:mid])
    req_text = " ".join(sentences[mid:])
    skill_text = " ".join(required_skills)

    facets = embed([role_text, req_text, skill_text], normalise=True)
    combined = np.mean(facets, axis=0)

    # Re-normalise the mean
    norm = np.linalg.norm(combined)
    return (combined / max(norm, 1e-12)).astype(np.float32)