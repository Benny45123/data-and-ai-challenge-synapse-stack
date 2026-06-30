"""
config.py — Centralised settings loaded from environment / .env file.
All tuneable constants live here; nothing is hardcoded in stage modules.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Vector Store ──────────────────────────────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "redrob-candidates"
    PINECONE_REGION: str = "us-east-1"

    # ── LLM ───────────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str
    LLM_MODEL: str = "claude-haiku-4-5"
    LLM_MAX_TOKENS: int = 512

    # ── Embedding / Cross-encoder models ──────────────────────────────────────
    DENSE_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_DIM: int = 384            # BGE-small; swap to 1024 for BGE-M3

    # ── Pipeline thresholds ───────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 500          # Stage 1 output
    TOP_K_RERANKING: int = 100          # Stage 2 output
    TOP_K_LTR: int = 25                 # Stage 3 / final output
    RRF_K: int = 60                     # RRF constant (standard value)
    CROSS_ENCODER_BATCH_SIZE: int = 32
    CE_MAX_TOKENS: int = 512            # JD summary + profile truncation limit

    # ── Caching ───────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    FEATURE_CACHE_TTL_SECONDS: int = 300

    # ── LTR model ─────────────────────────────────────────────────────────────
    LTR_MODEL_PATH: str = "models/ltr_model.pkl"

    # ── Fairness ──────────────────────────────────────────────────────────────
    MAX_EXPOSURE_GAP: float = 0.05      # Exposure parity tolerance ε
    EXPLORATION_FRACTION: float = 0.05  # Random exploration injection

    # ── SLOs ──────────────────────────────────────────────────────────────────
    MAX_SYNC_LATENCY_MS: int = 200


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()