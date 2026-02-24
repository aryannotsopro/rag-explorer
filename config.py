"""
config.py - Centralized configuration management using Pydantic Settings.

Sources of secrets (in priority order):
  1. Streamlit Cloud Secrets  (when deployed on share.streamlit.io)
  2. .env file                (local development)
  3. OS environment variables (CI / Docker)

Import `get_settings()` anywhere in the project to access typed, validated config.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _inject_streamlit_secrets() -> None:
    """
    If running on Streamlit Cloud, st.secrets contains the app's secrets as a
    dict-like object. Inject them into os.environ so Pydantic Settings can read
    them via its normal env-var mechanism.

    This is a no-op when the `streamlit` package is not installed or when
    `st.secrets` is empty (i.e., in unit tests or local .env usage).
    """
    try:
        import streamlit as st  # noqa: PLC0415

        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key.upper(), value)
    except Exception:
        # Streamlit not installed, no secrets configured, or running outside Streamlit
        pass

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Settings model
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ------------------------------------------------------------------ #
    # API Keys
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(..., description="OpenAI API key")
    pinecone_api_key: str = Field(..., description="Pinecone API key")

    # ------------------------------------------------------------------ #
    # Pinecone settings
    # ------------------------------------------------------------------ #
    pinecone_environment: str = Field(
        default="us-east-1",
        description="Pinecone cloud region/environment (e.g. us-east-1)",
    )
    pinecone_index_name: str = Field(
        default="rag-index",
        description="Name of the Pinecone index to use",
    )
    pinecone_cloud: str = Field(
        default="aws",
        description="Pinecone cloud provider (aws | gcp | azure)",
    )

    # ------------------------------------------------------------------ #
    # OpenAI / Embedding settings
    # ------------------------------------------------------------------ #
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Dimension of the embedding vectors (must match Pinecone index)",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model used for answer generation",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM (0 = deterministic)",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Maximum tokens in the LLM response",
    )

    # ------------------------------------------------------------------ #
    # Chunking / ingestion settings
    # ------------------------------------------------------------------ #
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=8000,
        description="Target character size of each text chunk",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Character overlap between consecutive chunks",
    )

    # ------------------------------------------------------------------ #
    # Retrieval settings
    # ------------------------------------------------------------------ #
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of chunks to retrieve per query",
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine-similarity score to include a result",
    )

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max concurrent OpenAI API requests (semaphore limit)",
    )
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Number of texts to embed in a single API call",
    )
    upsert_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of vectors to upsert to Pinecone in one batch",
    )

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG | INFO | WARNING | ERROR)",
    )
    upload_dir: str = Field(
        default="uploads",
        description="Local directory to temporarily store uploaded files",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_chunk_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}; got '{v}'")
        return v_upper


# --------------------------------------------------------------------------- #
# Singleton accessor — use @lru_cache so Settings is instantiated only once.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (lazily created, then cached)."""
    # Merge Streamlit Cloud secrets into os.environ before Pydantic reads them.
    _inject_streamlit_secrets()
    s = Settings()  # type: ignore[call-arg]
    # Apply the configured log level
    logging.getLogger().setLevel(s.log_level)
    logger.info(
        "Settings loaded | index=%s | embed=%s | llm=%s",
        s.pinecone_index_name,
        s.embedding_model,
        s.llm_model,
    )
    return s


# Convenience alias — most modules can just do: from config import settings
settings = get_settings  # callable, NOT the object, so tests can monkeypatch easily
