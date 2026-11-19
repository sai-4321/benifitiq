"""Central, typed settings for BenefitIQ.

Every module calls get_settings() instead of reading os.environ directly, so:
  * a typo or bad value in .env fails loudly at startup, in one place;
  * secrets are SecretStr and never appear in logs, tracebacks or repr();
  * tests can override any value with an environment variable.

Precedence (highest first): real environment variables -> .env file -> defaults below.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains the "app" package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Everything runs on Amazon Bedrock. "fake" embeddings exist only for tests/CI (no AWS calls).
EmbeddingProvider = Literal["bedrock", "fake"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
TITAN_V2_DIMENSIONS = (256, 512, 1024)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",          # unknown keys in .env are ignored instead of crashing
        case_sensitive=False,    # AWS_REGION and aws_region both work
    )

    # --- Amazon Bedrock (Claude + Titan) ---
    # boto3 finds credentials itself: AWS_BEARER_TOKEN_BEDROCK (Bedrock API key), "aws configure",
    # AWS_PROFILE, or an IAM role. The optional key below is only a convenience for .env users.
    aws_region: str = "us-east-1"
    aws_profile: str = ""
    aws_bearer_token_bedrock: SecretStr = SecretStr("")
    # Cross-region inference profile IDs ("us." prefix). Newer Claude models on Bedrock must be
    # called through an inference profile, not the bare model ID.
    bedrock_chat_model_id: str = "us.anthropic.claude-sonnet-5"
    bedrock_fallback_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # "none" = off
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_embed_dimensions: int = 1024
    bedrock_max_tokens: int = Field(default=2048, ge=256, le=8192)
    embedding_provider: EmbeddingProvider = "bedrock"

    # --- LangSmith tracing ---
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "benefitiq"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- App behaviour ---
    chroma_dir: str = "data/chroma"
    member_db_path: str = "data/synthetic/members.db"
    cache_dir: str = "data/cache"
    cache_ttl_seconds: int = Field(default=86400, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    hitl_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_json_retries: int = Field(default=2, ge=0, le=5)
    injection_llm_check: bool = False
    llm_timeout_seconds: int = Field(default=60, ge=5, le=300)
    log_level: LogLevel = "INFO"
    api_url: str = "http://localhost:8000"

    # ---------- validators ----------
    @field_validator("bedrock_embed_dimensions")
    @classmethod
    def _titan_dimensions(cls, v: int) -> int:
        # Not Literal[256, 512, 1024]: values from .env arrive as strings and Literal ints don't coerce.
        if v not in TITAN_V2_DIMENSIONS:
            raise ValueError(f"must be one of {TITAN_V2_DIMENSIONS} for Titan Text Embeddings V2")
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    @field_validator("bedrock_chat_model_id")
    @classmethod
    def _chat_model_not_empty(cls, v: str) -> str:
        if not v.strip() or v.strip().lower() == "none":
            raise ValueError("a main Claude model ID is required")
        return v.strip()

    # ---------- helpers ----------
    @property
    def fallback_enabled(self) -> bool:
        fb = self.bedrock_fallback_model_id.strip().lower()
        return fb not in ("", "none") and fb != self.bedrock_chat_model_id.lower()

    def resolve(self, relative: str) -> Path:
        """Turn a relative path from .env into an absolute path under the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def chroma_path(self) -> Path:
        return self.resolve(self.chroma_dir)

    @property
    def member_db(self) -> Path:
        return self.resolve(self.member_db_path)

    @property
    def cache_path(self) -> Path:
        return self.resolve(self.cache_dir)

    @property
    def raw_docs_path(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def chunks_file(self) -> Path:
        return self.chroma_path / "chunks.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the one shared Settings object (created on first call, then cached)."""
    return Settings()
