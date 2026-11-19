"""Creates the Claude chat model and the Titan embedding model on Amazon Bedrock.

get_llm() returns Claude Sonnet wrapped with .with_fallbacks([Claude Haiku]) so that if
the main model raises an error (throttling, timeout, model not enabled, outage)
LangChain automatically re-sends the same request to the smaller, cheaper Claude model.

Credentials are never read here. boto3 finds them on its own, in this order:
AWS_BEARER_TOKEN_BEDROCK (Bedrock API key) -> AWS_ACCESS_KEY_ID/SECRET -> AWS_PROFILE
-> ~/.aws/credentials from "aws configure" -> IAM role (when running on AWS).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings
from app.core.logging_config import get_logger

log = get_logger(__name__)


class ProviderNotConfigured(RuntimeError):
    """Raised when a Bedrock setting is missing or invalid."""


def _boto_config():
    """Timeouts + retry policy for every Bedrock call (botocore Config object)."""
    from botocore.config import Config

    s = get_settings()
    return Config(
        region_name=s.aws_region,
        connect_timeout=10,
        read_timeout=s.llm_timeout_seconds,
        # "adaptive" retry backs off automatically on ThrottlingException (HTTP 429).
        retries={"max_attempts": 2, "mode": "adaptive"},
    )


def _is_set(value: str) -> bool:
    """True for a real value; False for empty or the .env.example placeholder."""
    return bool(value) and "paste-your" not in value


def _export_bedrock_api_key() -> None:
    """boto3 reads AWS_BEARER_TOKEN_BEDROCK from the real environment, not from .env.

    pydantic-settings loads .env into our Settings object only, so copy the key across.
    A value already set in Windows (setx) or by Docker wins over the .env value.
    """
    key = get_settings().aws_bearer_token_bedrock.get_secret_value()
    if _is_set(key):
        os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", key)


def _common_kwargs() -> dict:
    _export_bedrock_api_key()
    s = get_settings()
    kwargs: dict = {"region_name": s.aws_region, "config": _boto_config()}
    if s.aws_profile:
        kwargs["credentials_profile_name"] = s.aws_profile
    return kwargs


# ---------------------------------------------------------------- chat models
def build_chat_model(model_id: str) -> BaseChatModel:
    """One Claude model on Bedrock, called through the Converse API."""
    if not model_id or model_id.strip().lower() == "none":
        raise ProviderNotConfigured("Bedrock model ID is empty")
    from langchain_aws import ChatBedrockConverse

    s = get_settings()
    # No temperature: recent Claude models may think adaptively and pick their own sampling.
    # Determinism comes from the strict JSON schema + validation, not from temperature=0.
    return ChatBedrockConverse(model_id=model_id, max_tokens=s.bedrock_max_tokens, **_common_kwargs())


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Main Claude model with automatic Claude fallback. Cached: built once per process."""
    s = get_settings()
    primary = build_chat_model(s.bedrock_chat_model_id)
    if not s.fallback_enabled:
        log.info("LLM: %s on Bedrock (no fallback)", s.bedrock_chat_model_id)
        return primary
    fallback = build_chat_model(s.bedrock_fallback_model_id)
    log.info("LLM: %s -> fallback %s (Bedrock %s)", s.bedrock_chat_model_id, s.bedrock_fallback_model_id, s.aws_region)
    return primary.with_fallbacks([fallback])  # type: ignore[return-value]


def active_provider_label() -> str:
    s = get_settings()
    label = f"bedrock:{s.bedrock_chat_model_id}"
    if s.fallback_enabled:
        label += f" (fallback: {s.bedrock_fallback_model_id})"
    return label


# ---------------------------------------------------------------- embeddings
class HashEmbeddings(Embeddings):
    """Offline, free, deterministic embeddings (feature hashing of words).

    NOT semantic - only used for unit tests, CI, and a no-key smoke test.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embedding_provider == "fake":
        return HashEmbeddings()
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(
        model_id=s.bedrock_embed_model_id,
        # normalize=True -> unit-length vectors, so cosine similarity in Chroma behaves well.
        normalize=True,
        model_kwargs={"dimensions": s.bedrock_embed_dimensions},
        **_common_kwargs(),
    )


def extract_text(message) -> str:
    """Plain text from an AIMessage.

    Claude on the Converse API can return a *list* of content blocks (text, and reasoning when it
    thinks). Keep only the text blocks so reasoning never leaks into answers or the JSON parser.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)
