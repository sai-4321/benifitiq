"""Settings tests: defaults, env overrides, validation and secret handling."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings, get_settings


def make(**env) -> Settings:
    # _env_file=None: ignore any local .env so tests behave the same on every machine.
    return Settings(_env_file=None, **env)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("AWS_REGION", "BEDROCK_EMBED_DIMENSIONS", "LOG_LEVEL", "EMBEDDING_PROVIDER",
                "BEDROCK_FALLBACK_MODEL_ID", "HITL_CONFIDENCE_THRESHOLD"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_point_to_claude_on_bedrock():
    s = make()
    assert s.aws_region == "us-east-1"
    assert s.bedrock_chat_model_id == "us.anthropic.claude-sonnet-5"
    assert "haiku" in s.bedrock_fallback_model_id
    assert s.bedrock_embed_model_id == "amazon.titan-embed-text-v2:0"
    assert s.bedrock_embed_dimensions == 1024
    assert s.fallback_enabled


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("BEDROCK_EMBED_DIMENSIONS", "512")   # arrives as a string
    monkeypatch.setenv("LOG_LEVEL", "debug")                # normalised to upper case
    s = make()
    assert (s.aws_region, s.bedrock_embed_dimensions, s.log_level) == ("us-west-2", 512, "DEBUG")


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


@pytest.mark.parametrize(
    "field, bad",
    [
        ("bedrock_embed_dimensions", 768),        # Titan V2 only supports 256/512/1024
        ("embedding_provider", "openai"),          # Bedrock only
        ("hitl_confidence_threshold", 1.5),        # must be 0..1
        ("retrieval_top_k", 0),
        ("log_level", "LOUD"),
        ("bedrock_chat_model_id", "none"),         # main model is required
    ],
)
def test_invalid_values_fail_fast(field, bad):
    with pytest.raises(ValidationError) as exc:
        make(**{field: bad})
    assert field in str(exc.value)


def test_fallback_can_be_disabled():
    assert not make(bedrock_fallback_model_id="none").fallback_enabled
    assert not make(bedrock_fallback_model_id="us.anthropic.claude-sonnet-5").fallback_enabled


def test_secrets_never_printed():
    s = make(aws_bearer_token_bedrock="bedrock-api-key-SECRET", langsmith_api_key="ls-SECRET")
    assert "SECRET" not in repr(s)
    assert "SECRET" not in str(s.model_dump())
    assert s.aws_bearer_token_bedrock.get_secret_value() == "bedrock-api-key-SECRET"


def test_relative_paths_resolve_under_project_root():
    s = make(chroma_dir="data/chroma")
    assert s.chroma_path == PROJECT_ROOT / "data" / "chroma"
    assert s.chunks_file.name == "chunks.jsonl"


def test_env_example_documents_every_setting():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").upper()
    undocumented = [name for name in Settings.model_fields if name.upper() + "=" not in example]
    assert undocumented == [], f"add these to .env.example: {undocumented}"
