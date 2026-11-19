"""Response cache and token/cost accounting tests."""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage
from pydantic import SecretStr

from app.config import get_settings
from app.core.cache import ResponseCache, make_key
from app.core.tracing import estimate_cost, redact_inputs, setup_tracing, sum_usage, usage_from_message


def test_cache_key_ignores_case_and_spacing_but_not_settings():
    k = make_key("What does  Part B cover?", "bedrock:sonnet", "v1", 5)
    assert k == make_key("what does part b cover?", "bedrock:sonnet", "v1", 5)
    assert k != make_key("what does part b cover?", "bedrock:haiku", "v1", 5)     # model changed
    assert k != make_key("what does part b cover?", "bedrock:sonnet", "v2", 5)    # prompt changed
    assert k != make_key("what does part b cover?", "bedrock:sonnet", "v1", 8)    # retrieval changed


def test_cache_roundtrip_and_ttl(tmp_path):
    cache = ResponseCache(str(tmp_path), ttl_seconds=3600)
    cache.set("k", {"answer": "cached"})
    assert cache.get("k") == {"answer": "cached"} and len(cache) == 1
    cache.clear()
    assert cache.get("k") is None
    expired = ResponseCache(str(tmp_path / "x"), ttl_seconds=-1)
    expired.set("k", {"a": 1})
    assert expired.get("k") is None


def test_cost_matches_inference_profile_ids():
    assert estimate_cost("us.anthropic.claude-sonnet-5", 1_000_000, 1_000_000) == 12.0
    assert estimate_cost("global.anthropic.claude-sonnet-5", 1_000_000, 0) == 2.0
    assert estimate_cost("us.anthropic.claude-haiku-4-5-20251001-v1:0", 1_000_000, 1_000_000) == 6.0
    assert estimate_cost("us.anthropic.claude-sonnet-5-5", 1_000_000, 0) == 2.0   # longest match wins
    assert estimate_cost("unknown-model", 10, 10) == 0.0 and estimate_cost(None, 10, 10) == 0.0


def test_usage_from_bedrock_message_and_sum():
    msg = AIMessage(content="hi", usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
                    response_metadata={"model_name": "us.anthropic.claude-sonnet-5"})
    rec = usage_from_message(msg, "answer")
    assert rec["model"] == "us.anthropic.claude-sonnet-5"
    assert rec["cost_usd"] == round(1000 / 1e6 * 2 + 200 / 1e6 * 10, 6)
    total = sum_usage([rec, rec])
    assert total["input_tokens"] == 2000 and total["cost_usd"] == round(rec["cost_usd"] * 2, 6)


def test_tracing_off_without_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    s = get_settings()
    monkeypatch.setattr(s, "langsmith_tracing", True)
    monkeypatch.setattr(s, "langsmith_api_key", SecretStr("paste-your-langsmith-key-here"))
    assert setup_tracing() is False and os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_on_exports_env(monkeypatch):
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
        monkeypatch.setenv(k, "x")
    s = get_settings()
    monkeypatch.setattr(s, "langsmith_tracing", True)
    monkeypatch.setattr(s, "langsmith_api_key", SecretStr("ls-test-key"))
    assert setup_tracing() is True
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key" and os.environ["LANGSMITH_PROJECT"] == "benefitiq"
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_raw_question_is_redacted_from_traces():
    assert redact_inputs({"text": "Patient John Smith SSN 123-45-6789"}) == {"input": "<redacted>"}
