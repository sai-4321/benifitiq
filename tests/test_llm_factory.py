"""Bedrock LLM layer tests - no AWS account and no network needed.

Building a boto3 client is local. Real calls are replaced with botocore's Stubber, which
returns canned Bedrock responses, so we test our code's behaviour, not AWS.
"""
from __future__ import annotations

import os

import pytest
from botocore.stub import Stubber
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableWithFallbacks
from pydantic import SecretStr

from app import llm_factory
from app.config import get_settings

SONNET = "us.anthropic.claude-sonnet-5"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def converse_reply(text: str, *, with_reasoning: bool = False) -> dict:
    """A minimal Bedrock Converse API response."""
    content = [{"text": text}]
    if with_reasoning:
        content.insert(0, {"reasoningContent": {"reasoningText": {"text": "thinking...", "signature": "sig"}}})
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1200, "outputTokens": 80, "totalTokens": 1280},
        "metrics": {"latencyMs": 900},
    }


@pytest.fixture(autouse=True)
def _offline_aws(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")       # dummy creds: boto3 needs *something*
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    # setenv + delenv makes monkeypatch remove anything a test exports into os.environ.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_ID", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    for f in (get_settings, llm_factory.get_llm, llm_factory.get_embeddings):
        f.cache_clear()
    yield
    for f in (get_settings, llm_factory.get_llm, llm_factory.get_embeddings):
        f.cache_clear()


# ---------------------------------------------------------------- building models
def test_build_chat_model_uses_converse_with_profile_id():
    from langchain_aws import ChatBedrockConverse

    m = llm_factory.build_chat_model(SONNET)
    assert isinstance(m, ChatBedrockConverse)
    assert (m.model_id, m.region_name) == (SONNET, "us-east-1")
    assert m.max_tokens == get_settings().bedrock_max_tokens


@pytest.mark.parametrize("bad", ["", "none", "NONE"])
def test_empty_model_id_rejected(bad):
    with pytest.raises(llm_factory.ProviderNotConfigured):
        llm_factory.build_chat_model(bad)


def test_get_llm_has_haiku_fallback_by_default():
    llm = llm_factory.get_llm()
    assert isinstance(llm, RunnableWithFallbacks)
    assert llm.fallbacks[0].model_id == HAIKU
    assert "fallback" in llm_factory.active_provider_label()


def test_fallback_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("BEDROCK_FALLBACK_MODEL_ID", "none")
    assert not isinstance(llm_factory.get_llm(), RunnableWithFallbacks)
    assert "fallback" not in llm_factory.active_provider_label()


def test_get_llm_is_built_once():
    assert llm_factory.get_llm() is llm_factory.get_llm()


# ---------------------------------------------------------------- credentials
def test_api_key_from_env_file_is_exported(monkeypatch):
    monkeypatch.setattr(get_settings(), "aws_bearer_token_bedrock", SecretStr("bedrock-api-key-123"))
    llm_factory._export_bedrock_api_key()
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "bedrock-api-key-123"


def test_placeholder_api_key_is_ignored(monkeypatch):
    monkeypatch.setattr(get_settings(), "aws_bearer_token_bedrock", SecretStr("paste-your-bedrock-api-key-here"))
    llm_factory._export_bedrock_api_key()
    assert "AWS_BEARER_TOKEN_BEDROCK" not in os.environ


def test_shell_api_key_wins_over_env_file(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "from-setx")
    monkeypatch.setattr(get_settings(), "aws_bearer_token_bedrock", SecretStr("from-dotenv"))
    llm_factory._export_bedrock_api_key()
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "from-setx"


# ---------------------------------------------------------------- calling Claude (stubbed)
def test_converse_reply_is_parsed_and_reasoning_dropped():
    m = llm_factory.build_chat_model(SONNET)
    with Stubber(m.client) as stub:
        stub.add_response("converse", converse_reply('{"answer": "Part B covers DME."}', with_reasoning=True))
        msg = m.invoke("What covers DME?")
    assert llm_factory.extract_text(msg) == '{"answer": "Part B covers DME."}'
    assert msg.usage_metadata["input_tokens"] == 1200


def test_throttled_sonnet_falls_back_to_haiku():
    primary, fallback = llm_factory.build_chat_model(SONNET), llm_factory.build_chat_model(HAIKU)
    with Stubber(primary.client) as sp, Stubber(fallback.client) as sf:
        for _ in range(3):   # botocore retries throttling before giving up
            sp.add_client_error("converse", "ThrottlingException", "Too many requests", 429)
        sf.add_response("converse", converse_reply("answered by haiku"))
        msg = primary.with_fallbacks([fallback]).invoke("hi")
    assert llm_factory.extract_text(msg) == "answered by haiku"


@pytest.mark.parametrize(
    "content, expected",
    [("plain", "plain"), ([{"type": "text", "text": "a"}, {"type": "reasoning_content"}, "b"], "ab")],
)
def test_extract_text(content, expected):
    assert llm_factory.extract_text(AIMessage(content=content)) == expected


# ---------------------------------------------------------------- embeddings
def test_titan_embeddings_config():
    from langchain_aws import BedrockEmbeddings

    emb = llm_factory.get_embeddings()
    assert isinstance(emb, BedrockEmbeddings)
    assert emb.model_id == "amazon.titan-embed-text-v2:0"
    assert emb.model_kwargs == {"dimensions": 1024}
    assert emb.normalize is True


def test_fake_embeddings_are_deterministic_and_unit_length(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    emb = llm_factory.get_embeddings()
    a, b = emb.embed_query("Part B covers DME"), emb.embed_query("Part B covers DME")
    assert a == b and len(a) == 384
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9
    assert emb.embed_documents(["x", "y"])[0] != emb.embed_documents(["x", "y"])[1]
