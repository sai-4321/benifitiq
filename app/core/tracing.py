"""LangSmith tracing setup + token/cost accounting.

LangChain and LangGraph send traces to LangSmith automatically when the
LANGSMITH_* environment variables are set. pydantic-settings reads .env but does
not put values into os.environ, so setup_tracing() copies them there.
Functions that are not LangChain objects (guardrails, retrieval) are traced
with the @traceable decorator from langsmith.
"""
from __future__ import annotations

import os

from app.config import get_settings
from app.core.logging_config import get_logger

log = get_logger(__name__)

# Approximate USD price per 1M tokens (input, output) for on-demand Bedrock usage.
# Check https://aws.amazon.com/bedrock/pricing/ for your region. Keys are matched as a
# substring of the model ID, so "us.anthropic.claude-sonnet-5" and
# "global.anthropic.claude-sonnet-5" both hit "claude-sonnet-5". Unknown models count as $0.
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-5-5": (5.00, 25.00),
    "titan-embed-text-v2": (0.02, 0.0),
}


def setup_tracing() -> bool:
    """Export LangSmith settings to environment variables. Returns True if tracing is on."""
    s = get_settings()
    key = s.langsmith_api_key.get_secret_value()
    enabled = bool(s.langsmith_tracing and key and "paste-your" not in key)
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
        log.info("LangSmith tracing ON -> project '%s'", s.langsmith_project)
    else:
        log.info("LangSmith tracing OFF (set LANGSMITH_TRACING=true and a key to enable)")
    return enabled


def estimate_cost(model_name: str | None, input_tokens: int, output_tokens: int) -> float:
    """Dollar estimate for one LLM call. Matches by substring, longest match wins."""
    if not model_name:
        return 0.0
    name = model_name.lower()
    best = ""
    for key in PRICES_PER_1M:
        if key in name and len(key) > len(best):
            best = key
    if not best:
        return 0.0
    p_in, p_out = PRICES_PER_1M[best]
    return round(input_tokens / 1_000_000 * p_in + output_tokens / 1_000_000 * p_out, 6)


def usage_from_message(msg, node: str) -> dict:
    """Read token counts from a LangChain AIMessage and turn them into a usage record."""
    meta = getattr(msg, "usage_metadata", None) or {}
    resp = getattr(msg, "response_metadata", None) or {}
    model = resp.get("model_name") or resp.get("model_id") or resp.get("model") or ""
    in_tok = int(meta.get("input_tokens", 0) or 0)
    out_tok = int(meta.get("output_tokens", 0) or 0)
    return {
        "node": node,
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": estimate_cost(model, in_tok, out_tok),
    }


def redact_inputs(_inputs: dict) -> dict:
    """Used as @traceable(process_inputs=...) on functions that receive raw user text, so the
    unmasked question never reaches LangSmith."""
    return {"input": "<redacted>"}


def sum_usage(records: list[dict]) -> dict:
    return {
        "input_tokens": sum(r.get("input_tokens", 0) for r in records),
        "output_tokens": sum(r.get("output_tokens", 0) for r in records),
        "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in records), 6),
    }
