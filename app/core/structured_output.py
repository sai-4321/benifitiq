"""Ask the LLM for JSON, validate it with Pydantic, and retry with feedback if it is wrong.

Flow:  call LLM -> pull out the JSON object -> schema.model_validate()
       if that fails -> show the model its own output + the exact validation error
       -> ask again (up to MAX_JSON_RETRIES extra times) -> otherwise raise.
Transient network/API errors are retried separately with exponential backoff (tenacity).
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.logging_config import get_logger
from app.core.tracing import usage_from_message
from app.llm_factory import extract_text

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """The model could not produce valid JSON after all retries."""


def extract_json(text: str) -> dict:
    """Find the JSON object in a model reply. Handles ```json fences and extra prose."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in model output")
        cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")
    return data


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _invoke(llm, messages):
    return llm.invoke(messages)


def invoke_structured(
    llm,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    node: str,
    max_retries: int | None = None,
) -> tuple[T, list[dict]]:
    """Return (validated_object, usage_records)."""
    retries = get_settings().max_json_retries if max_retries is None else max_retries
    schema_json = json.dumps(schema.model_json_schema(), indent=None)
    messages = [
        SystemMessage(
            content=f"{system_prompt}\n\nReturn ONLY one JSON object (no prose, no markdown) that matches "
            f"this JSON Schema:\n{schema_json}"
        ),
        HumanMessage(content=user_prompt),
    ]
    usage: list[dict] = []
    last_error = ""
    for attempt in range(retries + 1):
        reply = _invoke(llm, messages)
        usage.append(usage_from_message(reply, node))
        raw = extract_text(reply)
        try:
            return schema.model_validate(extract_json(raw)), usage
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)[:800]
            log.warning("[%s] invalid JSON on attempt %d/%d: %s", node, attempt + 1, retries + 1, last_error[:200])
            messages.append(AIMessage(content=raw))
            messages.append(
                HumanMessage(
                    content=f"Your previous reply was not valid. Error:\n{last_error}\n"
                    "Fix it and return ONLY the corrected JSON object."
                )
            )
    raise StructuredOutputError(f"{node}: no valid JSON after {retries + 1} attempts. Last error: {last_error}")
