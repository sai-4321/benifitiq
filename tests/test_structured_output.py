"""Structured output tests with a scripted fake LLM: valid, self-corrected and failed JSON."""
from __future__ import annotations

import pytest

from app.core.structured_output import StructuredOutputError, extract_json, invoke_structured
from app.schemas import AnswerJSON, RouteDecision
from tests.conftest import make_llm

VALID = {"answer": "Part B covers doctor visits.", "citations": [
    {"chunk_id": "10050-p21-c0", "source": "10050-Medicare-and-You.pdf", "page": 21, "quote": "doctors"}],
    "confidence": 0.9, "needs_human_review": False}


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json at all")


def test_valid_json_passes_first_time():
    llm = make_llm(VALID)
    result, usage = invoke_structured(llm, "sys", "user", AnswerJSON, node="t")
    assert result.confidence == 0.9
    assert len(usage) == 1


def test_broken_json_is_retried_and_fixed():
    llm = make_llm("this is not json", {"intent": "nonsense"}, {"intent": "policy", "service": None})
    result, usage = invoke_structured(llm, "sys", "user", RouteDecision, node="t", max_retries=2)
    assert result.intent == "policy"
    assert len(usage) == 3  # two failures + one success


def test_gives_up_after_max_retries():
    llm = make_llm("bad", "still bad")
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, "sys", "user", AnswerJSON, node="t", max_retries=1)


def test_schema_rejects_out_of_range_confidence():
    llm = make_llm({**VALID, "confidence": 1.7}, VALID)
    result, usage = invoke_structured(llm, "sys", "user", AnswerJSON, node="t", max_retries=1)
    assert result.confidence == 0.9 and len(usage) == 2


def test_error_feedback_is_sent_back_to_the_model():
    llm = make_llm("oops", VALID)
    seen = []
    original = llm.invoke

    def spy(messages, *a, **k):
        seen.append(list(messages))
        return original(messages, *a, **k)

    object.__setattr__(llm, "invoke", spy)
    invoke_structured(llm, "sys", "user", AnswerJSON, node="t", max_retries=1)
    second_call = seen[1]
    assert second_call[-2].content == "oops"                       # model sees its own bad output
    assert "not valid" in second_call[-1].content                  # ...and the error
    assert "JSON Schema" in second_call[0].content                 # schema is in the system prompt


def test_api_contracts_validate_member_id():
    from pydantic import ValidationError

    from app.schemas import AskRequest

    assert AskRequest(question="Does M1002 need prior auth?", member_id="M1002").member_id == "M1002"
    with pytest.raises(ValidationError):
        AskRequest(question="hi there", member_id="1002; DROP TABLE")
