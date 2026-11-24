"""Claims history inside the agents: intent detection, routing, prompt data, citations and PHI safety."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app import pipeline
from app.agents import nodes
from app.agents.claims_intent import claims_query
from app.integrations.claims import summarize
from app.schemas import AnswerJSON

FIX = Path(__file__).parent / "fixtures" / "bluebutton"
EOBS = [e["resource"] for name in ("eob_page1.json", "eob_page2.json")
        for e in json.loads((FIX / name).read_text())["entry"]]
CLAIMS = {**summarize(EOBS, service="mammogram", months=12, today=date(2026, 11, 24)), "available": True}

ROUTE_POLICY = {"intent": "policy", "service": None, "reason": "screening question"}
CLAIMS_ANSWER = {
    "answer": "Yes - [MEMBER_ID_1] had a screening mammogram on 2026-03-14. Part B covers one every 12 months.",
    "citations": [{"chunk_id": "TOOL:get_claims_summary", "source": "get_claims_summary", "page": 0,
                   "quote": "Screening mammography, bilateral"},
                  {"chunk_id": "10110-p15-c1", "source": "10110-Your-Guide-to-Medicare-Preventive-Services.pdf",
                   "page": 15, "quote": "once every 12 months"}],
    "confidence": 0.9, "needs_human_review": True}


@pytest.mark.parametrize("question, expected", [
    ("Has [MEMBER_ID_1] had a mammogram in the last year?", (True, "mammogram", 12)),
    ("Did [MEMBER_ID_1] get a flu shot in the past 6 months?", (True, "flu shot", 6)),
    ("When was [MEMBER_ID_1]'s last colonoscopy?", (True, "colorectal cancer screening", 24)),
    ("Show [MEMBER_ID_1]'s claims history", (True, None, 24)),
    ("How often does Medicare cover a screening mammogram?", (False, None, 24)),      # policy, no history cue
    ("What is the specialist copay for [MEMBER_ID_1]?", (False, None, 24)),
])
def test_claims_query(question, expected):
    assert claims_query(question) == expected


def lookup_with(claims):
    def fake(member_id, service=None, timeout=30.0, claims_args=None):
        fake.args = claims_args
        return {"member": {"member_id": member_id, "plan_id": "PLN-001", "plan_name": "Demo Silver PPO",
                           "status": "active"},
                "benefits": {"plan_id": "PLN-001", "specialist_copay": 50}, "prior_auth": None,
                "claims": claims, "tool_calls": []}
    return fake


def test_history_question_routes_to_lookup_and_cites_claims(wire_graph, monkeypatch):
    fake = lookup_with(CLAIMS)
    monkeypatch.setattr(nodes, "run_member_lookup", fake)
    wire_graph(ROUTE_POLICY, CLAIMS_ANSWER)
    resp, debug = pipeline.ask("Has member M1001 had a mammogram in the last year?")
    assert resp.intent == "both"                                     # upgraded from policy
    assert fake.args == {"months": 12, "service": "mammogram"}
    assert resp.status == "pending_approval"                         # member-specific -> human review
    assert any(c.chunk_id == "TOOL:get_claims_summary" for c in resp.answer.citations)


def test_claims_in_prompt_have_no_identifiers(wire_graph, monkeypatch):
    monkeypatch.setattr(nodes, "run_member_lookup", lookup_with(CLAIMS))
    llm = wire_graph(ROUTE_POLICY, CLAIMS_ANSWER)
    seen = []
    original = type(llm).invoke

    def spy(self, messages, *a, **k):
        seen.extend(getattr(m, "content", str(m)) for m in messages)
        return original(self, messages, *a, **k)

    object.__setattr__(llm, "invoke", spy.__get__(llm))
    pipeline.ask("Has member M1003 had a mammogram in the last year?")    # M1001 appears in the router prompt
    sent = "\n".join(seen)
    assert "2026-03-14" in sent and "TOOL:get_claims_summary" in sent
    for leaked in ("M1003", "-10000000000001", "9999999999", "123.45"):
        assert leaked not in sent


def test_unconnected_member_gets_clear_note_not_failure(wire_graph, monkeypatch):
    monkeypatch.setattr(nodes, "run_member_lookup", lookup_with({"available": False, "reason": "not_connected"}))
    wire_graph(ROUTE_POLICY, CLAIMS_ANSWER)
    resp, _ = pipeline.ask("Has member M1001 had a mammogram in the last year?")
    assert resp.status in ("pending_approval", "completed")
    state = {"tool_results": lookup_with({"available": False, "reason": "not_connected"})("M1001"),
             "tool_error": None}
    assert "has not connected Medicare claims" in nodes._format_member_data(state)
    # claims citation is not valid when claims were unavailable
    answer, dropped = nodes.validate_citations(AnswerJSON.model_validate(CLAIMS_ANSWER),
                                               {**state, "retrieved": []})
    assert all(c.chunk_id != "TOOL:get_claims_summary" for c in answer.citations) and dropped >= 1


def test_plain_member_question_does_not_read_claims(wire_graph, monkeypatch):
    fake = lookup_with(None)
    monkeypatch.setattr(nodes, "run_member_lookup", fake)
    wire_graph({"intent": "member", "service": None, "reason": ""},
               {"answer": "Specialist copay is $50.", "citations": [{"chunk_id": "TOOL:get_plan_benefits",
                "source": "get_plan_benefits", "page": 0, "quote": "50"}], "confidence": 0.9,
                "needs_human_review": False})
    pipeline.ask("What is the specialist copay for member M1001?")
    assert fake.args is None


def test_real_mcp_lookup_survives_unconnected_member(member_db):
    """Through the real MCP subprocess: no token stored -> claims unavailable, plan lookup still succeeds."""
    from app.agents.mcp_client import run_member_lookup

    result = run_member_lookup("M1002", claims_args={"months": 12, "service": "mammogram"})
    assert result["member"]["plan_id"] == "PLN-002"
    assert result["claims"] == {"available": False, "reason": "not_connected"}
