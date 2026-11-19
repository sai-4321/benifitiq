"""Graph-level tests: right path per intent, HITL pause/resume, and failure fallbacks."""
from __future__ import annotations

from langgraph.types import Command

from app.agents.graph import build_graph

POLICY_ANSWER = {
    "answer": "Medicare covers screening mammograms once every 12 months for women 40 and older.",
    "citations": [{"chunk_id": "10110-p15-c1", "source": "x", "page": 1,
                   "quote": "screening mammograms once every 12 months"}],
    "confidence": 0.92, "needs_human_review": False,
}
MEMBER_ANSWER = {
    "answer": "Member [MEMBER_ID_1] is on Demo Gold HMO; an MRI requires prior authorization.",
    "citations": [{"chunk_id": "TOOL:check_prior_auth", "source": "check_prior_auth", "page": 0,
                   "quote": "Prior authorization required"}],
    "confidence": 0.95, "needs_human_review": True,
}


def run(graph, question, member_id=None, thread="t1"):
    return graph.invoke({"question": question, "member_id": member_id},
                        {"configurable": {"thread_id": thread}})


def test_policy_route(wire_graph):
    wire_graph({"intent": "policy", "service": None, "reason": "general"}, POLICY_ANSWER)
    state = run(build_graph(), "How often are mammograms covered?")
    assert state["trace"] == ["router", "retriever", "summarizer", "human_review(skipped)"]
    assert state["final"]["citations"][0]["source"].startswith("10110")  # source fixed from real chunk
    assert "__interrupt__" not in state


def test_out_of_scope_route_skips_llm_answer(wire_graph):
    wire_graph({"intent": "out_of_scope", "service": None, "reason": "weather"})
    state = run(build_graph(), "What's the weather?")
    assert state["trace"] == ["router", "refuse"]
    assert "only help" in state["final"]["answer"]


def test_member_route_pauses_then_approves(wire_graph):
    wire_graph({"intent": "member", "service": "mri", "reason": "member prior auth"}, MEMBER_ANSWER)
    graph = build_graph()
    state = run(graph, "Does [MEMBER_ID_1] need prior auth for an MRI?", member_id="M1002", thread="hitl")
    assert state["trace"] == ["router", "lookup", "summarizer"]
    assert state["__interrupt__"]  # paused for human review
    assert state["tool_results"]["prior_auth"]["prior_auth_required"] is True

    resumed = graph.invoke(Command(resume={"approved": True, "reviewer": "nurse_lead"}),
                           {"configurable": {"thread_id": "hitl"}})
    assert resumed["final"]["answer"] == MEMBER_ANSWER["answer"]
    assert resumed["review_decision"]["reviewer"] == "nurse_lead"


def test_rejection_replaces_answer(wire_graph):
    wire_graph({"intent": "member", "service": "mri", "reason": ""}, MEMBER_ANSWER)
    graph = build_graph()
    run(graph, "q", member_id="M1002", thread="rej")
    resumed = graph.invoke(Command(resume={"approved": False, "reviewer": "sup", "note": "wrong plan year"}),
                           {"configurable": {"thread_id": "rej"}})
    assert "did not approve" in resumed["final"]["answer"] and "wrong plan year" in resumed["final"]["answer"]


def test_both_route_runs_retriever_and_lookup(wire_graph):
    wire_graph({"intent": "both", "service": "durable_medical_equipment", "reason": ""}, MEMBER_ANSWER)
    state = run(build_graph(), "walker for [MEMBER_ID_1]?", member_id="M1003", thread="both")
    assert state["trace"][:4] == ["router", "retriever", "lookup", "summarizer"]
    assert state["retrieved"] and state["tool_results"]["prior_auth"]["service"] == "durable_medical_equipment"


def test_router_failure_uses_heuristic(wire_graph):
    wire_graph("garbage", "garbage", "garbage", POLICY_ANSWER)  # router gets 3 bad replies
    state = run(build_graph(), "How often are mammograms covered?", thread="h")
    assert state["intent"] == "policy" and state["route_reason"] == "heuristic"


def test_invalid_citations_dropped_and_confidence_capped(wire_graph):
    bad = {**POLICY_ANSWER, "citations": [{"chunk_id": "made-up-id", "source": "x", "page": 9, "quote": "x"}]}
    wire_graph({"intent": "policy", "service": None, "reason": ""}, bad)
    state = run(build_graph(), "mammograms?", thread="cit")
    assert state["draft"]["citations"] == []
    assert state["draft"]["confidence"] <= 0.3
    assert state["__interrupt__"]  # low confidence -> human review
    assert any("invalid_citations" in e for e in state["errors"])


def test_unknown_member_handled(wire_graph):
    wire_graph({"intent": "member", "service": None, "reason": ""},
               {"answer": "Member not found.", "citations": [], "confidence": 0.2})
    state = run(build_graph(), "copay for [MEMBER_ID_1]?", member_id="M9999", thread="nf")
    assert "not found" in state["tool_error"]


def test_heuristic_router_picks_service_and_intent():
    from app.agents.nodes import heuristic_route

    assert heuristic_route("Is a walker covered?", None).service == "durable_medical_equipment"
    assert heuristic_route("What is my copay?", "M1001").intent == "member"
    assert heuristic_route("Does an MRI need prior auth?", "M1001").intent == "both"


def test_summarizer_failure_returns_safe_fallback(wire_graph):
    wire_graph({"intent": "policy", "service": None, "reason": ""}, "bad", "bad", "bad")
    state = run(build_graph(), "mammograms?", thread="sf")
    assert state["final"]["confidence"] == 0.0 and "couldn't generate" in state["final"]["answer"]
    assert any("summarizer_error" in e for e in state["errors"])
