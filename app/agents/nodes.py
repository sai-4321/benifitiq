"""The agents. Each function is one LangGraph node: it receives the shared state and
returns only the keys it wants to change.

router      -> decides the path (policy / member / both / out_of_scope)
retriever   -> hybrid search over Medicare booklets
lookup      -> calls MCP tools for the member's plan data
summarizer  -> writes the JSON answer with citations (validated + retried)
refuse      -> polite out-of-scope answer, no LLM call
human_review-> pauses the graph (interrupt) until a supervisor approves or rejects
"""
from __future__ import annotations

import json
import re

from langgraph.types import interrupt

from app.agents.mcp_client import ToolCallError, run_member_lookup
from app.agents.prompts import ROUTER_SYSTEM, SUMMARIZER_SYSTEM
from app.agents.state import AgentState
from app.config import get_settings
from app.core.logging_config import get_logger
from app.core.structured_output import StructuredOutputError, invoke_structured
from app.llm_factory import get_llm
from app.rag.hybrid_retriever import get_retriever
from app.schemas import AnswerJSON, Citation, RouteDecision

log = get_logger(__name__)

# Keyword fallback used only if the router LLM call fails completely.
SERVICE_KEYWORDS = {
    "mri": ["mri", "magnetic resonance"],
    "ct_scan": ["ct scan", "cat scan", "ct "],
    "specialist_visit": ["specialist", "cardiologist", "dermatologist", "orthopedic"],
    "physical_therapy": ["physical therapy", "physiotherapy", " pt "],
    "durable_medical_equipment": ["walker", "wheelchair", "dme", "durable medical", "hospital bed", "oxygen"],
    "home_health": ["home health"],
    "skilled_nursing": ["skilled nursing", "snf"],
    "outpatient_surgery": ["outpatient surgery", "surgery"],
    "sleep_study": ["sleep study", "sleep apnea"],
    "genetic_testing": ["genetic test", "genetic testing"],
}


def heuristic_route(question: str, member_id: str | None) -> RouteDecision:
    q = f" {question.lower()} "
    service = next((svc for svc, words in SERVICE_KEYWORDS.items() if any(w in q for w in words)), None)
    if member_id:
        member_only = re.search(r"\b(copay|deductible|out-of-pocket|oop|status|active|plan type)\b", q)
        return RouteDecision(intent="member" if member_only else "both", service=service, reason="heuristic")
    return RouteDecision(intent="policy", service=service, reason="heuristic")


# ------------------------------------------------------------------ router
def router_node(state: AgentState) -> dict:
    question, member_id = state["question"], state.get("member_id")
    member_note = "A member ID WAS provided." if member_id else "No member ID was provided."
    try:
        decision, usage = invoke_structured(
            get_llm(), ROUTER_SYSTEM, f"{member_note}\n<question>\n{question}\n</question>", RouteDecision,
            node="router",
        )
        errors: list[str] = []
    except Exception as exc:  # noqa: BLE001 - routing must never take the app down
        log.warning("Router LLM failed, using keyword heuristic: %s", exc)
        decision, usage, errors = heuristic_route(question, member_id), [], [f"router_fallback: {exc}"[:200]]
    log.info("Route -> %s (service=%s)", decision.intent, decision.service)
    return {"intent": decision.intent, "service": decision.service, "route_reason": decision.reason,
            "usage": usage, "errors": errors, "trace": ["router"]}


# ------------------------------------------------------------------ retriever
def retriever_node(state: AgentState) -> dict:
    try:
        hits = get_retriever().retrieve(state["question"], top_k=get_settings().retrieval_top_k)
        return {"retrieved": hits, "trace": ["retriever"]}
    except Exception as exc:  # noqa: BLE001
        log.error("Retrieval failed: %s", exc)
        return {"retrieved": [], "errors": [f"retrieval_error: {exc}"[:200]], "trace": ["retriever"]}


# ------------------------------------------------------------------ lookup (MCP)
def lookup_node(state: AgentState) -> dict:
    member_id = state.get("member_id")
    if not member_id:
        return {"tool_results": None, "tool_error": "No member ID was provided. Ask the user for the member ID "
                "(format M1234).", "trace": ["lookup"]}
    try:
        results = run_member_lookup(member_id, state.get("service"))
        return {"tool_results": results, "tool_error": None, "trace": ["lookup"]}
    except ToolCallError as exc:
        log.warning("Lookup failed: %s", exc)
        return {"tool_results": None, "tool_error": str(exc)[:300], "errors": [f"lookup_error: {exc}"[:200]],
                "trace": ["lookup"]}


# ------------------------------------------------------------------ summarizer
def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no policy excerpts were retrieved)"
    return "\n\n".join(
        f"[chunk_id={c['id']} | source={c['source']} | page={c['page']}]\n{c['text']}" for c in chunks
    )


def _format_member_data(state: AgentState) -> str:
    if state.get("tool_error"):
        return f"TOOL ERROR: {state['tool_error']}"
    results = state.get("tool_results")
    if not results:
        return "(no member data requested)"
    view = {
        "TOOL:get_member_plan": results.get("member"),
        "TOOL:get_plan_benefits": results.get("benefits"),
    }
    if results.get("prior_auth"):
        view["TOOL:check_prior_auth"] = results["prior_auth"]
    return json.dumps(view, indent=2)


def validate_citations(answer: AnswerJSON, state: AgentState) -> tuple[AnswerJSON, int]:
    """Drop citations that point at chunks we never retrieved; fix source/page from the real chunk."""
    by_id = {c["id"]: c for c in state.get("retrieved") or []}
    tool_ids = set()
    if state.get("tool_results"):
        tool_ids = {"TOOL:get_member_plan", "TOOL:get_plan_benefits"}
        if state["tool_results"].get("prior_auth"):
            tool_ids.add("TOOL:check_prior_auth")
    kept: list[Citation] = []
    dropped = 0
    for cit in answer.citations:
        if cit.chunk_id in by_id:
            real = by_id[cit.chunk_id]
            kept.append(cit.model_copy(update={"source": real["source"], "page": real["page"]}))
        elif cit.chunk_id in tool_ids:
            kept.append(cit.model_copy(update={"source": cit.chunk_id.split(":", 1)[1], "page": 0}))
        else:
            dropped += 1
    return answer.model_copy(update={"citations": kept}), dropped


def summarizer_node(state: AgentState) -> dict:
    s = get_settings()
    user_prompt = (
        f"<question>\n{state['question']}\n</question>\n\n"
        f"<context>\n{_format_context(state.get('retrieved') or [])}\n</context>\n\n"
        f"<member_data>\n{_format_member_data(state)}\n</member_data>"
    )
    errors: list[str] = []
    try:
        answer, usage = invoke_structured(get_llm(), SUMMARIZER_SYSTEM, user_prompt, AnswerJSON, node="summarizer")
    except (StructuredOutputError, Exception) as exc:  # noqa: BLE001
        log.error("Summarizer failed: %s", exc)
        fallback = AnswerJSON(
            answer="I couldn't generate a reliable answer right now. Please try again in a moment, or check "
                   "the Medicare.gov booklets directly.",
            citations=[], confidence=0.0, needs_human_review=False,
        )
        return {"draft": fallback.model_dump(), "review_required": False, "review_reasons": [],
                "errors": [f"summarizer_error: {exc}"[:200]], "trace": ["summarizer"]}

    answer, dropped = validate_citations(answer, state)
    if dropped:
        errors.append(f"dropped_{dropped}_invalid_citations")
    if not answer.citations and answer.confidence > 0.3:
        answer = answer.model_copy(update={"confidence": 0.3})  # uncited answers are never "confident"

    reasons: list[str] = []
    if state.get("intent") in ("member", "both") and state.get("tool_results"):
        reasons.append("Member-specific answer (uses this member's plan data)")
    if answer.confidence < s.hitl_confidence_threshold:
        reasons.append(f"Low confidence ({answer.confidence:.2f} < {s.hitl_confidence_threshold})")
    if answer.needs_human_review:
        reasons.append("Model flagged this answer for review")
    return {"draft": answer.model_dump(), "review_required": bool(reasons), "review_reasons": reasons,
            "usage": usage, "errors": errors, "trace": ["summarizer"]}


# ------------------------------------------------------------------ refuse
def refuse_node(state: AgentState) -> dict:
    answer = AnswerJSON(
        answer="I can only help with health-plan benefits, Medicare coverage, and member plan questions. "
               "Please rephrase your question about coverage, costs, enrollment, or prior authorization.",
        citations=[], confidence=1.0, needs_human_review=False,
    )
    return {"draft": answer.model_dump(), "final": answer.model_dump(), "review_required": False,
            "review_reasons": [], "trace": ["refuse"]}


# ------------------------------------------------------------------ human in the loop
def human_review_node(state: AgentState) -> dict:
    if not state.get("review_required"):
        return {"final": state["draft"], "trace": ["human_review(skipped)"]}

    # Pause here. The API returns "pending_approval"; the graph resumes when
    # /approve calls graph.invoke(Command(resume={...})). Code above this line re-runs
    # on resume, so it must have no side effects.
    decision = interrupt({"draft": state["draft"], "reasons": state.get("review_reasons", [])})

    if decision.get("approved"):
        final = dict(state["draft"])
    else:
        note = decision.get("note") or "no note given"
        final = AnswerJSON(
            answer=f"A reviewer ({decision.get('reviewer', 'unknown')}) did not approve this answer. Reason: {note}. "
                   "Please escalate to a benefits specialist.",
            citations=[], confidence=0.0, needs_human_review=False,
        ).model_dump()
    return {"final": final, "review_decision": decision, "trace": ["human_review"]}
