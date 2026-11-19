"""The end-to-end request pipeline, shared by the FastAPI app and the eval script.

ask():     PII mask -> injection check -> cache -> LangGraph -> (pause for review | answer) -> unmask
approve(): resume a paused run with the reviewer's decision
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from langgraph.types import Command
from langsmith import traceable

from app.agents.graph import get_graph
from app.agents.prompts import PROMPT_VERSION
from app.config import get_settings
from app.core.cache import get_cache, make_key
from app.core.logging_config import get_logger
from app.core.tracing import redact_inputs, sum_usage
from app.guardrails import check_injection, extract_member_id, mask_pii, unmask
from app.guardrails.llm_classifier import claude_injection_classifier
from app.llm_factory import active_provider_label
from app.schemas import AnswerJSON, AskResponse, SourceRef

log = get_logger(__name__)


@dataclass
class PendingRun:
    mapping: dict[str, str]
    masked_question: str
    pii_counts: dict[str, int]
    started: float
    created_at: float = field(default_factory=time.time)


@dataclass
class Metrics:
    requests: int = 0
    cache_hits: int = 0
    blocked: int = 0
    pending_created: int = 0
    approvals: int = 0
    rejections: int = 0
    errors: int = 0
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    def snapshot(self) -> dict:
        avg = self.total_latency_ms / self.requests if self.requests else 0
        return {**self.__dict__, "avg_latency_ms": round(avg), "total_cost_usd": round(self.total_cost_usd, 5),
                "pending_now": len(PENDING)}


PENDING: dict[str, PendingRun] = {}
METRICS = Metrics()
_LOCK = threading.Lock()


def _unmask_answer(answer: dict, mapping: dict[str, str]) -> AnswerJSON:
    a = AnswerJSON.model_validate(answer)
    return a.model_copy(update={
        "answer": unmask(a.answer, mapping),
        "follow_up_questions": [unmask(q, mapping) for q in a.follow_up_questions],
    })


def _sources(state: dict) -> list[SourceRef]:
    return [SourceRef(chunk_id=c["id"], source=c["source"], page=c["page"], score=c.get("rrf_score", 0.0))
            for c in state.get("retrieved") or []]


def _build_response(state: dict, thread_id: str, run: PendingRun, *, cached: bool = False) -> AskResponse:
    usage = sum_usage(state.get("usage") or [])
    paused = bool(state.get("__interrupt__"))
    raw = state.get("draft") if paused else state.get("final") or state.get("draft")
    decision = state.get("review_decision") or {}
    return AskResponse(
        status="pending_approval" if paused else "completed",
        thread_id=thread_id,
        intent=state.get("intent"),
        answer=_unmask_answer(raw, run.mapping) if raw else None,
        review_reasons=state.get("review_reasons") or [],
        reviewer=decision.get("reviewer"),
        masked_question=run.masked_question,
        pii_found=run.pii_counts,
        sources_considered=_sources(state),
        cached=cached,
        latency_ms=int((time.perf_counter() - run.started) * 1000),
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cost_usd=usage["cost_usd"],
        provider=active_provider_label(),
        message="Waiting for supervisor approval." if paused else None,
    )


def _record(resp: AskResponse) -> None:
    with _LOCK:
        METRICS.requests += 1
        METRICS.total_latency_ms += resp.latency_ms
        METRICS.total_cost_usd += resp.cost_usd
        METRICS.total_tokens += resp.input_tokens + resp.output_tokens


@traceable(name="benefitiq_ask", run_type="chain", process_inputs=redact_inputs)
def ask(question: str, member_id: str | None = None, *, use_cache: bool = True, auto_approve: bool = False,
        include_debug: bool = False) -> tuple[AskResponse, dict]:
    """Returns (response, debug). debug holds raw state for the eval script; the API ignores it."""
    s = get_settings()
    started = time.perf_counter()

    # 1. PII / PHI masking first - nothing after this line (classifier, LLM, cache, logs) sees raw identifiers
    member_id = member_id or extract_member_id(question)
    masked = mask_pii(question)
    run = PendingRun(mapping=masked.mapping, masked_question=masked.text, pii_counts=masked.counts,
                     started=started)

    # 2. Prompt-injection guard (rules always; Claude classifier if INJECTION_LLM_CHECK=true)
    inj = check_injection(masked.text, classifier=claude_injection_classifier if s.injection_llm_check else None)
    if inj.blocked:
        resp = AskResponse(status="blocked", message="This request was blocked by the safety filter "
                           f"({', '.join(inj.reasons)}). Please ask a benefits or coverage question.",
                           pii_found=masked.counts, latency_ms=int((time.perf_counter() - started) * 1000))
        with _LOCK:
            METRICS.blocked += 1
        _record(resp)
        return resp, {"blocked_reasons": inj.reasons}

    # 3. Cache (only general policy answers are ever stored)
    key = make_key(masked.text, active_provider_label(), PROMPT_VERSION, s.retrieval_top_k)
    if use_cache and not member_id:
        hit = get_cache().get(key)
        if hit:
            resp = AskResponse.model_validate(hit).model_copy(update={
                "cached": True, "latency_ms": int((time.perf_counter() - started) * 1000),
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "thread_id": None})
            with _LOCK:
                METRICS.cache_hits += 1
            _record(resp)
            return resp, {}

    # 4. Run the multi-agent graph
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "run_name": "benefitiq_graph"}
    graph = get_graph()
    try:
        state = graph.invoke({"question": masked.text, "member_id": member_id}, config)
        if state.get("__interrupt__") and auto_approve:
            state = graph.invoke(Command(resume={"approved": True, "reviewer": "auto-eval", "note": None}), config)
    except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the user
        log.exception("Graph failed")
        with _LOCK:
            METRICS.errors += 1
        resp = AskResponse(status="error", message=f"Something went wrong: {type(exc).__name__}. Please retry.",
                           latency_ms=int((time.perf_counter() - started) * 1000))
        _record(resp)
        return resp, {"exception": repr(exc)}

    resp = _build_response(state, thread_id, run)
    if resp.status == "pending_approval":
        PENDING[thread_id] = run
        with _LOCK:
            METRICS.pending_created += 1
    elif use_cache and not member_id and resp.intent in ("policy", "out_of_scope") and not state.get("errors"):
        get_cache().set(key, resp.model_dump())
    _record(resp)
    debug = {"state": state} if include_debug else {}
    return resp, debug


@traceable(name="benefitiq_approve", run_type="chain")
def approve(thread_id: str, approved: bool, reviewer: str, note: str | None = None) -> AskResponse:
    run = PENDING.pop(thread_id, None)
    if run is None:
        raise KeyError(thread_id)
    run.started = time.perf_counter()
    config = {"configurable": {"thread_id": thread_id}}
    state = get_graph().invoke(
        Command(resume={"approved": approved, "reviewer": reviewer, "note": note}), config
    )
    with _LOCK:
        if approved:
            METRICS.approvals += 1
        else:
            METRICS.rejections += 1
    return _build_response(state, thread_id, run)
