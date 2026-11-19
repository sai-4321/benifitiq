"""AgentState = the shared notebook every LangGraph node reads from and writes to.

Each node returns a dict with ONLY the keys it changed; LangGraph merges it in.
Keys annotated with operator.add are APPENDED to instead of overwritten, so every
node can add its own usage/error records without erasing the others.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---- input (already PII-masked by the API) ----
    question: str
    member_id: str | None

    # ---- router ----
    intent: str  # policy | member | both | out_of_scope
    service: str | None
    route_reason: str

    # ---- retriever ----
    retrieved: list[dict]

    # ---- lookup (MCP) ----
    tool_results: dict[str, Any] | None
    tool_error: str | None

    # ---- summarizer ----
    draft: dict  # AnswerJSON as a dict
    review_required: bool
    review_reasons: list[str]

    # ---- human-in-the-loop ----
    review_decision: dict | None
    final: dict

    # ---- bookkeeping (appended by every node) ----
    usage: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]  # node names in the order they ran
