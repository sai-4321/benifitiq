"""Pydantic models = the contracts between the UI, the API, the agents and the LLM.

If the LLM returns JSON that does not match AnswerJSON, validation fails and
app/core/structured_output.py asks the model to fix it (retry).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["policy", "member", "both", "out_of_scope"]

# Services our synthetic prior-authorization table knows about.
SERVICES = (
    "mri",
    "ct_scan",
    "specialist_visit",
    "physical_therapy",
    "durable_medical_equipment",
    "home_health",
    "skilled_nursing",
    "outpatient_surgery",
    "sleep_study",
    "genetic_testing",
)
Service = Literal[
    "mri",
    "ct_scan",
    "specialist_visit",
    "physical_therapy",
    "durable_medical_equipment",
    "home_health",
    "skilled_nursing",
    "outpatient_surgery",
    "sleep_study",
    "genetic_testing",
]

MEMBER_ID_PATTERN = r"^M\d{4}$"


# ---------------- LLM output contracts ----------------
class RouteDecision(BaseModel):
    """What the router agent must return."""

    intent: Intent
    service: Service | None = Field(
        default=None, description="Medical service mentioned in the question, if any, from the allowed list."
    )
    reason: str = Field(default="", max_length=300)


class Citation(BaseModel):
    chunk_id: str = Field(description="ID of the context chunk (e.g. 10050-p23-c1) or TOOL:<tool_name>")
    source: str = Field(description="Document file name or MCP tool name")
    page: int = Field(ge=0, description="PDF page number (0 for tool results)")
    quote: str = Field(default="", max_length=400, description="Short supporting quote")


class AnswerJSON(BaseModel):
    """What the summarizer agent must return."""

    answer: str = Field(min_length=1, max_length=3000)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False
    follow_up_questions: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, v: str) -> str:
        return v.strip()


class GroundednessVerdict(BaseModel):
    """What the evaluation judge must return."""

    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


class InjectionVerdict(BaseModel):
    is_injection: bool
    reason: str = ""


# ---------------- API contracts ----------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    member_id: str | None = Field(default=None, pattern=MEMBER_ID_PATTERN)
    use_cache: bool = True


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool
    reviewer: str = Field(min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=500)


class SourceRef(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    status: Literal["completed", "pending_approval", "blocked", "error"]
    thread_id: str | None = None
    intent: Intent | None = None
    answer: AnswerJSON | None = None
    review_reasons: list[str] = Field(default_factory=list)
    reviewer: str | None = None
    masked_question: str | None = None
    pii_found: dict[str, int] = Field(default_factory=dict)
    sources_considered: list[SourceRef] = Field(default_factory=list)
    cached: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    provider: str | None = None
    message: str | None = None
