"""Prompt-injection / jailbreak guard. Runs on the (already PII-masked) question before any LLM call.

Layer 1 - rules (always on, free, ~0 ms): regexes for well-known attack phrasing and
          bulk-data exfiltration attempts. Each rule has a name, so a block is explainable.
Layer 2 - classifier (optional): any callable that returns (is_injection, reason). The Claude
          classifier is plugged in once structured output exists (BIQ-10/12). If the classifier
          errors, the rules-only result stands and a warning is logged - the guard never crashes
          the request.

Indirect injection (malicious text inside retrieved documents) is a separate defence in the
summarizer prompt: retrieved text is wrapped in <context> tags and declared to be data.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from langsmith import traceable

from app.core.logging_config import get_logger
from app.core.tracing import redact_inputs

log = get_logger(__name__)

# (rule name, pattern). Patterns need an attack *verb* near an attack *target*, which keeps
# normal benefit questions ("can a plan ignore a prior authorization request?") from being blocked.
RULES: list[tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(instructions?|prompts?|guidelines|directives"
        r"|guardrails|safety rules|everything (above|before))\b", re.I)),
    ("reveal_prompt", re.compile(
        r"\b(reveal|show|print|repeat|leak|output)\b.{0,30}\b(system prompt|your prompt|hidden prompt"
        r"|your instructions|initial prompt|developer message)\b", re.I)),
    ("role_hijack", re.compile(
        r"\b(you are now|act as|pretend to be|roleplay as|from now on you)\b.{0,40}"
        r"\b(dan|developer mode|unfiltered|jailbroken|admin|root|no restrictions)\b", re.I)),
    ("jailbreak_keyword", re.compile(r"\b(jailbreak|DAN mode|developer mode enabled|do anything now)\b", re.I)),
    ("bulk_data_exfiltration", re.compile(
        r"\b(list|dump|export|show|give me|print)\b.{0,25}\b(all|every|entire)\b.{0,25}"
        r"\b(members?|patients?|ssns?|social security|records|database|table)\b", re.I)),
    ("sql_or_code", re.compile(
        # SQL shape, not English: "select * from x" / "select a, b from x" - but not
        # "select the best plan from the list".
        r"(\bselect\s+(\*|\w+(\s*,\s*\w+)*)\s+from\s+\w+|\bdrop\s+table\b|\bunion\s+select\b"
        r"|<script\b|\bos\.system\b|\bexec\()", re.I)),
    ("fake_system_tag", re.compile(r"(\[/?system\]|<\|?system\|?>|^\s*system\s*:|BEGIN SYSTEM PROMPT)", re.I | re.M)),
]

# A classifier takes the text and returns (is_injection, short_reason).
Classifier = Callable[[str], tuple[bool, str]]


@dataclass
class InjectionResult:
    blocked: bool
    reasons: list[str] = field(default_factory=list)


@traceable(name="check_injection", run_type="tool", process_inputs=redact_inputs)
def check_injection(text: str, classifier: Classifier | None = None) -> InjectionResult:
    """Block if any rule matches; otherwise ask the optional classifier."""
    reasons = [name for name, pattern in RULES if pattern.search(text)]
    if reasons:
        log.info("Injection blocked by rules: %s", reasons)   # rule names only, never the text
        return InjectionResult(blocked=True, reasons=reasons)
    if classifier is not None:
        try:
            is_injection, why = classifier(text)
            if is_injection:
                log.info("Injection blocked by classifier")
                return InjectionResult(blocked=True, reasons=[f"classifier: {why}"])
        except Exception as exc:  # noqa: BLE001 - a guard must never crash the request
            log.warning("Injection classifier failed; rules-only result used: %s", type(exc).__name__)
    return InjectionResult(blocked=False)
