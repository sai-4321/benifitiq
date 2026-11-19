"""PHI/PII masking. Runs BEFORE any text is sent to an LLM, logged, or cached.

Each detected value is replaced by a placeholder like [SSN_1]. The mapping
placeholder -> original value stays inside our API process (never sent to the LLM),
so the final answer can be un-masked for the authorized user.

Example:
    "Patient John Smith (M1003), SSN 123-45-6789"
 -> "Patient [NAME_1] ([MEMBER_ID_1]), SSN [SSN_1]"   mapping {"[NAME_1]": "John Smith", ...}

Production note: swap the regexes for Amazon Comprehend Medical (DetectPHI) or
Amazon Bedrock Guardrails sensitive-information filters, which use ML models and
catch far more name/address variants.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: more specific patterns first (SSN before phone, MBI before member ID).
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Medicare Beneficiary Identifier format, e.g. 1EG4-TE5-MK73 or 1EG4TE5MK73
    ("MBI", re.compile(r"\b[1-9][A-Z][A-Z0-9]\d-?[A-Z][A-Z0-9]\d-?[A-Z]{2}\d{2}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")),
    ("DOB", re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b")),
    ("DOB", re.compile(r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b")),
    ("MEMBER_ID", re.compile(r"\bM\d{4}\b")),
    # Names that follow a cue word: "patient John Smith", "member named Mary Lee", "Dr. Patel".
    # Only the <value> group is replaced; the cue word stays so the sentence still reads well.
    (
        "NAME",
        re.compile(
            r"\b(?i:patient|member|named|name is|mr\.|mrs\.|ms\.|dr\.)\s+"
            r"(?P<value>[A-Z][a-z]+(?: [A-Z][a-z]+)?)"
        ),
    ),
]

MEMBER_ID_RE = re.compile(r"\bM\d{4}\b")


@dataclass
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    counts: dict[str, int] = field(default_factory=dict)  # entity type -> how many


def mask_pii(text: str) -> MaskResult:
    """Replace PII with numbered placeholders. Same value -> same placeholder."""
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    counts: dict[str, int] = {}
    masked = text
    for label, pattern in PATTERNS:

        def _sub(m: re.Match, label: str = label) -> str:
            has_group = "value" in m.re.groupindex
            value = m.group("value") if has_group else m.group(0)
            prefix = m.group(0)[: m.start("value") - m.start()] if has_group else ""
            if value not in reverse:
                counts[label] = counts.get(label, 0) + 1
                placeholder = f"[{label}_{counts[label]}]"
                mapping[placeholder] = value
                reverse[value] = placeholder
            return prefix + reverse[value]

        masked = pattern.sub(_sub, masked)
    return MaskResult(text=masked, mapping=mapping, counts=counts)


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Put original values back (only for the authorized user's screen)."""
    for placeholder, value in mapping.items():
        text = text.replace(placeholder, value)
    return text


def extract_member_id(text: str) -> str | None:
    m = MEMBER_ID_RE.search(text)
    return m.group(0) if m else None
