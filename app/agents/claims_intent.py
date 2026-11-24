"""Decide whether a (masked) question asks about a member's claims history, and for which service.

Deterministic on purpose: cheap, testable, and it never needs the LLM to decide whether to read claims.
"""
from __future__ import annotations

import re

from app.integrations.claims import ALIASES, SERVICE_CODES, normalize_service

HISTORY = re.compile(
    r"\b(has|have|had|did|got|gotten|received|already|last|recent(ly)?|history|claims?|when was|"
    r"been (seen|screened|vaccinated)|in the (past|last))\b", re.I)
_KNOWN = sorted(set(SERVICE_CODES) | set(ALIASES), key=len, reverse=True)
SERVICE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _KNOWN) + r")s?\b", re.I)
MONTHS = re.compile(r"\b(?:last|past)\s+(\d{1,2})\s+months?\b|\b(?:last|past|this)\s+(year)\b", re.I)


def claims_query(question: str) -> tuple[bool, str | None, int]:
    """(wants_claims, service, months). Needs a history cue, e.g. 'Has [MEMBER_ID_1] had a mammogram?'."""
    if not HISTORY.search(question or ""):
        return False, None, 24
    m = SERVICE.search(question)
    service = normalize_service(m.group(1)) if m else None
    wants = service is not None or bool(re.search(r"\bclaims?\b|\bhistory\b", question, re.I))
    months = 24
    w = MONTHS.search(question)
    if w:
        months = 12 if w.group(2) else max(1, min(60, int(w.group(1))))
    return wants, service, months
