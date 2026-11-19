"""Evaluation metrics. Pure functions (easy to unit test) + one LLM-as-judge function.

Keyword format: each expected keyword may hold alternatives separated by "|",
e.g. "doesn't cover|does not cover" - any alternative counts as a match.
"""
from __future__ import annotations

import math
import re
import statistics

from app.agents.prompts import JUDGE_SYSTEM
from app.core.structured_output import invoke_structured
from app.schemas import GroundednessVerdict


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'"))


def keyword_hit(text: str, keyword: str) -> bool:
    t = _norm(text)
    return any(_norm(alt.strip()) in t for alt in keyword.split("|") if alt.strip())


def keyword_recall(answer: str, keywords: list[str]) -> float | None:
    """Fraction of expected keywords found in the answer (answer correctness proxy)."""
    if not keywords:
        return None
    return sum(keyword_hit(answer, k) for k in keywords) / len(keywords)


def doc_hit_at_k(retrieved: list[dict], expected_sources: list[str]) -> float | None:
    """1.0 if any top-k chunk comes from an expected document."""
    if not expected_sources:
        return None
    return 1.0 if any(c["source"] in expected_sources for c in retrieved) else 0.0


def evidence_rank(retrieved: list[dict], keywords: list[str]) -> int | None:
    """Rank (1-based) of the first chunk containing ANY expected keyword, or None."""
    for i, c in enumerate(retrieved, start=1):
        if any(keyword_hit(c["text"], k) for k in keywords):
            return i
    return None


def evidence_hit_and_rr(retrieved: list[dict], keywords: list[str]) -> tuple[float | None, float | None]:
    """(hit@k, reciprocal rank) using 'chunk contains an expected keyword' as relevance."""
    if not keywords:
        return None, None
    rank = evidence_rank(retrieved, keywords)
    return (1.0, 1.0 / rank) if rank else (0.0, 0.0)


def _token_overlap(quote: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", quote.lower()))
    if not q:
        return 0.0
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(q & t) / len(q)


def citation_accuracy(citations: list[dict], retrieved: list[dict], tool_results: dict | None) -> float | None:
    """Share of citations that point at a real retrieved chunk (or called tool) AND whose quote
    is actually in that chunk (>= 60% word overlap)."""
    if not citations:
        return None
    by_id = {c["id"]: c for c in retrieved}
    good = 0
    for cit in citations:
        cid = cit.get("chunk_id", "")
        if cid in by_id:
            if not cit.get("quote") or _token_overlap(cit["quote"], by_id[cid]["text"]) >= 0.6:
                good += 1
        elif cid.startswith("TOOL:") and tool_results:
            good += 1
    return good / len(citations)


def judge_groundedness(llm, answer: str, context: str) -> GroundednessVerdict:
    prompt = f"<context>\n{context}\n</context>\n\n<answer>\n{answer}\n</answer>"
    verdict, _ = invoke_structured(llm, JUDGE_SYSTEM, prompt, GroundednessVerdict, node="judge", max_retries=1)
    return verdict


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(statistics.mean(vals), 3) if vals else None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]
