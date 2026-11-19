"""Run every golden question through the full pipeline and print quality metrics.

Usage:
    python evals/run_eval.py                 # all questions, with LLM-as-judge groundedness
    python evals/run_eval.py --no-judge      # skip the judge (cheaper, faster)
    python evals/run_eval.py --limit 10      # first 10 questions only
    python evals/run_eval.py --category preventive

Human review is auto-approved and the cache is OFF so every question really runs.
Results are saved to evals/results/eval_<timestamp>.csv and summary_<timestamp>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app import pipeline  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.core.tracing import setup_tracing  # noqa: E402
from app.llm_factory import active_provider_label, get_llm  # noqa: E402
from evals.metrics import (  # noqa: E402
    citation_accuracy,
    doc_hit_at_k,
    evidence_hit_and_rr,
    judge_groundedness,
    keyword_recall,
    mean,
    percentile,
)

DATASET = ROOT / "evals" / "golden_dataset.jsonl"
RESULTS = ROOT / "evals" / "results"


def load_dataset(category: str | None, limit: int | None) -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if category:
        rows = [r for r in rows if r["category"] == category]
    return rows[:limit] if limit else rows


def evaluate_one(item: dict, use_judge: bool, pause: float) -> dict:
    resp, debug = pipeline.ask(item["question"], item.get("member_id"), use_cache=False, auto_approve=True,
                               include_debug=True)
    state = debug.get("state", {})
    retrieved = state.get("retrieved") or []
    answer = resp.answer.answer if resp.answer else ""
    citations = [c.model_dump() for c in resp.answer.citations] if resp.answer else []
    route = "blocked" if resp.status == "blocked" else (resp.intent or resp.status)

    row = {
        "id": item["id"], "category": item["category"], "question": item["question"],
        "expected_route": item["expected_route"], "route": route,
        "route_correct": float(route == item["expected_route"]),
        "status": resp.status, "latency_ms": resp.latency_ms,
        "tokens": resp.input_tokens + resp.output_tokens, "cost_usd": resp.cost_usd,
        "confidence": resp.answer.confidence if resp.answer else None,
        "answer": answer[:500],
    }
    if item.get("must_block"):
        row["blocked_ok"] = float(resp.status == "blocked")
        return row

    hit, rr = evidence_hit_and_rr(retrieved, item["expected_keywords"]) if retrieved else (None, None)
    row.update({
        "doc_hit@k": doc_hit_at_k(retrieved, item["expected_sources"]) if retrieved else None,
        "evidence_hit@k": hit if item["expected_sources"] else None,
        "mrr": rr if item["expected_sources"] else None,
        "keyword_recall": keyword_recall(answer, item["expected_keywords"]),
        "citation_accuracy": citation_accuracy(citations, retrieved, state.get("tool_results")),
        "n_citations": len(citations),
    })
    if use_judge and answer and item["expected_route"] != "out_of_scope":
        context = "\n\n".join(c["text"] for c in retrieved)
        if state.get("tool_results"):
            context += "\n\nMEMBER DATA:\n" + json.dumps(state["tool_results"], default=str)
        try:
            verdict = judge_groundedness(get_llm(), answer, context or "(empty)")
            row["groundedness"] = verdict.score
            row["unsupported"] = "; ".join(verdict.unsupported_claims)[:300]
        except Exception as exc:  # noqa: BLE001
            row["groundedness"] = None
            row["unsupported"] = f"judge_error: {exc}"[:200]
    if pause:
        time.sleep(pause)  # helps stay under Bedrock on-demand rate limits (requests/min quota)
    return row


def summarize(df: pd.DataFrame) -> dict:
    def col(name: str) -> list:
        return df[name].tolist() if name in df else []

    answered = df[df["status"].isin(["completed", "pending_approval"])]
    return {
        "questions": int(len(df)),
        "route_accuracy": mean(col("route_correct")),
        "injection_block_rate": mean(col("blocked_ok")),
        "doc_hit@k": mean(col("doc_hit@k")),
        "evidence_hit@k": mean(col("evidence_hit@k")),
        "mrr": mean(col("mrr")),
        "keyword_recall": mean(col("keyword_recall")),
        "citation_accuracy": mean(col("citation_accuracy")),
        "groundedness": mean(col("groundedness")),
        "latency_p50_ms": percentile(answered["latency_ms"].tolist(), 50),
        "latency_p95_ms": percentile(answered["latency_ms"].tolist(), 95),
        "avg_tokens": round(float(answered["tokens"].mean()), 1) if len(answered) else 0,
        "total_cost_usd": round(float(df["cost_usd"].sum()), 5),
        "errors": int((df["status"] == "error").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions")
    args = parser.parse_args()

    setup_logging("WARNING")
    setup_tracing()
    items = load_dataset(args.category, args.limit)
    print(f"Evaluating {len(items)} questions with {active_provider_label()} "
          f"(top_k={get_settings().retrieval_top_k}, judge={'off' if args.no_judge else 'on'})\n")

    rows = []
    for i, item in enumerate(items, start=1):
        row = evaluate_one(item, not args.no_judge, args.pause)
        rows.append(row)
        flag = "OK " if row["route_correct"] else "XX "
        kr = row.get("keyword_recall")
        print(f"[{i:02d}/{len(items)}] {flag}{item['id']} {row['route']:<12} "
              f"kw={'-' if kr is None else f'{kr:.2f}'} {row['latency_ms']:>6} ms  {item['question'][:60]}")

    df = pd.DataFrame(rows)
    summary = summarize(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(RESULTS / f"eval_{stamp}.csv", index=False)
    (RESULTS / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ BenefitIQ evaluation summary ================")
    for key, value in summary.items():
        print(f"  {key:<22} {value}")
    print("\nBy category (keyword recall / route accuracy):")
    by_cat = df.groupby("category").agg(n=("id", "count"), route_acc=("route_correct", "mean"))
    if "keyword_recall" in df:
        by_cat["kw_recall"] = df.groupby("category")["keyword_recall"].mean()
    print(by_cat.round(2).to_string())
    print(f"\nSaved: evals/results/eval_{stamp}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
