"""BenefitIQ web UI. Talks to the FastAPI backend over HTTP only (no direct imports),
exactly like a real front end would.

Run:  streamlit run ui/streamlit_app.py      -> opens http://localhost:8501
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 180

SAMPLES = [
    ("What does Medicare Part B cover?", ""),
    ("How often does Medicare cover a screening mammogram?", ""),
    ("What is the specialist copay for member M1001?", "M1001"),
    ("Does member M1002 need prior authorization for an MRI?", "M1002"),
    ("Member M1003 needs a walker. Does Medicare cover it and does their plan need prior auth?", "M1003"),
]

st.set_page_config(page_title="BenefitIQ", page_icon="🩺", layout="wide")
st.session_state.setdefault("result", None)
st.session_state.setdefault("question", "")
st.session_state.setdefault("member_id", "")


def call_api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is uvicorn running?"
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", exc.response.text) if exc.response is not None else str(exc)
        return None, f"API error: {detail}"
    except requests.Timeout:
        return None, "The API took too long to answer. Try again."


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("System status")
    health, err = call_api("GET", "/health")
    if err:
        st.error(err)
    else:
        st.success(f"API {health['status']}") if health["status"] == "ok" else st.warning("API degraded")
        st.caption(f"LLM: {health['provider']}  \nEmbeddings: {health['embedding_provider']}  \n"
                   f"Indexed chunks: {health['indexed_chunks']}  \nMember DB: {health['member_db']}  \n"
                   f"LangSmith: {health['langsmith_tracing']}")
    metrics, _ = call_api("GET", "/metrics")
    if metrics:
        st.header("Session metrics")
        c1, c2 = st.columns(2)
        c1.metric("Requests", metrics["requests"])
        c2.metric("Cache hits", metrics["cache_hits"])
        c1.metric("Blocked", metrics["blocked"])
        c2.metric("Pending", metrics["pending_now"])
        st.metric("Avg latency", f"{metrics['avg_latency_ms']} ms")
        st.metric("Total cost", f"${metrics['total_cost_usd']:.4f}")
    st.header("Demo questions")
    for q, mid in SAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.question, st.session_state.member_id = q, mid
    st.caption("All member data is synthetic. Policy text is from public Medicare.gov booklets.")

# ---------------- main ----------------
st.title("🩺 BenefitIQ - Health Plan Benefits & Policy Assistant")
st.write("Ask a Medicare coverage question, or include a synthetic member ID (M1001-M1050) for plan details.")

with st.form("ask_form"):
    question = st.text_area("Question", key="question", height=90)
    c1, c2 = st.columns([1, 3])
    member_id = c1.text_input("Member ID (optional)", key="member_id", placeholder="M1001")
    use_cache = c2.checkbox("Use response cache", value=True)
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    payload = {"question": question.strip(), "use_cache": use_cache}
    if member_id.strip():
        payload["member_id"] = member_id.strip().upper()
    with st.spinner("Agents working: router -> retriever / MCP lookup -> summarizer ..."):
        data, err = call_api("POST", "/ask", json=payload)
    if err:
        st.error(err)
    st.session_state.result = data


def render(res: dict) -> None:
    status = res["status"]
    if status == "blocked":
        st.error(f"🛡️ {res['message']}")
        return
    if status == "error":
        st.error(res.get("message", "Error"))
        return

    if status == "pending_approval":
        st.warning("⏸️ Human review required before this answer is released.")
        for r in res["review_reasons"]:
            st.write(f"- {r}")
    elif res.get("cached"):
        st.info("⚡ Served from cache ($0, no LLM call)")

    ans = res.get("answer") or {}
    st.subheader("Draft answer (not yet released)" if status == "pending_approval" else "Answer")
    st.write(ans.get("answer", ""))
    st.progress(min(max(ans.get("confidence", 0.0), 0.0), 1.0), text=f"Confidence {ans.get('confidence', 0):.2f}")
    if res.get("reviewer"):
        st.caption(f"Reviewed by: {res['reviewer']}")

    if ans.get("citations"):
        st.subheader("Citations")
        for c in ans["citations"]:
            where = f"{c['source']} - page {c['page']}" if c["page"] else f"Member tool: {c['source']}"
            st.markdown(f"**{where}**  \n> {c.get('quote', '')}")

    cols = st.columns(5)
    cols[0].metric("Route", res.get("intent") or "-")
    cols[1].metric("Latency", f"{res['latency_ms']} ms")
    cols[2].metric("Tokens", res["input_tokens"] + res["output_tokens"])
    cols[3].metric("Cost", f"${res['cost_usd']:.5f}")
    cols[4].metric("PII masked", sum(res.get("pii_found", {}).values()))

    with st.expander("What the LLM saw (masked question) and retrieved sources"):
        st.code(res.get("masked_question") or "", language=None)
        for s in res.get("sources_considered", []):
            st.write(f"{s['chunk_id']}  |  {s['source']} p.{s['page']}  |  RRF {s['score']:.4f}")

    if status == "pending_approval":
        st.divider()
        st.subheader("Supervisor review")
        reviewer = st.text_input("Reviewer name", value="Supervisor")
        note = st.text_input("Note (required when rejecting)")
        a, b = st.columns(2)
        decision = None
        if a.button("✅ Approve", type="primary", use_container_width=True):
            decision = True
        if b.button("❌ Reject", use_container_width=True):
            decision = False
        if decision is not None:
            data, err = call_api("POST", "/approve", json={"thread_id": res["thread_id"], "approved": decision,
                                                         "reviewer": reviewer or "Supervisor", "note": note or None})
            if err:
                st.error(err)
            else:
                st.session_state.result = data
                st.rerun()


if st.session_state.result:
    render(st.session_state.result)
