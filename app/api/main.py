"""FastAPI backend.

GET  /health   - is the app ready? (index size, DB present, provider)
POST /ask      - ask a question -> completed answer, blocked, or pending_approval
POST /approve  - supervisor approves/rejects a pending answer
GET  /pending  - list paused runs waiting for review
GET  /metrics  - request counts, cache hits, latency, cost

Run:  uvicorn app.api.main:app --reload --port 8000     Docs: http://localhost:8000/docs
The endpoints are plain `def` (not async): FastAPI runs them in a worker thread, which is
right for our blocking LLM/MCP calls.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__, pipeline
from app.config import get_settings
from app.core.logging_config import get_logger, setup_logging
from app.core.tracing import setup_tracing
from app.llm_factory import active_provider_label
from app.schemas import ApproveRequest, AskRequest, AskResponse

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    setup_logging(s.log_level)
    setup_tracing()
    log.info("BenefitIQ API starting. Provider: %s", active_provider_label())
    yield


app = FastAPI(title="BenefitIQ API", version=__version__, lifespan=lifespan,
              description="Health Plan Benefits & Policy Assistant (synthetic demo data only)")


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"status": "error", "message": "Internal error. Please retry."})


@app.get("/health")
def health() -> dict:
    s = get_settings()
    chunks = 0
    if s.chunks_file.exists():
        with s.chunks_file.open(encoding="utf-8") as f:
            chunks = sum(1 for line in f if line.strip())
    ready = chunks > 0 and s.member_db.exists()
    return {"status": "ok" if ready else "degraded", "version": __version__, "provider": active_provider_label(),
            "embedding_provider": s.embedding_provider, "indexed_chunks": chunks,
            "member_db": s.member_db.exists(), "langsmith_tracing": s.langsmith_tracing}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    resp, _ = pipeline.ask(req.question, req.member_id, use_cache=req.use_cache)
    return resp


@app.post("/approve", response_model=AskResponse)
def approve(req: ApproveRequest) -> AskResponse:
    try:
        return pipeline.approve(req.thread_id, req.approved, req.reviewer, req.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="No pending answer with that thread_id (already "
                            "reviewed, or the server restarted).") from None


@app.get("/pending")
def pending() -> list[dict]:
    return [{"thread_id": tid, "masked_question": run.masked_question} for tid, run in pipeline.PENDING.items()]


@app.get("/metrics")
def metrics() -> dict:
    return pipeline.METRICS.snapshot()
