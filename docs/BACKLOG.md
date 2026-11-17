# BenefitIQ backlog

Tickets are tracked as GitHub Issues (`BIQ-n` = issue #n). Each ticket gets its own branch and pull request.
Branch names: `feature/BIQ-n-short-name`, `chore/...`, `docs/...`. Commits follow Conventional Commits.

## Sprint 1 - Foundation (milestone v0.1.0)
| ID | Title | Acceptance criteria |
|---|---|---|
| BIQ-1 | Project skeleton, tooling and CI | `requirements.txt` pinned; `pyproject.toml` with pytest + ruff config; package folders; GitHub Actions runs ruff + pytest on every PR; one smoke test passes |
| BIQ-2 | Typed settings from `.env` | `app/config.py` with pydantic-settings; `.env.example` documents every setting; invalid values fail fast; tests |
| BIQ-3 | Claude on Bedrock LLM layer | `ChatBedrockConverse` Sonnet 5 primary, Haiku 4.5 fallback via `.with_fallbacks`; Titan V2 embeddings; offline fake embeddings for CI; botocore timeouts + adaptive retries; tests with no AWS calls |
| BIQ-4 | PII/PHI masking guardrail | Masks SSN, MBI, member ID, email, phone, DOB, cued names; reversible mapping; tests |
| BIQ-5 | Prompt-injection guardrail | Rule-based detector (+ optional LLM check); no false positives on normal benefit questions; tests |

## Sprint 2 - Core AI (milestone v0.2.0)
| ID | Title | Acceptance criteria |
|---|---|---|
| BIQ-6 | Download public Medicare documents | Script downloads 4 Medicare.gov PDFs with checksums and a clear manual fallback |
| BIQ-7 | Synthetic member database | Faker + SQLite: plans, members, prior-auth rules; deterministic seed; no real data |
| BIQ-8 | PDF loading and chunking | Page-level loading, 800/120 recursive chunks with stable IDs and file+page metadata; tests |
| BIQ-9 | Hybrid retrieval | Chroma vector search + BM25 fused with RRF (k=60); ingest script; degrades to BM25 if embeddings fail; tests |
| BIQ-10 | Structured output + cache + tracing | Pydantic schemas; JSON extract/validate/retry; diskcache with TTL; LangSmith tracing with Bedrock cost estimate; tests |
| BIQ-11 | MCP member/plan tool server | FastMCP server with 3 typed tools over stdio, read-only parameterized SQL, minimum-necessary fields; tests incl. real stdio round trip |
| BIQ-12 | LangGraph multi-agent workflow | Router, retriever, lookup, summarizer, refuse, human-review interrupt; shared state; checkpointer; routing tests |

## Sprint 3 - Ship it (milestone v1.0.0)
| ID | Title | Acceptance criteria |
|---|---|---|
| BIQ-13 | FastAPI service | `/health`, `/ask`, `/approve`, `/pending`, `/metrics`; PII never reaches the LLM (spy test); API tests |
| BIQ-14 | Streamlit UI | Chat, citations, "what the LLM saw", supervisor approve/reject |
| BIQ-15 | Evaluation harness | 44-question golden set; hit@k, MRR, citation accuracy, keyword recall, groundedness judge, latency, cost; CSV/JSON output |
| BIQ-16 | Docker + release | Dockerfile, docker-compose, CI docker build, full README with architecture and AWS production mapping, CHANGELOG, tag v1.0.0 |
