# Changelog

All notable changes to BenefitIQ. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-11-19 - Sprint 3: Ship it

### Added
- FastAPI service: `/health`, `/ask`, `/approve`, `/pending`, `/metrics`, generic 500 handler (BIQ-13, #31)
- Streamlit UI with citations, masked-prompt view, supervisor approval and live metrics (BIQ-14, #32)
- Evaluation harness: 44-question golden set; retrieval, answer, groundedness and system metrics (BIQ-15, #33)
- Docker image (non-root, healthcheck), docker-compose for API + UI, CI image build with smoke test,
  full README and AWS setup guide (BIQ-16, #34)

## [0.2.0] - 2026-11-19 - Sprint 2: Core AI

### Added
- Public Medicare.gov document downloader with atomic writes, PDF validation and SHA-256 manifest;
  `docs/DATA_SOURCES.md` (BIQ-6, #23)
- Synthetic member database (Faker + SQLite, fixed seed, every row flagged synthetic) (BIQ-7, #24)
- Page-level PDF loading and 800/120 page-bounded chunking with stable citation ids (BIQ-8, #25)
- Hybrid retrieval: ChromaDB + BM25 fused with Reciprocal Rank Fusion, keyword-only fallback, ingest script (BIQ-9, #26)
- Pydantic contracts, validate-and-retry JSON output, diskcache response cache, LangSmith tracing with redacted
  inputs and Bedrock cost estimates (BIQ-10, #27)
- MCP server with three typed, read-only member/plan tools and a stdio client (BIQ-11, #28)
- LangGraph multi-agent workflow with interrupt-based human review, citation validation, Claude injection
  classifier and the end-to-end request pipeline (BIQ-12, #29)

### Changed
- PII masking now runs before the prompt-injection guard, so no downstream component sees raw identifiers (#29)

### Fixed
- Raw member ID from tool results no longer reaches the LLM prompt (#29)

## [0.1.0] - 2026-11-19 - Sprint 1: Foundation

### Added
- Project skeleton, pinned tooling, pytest + ruff config and GitHub Actions CI (BIQ-1, #17)
- Typed settings from `.env` with validation and `SecretStr` keys; `.env.example` (BIQ-2, #18)
- Claude on Amazon Bedrock LLM layer: Sonnet 5 via Converse with Haiku 4.5 fallback, Titan Text Embeddings V2,
  offline hash embeddings, botocore timeouts and adaptive retries, `scripts/check_bedrock.py` (BIQ-3, #19)
- PII/PHI masking guardrail with reversible placeholders and member-ID extraction (BIQ-4, #20)
- Prompt-injection guardrail with seven named rules and a pluggable classifier hook (BIQ-5, #21)

### Fixed
- SQL injection rule no longer blocks plain English such as "select the best plan from the list" (#21)

[Unreleased]: https://github.com/sai-4321/benifitiq/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/sai-4321/benifitiq/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/sai-4321/benifitiq/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sai-4321/benifitiq/releases/tag/v0.1.0
