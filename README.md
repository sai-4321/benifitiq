# BenefitIQ

Health plan benefits & policy assistant for care teams: RAG + LangGraph multi-agent + MCP tools,
running on Claude (Amazon Bedrock).

> Work in progress. Built ticket by ticket - see [docs/BACKLOG.md](docs/BACKLOG.md) and the Issues tab.

**Data policy:** public Medicare.gov documents and synthetic members only. No real patient data.

## Configuration

All settings live in one typed class, `app/config.py`, loaded from `.env`.

```powershell
Copy-Item .env.example .env   # Windows PowerShell   (macOS/Linux: cp .env.example .env)
```

Then paste your Bedrock API key and region. `.env` is git-ignored - never commit it.
Real environment variables override `.env`, which overrides the defaults in code.

## AI models (all on Amazon Bedrock)

| Role | Model | Setting |
|---|---|---|
| Main LLM | Claude Sonnet 5 | `BEDROCK_CHAT_MODEL_ID=us.anthropic.claude-sonnet-5` |
| Automatic fallback | Claude Haiku 4.5 | `BEDROCK_FALLBACK_MODEL_ID` (`none` = off) |
| Embeddings | Amazon Titan Text Embeddings V2 (1024-d) | `BEDROCK_EMBED_MODEL_ID` |

If Sonnet fails (throttling, timeout, outage), botocore retries with back-off and LangChain re-sends the
same request to Haiku. Check your AWS setup once with:

```powershell
python scripts/check_bedrock.py
```

Tests never call AWS: they stub Bedrock with botocore's `Stubber`.

## Guardrails: PII/PHI masking

Before any text reaches Claude, logs or the cache, `app/guardrails/pii.py` swaps identifiers for placeholders:

```
Patient John Smith (M1003), SSN 123-45-6789, call 469-555-0134
-> Patient [NAME_1] ([MEMBER_ID_1]), SSN [SSN_1], call [PHONE_1]
```

Detected: SSN, Medicare Beneficiary Identifier (MBI), email, phone, date of birth, member ID, and names after
cue words (patient, member, named, Dr., Mr./Mrs./Ms.). The placeholder map never leaves the API process, so the
answer can be un-masked only for the authorized user. Production upgrade: Amazon Comprehend Medical `DetectPHI`
or Amazon Bedrock Guardrails sensitive-information filters.

## Guardrails: prompt injection

`app/guardrails/injection.py` checks every (masked) question before any LLM call:

1. **Rules** (always on, ~0 ms): named regex rules for instruction override, system-prompt leaks, role hijack,
   jailbreak keywords, bulk data exfiltration ("list all members' SSNs"), SQL/code and fake system tags.
2. **Classifier** (optional): a pluggable `(text) -> (is_injection, reason)` hook for attacks the rules miss.
   If it errors, the rules-only result stands - the guard never crashes a request.

Blocks are explainable (rule names are returned and logged; the attack text never is). The test suite checks
both directions: every attack is blocked, and look-alike normal questions ("Can a plan ignore a prior
authorization request?") are not.

## Retrieval (hybrid RAG)

```powershell
python scripts/download_docs.py   # 4 public Medicare.gov PDFs -> data/raw/
python scripts/ingest.py          # pages -> 800/120 chunks -> Titan embeddings -> ChromaDB + chunks.jsonl
```

Each question runs two searches and fuses them with Reciprocal Rank Fusion (`score = sum 1/(60 + rank)`):
vector search (Titan embeddings in ChromaDB) finds meaning ("mobility aid" -> walkers), BM25 finds exact terms
("Part B", "63 days"). If the embedding call fails, keyword results still come back.
Production mapping: Amazon OpenSearch Serverless hybrid query, or Bedrock Knowledge Bases.
