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
