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
