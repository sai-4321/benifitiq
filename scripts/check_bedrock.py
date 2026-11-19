"""Live smoke check for your AWS setup (costs a fraction of a cent).

Run once after filling in .env:
    python scripts/check_bedrock.py

It calls Claude Sonnet, the Haiku fallback and Titan embeddings once each and prints a
clear hint for the most common setup errors. Not part of pytest / CI (needs AWS).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.llm_factory import build_chat_model, extract_text, get_embeddings  # noqa: E402

HINTS = {
    "use case details": "Submit Anthropic's one-time use-case form (Bedrock console -> Model catalog -> Claude).",
    "on-demand throughput": "Use the inference profile ID with the 'us.' prefix in BEDROCK_CHAT_MODEL_ID.",
    "not authorized": "Your key/user needs the AmazonBedrockLimitedAccess policy.",
    "security token": "API key expired or wrong. Generate a new one in Bedrock console -> API keys.",
    "Throttling": "Rate limit hit. Wait a minute, or request a quota increase in Service Quotas.",
}


def hint_for(err: Exception) -> str:
    text = str(err)
    for needle, hint in HINTS.items():
        if needle.lower() in text.lower():
            return hint
    return "See the error table in the build guide."


def check(name: str, fn) -> bool:
    try:
        result = fn()
        print(f"  OK    {name}: {result}")
        return True
    except NoCredentialsError:
        print(f"  FAIL  {name}: no AWS credentials. Set AWS_BEARER_TOKEN_BEDROCK in .env or run 'aws configure'.")
    except (ClientError, BotoCoreError) as err:
        print(f"  FAIL  {name}: {err}\n        -> {hint_for(err)}")
    return False


def main() -> int:
    setup_logging("WARNING")
    # langchain-aws logs a full traceback for embedding errors; we print our own short message instead.
    logging.getLogger("langchain_aws").setLevel(logging.CRITICAL)
    s = get_settings()
    print(f"Region: {s.aws_region}")
    prompt = "Reply with exactly: Bedrock OK"
    results = [
        check(f"chat    {s.bedrock_chat_model_id}",
              lambda: extract_text(build_chat_model(s.bedrock_chat_model_id).invoke(prompt))),
    ]
    if s.fallback_enabled:
        results.append(check(f"fallback {s.bedrock_fallback_model_id}",
                             lambda: extract_text(build_chat_model(s.bedrock_fallback_model_id).invoke(prompt))))
    results.append(check(f"embed   {s.bedrock_embed_model_id}",
                         lambda: f"{len(get_embeddings().embed_query('Medicare Part B'))} dimensions"))
    print("All good - Bedrock is ready." if all(results) else "Fix the FAIL lines above, then run again.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
