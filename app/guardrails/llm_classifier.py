"""Claude-based prompt-injection classifier: layer 2 of the injection guard (INJECTION_LLM_CHECK=true).

Plugs into check_injection(text, classifier=...). It only ever sees the PII-masked question.
"""
from __future__ import annotations


def claude_injection_classifier(text: str) -> tuple[bool, str]:
    from app.agents.prompts import INJECTION_SYSTEM
    from app.core.structured_output import invoke_structured
    from app.llm_factory import get_llm
    from app.schemas import InjectionVerdict

    verdict, _ = invoke_structured(
        get_llm(), INJECTION_SYSTEM, f"<user_text>\n{text}\n</user_text>", InjectionVerdict,
        node="injection_classifier", max_retries=1,
    )
    return verdict.is_injection, verdict.reason
