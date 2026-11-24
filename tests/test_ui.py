"""Streamlit UI smoke tests with AppTest (runs the script headless; no browser, no server)."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


def test_ui_renders_and_explains_missing_api(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")     # nothing listens here
    at = AppTest.from_file(UI, default_timeout=30).run()
    assert not at.exception
    assert "BenefitIQ" in at.title[0].value
    assert any("Cannot reach the API" in e.value for e in at.error)
    assert len(at.button) >= 5                               # demo question buttons


def test_demo_button_fills_the_form(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(UI, default_timeout=30).run()
    demo = next(b for b in at.button if b.label.startswith("Does member M1002"))
    demo.click().run()
    assert at.session_state.member_id == "M1002"
    assert "prior authorization" in at.session_state.question


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


HEALTH = {"status": "ok", "version": "1.0.0", "provider": "bedrock:test", "embedding_provider": "fake",
          "indexed_chunks": 5, "member_db": True, "langsmith_tracing": False}
METRICS = {"requests": 2, "cache_hits": 0, "blocked": 0, "pending_now": 0, "avg_latency_ms": 100,
           "total_cost_usd": 0.03}


def test_sidebar_shows_status_without_leaking_widget_help(monkeypatch):
    """Regression: a one-line `st.success(...) if ok else st.warning(...)` made Streamlit 'magic' print
    DeltaGenerator's docstring in the sidebar."""
    import requests

    def fake_request(method, url, **kwargs):
        return _FakeResponse(HEALTH if url.endswith("/health") else METRICS)

    monkeypatch.setattr(requests, "request", fake_request)
    at = AppTest.from_file(UI, default_timeout=30).run()
    assert not at.exception
    assert any("API ok" in s.value for s in at.success)
    page_text = " ".join(str(getattr(el, "value", "")) for el in at.main) + " ".join(
        str(getattr(el, "value", "")) for el in at.sidebar)
    assert "DeltaGenerator" not in page_text
    assert len(at.get("help")) == 0                      # no st.help / magic object dumps anywhere
