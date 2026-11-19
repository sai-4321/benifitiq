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
