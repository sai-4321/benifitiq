"""Medicare claims (Blue Button FHIR) tests - fixtures only, no network."""
from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.config import get_settings
from app.integrations import bluebutton as bb
from app.integrations import claims

FIX = Path(__file__).parent / "fixtures" / "bluebutton"
PAGE1 = json.loads((FIX / "eob_page1.json").read_text())
PAGE2 = json.loads((FIX / "eob_page2.json").read_text())
TODAY = date(2026, 11, 24)


class Resp:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload or {}, status

    def json(self):
        return self.payload


class FakeSession:
    """GET answers come from a queue; POST is the token endpoint (refresh)."""

    def __init__(self, gets, posts=()):
        self.gets, self.posts, self.get_calls, self.post_calls = list(gets), list(posts), [], []

    def get(self, url, **kw):
        self.get_calls.append((url, kw))
        return self.gets.pop(0)

    def post(self, url, **kw):
        self.post_calls.append((url, kw))
        return self.posts.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def connected(monkeypatch, tmp_path):
    s = get_settings()
    monkeypatch.setattr(s, "bluebutton_client_id", "cid")
    monkeypatch.setattr(s, "bluebutton_client_secret", SecretStr("secret"))
    monkeypatch.setattr(s, "bluebutton_token_dir", str(tmp_path / "bb"))
    bb.save_token("M1001", bb.Token(access_token="at-1", patient="-10000000000001", expires_at=9e12,
                                    refresh_token="rt-1"))


def all_eobs():
    return [e["resource"] for e in PAGE1["entry"] + PAGE2["entry"]]


def test_parse_keeps_only_minimum_necessary_fields():
    lines = claims.parse_eob(PAGE1["entry"][1]["resource"])
    assert [ln.code for ln in lines] == ["G0008", "90662"]
    assert lines[0].service_date == "2025-10-02" and lines[0].claim_type == "professional (Part B)"
    dumped = json.dumps(claims.summarize(all_eobs(), today=TODAY))
    for leaked in ("123.45", "80.0", "9999999999", "Z1231", "-10000000000001"):   # amounts, NPI, diagnosis, patient
        assert leaked not in dumped


def test_pde_and_serviced_date_are_supported():
    line = claims.parse_eob(PAGE2["entry"][2]["resource"])[0]
    assert (line.claim_type, line.service_date, line.code) == ("prescription drug (Part D)", "2026-05-05",
                                                               "00093721401")


@pytest.mark.parametrize("asked, expected", [("mammogram", "mammogram"), ("Mammography", "mammogram"),
                                             ("flu", "flu shot"), ("colonoscopy", "colorectal cancer screening"),
                                             (None, None)])
def test_service_aliases(asked, expected):
    assert claims.normalize_service(asked) == expected


def test_summary_filters_by_service_and_window_newest_first():
    s = claims.summarize(all_eobs(), service="mammogram", months=12, today=TODAY)
    assert s["matching_lines"] == 1 and s["most_recent"] == "2026-03-14"     # 2023 mammogram is outside 12 months
    s24 = claims.summarize(all_eobs(), service="mammogram", months=48, today=TODAY)
    assert [ln["service_date"] for ln in s24["lines"]] == ["2026-03-14", "2023-02-01"]


def test_code_map_matches_even_when_description_differs():
    s = claims.summarize(all_eobs(), service="flu shot", months=24, today=TODAY)
    assert {ln["code"] for ln in s["lines"]} == {"G0008", "90662"}


def test_no_match_is_explicit_not_an_error():
    s = claims.summarize(all_eobs(), service="bone density", months=24, today=TODAY)
    assert s["matching_lines"] == 0 and s["most_recent"] is None and s["claims_in_window"] >= 4


def test_fetch_follows_next_links_and_sends_bearer_token():
    session = FakeSession([Resp(PAGE1), Resp(PAGE2)])
    eobs = claims.fetch_eobs(session, "M1001")
    assert len(eobs) == 5 and len(session.get_calls) == 2
    first_url, first_kw = session.get_calls[0]
    assert first_url == "https://sandbox.bluebutton.cms.gov/v2/fhir/ExplanationOfBenefit/"
    assert first_kw["params"] == {"patient": "-10000000000001"}
    assert first_kw["headers"]["Authorization"] == "Bearer at-1"
    assert session.get_calls[1][0].endswith("startIndex=2") and session.get_calls[1][1]["params"] is None


def test_401_refreshes_once_then_retries():
    session = FakeSession([Resp(status=401), Resp(PAGE2)],
                          posts=[Resp({"access_token": "at-2", "expires_in": 3600, "refresh_token": "rt-2"})])
    eobs = claims.fetch_eobs(session, "M1001")
    assert len(eobs) == 3 and session.get_calls[1][1]["headers"]["Authorization"] == "Bearer at-2"
    assert bb.load_token("M1001").refresh_token == "rt-2"


def test_paging_is_bounded(monkeypatch):
    monkeypatch.setattr(claims, "MAX_PAGES", 2)
    session = FakeSession([Resp(PAGE1), Resp(PAGE1), Resp(PAGE1)])
    claims.fetch_eobs(session, "M1001")
    assert len(session.get_calls) == 2


def test_http_error_is_clean():
    with pytest.raises(bb.BlueButtonError, match="HTTP 503"):
        claims.fetch_eobs(FakeSession([Resp(status=503)]), "M1001")


def test_mcp_tool_returns_summary(monkeypatch):
    from app.mcp_server import server

    monkeypatch.setattr(server, "_http_session", lambda: FakeSession([Resp(PAGE1), Resp(PAGE2)]))
    out = asyncio.run(server.mcp.call_tool("get_claims_summary",
                                           {"member_id": "M1001", "service": "mammogram", "months": 60}))
    content = out[0] if isinstance(out, tuple) else out          # (content, structured) on newer mcp versions
    result = json.loads(content[0].text)                         # same JSON text the stdio client receives
    assert result["matching_lines"] >= 1 and result["lines"][0]["code"] == "77067"


def test_mcp_tool_not_connected_and_bad_input(monkeypatch):
    from mcp.server.fastmcp.exceptions import ToolError

    from app.mcp_server import server

    monkeypatch.setattr(server, "_http_session", lambda: FakeSession([]))
    with pytest.raises(ToolError, match="has not connected"):
        asyncio.run(server.mcp.call_tool("get_claims_summary", {"member_id": "M1002"}))
    with pytest.raises(ToolError):
        asyncio.run(server.mcp.call_tool("get_claims_summary", {"member_id": "M1001", "service": "x'; DROP"}))
    with pytest.raises(ToolError):
        asyncio.run(server.mcp.call_tool("get_claims_summary", {"member_id": "M1001", "months": 999}))
