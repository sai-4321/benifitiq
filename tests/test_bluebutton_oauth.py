"""Blue Button OAuth tests - no network: a fake HTTP session records requests and returns canned tokens."""
from __future__ import annotations

import json
import os
import stat
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr

from app.config import get_settings
from app.integrations import bluebutton as bb


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, *responses: FakeResponse):
        self.responses, self.calls = list(responses), []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def configured(monkeypatch, tmp_path):
    s = get_settings()
    monkeypatch.setattr(s, "bluebutton_client_id", "client-123")
    monkeypatch.setattr(s, "bluebutton_client_secret", SecretStr("secret-xyz"))
    monkeypatch.setattr(s, "bluebutton_token_dir", str(tmp_path / "bb"))
    return s


def test_pkce_matches_rfc7636_test_vector():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert bb.code_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    assert 43 <= len(bb.make_code_verifier()) <= 128


def test_state_has_enough_entropy_and_is_random():
    a, b = bb.make_state(), bb.make_state()
    assert a != b and len(a) >= 40                      # 32 bytes -> 43 url-safe chars, > 122 bits


def test_authorize_url_has_required_parameters():
    req = bb.authorize_url()
    url = urlparse(req.url)
    q = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert url.geturl().startswith("https://sandbox.bluebutton.cms.gov/v2/o/authorize/")
    assert q["response_type"] == "code" and q["client_id"] == "client-123"
    assert q["code_challenge_method"] == "S256" and q["code_challenge"] == bb.code_challenge(req.verifier)
    assert q["state"] == req.state and q["redirect_uri"] == "http://localhost:8765/callback"
    assert "secret-xyz" not in req.url


def test_missing_configuration_explains_fix(monkeypatch, configured):
    monkeypatch.setattr(configured, "bluebutton_client_secret", SecretStr("paste-your-bluebutton-client-secret-here"))
    with pytest.raises(bb.BlueButtonError, match="BLUEBUTTON_CLIENT_ID"):
        bb.authorize_url()


def test_callback_parsing_checks_state_and_errors():
    good = "http://localhost:8765/callback?code=abc&state=S" + "x" * 40
    assert bb.parse_callback(good, "S" + "x" * 40) == "abc"
    with pytest.raises(bb.BlueButtonError, match="State mismatch"):
        bb.parse_callback(good, "S" + "y" * 40)                     # CSRF / stale callback
    with pytest.raises(bb.BlueButtonError, match="not granted"):
        bb.parse_callback("http://localhost:8765/callback?error=access_denied&state=s", "s")
    with pytest.raises(bb.BlueButtonError, match="No authorization code"):
        bb.parse_callback("http://localhost:8765/callback?state=s", "s")


def test_exchange_code_uses_basic_auth_and_pkce_verifier():
    session = FakeSession(FakeResponse({"access_token": "at-1", "expires_in": 3600, "patient": "-10000000000001",
                                        "refresh_token": "rt-1", "scope": "patient/ExplanationOfBenefit.read"}))
    token = bb.exchange_code(session, "the-code", "the-verifier")
    url, kwargs = session.calls[0]
    assert url == "https://sandbox.bluebutton.cms.gov/v2/o/token/"
    assert kwargs["auth"] == ("client-123", "secret-xyz")
    assert kwargs["data"]["grant_type"] == "authorization_code" and kwargs["data"]["code_verifier"] == "the-verifier"
    assert token.patient == "-10000000000001" and not token.expired
    assert "at-1" not in repr(token) and "rt-1" not in repr(token)


def test_token_error_is_clean():
    with pytest.raises(bb.BlueButtonError, match="HTTP 400"):
        bb.exchange_code(FakeSession(FakeResponse({"error": "invalid_grant"}, status=400)), "c", "v")


def test_store_load_and_owner_only_permissions(configured):
    token = bb.Token(access_token="at", patient="-1", expires_at=9e12, refresh_token="rt")
    path = bb.save_token("M1001", token)
    assert bb.load_token("M1001") == token and bb.is_connected("M1001")
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert bb.disconnect("M1001") and not bb.is_connected("M1001")


@pytest.mark.parametrize("bad", ["../../etc/passwd", "M1", "m1001", "M1001.json"])
def test_member_id_cannot_escape_token_folder(bad):
    with pytest.raises(bb.BlueButtonError):
        bb.load_token(bad)
    assert not bb.is_connected(bad)


def test_expired_token_is_refreshed_and_new_refresh_token_saved():
    bb.save_token("M1002", bb.Token(access_token="old", patient="-7", expires_at=0, refresh_token="rt-old"))
    session = FakeSession(FakeResponse({"access_token": "new", "expires_in": 3600, "refresh_token": "rt-new"}))
    token = bb.valid_token(session, "M1002")
    assert token.access_token == "new" and token.patient == "-7"        # patient carried over
    assert session.calls[0][1]["data"] == {"grant_type": "refresh_token", "refresh_token": "rt-old"}
    assert json.loads((get_settings().bluebutton_token_path / "M1002.json").read_text())["refresh_token"] == "rt-new"


def test_expired_without_refresh_token_asks_to_reconnect():
    bb.save_token("M1003", bb.Token(access_token="old", patient="-7", expires_at=0))
    with pytest.raises(bb.BlueButtonError, match="Reconnect"):
        bb.valid_token(FakeSession(), "M1003")


def test_not_connected_member():
    with pytest.raises(bb.BlueButtonError, match="has not connected"):
        bb.valid_token(FakeSession(), "M1004")
