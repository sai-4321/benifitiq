"""CMS Blue Button 2.0 OAuth client (sandbox) - consent-based access to synthetic Medicare claims.

Blue Button follows OAuth 2.0 authorization code with PKCE (S256 only) for confidential clients:

  1. authorize_url()  -> the enrollee logs in on Medicare's page and approves access
  2. exchange_code()  -> our server swaps the one-time code for an access token (+ patient id)
  3. refresh()        -> new access token later (13-month / research apps get refresh tokens)

Tokens are stored per BenefitIQ member under data/bluebutton/ (git-ignored, owner-only permissions)
and are never logged. Docs: https://bluebutton.cms.gov/api-documentation/authorization/
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlparse

from app.config import get_settings
from app.core.logging_config import get_logger

log = get_logger(__name__)

MEMBER_ID_RE = re.compile(r"^M\d{4}$")


class BlueButtonError(RuntimeError):
    """Configuration, consent or token problem - message is safe to show (no secrets)."""


class HttpSession(Protocol):
    def post(self, url: str, **kwargs): ...


# ---------------------------------------------------------------- PKCE + state
def make_code_verifier() -> str:
    """43-128 chars from the unreserved set (RFC 7636). token_urlsafe(64) -> 86 chars."""
    return secrets.token_urlsafe(64)


def code_challenge(verifier: str) -> str:
    """S256: BASE64URL(SHA256(ASCII(verifier))) without '=' padding (RFC 7636)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def make_state() -> str:
    """Blue Button requires >= 122 bits of entropy; 32 random bytes = 256 bits."""
    return secrets.token_urlsafe(32)


@dataclass
class AuthRequest:
    url: str
    state: str
    verifier: str


def _urls() -> tuple[str, str]:
    s = get_settings()
    base = f"{s.bluebutton_base_url.rstrip('/')}/{s.bluebutton_api_version}"
    return f"{base}/o/authorize/", f"{base}/o/token/"


def _client() -> tuple[str, str]:
    s = get_settings()
    secret = s.bluebutton_client_secret.get_secret_value()
    if not s.bluebutton_client_id or not secret or "paste-your" in secret:
        raise BlueButtonError("Blue Button is not configured: set BLUEBUTTON_CLIENT_ID and BLUEBUTTON_CLIENT_SECRET "
                              "in .env (see docs/BLUE_BUTTON_SETUP.md).")
    return s.bluebutton_client_id, secret


def authorize_url() -> AuthRequest:
    client_id, _ = _client()
    verifier, state = make_code_verifier(), make_state()
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": get_settings().bluebutton_redirect_uri,
        "state": state,
        "code_challenge": code_challenge(verifier),
        "code_challenge_method": "S256",
    }
    return AuthRequest(url=f"{_urls()[0]}?{urlencode(params)}", state=state, verifier=verifier)


def parse_callback(callback_url: str, expected_state: str) -> str:
    """Return the authorization code from the redirect URL; reject CSRF/denied/malformed callbacks."""
    query = parse_qs(urlparse(callback_url.strip()).query)
    if "error" in query:
        raise BlueButtonError(f"Access was not granted: {query['error'][0]}")
    state = query.get("state", [""])[0]
    if not secrets.compare_digest(state, expected_state):
        raise BlueButtonError("State mismatch - this callback was not started by this session. Try again.")
    code = query.get("code", [""])[0]
    if not code:
        raise BlueButtonError("No authorization code in the callback URL.")
    return code


# ---------------------------------------------------------------- tokens
@dataclass
class Token:
    access_token: str
    patient: str                       # Blue Button FHIR Patient id of the synthetic enrollee
    expires_at: float
    refresh_token: str | None = None
    scope: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: dict, patient: str | None = None) -> Token:
        patient = str(data.get("patient") or patient or "")
        if "access_token" not in data or not patient:
            raise BlueButtonError("Unexpected token response from Blue Button (missing access_token or patient).")
        return cls(access_token=data["access_token"], patient=patient,
                   expires_at=time.time() + float(data.get("expires_in", 3600)),
                   refresh_token=data.get("refresh_token"), scope=data.get("scope", ""))

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - 60       # refresh a minute early

    def __repr__(self) -> str:                           # never print tokens
        return f"Token(patient={self.patient!r}, expired={self.expired}, refresh={bool(self.refresh_token)})"


def _token_request(session: HttpSession, data: dict, patient: str | None = None) -> Token:
    client_id, secret = _client()
    resp = session.post(_urls()[1], data=data, auth=(client_id, secret),
                        timeout=get_settings().bluebutton_timeout_seconds)
    if getattr(resp, "status_code", 200) >= 400:
        raise BlueButtonError(f"Blue Button token request failed (HTTP {resp.status_code}). "
                              "If this was a refresh, reconnect the member.")
    return Token.from_response(resp.json(), patient=patient)


def exchange_code(session: HttpSession, code: str, verifier: str) -> Token:
    return _token_request(session, {
        "grant_type": "authorization_code", "code": code, "code_verifier": verifier,
        "redirect_uri": get_settings().bluebutton_redirect_uri,
    })


def refresh(session: HttpSession, token: Token) -> Token:
    if not token.refresh_token:
        raise BlueButtonError("Access expired and this app has no refresh token. Reconnect the member.")
    # Refresh tokens are single use: the response carries a new one, which the caller must save.
    return _token_request(session, {"grant_type": "refresh_token", "refresh_token": token.refresh_token},
                          patient=token.patient)


# ---------------------------------------------------------------- per-member store
def _token_file(member_id: str) -> Path:
    if not MEMBER_ID_RE.fullmatch(member_id):
        raise BlueButtonError("Invalid member id.")            # also blocks path tricks like ../
    return get_settings().bluebutton_token_path / f"{member_id}.json"


def save_token(member_id: str, token: Token) -> Path:
    path = _token_file(member_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(token)), encoding="utf-8")
    os.chmod(tmp, 0o600)                                      # owner read/write only
    tmp.replace(path)
    log.info("Blue Button token saved for %s", member_id)    # never the token itself
    return path


def load_token(member_id: str) -> Token | None:
    path = _token_file(member_id)
    if not path.exists():
        return None
    return Token(**json.loads(path.read_text(encoding="utf-8")))


def is_connected(member_id: str) -> bool:
    try:
        return load_token(member_id) is not None
    except BlueButtonError:
        return False


def disconnect(member_id: str) -> bool:
    path = _token_file(member_id)
    if path.exists():
        path.unlink()
        return True
    return False


def valid_token(session: HttpSession, member_id: str) -> Token:
    """Stored token for the member, refreshed (and re-saved) if it has expired."""
    token = load_token(member_id)
    if token is None:
        raise BlueButtonError(f"Member {member_id} has not connected Medicare claims (Blue Button).")
    if token.expired:
        token = refresh(session, token)
        save_token(member_id, token)
    return token
