"""Medicare claims history from CMS Blue Button 2.0 (FHIR R4, CARIN Blue Button profiles).

fetch_eobs()      GET ExplanationOfBenefit?patient=<id>, following Bundle "next" links (bounded),
                  refreshing the token once on HTTP 401.
parse_eob()       one EOB -> claim lines with ONLY minimum-necessary facts:
                  service date, claim type, procedure/drug code and its description.
                  No amounts, provider NPIs, diagnoses or patient identifiers.
summarize()       filter by look-back window and optional service, newest first, capped.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Protocol

from app.config import get_settings
from app.core.logging_config import get_logger
from app.integrations import bluebutton as bb

log = get_logger(__name__)

MAX_PAGES = 10
MAX_LINES = 25

# Common Medicare preventive services -> HCPCS/CPT codes. Lets "mammogram" match claim lines even when the
# code description uses different words ("Screening digital breast tomosynthesis").
SERVICE_CODES: dict[str, set[str]] = {
    "mammogram": {"77067", "77063", "77065", "77066", "G0202", "G0279"},
    "flu shot": {"G0008", "90653", "90656", "90658", "90661", "90662", "90673", "90674", "90682", "90686",
                 "90688", "90694", "90756"},
    "colorectal cancer screening": {"G0105", "G0121", "G0104", "G0106", "G0120", "G0328", "81528", "45378",
                                    "82270", "G0327"},
    "bone density": {"77080", "77081", "77078", "G0130"},
    "diabetes screening": {"82947", "82950", "82951", "83036"},
    "cardiovascular screening": {"80061", "82465", "83718", "84478"},
    "lung cancer screening": {"71271", "G0296"},
    "wellness visit": {"G0438", "G0439", "G0402"},
    "depression screening": {"G0444"},
}
ALIASES = {"mammography": "mammogram", "flu": "flu shot", "influenza": "flu shot", "colonoscopy":
           "colorectal cancer screening", "colorectal": "colorectal cancer screening", "bone mass": "bone density",
           "dexa": "bone density", "diabetes": "diabetes screening", "cholesterol": "cardiovascular screening",
           "lung": "lung cancer screening", "annual wellness": "wellness visit", "welcome to medicare":
           "wellness visit", "depression": "depression screening"}

CLAIM_TYPE_LABELS = {"CARRIER": "professional (Part B)", "OUTPATIENT": "outpatient (Part B)",
                     "INPATIENT": "inpatient (Part A)", "SNF": "skilled nursing (Part A)", "HHA": "home health",
                     "HOSPICE": "hospice", "DME": "durable medical equipment", "PDE": "prescription drug (Part D)"}


class HttpSession(Protocol):
    def get(self, url: str, **kwargs): ...
    def post(self, url: str, **kwargs): ...


@dataclass(frozen=True)
class ClaimLine:
    service_date: str            # ISO date
    claim_type: str
    code: str
    description: str


# ---------------------------------------------------------------- fetch
def _fhir_base() -> str:
    s = get_settings()
    return f"{s.bluebutton_base_url.rstrip('/')}/{s.bluebutton_api_version}/fhir"


def fetch_eobs(session: HttpSession, member_id: str) -> list[dict]:
    """All ExplanationOfBenefit resources for the member's connected enrollee (bounded paging)."""
    token = bb.valid_token(session, member_id)
    url: str | None = f"{_fhir_base()}/ExplanationOfBenefit/"
    params: dict | None = {"patient": token.patient}
    resources: list[dict] = []
    refreshed = False
    pages = 0
    while url and pages < MAX_PAGES:
        resp = session.get(url, params=params, headers={"Authorization": f"Bearer {token.access_token}",
                                                         "Accept": "application/fhir+json"},
                           timeout=get_settings().bluebutton_timeout_seconds)
        if resp.status_code == 401 and not refreshed:          # token revoked/expired early -> refresh once
            token = bb.refresh(session, token)
            bb.save_token(member_id, token)
            refreshed = True
            continue
        if resp.status_code >= 400:
            raise bb.BlueButtonError(f"Blue Button claims request failed (HTTP {resp.status_code}).")
        bundle = resp.json()
        resources += [e["resource"] for e in bundle.get("entry", []) if "resource" in e]
        url = next((link["url"] for link in bundle.get("link", []) if link.get("relation") == "next"), None)
        params = None                                           # "next" links already carry the query
        pages += 1
    if url:
        log.warning("Claims paging stopped after %d pages", MAX_PAGES)
    return resources


# ---------------------------------------------------------------- parse
def _claim_type(eob: dict) -> str:
    for coding in eob.get("type", {}).get("coding", []):
        if coding.get("system", "").endswith("/eob-type"):
            return CLAIM_TYPE_LABELS.get(coding.get("code", ""), coding.get("code", "unknown").lower())
    codings = eob.get("type", {}).get("coding", [])
    return codings[0].get("code", "unknown") if codings else "unknown"


def _item_date(item: dict, eob: dict) -> str | None:
    raw = (item.get("servicedDate") or item.get("servicedPeriod", {}).get("start")
           or eob.get("billablePeriod", {}).get("start") or "")
    return raw[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", raw) else None


def parse_eob(eob: dict) -> list[ClaimLine]:
    claim_type = _claim_type(eob)
    lines = []
    for item in eob.get("item", []):
        when = _item_date(item, eob)
        codings = item.get("productOrService", {}).get("coding", [])
        coding = next((c for c in codings if c.get("code")), None)
        if not when or not coding:
            continue
        description = coding.get("display") or item.get("productOrService", {}).get("text") or "(no description)"
        lines.append(ClaimLine(when, claim_type, coding["code"], description[:120]))
    return lines


# ---------------------------------------------------------------- summarize
def normalize_service(service: str | None) -> str | None:
    if not service:
        return None
    key = service.strip().lower()
    for alias, canonical in ALIASES.items():
        if alias in key:
            return canonical
    return key


def matches(line: ClaimLine, service: str | None) -> bool:
    if not service:
        return True
    codes = SERVICE_CODES.get(service, set())
    return line.code.upper() in codes or service in line.description.lower()


def summarize(eobs: list[dict], service: str | None = None, months: int = 24,
              today: date | None = None) -> dict:
    today = today or date.today()
    since = today - timedelta(days=round(months * 30.44))
    service = normalize_service(service)
    lines = {ln for eob in eobs for ln in parse_eob(eob)}           # set: same line on two pages counted once
    window = [ln for ln in lines if since.isoformat() <= ln.service_date <= today.isoformat()]
    hits = sorted((ln for ln in window if matches(ln, service)), key=lambda ln: ln.service_date, reverse=True)
    return {
        "source": "CMS Blue Button 2.0 sandbox (synthetic Medicare claims)",
        "window": {"from": since.isoformat(), "to": today.isoformat(), "months": months},
        "service_filter": service,
        "claims_in_window": len(window),
        "matching_lines": len(hits),
        "most_recent": hits[0].service_date if hits else None,
        "lines": [asdict(ln) for ln in hits[:MAX_LINES]],
        "truncated": len(hits) > MAX_LINES,
    }


def get_claims_summary(session: HttpSession, member_id: str, service: str | None = None,
                       months: int = 24) -> dict:
    return summarize(fetch_eobs(session, member_id), service=service, months=months)
