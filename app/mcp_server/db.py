"""Safe, read-only queries against the synthetic member database.

Every query is parameterized ("?" placeholders) - user/LLM text is never pasted
into SQL, so SQL injection is impossible through the tools.
Only the minimum necessary fields are returned (no names, DOBs or phones):
the LLM never needs them to answer a benefits question (HIPAA "minimum necessary").
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class NotFoundError(LookupError):
    pass


def db_path() -> Path:
    from app.config import PROJECT_ROOT

    raw = os.environ.get("MEMBER_DB_PATH", "data/synthetic/members.db")
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


@contextmanager
def connect():
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python scripts/generate_synthetic_data.py")
    # as_uri() gives file:///C:/... on Windows and file:///home/... on Linux; mode=ro = read-only
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def get_member_plan(member_id: str) -> dict:
    with connect() as con:
        row = con.execute(
            """SELECT m.member_id, m.status, m.coverage_start, p.plan_id, p.plan_name, p.plan_type
               FROM members m JOIN plans p ON p.plan_id = m.plan_id WHERE m.member_id = ?""",
            (member_id,),
        ).fetchone()
    if row is None:
        raise NotFoundError(f"Member {member_id} not found")
    return dict(row) | {"synthetic": True}


def get_plan_benefits(plan_id: str) -> dict:
    with connect() as con:
        row = con.execute(
            """SELECT plan_id, plan_name, plan_type, deductible, pcp_copay, specialist_copay, er_copay,
                      urgent_care_copay, oop_max, coinsurance_pct, referral_required
               FROM plans WHERE plan_id = ?""",
            (plan_id,),
        ).fetchone()
    if row is None:
        raise NotFoundError(f"Plan {plan_id} not found")
    data = dict(row)
    data["referral_required"] = bool(data["referral_required"])
    return data | {"currency": "USD", "synthetic": True}


def check_prior_auth(plan_id: str, service: str) -> dict:
    with connect() as con:
        row = con.execute(
            "SELECT plan_id, service, prior_auth_required, notes FROM prior_auth_rules WHERE plan_id=? AND service=?",
            (plan_id, service),
        ).fetchone()
    if row is None:
        raise NotFoundError(f"No prior-auth rule for plan {plan_id} and service {service}")
    data = dict(row)
    data["prior_auth_required"] = bool(data["prior_auth_required"])
    return data | {"synthetic": True}
