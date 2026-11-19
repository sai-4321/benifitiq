"""Synthetic database tests: shape, determinism and 'everything is fake' guarantees."""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from app.data.synthetic import NUM_MEMBERS, PLANS, PRIOR_AUTH, build_database


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "members.db"
    counts = build_database(path)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    yield con, counts, path
    con.close()


def test_counts(db):
    con, counts, _ = db
    assert counts == {"plans": 4, "members": NUM_MEMBERS, "prior_auth_rules": len(PRIOR_AUTH) * len(PLANS)}
    assert con.execute("SELECT COUNT(*) FROM members").fetchone()[0] == NUM_MEMBERS


def test_documented_sample_ids(db):
    con, _, _ = db
    plan = {r["member_id"]: r["plan_id"] for r in con.execute("SELECT member_id, plan_id FROM members")}
    assert [plan[f"M100{i}"] for i in range(1, 5)] == ["PLN-001", "PLN-002", "PLN-003", "PLN-004"]
    inactive = [r[0] for r in con.execute("SELECT member_id FROM members WHERE status='inactive' ORDER BY 1")]
    assert inactive == ["M1010", "M1020", "M1030", "M1040", "M1050"]


def test_every_row_is_flagged_synthetic(db):
    con, _, _ = db
    for table in ("plans", "members", "prior_auth_rules"):
        assert con.execute(f"SELECT MIN(synthetic) FROM {table}").fetchone()[0] == 1


def test_every_plan_has_every_service_rule(db):
    con, _, _ = db
    rows = con.execute("SELECT plan_id, COUNT(*) FROM prior_auth_rules GROUP BY plan_id").fetchall()
    assert {r[0]: r[1] for r in rows} == {p[0]: len(PRIOR_AUTH) for p in PLANS}


def test_foreign_keys_are_consistent(db):
    con, _, _ = db
    orphans = con.execute("SELECT COUNT(*) FROM members m LEFT JOIN plans p USING(plan_id) WHERE p.plan_id IS NULL")
    assert orphans.fetchone()[0] == 0


def test_build_is_deterministic(tmp_path):
    def dump(path):
        con = sqlite3.connect(path)
        rows = con.execute("SELECT * FROM members ORDER BY member_id").fetchall()
        con.close()
        return hashlib.sha256(repr(rows).encode()).hexdigest()

    build_database(tmp_path / "a.db")
    build_database(tmp_path / "b.db")
    assert dump(tmp_path / "a.db") == dump(tmp_path / "b.db")


def test_rebuild_replaces_old_data(db):
    _, _, path = db
    build_database(path)       # second run must not duplicate rows
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM members").fetchone()[0] == NUM_MEMBERS
    con.close()
