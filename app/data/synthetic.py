"""Synthetic member/plan database - 100% FAKE data, never real patients.

Names, DOBs and phones come from Faker with a fixed seed, so every run builds exactly the
same database (repeatable tests and evals). Every row carries synthetic=1.

Sample IDs (stable): M1001 -> PLN-001 PPO, M1002 -> PLN-002 HMO, M1003 -> PLN-003 EPO,
M1004 -> PLN-004 Medicare Advantage HMO; every 10th member (M1010, M1020 ...) is inactive.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from faker import Faker

SEED = 42
NUM_MEMBERS = 50

PLANS = [
    # plan_id, name, type, deductible, pcp, specialist, er, urgent_care, oop_max, coinsurance %, referral needed
    ("PLN-001", "Demo Silver PPO", "PPO", 1500, 25, 50, 300, 75, 6500, 20, 0),
    ("PLN-002", "Demo Gold HMO", "HMO", 500, 15, 35, 250, 50, 4500, 10, 1),
    ("PLN-003", "Demo Bronze EPO", "EPO", 4000, 40, 80, 500, 100, 8500, 30, 0),
    ("PLN-004", "Demo Medicare Advantage HMO", "MA-HMO", 0, 0, 40, 120, 45, 5000, 20, 1),
]

# service -> {plan_id: (prior_auth_required, note)}
PRIOR_AUTH = {
    "mri": {p: (1, "Prior authorization required for all non-emergency MRI scans.") for p in
            ("PLN-001", "PLN-002", "PLN-003", "PLN-004")},
    "ct_scan": {
        "PLN-001": (0, "No prior authorization needed for CT scans in network."),
        "PLN-002": (1, "Prior authorization required for non-emergency CT scans."),
        "PLN-003": (1, "Prior authorization required for non-emergency CT scans."),
        "PLN-004": (1, "Prior authorization required for non-emergency CT scans."),
    },
    "specialist_visit": {
        "PLN-001": (0, "No prior authorization or referral needed."),
        "PLN-002": (0, "No prior authorization, but a PCP referral is required."),
        "PLN-003": (0, "No prior authorization; must use in-network specialists."),
        "PLN-004": (0, "No prior authorization, but a PCP referral is required."),
    },
    "physical_therapy": {
        "PLN-001": (1, "Prior authorization required after the first 12 visits per year."),
        "PLN-002": (1, "Prior authorization required before the first visit."),
        "PLN-003": (1, "Prior authorization required after the first 20 visits per year."),
        "PLN-004": (1, "Prior authorization required before the first visit."),
    },
    "durable_medical_equipment": {
        "PLN-001": (1, "Prior authorization required for items costing more than $500."),
        "PLN-002": (1, "Prior authorization required for all DME items."),
        "PLN-003": (1, "Prior authorization required for all DME items."),
        "PLN-004": (1, "Prior authorization required for items costing more than $500."),
    },
    "home_health": {p: (1, "Prior authorization required; physician plan of care needed.") for p in
                    ("PLN-001", "PLN-002", "PLN-003", "PLN-004")},
    "skilled_nursing": {p: (1, "Prior authorization required before admission.") for p in
                        ("PLN-001", "PLN-002", "PLN-003", "PLN-004")},
    "outpatient_surgery": {p: (1, "Prior authorization required for scheduled outpatient surgery.") for p in
                           ("PLN-001", "PLN-002", "PLN-003", "PLN-004")},
    "sleep_study": {
        "PLN-001": (0, "No prior authorization needed for in-lab sleep studies."),
        "PLN-002": (1, "Prior authorization required."),
        "PLN-003": (1, "Prior authorization required."),
        "PLN-004": (1, "Prior authorization required."),
    },
    "genetic_testing": {p: (1, "Prior authorization required; genetic counseling recommended.") for p in
                        ("PLN-001", "PLN-002", "PLN-003", "PLN-004")},
}

SCHEMA = """
DROP TABLE IF EXISTS members;
DROP TABLE IF EXISTS plans;
DROP TABLE IF EXISTS prior_auth_rules;
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY, plan_name TEXT NOT NULL, plan_type TEXT NOT NULL,
    deductible INTEGER, pcp_copay INTEGER, specialist_copay INTEGER, er_copay INTEGER,
    urgent_care_copay INTEGER, oop_max INTEGER, coinsurance_pct INTEGER, referral_required INTEGER,
    synthetic INTEGER DEFAULT 1
);
CREATE TABLE members (
    member_id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, dob TEXT, phone TEXT,
    plan_id TEXT REFERENCES plans(plan_id), coverage_start TEXT, status TEXT, synthetic INTEGER DEFAULT 1
);
CREATE TABLE prior_auth_rules (
    plan_id TEXT REFERENCES plans(plan_id), service TEXT, prior_auth_required INTEGER, notes TEXT,
    synthetic INTEGER DEFAULT 1, PRIMARY KEY (plan_id, service)
);
"""


def build_database(db_path: Path) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fake = Faker("en_US")
    Faker.seed(SEED)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO plans VALUES (?,?,?,?,?,?,?,?,?,?,?,1)", PLANS)
        members = []
        for i in range(NUM_MEMBERS):
            plan_id = PLANS[i % len(PLANS)][0]          # deterministic: M1001->PLN-001, M1002->PLN-002 ...
            status = "inactive" if i % 10 == 9 else "active"  # M1010, M1020 ... are inactive
            members.append((
                f"M{1001 + i}",
                fake.first_name(),
                fake.last_name(),
                fake.date_of_birth(minimum_age=26, maximum_age=90).isoformat(),
                fake.numerify("###-###-####"),
                plan_id,
                f"202{4 + i % 3}-01-01",
                status,
            ))
        con.executemany("INSERT INTO members VALUES (?,?,?,?,?,?,?,?,1)", members)
        rules = [(plan, svc, req, note) for svc, by_plan in PRIOR_AUTH.items()
                 for plan, (req, note) in by_plan.items()]
        con.executemany("INSERT INTO prior_auth_rules VALUES (?,?,?,?,1)", rules)
        con.commit()
    finally:
        con.close()
    return {"plans": len(PLANS), "members": NUM_MEMBERS, "prior_auth_rules": len(rules)}

