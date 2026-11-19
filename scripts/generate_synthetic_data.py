"""Create data/synthetic/members.db - 100% FAKE members, plans and prior-auth rules.

Usage:  python scripts/generate_synthetic_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.data.synthetic import build_database  # noqa: E402


def main() -> None:
    db_path = get_settings().member_db
    counts = build_database(db_path)
    print(f"Created {db_path}")
    for table, n in counts.items():
        print(f"  {table:<18} {n} rows")
    print("Sample: M1001 -> PLN-001 (PPO), M1002 -> PLN-002 (HMO), M1003 -> PLN-003 (EPO), "
          "M1004 -> PLN-004 (MA-HMO); M1010 is inactive.")


if __name__ == "__main__":
    main()
