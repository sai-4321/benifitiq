"""Download the four public Medicare.gov booklets into data/raw/.

Usage:  python scripts/download_docs.py          (skips files already downloaded)
        python scripts/download_docs.py --force  (re-download everything)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.data.documents import SOURCES, fetch_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    raw = get_settings().raw_docs_path
    with requests.Session() as session:
        entries, errors = fetch_all(session, raw, force=args.force)
    for e in entries:
        print(f"  OK  {e.file:<62} {e.pages:>4} pages  {e.size_mb:5.1f} MB  sha256:{e.sha256_16}")
    for line in errors:
        print(f"  FAILED {line}")
    print(f"\n{len(entries)}/{len(SOURCES)} documents ready in {raw}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
