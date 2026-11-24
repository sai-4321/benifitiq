"""Connect a BenefitIQ member to a synthetic Medicare enrollee in the CMS Blue Button 2.0 sandbox.

Usage:
    python scripts/connect_bluebutton.py --member M1001          # connect (OAuth consent)
    python scripts/connect_bluebutton.py --member M1001 --status
    python scripts/connect_bluebutton.py --member M1001 --disconnect

Works on a laptop and in GitHub Codespaces ("paste mode"): after you approve access, your browser is
sent to the redirect URI (http://localhost:8765/callback). That page may not load - that's fine.
Copy the FULL address from the browser bar and paste it here.
Sandbox login: a synthetic user such as BBUser00000 / PW00000! (never real Medicare credentials).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from app.integrations import bluebutton as bb  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Connect a member to Blue Button (sandbox).")
    p.add_argument("--member", required=True, help="BenefitIQ member id, e.g. M1001")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true")
    g.add_argument("--disconnect", action="store_true")
    args = p.parse_args()
    try:
        if args.status:
            token = bb.load_token(args.member)
            print(f"{args.member}: " + ("connected" if token else "not connected"))
            return 0
        if args.disconnect:
            print(f"{args.member}: " + ("disconnected" if bb.disconnect(args.member) else "was not connected"))
            return 0
        req = bb.authorize_url()
        print("\n1. Open this URL in your browser and log in with a SYNTHETIC sandbox user (e.g. BBUser00000):\n")
        print(f"   {req.url}\n")
        print("2. Click 'Connect' / 'Allow'. Your browser then goes to localhost - the page may not load.")
        callback = input("3. Paste the FULL address from the browser bar here:\n   > ")
        code = bb.parse_callback(callback, req.state)
        with requests.Session() as session:
            token = bb.exchange_code(session, code, req.verifier)
        bb.save_token(args.member, token)
        print(f"\nConnected {args.member} to Blue Button (refresh token: {'yes' if token.refresh_token else 'no'}).")
        return 0
    except bb.BlueButtonError as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
