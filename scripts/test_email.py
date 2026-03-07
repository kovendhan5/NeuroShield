#!/usr/bin/env python3
"""Quick email test — verifies SMTP credentials without touching the orchestrator.

Usage:
    python scripts/test_email.py          # sends a test email
    python scripts/test_email.py --check  # just validates env vars are set
"""

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils.notifications import send_email_alert


def check_config() -> bool:
    """Return True if all email env vars are set."""
    required = ["ALERT_EMAIL_FROM", "ALERT_EMAIL_TO", "ALERT_EMAIL_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"[FAIL] Missing environment variables: {', '.join(missing)}")
        print("       Copy .env.example to .env and fill in values.")
        return False
    print(f"[OK] Email configured: {os.getenv('ALERT_EMAIL_FROM')} → {os.getenv('ALERT_EMAIL_TO')}")
    return True


def main() -> None:
    if "--check" in sys.argv:
        sys.exit(0 if check_config() else 1)

    if not check_config():
        sys.exit(1)

    print("\nSending test email...")
    ok = send_email_alert(
        subject="Test Alert — Email Working",
        body=(
            "If you see this, NeuroShield email alerts are configured correctly.\n\n"
            "Healing actions, escalations, and self-CI failures will be "
            "delivered to this address."
        ),
    )
    if ok:
        print(f"[OK] Test email sent to {os.getenv('ALERT_EMAIL_TO')}")
    else:
        print("[FAIL] Email failed — check credentials and network.")
        sys.exit(1)


if __name__ == "__main__":
    main()
