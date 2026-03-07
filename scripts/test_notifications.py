#!/usr/bin/env python3
"""Test all notification channels — email and Slack.

Usage:
    python scripts/test_notifications.py
"""

import os
import sys

# Force UTF-8 stdout on Windows so emoji prints correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils.notifications import (
    send_email_alert,
    send_slack_notification,
    send_healing_notification,
)

print("Testing notifications...")
print()

# Test email
print("1. Email:")
result = send_email_alert(
    "Test: NeuroShield Notifications Working",
    "Email notifications are configured correctly.",
)
print(f"   {'✅ SENT' if result else '⚪ NOT CONFIGURED'}")
print()

# Test Slack
print("2. Slack:")
result = send_slack_notification(
    title="Test Notification",
    message="Slack notifications are configured correctly.",
    color='#36a64f',
    fields=[
        {"title": "Status", "value": "Working ✅"},
        {"title": "Project", "value": "NeuroShield AIOps"},
    ],
)
print(f"   {'✅ SENT' if result else '⚪ NOT CONFIGURED'}")
print()

# Test combined healing notification
print("3. Healing notification (both channels):")
result = send_healing_notification(
    action='retry_build',
    reason='Test: Build failure simulation',
    result='SUCCESS',
    telemetry={
        'prometheus_cpu_usage': 45.2,
        'prometheus_memory_usage': 67.8,
        'jenkins_last_build_status': 'SUCCESS',
    },
)
print(f"   {'✅ SENT' if result else '⚪ NOT CONFIGURED'}")
