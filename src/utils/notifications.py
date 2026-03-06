"""NeuroShield Notification Utilities.

Provides desktop notifications, email alerts, and dashboard alert banner writing.
All methods fail gracefully when dependencies or configuration are missing.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def send_desktop_notification(title: str, message: str) -> bool:
    """Send a Windows desktop toast notification.

    Tries plyer first, falls back to PowerShell WinRT toast.
    Returns True if notification was attempted successfully.
    """
    # Try plyer first
    try:
        from plyer import notification as _notif  # type: ignore
        _notif.notify(
            title=title,
            message=message,
            app_icon=None,
            timeout=10,
        )
        logger.info("[NOTIFY] Desktop notification sent via plyer: %s", title)
        return True
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("[NOTIFY] plyer notification failed: %s", exc)

    # Fallback: PowerShell WinRT toast (Windows 10/11 only)
    try:
        import subprocess
        # Sanitize strings to avoid PowerShell injection
        safe_title = title.replace('"', "'").replace("`", "").replace("\n", " ")[:100]
        safe_msg = message.replace('"', "'").replace("`", "").replace("\n", " ")[:200]
        ps_cmd = (
            "[Windows.UI.Notifications.ToastNotificationManager,"
            "Windows.UI.Notifications,ContentType=WindowsRuntime] > $null; "
            "$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02; "
            "$xml = [Windows.UI.Notifications.ToastNotificationManager]"
            "::GetTemplateContent($template); "
            f'$xml.GetElementsByTagName("text")[0].AppendChild($xml.CreateTextNode("{safe_title}")); '
            f'$xml.GetElementsByTagName("text")[1].AppendChild($xml.CreateTextNode("{safe_msg}")); '
            "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml); "
            '[Windows.UI.Notifications.ToastNotificationManager]'
            '::CreateToastNotifier("NeuroShield").Show($toast);'
        )
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True,
            timeout=10,
        )
        logger.info("[NOTIFY] PowerShell toast notification sent: %s", title)
        return True
    except Exception as exc:
        logger.warning("[NOTIFY] PowerShell notification failed: %s", exc)
        return False


def send_email_alert(subject: str, body: str) -> bool:
    """Send an HTML email alert via Gmail SMTP.

    Reads credentials from environment variables:
        ALERT_EMAIL_FROM     — sender Gmail address
        ALERT_EMAIL_TO       — recipient address
        ALERT_EMAIL_PASSWORD — Gmail app password (not regular password)

    Returns True if email was sent, False otherwise (missing config, network error, etc.)
    """
    smtp_user = os.getenv("ALERT_EMAIL_FROM")
    smtp_pass = os.getenv("ALERT_EMAIL_PASSWORD")
    to_email = os.getenv("ALERT_EMAIL_TO")

    if not all([smtp_user, smtp_pass, to_email]):
        logger.debug("[EMAIL] Email not configured (ALERT_EMAIL_FROM/TO/PASSWORD not set)")
        return False

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = f"[NeuroShield Alert] {subject}"

    html_body = f"""
    <html><body>
    <h2 style="color:red">&#9888; NeuroShield Escalation Alert</h2>
    <p><b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><b>Issue:</b> {subject}</p>
    <pre style="background:#f5f5f5; padding:12px; border-radius:4px;">{body}</pre>
    <hr>
    <p>View dashboard: <a href="http://localhost:8501">localhost:8501</a></p>
    </body></html>
    """
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("[EMAIL] Alert sent to %s", to_email)
        return True
    except Exception as exc:
        logger.warning("[EMAIL] Email alert failed: %s", exc)
        return False


def write_active_alert(
    title: str,
    message: str,
    severity: str = "HIGH",
    details: Optional[Dict[str, Any]] = None,
    alert_path: str = "data/active_alert.json",
) -> None:
    """Write an active escalation alert to data/active_alert.json.

    The dashboard reads this file and shows a red banner with a resolve button.
    """
    alert: Dict[str, Any] = {
        "active": True,
        "timestamp": datetime.now().isoformat(),
        "title": title,
        "message": message,
        "severity": severity,
        "details": details or {},
    }
    try:
        p = Path(alert_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(alert, indent=2), encoding="utf-8")
        logger.info("[ALERT] Active alert written to %s", alert_path)
    except Exception as exc:
        logger.warning("[ALERT] Failed to write active alert: %s", exc)
