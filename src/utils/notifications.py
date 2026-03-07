"""NeuroShield Notification Utilities.

Provides email alerts and dashboard alert banner writing.
Desktop notifications are disabled — email is more reliable and professional.
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


def send_email_alert(subject: str, body: str,
                     html_body: str = None) -> bool:
    """Send an email alert via Gmail SMTP.

    Reads credentials from environment variables:
        ALERT_EMAIL_FROM     — sender Gmail address
        ALERT_EMAIL_TO       — recipient address
        ALERT_EMAIL_PASSWORD — Gmail app password (not regular password)

    Returns True if email was sent, False otherwise.
    """
    smtp_user = os.getenv("ALERT_EMAIL_FROM")
    smtp_pass = os.getenv("ALERT_EMAIL_PASSWORD")
    to_email = os.getenv("ALERT_EMAIL_TO")

    # Silent no-op if email not configured
    if not all([smtp_user, smtp_pass, to_email]):
        logger.info(
            "Email not configured — set ALERT_EMAIL_FROM, "
            "ALERT_EMAIL_TO, ALERT_EMAIL_PASSWORD in .env")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"NeuroShield AIOps <{smtp_user}>"
        msg["To"] = to_email
        msg["Subject"] = f"[NeuroShield] {subject}"

        # Plain text fallback
        msg.attach(MIMEText(body, "plain"))

        # HTML version if provided
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        logger.info("Email sent to %s: %s", to_email, subject)
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "Email authentication failed. "
            "Use Gmail App Password, not your account password. "
            "Enable at: myaccount.google.com/apppasswords")
        return False
    except smtplib.SMTPException as e:
        logger.error("SMTP error: %s", e)
        return False
    except Exception as e:
        logger.error("Email failed: %s", e)
        return False


def send_healing_notification(action: str, reason: str,
                              result: str,
                              telemetry: dict) -> bool:
    """Send an email notification after a healing action completes."""
    subject = f"Auto-Healed: {action} — {result}"

    cpu = telemetry.get("prometheus_cpu_usage", "N/A")
    mem = telemetry.get("prometheus_memory_usage", "N/A")
    build = telemetry.get("jenkins_last_build_status", "N/A")

    plain = (
        f"NeuroShield executed healing action\n\n"
        f"Action: {action}\n"
        f"Reason: {reason}\n"
        f"Result: {result}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"System State:\n"
        f"  CPU: {cpu}%\n"
        f"  Memory: {mem}%\n"
        f"  Build: {build}\n\n"
        f"Dashboard: http://localhost:8501"
    )

    html = f"""
    <html><body style="font-family:Arial;max-width:600px;margin:auto">
    <div style="background:#1a1a2e;color:white;padding:20px;
                border-radius:8px 8px 0 0">
        <h2 style="margin:0">&#128737; NeuroShield — Auto-Healed</h2>
    </div>
    <div style="background:#f9f9f9;padding:20px;
                border:1px solid #ddd;border-radius:0 0 8px 8px">
        <table style="width:100%;border-collapse:collapse">
            <tr>
                <td style="padding:8px;font-weight:bold;width:30%">
                    Action</td>
                <td style="padding:8px;color:#0066cc">{action}</td>
            </tr>
            <tr style="background:#fff">
                <td style="padding:8px;font-weight:bold">Reason</td>
                <td style="padding:8px">{reason}</td>
            </tr>
            <tr>
                <td style="padding:8px;font-weight:bold">Result</td>
                <td style="padding:8px;
                    color:{'green' if 'success' in result.lower()
                           else 'red'}">{result}</td>
            </tr>
            <tr style="background:#fff">
                <td style="padding:8px;font-weight:bold">Time</td>
                <td style="padding:8px">
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </td>
            </tr>
        </table>
        <h3>System State</h3>
        <table style="width:100%;border-collapse:collapse">
            <tr>
                <td style="padding:6px;background:#eee;width:33%">
                    CPU</td>
                <td style="padding:6px">{cpu}%</td>
            </tr>
            <tr>
                <td style="padding:6px;background:#eee">Memory</td>
                <td style="padding:6px">{mem}%</td>
            </tr>
            <tr>
                <td style="padding:6px;background:#eee">
                    Build Status</td>
                <td style="padding:6px">{build}</td>
            </tr>
        </table>
        <br>
        <a href="http://localhost:8501"
           style="background:#1a1a2e;color:white;padding:10px 20px;
                  text-decoration:none;border-radius:4px">
            View Dashboard
        </a>
    </div>
    </body></html>
    """
    email_sent = send_email_alert(subject, plain, html)

    # Also send to Slack
    color = '#36a64f' if 'success' in result.lower() else '#cc0000'
    slack_sent = send_slack_notification(
        title=f"Auto-Healed: {action}",
        message=reason,
        color=color,
        fields=[
            {"title": "Action", "value": action},
            {"title": "Result", "value": result},
            {"title": "CPU",
             "value": f"{telemetry.get('prometheus_cpu_usage', 'N/A')}%"},
            {"title": "Memory",
             "value": f"{telemetry.get('prometheus_memory_usage', 'N/A')}%"},
            {"title": "Build",
             "value": telemetry.get('jenkins_last_build_status', 'N/A')},
            {"title": "Dashboard",
             "value": "http://localhost:8501"}
        ]
    )
    return email_sent or slack_sent


def send_escalation_alert(reason: str, report_path: str,
                          telemetry: dict) -> bool:
    """Send an escalation email when human intervention is required."""
    subject = "ESCALATION: Human Intervention Required"

    plain = (
        f"NeuroShield requires human intervention\n\n"
        f"Reason: {reason}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Report: {report_path}\n\n"
        f"NeuroShield could not automatically resolve this issue.\n"
        f"Please review the incident report and take action.\n\n"
        f"Dashboard: http://localhost:8501\n"
        f"Jenkins: http://localhost:8080"
    )

    cpu = telemetry.get("prometheus_cpu_usage", "N/A")
    mem = telemetry.get("prometheus_memory_usage", "N/A")
    build = telemetry.get("jenkins_last_build_status", "N/A")

    html = f"""
    <html><body style="font-family:Arial;max-width:600px;margin:auto">
    <div style="background:#cc0000;color:white;padding:20px;
                border-radius:8px 8px 0 0">
        <h2 style="margin:0">&#128680; NeuroShield — Escalation Alert</h2>
        <p style="margin:5px 0 0 0">Human intervention required</p>
    </div>
    <div style="background:#fff8f8;padding:20px;
                border:2px solid #cc0000;
                border-radius:0 0 8px 8px">
        <p style="color:#cc0000;font-weight:bold;font-size:16px">
            &#9888; {reason}
        </p>
        <p>NeuroShield attempted automatic remediation but could
           not resolve this incident. Manual investigation required.
        </p>
        <h3>System State at Escalation</h3>
        <table style="width:100%;border-collapse:collapse">
            <tr style="background:#f5f5f5">
                <td style="padding:8px;font-weight:bold">CPU</td>
                <td style="padding:8px">{cpu}%</td>
            </tr>
            <tr>
                <td style="padding:8px;font-weight:bold">Memory</td>
                <td style="padding:8px">{mem}%</td>
            </tr>
            <tr style="background:#f5f5f5">
                <td style="padding:8px;font-weight:bold">
                    Build Status</td>
                <td style="padding:8px">{build}</td>
            </tr>
            <tr>
                <td style="padding:8px;font-weight:bold">
                    Report Generated</td>
                <td style="padding:8px">{report_path}</td>
            </tr>
        </table>
        <br>
        <a href="http://localhost:8080"
           style="background:#cc0000;color:white;
                  padding:10px 20px;text-decoration:none;
                  border-radius:4px;margin-right:10px">
            Jenkins Dashboard
        </a>
        <a href="http://localhost:8501"
           style="background:#1a1a2e;color:white;
                  padding:10px 20px;text-decoration:none;
                  border-radius:4px">
            NeuroShield Dashboard
        </a>
    </div>
    </body></html>
    """
    email_sent = send_email_alert(subject, plain, html)

    slack_sent = send_slack_notification(
        title="\U0001f6a8 ESCALATION \u2014 Human Required",
        message=reason,
        color='#cc0000',
        fields=[
            {"title": "Reason", "value": reason},
            {"title": "CPU",
             "value": f"{telemetry.get('prometheus_cpu_usage', 'N/A')}%"},
            {"title": "Memory",
             "value": f"{telemetry.get('prometheus_memory_usage', 'N/A')}%"},
            {"title": "Report", "value": report_path},
            {"title": "Jenkins",
             "value": "http://localhost:8080"},
            {"title": "Dashboard",
             "value": "http://localhost:8501"}
        ]
    )
    return email_sent or slack_sent


def send_self_ci_failure_alert(build_number: int,
                               stages_failed: list) -> bool:
    """Send a critical email when NeuroShield's own CI pipeline fails."""
    subject = f"CRITICAL: NeuroShield Self-CI Failed (Build #{build_number})"

    plain = (
        f"NeuroShield's own CI pipeline has failed\n\n"
        f"Build: #{build_number}\n"
        f"Failed stages: {', '.join(stages_failed)}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"This means NeuroShield's own tests or models are broken.\n"
        f"Check: http://localhost:8080/job/neuroshield-ci/\n"
    )
    return send_email_alert(subject, plain)


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


def send_slack_notification(title: str, message: str,
                             color: str = '#1a1a2e',
                             fields: list = None) -> bool:
    """Send a Slack notification via Incoming Webhook.

    Reads SLACK_WEBHOOK_URL from environment.  Returns True if sent.
    """
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')

    if not webhook_url:
        logger.info(
            "Slack not configured \u2014 set SLACK_WEBHOOK_URL in .env")
        return False

    try:
        import urllib.request
        import json as _json

        attachment = {
            "color": color,
            "title": f"\U0001f6e1\ufe0f NeuroShield \u2014 {title}",
            "text": message,
            "footer": "NeuroShield AIOps",
            "ts": int(datetime.now().timestamp())
        }

        if fields:
            attachment["fields"] = [
                {"title": f["title"],
                 "value": f["value"],
                 "short": f.get("short", True)}
                for f in fields
            ]

        payload = _json.dumps({
            "attachments": [attachment]
        }).encode('utf-8')

        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200

    except Exception as e:
        logger.error(f"Slack notification failed: {e}")
        return False


# Keep as no-op so existing code doesn't break during transition
def send_desktop_notification(title: str, message: str) -> bool:
    """Disabled — use email notifications instead."""
    logger.debug("Desktop notifications disabled. Use email instead: %s", title)
    return False
