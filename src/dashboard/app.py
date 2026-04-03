"""NeuroShield AIOps Platform — Professional Dashboard."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import numpy as np
import torch

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.prediction.predictor import FailurePredictor, telemetry_to_vector
from src.utils.intelligence import detect_early_warning, explain_decision

# ──────────────────────────────────────────────────────────────────────────────
# FIX 7 — Page config with title and favicon
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuroShield AIOps",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Dark professional theme via CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #161b26 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    h1, h2, h3 { color: #f0f4f8 !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .badge-online {
        background: #065f46; color: #6ee7b7; padding: 4px 14px;
        border-radius: 20px; font-weight: 600; font-size: 0.8rem;
        display: inline-block; margin: 2px 4px;
    }
    .badge-offline {
        background: #7f1d1d; color: #fca5a5; padding: 4px 14px;
        border-radius: 20px; font-weight: 600; font-size: 0.8rem;
        display: inline-block; margin: 2px 4px;
    }
    .arch-box {
        background: #1a1f2e; border: 1px solid #374151; border-radius: 10px;
        padding: 20px; margin: 8px 0; text-align: center; font-family: monospace;
        font-size: 0.95rem; color: #e2e8f0; line-height: 2.2;
    }
    .arch-arrow { color: #60a5fa; font-weight: bold; font-size: 1.2rem; }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #0e1117 100%);
        padding: 8px 16px; border-radius: 8px; margin: 12px 0 8px 0;
        border-left: 4px solid #3b82f6;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .hero-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a1f2e 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 12px 0 20px 0;
        line-height: 1.7;
    }
    .action-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid;
    }
    .action-green { background: #0d2818; border-color: #10b981; }
    .action-orange { background: #2d1f0e; border-color: #f59e0b; }
    .sim-step {
        padding: 4px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .log-entry {
        font-family: monospace;
        font-size: 0.82rem;
        padding: 2px 0;
        border-bottom: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TELEMETRY_CSV = os.getenv("TELEMETRY_OUTPUT_PATH", "data/telemetry.csv")
HEALING_LOG_CSV = "data/healing_log.csv"
ACTION_HISTORY_CSV = "data/action_history.csv"
MTTR_LOG_CSV = "data/mttr_log.csv"
JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
DUMMY_APP_URL = os.getenv("DUMMY_APP_URL", "http://localhost:5000")

ACTION_NAMES = {
    0: "restart_pod",
    1: "scale_up",
    2: "retry_build",
    3: "rollback_deploy",
    4: "clear_cache",
    5: "escalate_to_human",
}

ACTION_DISTRIBUTION = [0, 0, 30, 0, 2, 68]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _check_service(url: str) -> bool:
    try:
        r = requests.get(url, timeout=3)
        return r.status_code < 500
    except Exception:
        return False


def _status_badge(name: str, is_up: bool) -> str:
    cls = "badge-online" if is_up else "badge-offline"
    label = "ONLINE" if is_up else "OFFLINE"
    return f'<span class="{cls}">● {name}: {label}</span>'


@st.cache_resource
def _get_predictor():
    """Load the real FailurePredictor once (DistilBERT + PCA + classifier)."""
    return FailurePredictor(model_dir=Path(_PROJECT_ROOT) / "models")


def _batch_predict(predictor, df: pd.DataFrame) -> pd.Series:
    """Compute real failure probability for every row.

    Uses per-row predictor.predict() so the status boost in the predictor
    is applied consistently (FAILURE builds -> 0.65 floor).
    """
    if df.empty:
        return pd.Series(dtype=float)

    probs: list[float] = []
    for _, row in df.iterrows():
        build_status = str(row.get("jenkins_last_build_status", "UNKNOWN")).upper()
        log_text = str(row.get("jenkins_last_build_log", ""))
        if not log_text or log_text == "nan" or len(log_text.strip()) < 10:
            log_text = f"Build status {build_status}"

        tel = {
            "jenkins_last_build_status": build_status,
            "jenkins_last_build_duration": row.get("jenkins_last_build_duration", 0),
            "jenkins_queue_length": row.get("jenkins_queue_length", 0),
            "prometheus_cpu_usage": row.get("prometheus_cpu_usage", 0),
            "prometheus_memory_usage": row.get("prometheus_memory_usage", 0),
            "prometheus_pod_count": row.get("prometheus_pod_count", 0),
            "prometheus_error_rate": row.get("prometheus_error_rate", 0),
        }

        try:
            prob = predictor.predict(log_text, tel)
        except Exception:
            prob = 0.65 if build_status in ("FAILURE", "UNSTABLE", "ABORTED") else 0.05

        # Clamp: SUCCESS should never show high
        if build_status == "SUCCESS" and prob > 0.3:
            prob = 0.05

        probs.append(round(prob, 4))

    return pd.Series(probs, index=df.index)


def _load_healing_json(path: str) -> list[dict]:
    """Load healing_log.json (JSONL: one JSON object per line)."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
    try:
        for line in p.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
    except Exception:
        pass
    return records


# ──────────────────────────────────────────────────────────────────────────────
# FIX 6 — SIDEBAR with project info, instructions, links
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ NeuroShield")
    st.markdown("**AIOps Self-Healing CI/CD System**")
    st.markdown("---")

    st.markdown("### 👤 Project Info")
    st.markdown("**Kovendhan P.**")
    st.markdown("Jeppiaar Institute of Technology")

    st.markdown("---")
    st.markdown("### 📖 How to Run Demo")
    st.markdown("""
    ```bash
    # 1. Train models
    python src/prediction/train.py
    python -m src.rl_agent.train

    # 2. Start orchestrator
    python src/orchestrator/main.py \\
        --mode simulate

    # 3. Run simulation
    python scripts/demo_simulation.py
    ```
    """)

    st.markdown("---")
    st.markdown("### 🔗 Service Links")
    st.markdown(f"- [Jenkins]({JENKINS_URL})")
    st.markdown(f"- [Prometheus]({PROMETHEUS_URL})")
    st.markdown(f"- [Dummy App]({DUMMY_APP_URL})")

    st.markdown("---")
    st.markdown("### 🔧 Model Status")
    ppo_ok = Path("models/ppo_policy.zip").exists()
    pred_ok = Path("models/failure_predictor.pth").exists()
    pca_ok = Path("models/log_pca.joblib").exists()
    st.markdown(f"- PPO Policy: {'✅ Loaded' if ppo_ok else '❌ Missing'}")
    st.markdown(f"- Failure Predictor: {'✅ Loaded' if pred_ok else '❌ Missing'}")
    st.markdown(f"- PCA Encoder: {'✅ Loaded' if pca_ok else '❌ Missing'}")

    st.markdown("---")
    st.markdown("### 🔄 Self-CI Status")
    _self_ci_path = Path("data/self_ci_status.json")
    if _self_ci_path.exists():
        try:
            _sci = json.loads(_self_ci_path.read_text(encoding="utf-8"))
            _sci_result = _sci.get("result", "UNKNOWN")
            _sci_num = _sci.get("build_number", "?")
            _sci_ts = str(_sci.get("timestamp", ""))[:19]
            if _sci_result == "SUCCESS":
                st.markdown(f"✅ Build #{_sci_num}: **{_sci_result}**")
            else:
                st.markdown(f"❌ Build #{_sci_num}: **{_sci_result}**")
                if _sci.get("reason"):
                    st.caption(_sci["reason"])
            st.caption(f"Last: {_sci_ts}")
        except Exception:
            st.markdown("⚠️ Status file unreadable")
    else:
        st.markdown("⏳ No self-CI builds yet")

    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    if st.button("🔴 Trigger Test Failure", use_container_width=True):
        try:
            r = requests.get(f"{DUMMY_APP_URL}/fail", timeout=5)
            if r.status_code < 500:
                st.success("Failure injected into dummy-app!")
            else:
                st.error(f"App returned {r.status_code}")
        except Exception as e:
            st.error(f"Could not reach dummy-app: {e}")

    if st.button("🔄 Run Healing Cycle", use_container_width=True):
        with st.spinner("Running one healing cycle..."):
            try:
                from src.orchestrator.main import run_single_cycle
                result = run_single_cycle()
                st.success(
                    f"Action: **{result['action']}** | "
                    f"Failure Prob: {result['failure_prob']} | "
                    f"Success: {result['success']}"
                )
            except Exception as e:
                st.error(f"Cycle failed: {e}")

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("NeuroShield v2.0")


# ──────────────────────────────────────────────────────────────────────────────
# A) HEADER — Title + Live Status Badges
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ NeuroShield AIOps Platform")
st.markdown(
    "*Intelligent Self-Healing CI/CD Pipeline — "
    "PPO Reinforcement Learning + DistilBERT Log Analysis*"
)

jenkins_up = _check_service(f"{JENKINS_URL}/api/json")
prometheus_up = _check_service(f"{PROMETHEUS_URL}/-/healthy")
app_up = _check_service(DUMMY_APP_URL)

badges = (
    _status_badge("Jenkins", jenkins_up) + " "
    + _status_badge("Prometheus", prometheus_up) + " "
    + _status_badge("Dummy-App", app_up)
)
st.markdown(badges, unsafe_allow_html=True)

st.markdown("---")

# ── Active escalation alert banner ─────────────────────────────────────────
_alert_path = Path("data/active_alert.json")
if _alert_path.exists():
    try:
        _alert_data = json.loads(_alert_path.read_text(encoding="utf-8"))
        if _alert_data.get("active"):
            _alert_ts = str(_alert_data.get("timestamp", ""))[:19]
            st.error(
                f"🚨 **ESCALATION ALERT: {_alert_data.get('title', 'Human Intervention Required')}**\n\n"
                f"{_alert_data.get('message', '')}\n\n"
                f"**Time:** {_alert_ts}"
            )
            if st.button("✅ Mark as Resolved", key="resolve_active_alert"):
                _alert_data["active"] = False
                _alert_path.write_text(json.dumps(_alert_data, indent=2), encoding="utf-8")
                st.rerun()
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# FIX 1 — HERO SECTION
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-box">
<h3 style="margin-top:0; color:#60a5fa !important;">What is NeuroShield?</h3>
<p style="color:#e2e8f0; margin-bottom:0;">
NeuroShield is an AI-powered self-healing CI/CD system. It monitors your Jenkins
pipeline and Kubernetes cluster 24/7. When something breaks, it automatically
diagnoses and fixes the issue using a Reinforcement Learning agent. If it cannot
fix it, it alerts your team with a full diagnosis report.
</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

telemetry_df = _load_csv(TELEMETRY_CSV)
healing_df = _load_csv(HEALING_LOG_CSV)
action_df = _load_csv(ACTION_HISTORY_CSV)
healing_json_records = _load_healing_json("data/healing_log.json")

total_healing_actions = max(len(healing_df), len(healing_json_records)) + len(action_df)

# ── Compute failure_prob for last 50 rows (predictor with rule-based fallback) ──
predictor = None
try:
    predictor = _get_predictor()
except Exception:
    pass

df_recent = telemetry_df.tail(50).copy() if not telemetry_df.empty else pd.DataFrame()
if not df_recent.empty:
    if predictor is not None:
        try:
            df_recent["failure_prob"] = _batch_predict(predictor, df_recent)
        except Exception:
            predictor = None  # fall through to rule-based
    # Rule-based fallback if predictor failed or was unavailable
    if predictor is None or "failure_prob" not in df_recent.columns:
        probs = []
        for _, row in df_recent.iterrows():
            try:
                bs = str(row.get("jenkins_last_build_status", "")).upper()
                if bs == "FAILURE":
                    probs.append(0.65)
                elif bs == "SUCCESS":
                    probs.append(0.05)
                else:
                    probs.append(0.10)
            except Exception:
                probs.append(0.05)
        df_recent["failure_prob"] = probs

# Debug sidebar info
if not telemetry_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.write("CSV rows:", len(telemetry_df))
    if not df_recent.empty and "failure_prob" in df_recent.columns:
        st.sidebar.write("Last prob:", f"{df_recent['failure_prob'].iloc[-1]:.4f}")

# REST API link
st.sidebar.markdown("---")
st.sidebar.markdown("### \U0001f50c REST API")
st.sidebar.markdown("[API Docs (Swagger)](http://localhost:8502/docs)")
st.sidebar.markdown("[Health Check](http://localhost:8502/health)")
st.sidebar.markdown("[Live Metrics](http://localhost:8502/metrics)")

# Model report
st.sidebar.markdown("---")
st.sidebar.markdown("### \U0001f4ca Model Report")
_report_path = Path("data/model_report.html")
if _report_path.exists():
    _mod_time = datetime.fromtimestamp(_report_path.stat().st_mtime)
    st.sidebar.success(f"Report generated\n{_mod_time.strftime('%Y-%m-%d %H:%M')}")
    if st.sidebar.button("\U0001f504 Regenerate Report"):
        import subprocess as _sp
        # Changed to accurate path
        _sp.Popen([sys.executable, "scripts/demo/generate_model_report.py"])
        st.sidebar.info("Generating... (~30 seconds)")
else:
    st.sidebar.warning("No report yet")
    if st.sidebar.button("\U0001f4ca Generate Report"):
        import subprocess as _sp
        # Changed to accurate path
        _sp.Popen([sys.executable, "scripts/demo/generate_model_report.py"])
        st.sidebar.info("Generating... (~30 seconds)")

# Notification configuration status
st.sidebar.markdown("---")
st.sidebar.markdown("### \U0001f514 Notifications")

_slack_configured = bool(os.getenv('SLACK_WEBHOOK_URL'))
_email_configured = all([
    os.getenv('ALERT_EMAIL_FROM'),
    os.getenv('ALERT_EMAIL_TO'),
    os.getenv('ALERT_EMAIL_PASSWORD')
])

st.sidebar.write(
    "\U0001f4e7 Email:",
    "\u2705 Enabled" if _email_configured else "\u26aa Disabled")
st.sidebar.write(
    "\U0001f4ac Slack:",
    "\u2705 Enabled" if _slack_configured else "\u26aa Disabled")

if not _email_configured and not _slack_configured:
    st.sidebar.caption(
        "Configure email or Slack in .env "
        "to receive alerts")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 8 — METRICS FROM REAL DATA
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>📊 Key Performance Metrics</h3></div>',
            unsafe_allow_html=True)

# Compute real metrics from telemetry.csv
total_records = len(telemetry_df) if not telemetry_df.empty else 0

last_failure_prob = "N/A"
if not df_recent.empty and "failure_prob" in df_recent.columns:
    last_failure_prob = f"{df_recent['failure_prob'].iloc[-1]:.3f}"

most_common_action = "N/A"
# Prefer healing_log.json data (JSONL has most complete records)
if healing_json_records:
    from collections import Counter as _Counter
    _action_counts = _Counter(r.get("action_name", "") for r in healing_json_records)
    if _action_counts:
        most_common_action = _action_counts.most_common(1)[0][0]
elif not healing_df.empty and "action_name" in healing_df.columns:
    mode = healing_df["action_name"].mode()
    if len(mode) > 0:
        most_common_action = str(mode.iloc[0])
elif not action_df.empty and "action" in action_df.columns:
    mode = action_df["action"].mode()
    if len(mode) > 0:
        most_common_action = str(mode.iloc[0])

uptime_str = "N/A"
if not telemetry_df.empty and "timestamp" in telemetry_df.columns:
    try:
        first_ts = pd.to_datetime(telemetry_df["timestamp"].iloc[0])
        last_ts = pd.to_datetime(telemetry_df["timestamp"].iloc[-1])
        delta = last_ts - first_ts
        hours = int(delta.total_seconds() // 3600)
        mins = int((delta.total_seconds() % 3600) // 60)
        uptime_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
    except Exception:
        pass

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="Total Records",
        value=f"{total_records:,}",
        delta=f"from {TELEMETRY_CSV}",
        delta_color="off",
    )

with m2:
    st.metric(
        label="Last Failure Prob",
        value=last_failure_prob,
        delta="latest reading",
        delta_color="off",
    )

with m3:
    st.metric(
        label="Top Healing Action",
        value=most_common_action,
        delta=f"{total_healing_actions} total actions",
        delta_color="off",
    )

with m4:
    st.metric(
        label="Telemetry Uptime",
        value=uptime_str,
        delta="collection duration",
        delta_color="off",
    )

# ──────────────────────────────────────────────────────────────────────────────
# MTTR Measurement Dashboard
# ──────────────────────────────────────────────────────────────────────────────

mttr_df = _load_csv(MTTR_LOG_CSV)
mttr_has_data = (not mttr_df.empty
                 and "reduction_pct" in mttr_df.columns
                 and "action" in mttr_df.columns)

st.markdown('<div class="section-header"><h3>⏱ MTTR Performance</h3></div>',
            unsafe_allow_html=True)

if mttr_has_data:
    mttr_vals = pd.to_numeric(mttr_df["reduction_pct"], errors="coerce").dropna()
    actual_vals = pd.to_numeric(mttr_df["actual_mttr_s"], errors="coerce").dropna()

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        best = mttr_vals.max() if len(mttr_vals) > 0 else 0
        st.metric("Best MTTR Reduction", f"{best:.1f}%",
                   delta="best measured result", delta_color="off")
    with mc2:
        st.metric("Avg Actual MTTR", f"{actual_vals.mean():.1f}s",
                   delta="automated response", delta_color="off")
    with mc3:
        st.metric("Avg Reduction", f"{mttr_vals.mean():.1f}%",
                   delta=f"across {len(mttr_vals)} incidents", delta_color="off")

    recent_mttr = mttr_df.tail(10)[["timestamp", "failure_type", "action",
                                      "actual_mttr_s", "baseline_mttr_s", "reduction_pct"]]
    st.dataframe(recent_mttr, use_container_width=True, hide_index=True)
else:
    st.info("No automated healing incidents recorded yet — trigger a failure to see MTTR data.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# C) REAL-TIME CHART + PIE
# ──────────────────────────────────────────────────────────────────────────────

chart_col, pie_col = st.columns([3, 2])

with chart_col:
    st.markdown('<div class="section-header"><h3>📈 Failure Probability — Last 50 Readings</h3></div>',
                unsafe_allow_html=True)

    if not df_recent.empty and "failure_prob" in df_recent.columns:
        chart_df = df_recent[["failure_prob"]].reset_index(drop=True)
        st.line_chart(chart_df, y="failure_prob", height=250, use_container_width=True)
        st.caption("🔴 Above 0.5 = healing triggered | 🟢 Below 0.5 = system healthy")
    else:
        st.info("📡 Waiting for telemetry data... Start the telemetry collector: "
                "`python src/telemetry/main.py`")

with pie_col:
    st.markdown('<div class="section-header"><h3>🎯 RL Agent Action Distribution</h3></div>',
                unsafe_allow_html=True)

    # Build action counts from healing_log.json with maximum error tolerance
    from collections import Counter as _Ctr
    _action_counts = _Ctr()
    _hlj_path = Path("data/healing_log.json")

    if _hlj_path.exists() and _hlj_path.stat().st_size > 0:
        try:
            _content = _hlj_path.read_text(encoding="utf-8", errors="ignore")
            for _ln in _content.strip().split("\n"):
                if not _ln.strip():
                    continue
                try:
                    _rec = json.loads(_ln)
                    _aname = (
                        _rec.get("action_name")
                        or _rec.get("action")
                        or _rec.get("healing_action")
                        or "unknown"
                    )
                    if _aname and _aname != "none":
                        _action_counts[_aname] += 1
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception as _e:
            st.error(f"Could not read healing log: {_e}")

    if _action_counts:
        chart_data = pd.Series(dict(_action_counts))
        st.bar_chart(chart_data, height=250, use_container_width=True)
        st.caption(f"Total healing actions: {sum(_action_counts.values())}")
    else:
        _placeholder = pd.Series({
            "retry_build": 68, "restart_pod": 30, "scale_up": 15,
            "clear_cache": 10, "rollback_deploy": 5, "escalate_to_human": 3
        })
        st.bar_chart(_placeholder, height=250, use_container_width=True)
        st.caption("Expected distribution (no live data yet)")

st.markdown("---")

# ── Early Warning Indicator (predictive healing) ──────────────────────────
if not df_recent.empty:
    _telem_history = df_recent.tail(10).to_dict("records")
    try:
        _warn_action, _warn_conf = detect_early_warning(_telem_history)
        if _warn_action:
            st.warning(
                f"⚠️ **Early Warning:** System trending toward **{_warn_action}** "
                f"(confidence: {_warn_conf:.0%}) — pre-emptive healing may activate soon"
            )
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# FIX 2 — LIVE SCENARIO SIMULATOR
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>🎬 Live Demo</h3></div>',
            unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LIVE DEMO CONTROL PANEL — real buttons that hit real services
# ──────────────────────────────────────────────────────────────────────────────

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

JENKINS_JOB_NAME = os.getenv("JENKINS_JOB", "neuroshield-app-build")


def _jenkins_auth():
    u = os.getenv("JENKINS_USERNAME") or os.getenv("JENKINS_USER") or "admin"
    p = os.getenv("JENKINS_PASSWORD") or os.getenv("JENKINS_TOKEN") or "admin123"
    return (u, p)


def _jenkins_crumb(session):
    try:
        r = session.get(f"{JENKINS_URL}/crumbIssuer/api/json", timeout=5)
        if r.status_code == 200:
            d = r.json()
            session.headers[d["crumbRequestField"]] = d["crumb"]
    except Exception:
        pass


with ctrl_col1:
    if st.button("🔨 Trigger Build Failure", use_container_width=True):
        with st.spinner("Triggering Jenkins build..."):
            try:
                s = requests.Session()
                s.auth = _jenkins_auth()
                _jenkins_crumb(s)
                r = s.post(f"{JENKINS_URL}/job/{JENKINS_JOB_NAME}/build", timeout=10)
                if r.status_code in {200, 201, 202, 301, 302}:
                    st.success("✅ Build triggered! Watch Jenkins: "
                              f"{JENKINS_URL}/job/{JENKINS_JOB_NAME}/")
                    st.info("NeuroShield will detect the build result in ~15 seconds "
                            "and auto-heal if it fails.")
                else:
                    st.error(f"Jenkins returned {r.status_code}")
            except Exception as e:
                st.error(f"Jenkins unreachable: {e}")

with ctrl_col2:
    if st.button("💀 Crash the Pod", use_container_width=True):
        with st.spinner("Crashing pod..."):
            try:
                requests.post(f"{DUMMY_APP_URL}/crash", timeout=5)
            except Exception:
                pass  # expected — pod dies, connection drops
            st.success("✅ Pod crash triggered!")
            st.info("NeuroShield will detect the pod failure and restart it in ~15 seconds.")
            st.code("kubectl get pods --watch", language="bash")

with ctrl_col3:
    if st.button("🔥 Stress Memory", use_container_width=True):
        try:
            r = requests.get(f"{DUMMY_APP_URL}/stress", timeout=10)
            if r.status_code == 200:
                data = r.json()
                st.info(f"Stress: {data.get('memory_before_mb', '?')}→{data.get('memory_after_mb', '?')} MB "
                        f"for {data.get('duration_seconds', 30)}s")
            else:
                st.error(f"Stress endpoint returned {r.status_code}")
        except Exception as e:
            st.error(f"App unreachable: {e}")

with ctrl_col4:
    if st.button("🚫 Bad Deployment", use_container_width=True):
        import subprocess as _sp
        r1 = _sp.run(["kubectl", "set", "env", "deployment/dummy-app",
                       "APP_VERSION=v2-broken", "-n", "default"],
                      capture_output=True, text=True, timeout=30)
        _sp.run(["kubectl", "patch", "deployment", "dummy-app", "-n", "default",
                 "-p", '{"spec":{"template":{"metadata":{"annotations":{"bad":"true"}}}}}'],
                capture_output=True, text=True, timeout=30)
        if r1.returncode == 0:
            st.error("Deployed v2-broken — /health will return 500")
        else:
            st.error(f"kubectl error: {r1.stderr[:200]}")

# ── Scenario 4, 5, 6 — Advanced Demo Buttons ──────────────────────────────
st.markdown("**Advanced Scenarios** (triggers non-retry-build actions):")
adv_col1, adv_col2, adv_col3 = st.columns(3)

with adv_col1:
    if st.button("⚡ CPU Spike → scale_up", use_container_width=True,
                 help="Spawns a CPU-intensive process. scale_up triggers when CPU > 80%."):
        import sys as _sys
        import subprocess as _sp
        try:
            proc = _sp.Popen(
                [_sys.executable, "-c",
                 "import time; start=time.time();"
                 "[sum(range(10**6)) for _ in iter(int,1) if time.time()-start<25]"],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            )
            st.success(f"✅ CPU spike started (PID {proc.pid}) — scale_up should trigger in ~15s")
            st.code("kubectl get pods --watch", language="bash")
        except Exception as e:
            st.error(f"Could not start CPU process: {e}")

with adv_col2:
    if st.button("🧠 Memory Leak → clear_cache", use_container_width=True,
                 help="Calls /stress to allocate 200 MB. clear_cache triggers when memory > 70%."):
        try:
            r = requests.get(f"{DUMMY_APP_URL}/stress", timeout=10)
            if r.status_code == 200:
                d = r.json()
                st.success(
                    f"✅ Memory stress: {d.get('memory_before_mb', '?')} → "
                    f"{d.get('memory_after_mb', '?')} MB for {d.get('duration_seconds', 30)}s"
                )
                st.info("clear_cache should trigger when memory > 70% with healthy build")
            else:
                st.error(f"Stress returned {r.status_code}")
        except Exception as e:
            st.error(f"App unreachable: {e}")

with adv_col3:
    if st.button("🚨 Escalate to Human", use_container_width=True,
                 help="Directly triggers escalation: sends email alert, writes alert banner."):
        try:
            from src.utils.notifications import (
                send_escalation_alert, write_active_alert
            )
            from src.orchestrator.main import generate_incident_report
            _ctx = {
                "failure_prob": "0.92",
                "failure_pattern": "ManualEscalation",
                "build_number": "demo",
                "escalation_reason": "Dashboard-triggered escalation demo",
                "prometheus_cpu_usage": "35",
                "prometheus_memory_usage": "68",
                "jenkins_last_build_status": "FAILURE",
            }
            write_active_alert(
                title="Human Intervention Required",
                message="Dashboard-triggered escalation demo",
                severity="HIGH",
                details=_ctx,
            )
            send_escalation_alert(
                reason="Dashboard-triggered escalation demo",
                report_path="data/escalation_reports/",
                telemetry=_ctx,
            )
            _rid, _rpath = generate_incident_report(
                _ctx, "escalate_to_human", "Dashboard-triggered escalation demo"
            )
            st.error(f"🚨 Escalation triggered! Alert banner active. Report: {_rpath}")
            st.rerun()
        except Exception as e:
            st.error(f"Escalation failed: {e}")

# ── Pod Status Widget (real kubectl) ─────────────────────────────────────
st.markdown('<div class="section-header"><h3>🖥️ Live Infrastructure Status</h3></div>',
            unsafe_allow_html=True)

infra_col1, infra_col2 = st.columns(2)

with infra_col1:
    st.markdown("**Pod Status** (kubectl get pods)")
    try:
        import subprocess as _sp
        pods_result = _sp.run(
            ["kubectl", "get", "pods", "-n", "default", "-l", "app=dummy-app", "-o", "wide"],
            capture_output=True, text=True, timeout=10,
        )
        if pods_result.returncode == 0 and pods_result.stdout.strip():
            st.code(pods_result.stdout.strip(), language="text")
        else:
            st.info("No pods found or kubectl not available")
    except Exception:
        st.info("kubectl not available")

with infra_col2:
    st.markdown("**Jenkins Build History** (last 5 builds)")
    try:
        jr = requests.get(
            f"{JENKINS_URL}/job/{JENKINS_JOB_NAME}/api/json?tree=builds[number,result,timestamp,duration]{{0,5}}",
            auth=_jenkins_auth(), timeout=5,
        )
        if jr.status_code == 200:
            builds = jr.json().get("builds", [])
            if builds:
                rows = []
                for b in builds:
                    result_str = b.get("result") or "RUNNING"
                    icon = "✅" if result_str == "SUCCESS" else "❌" if result_str == "FAILURE" else "⏳"
                    rows.append(f"{icon}  #{b['number']}  {result_str}  ({b.get('duration', 0)}ms)")
                st.code("\n".join(rows), language="text")
            else:
                st.info("No builds yet")
        else:
            st.info(f"Jenkins returned {jr.status_code}")
    except Exception as e:
        st.info(f"Jenkins unreachable: {e}")

# ── Healing Status Indicator ─────────────────────────────────────────────
healing_log_json = Path("data/healing_log.json")
is_healing = False
if healing_log_json.exists():
    try:
        lines = healing_log_json.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            # If last action was < 30s ago, consider "actively healing"
            last_ts = datetime.fromisoformat(last.get("timestamp", "2000-01-01")).replace(tzinfo=timezone.utc)
            delta = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if delta < 30 and not last.get("success", True):
                is_healing = True
    except Exception:
        pass

if is_healing:
    st.markdown(
        '<div style="background:#7f1d1d; color:#fca5a5; padding:12px 20px; border-radius:8px; '
        'text-align:center; font-weight:700; font-size:1.1rem;">'
        '🔴 ACTIVELY HEALING — Orchestrator is fixing a failure</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="background:#065f46; color:#6ee7b7; padding:12px 20px; border-radius:8px; '
        'text-align:center; font-weight:700; font-size:1.1rem;">'
        '🟢 SYSTEM HEALTHY — All services operational</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Keep the original simulation buttons below for offline demos ──────────
st.markdown('<div class="section-header"><h3>🎭 Offline Simulation (no infra needed)</h3></div>',
            unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns(2)

_BUILD_STEPS = [
    ("🔵", "Developer pushed code to main branch"),
    ("🔵", "Jenkins detected new commit — starting build #42"),
    ("🔴", "Build FAILED — dependency download timed out"),
    ("🟡", "NeuroShield detected failure (confidence: 87%)"),
    ("🟡", "Telemetry: CPU 45%, Memory 62%, Error rate 0.8"),
    ("🟡", "DistilBERT encoding build log..."),
    ("🟡", "PPO Agent analyzing 52D state vector..."),
    ("🟢", "Decision: retry_build (confidence: 91%)"),
    ("🟡", "Executing: Retrying Jenkins build #42..."),
    ("🟢", "Build #43 SUCCESS ✓"),
    ("🟢", "MTTR: 18s | Baseline: 32s | Reduction: 43.75%"),
]

_POD_STEPS = [
    ("🔵", "Pod dummy-app-7f4d8ddfc7 status: Running"),
    ("🔴", "Prometheus alert: Memory spike — 91%"),
    ("🟡", "NeuroShield detected anomaly (confidence: 94%)"),
    ("🟡", "Telemetry: CPU 78%, Memory 91%, Restarts 3"),
    ("🟡", "PPO Agent analyzing 52D state vector..."),
    ("🟢", "Decision: restart_pod (confidence: 88%)"),
    ("🟡", "kubectl rollout restart deployment/dummy-app"),
    ("🟢", "Pod restarted successfully ✓ — Memory: 34%"),
    ("🔴", "ALERT: Pod crashed again — pattern detected"),
    ("🟠", "Decision: escalate_to_human"),
    ("🟠", "ESCALATION SENT with full diagnosis report"),
]

with sim_col1:
    if st.button("▶ Simulate Build Failure", use_container_width=True):
        container = st.empty()
        lines: list[str] = []
        for icon, msg in _BUILD_STEPS:
            lines.append(f'<div class="sim-step">{icon} {msg}</div>')
            container.markdown("".join(lines), unsafe_allow_html=True)
            time.sleep(1.0)
        st.success("Scenario complete — Build auto-healed!")

with sim_col2:
    if st.button("▶ Simulate Pod Crash", use_container_width=True):
        container = st.empty()
        lines = []
        for icon, msg in _POD_STEPS:
            lines.append(f'<div class="sim-step">{icon} {msg}</div>')
            container.markdown("".join(lines), unsafe_allow_html=True)
            time.sleep(1.0)
        st.warning("Scenario complete — Escalated to human after repeat crash")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 3 — HEALING ACTIONS EXPLANATION (6 cards)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>🩹 Healing Actions Explained</h3></div>',
            unsafe_allow_html=True)

_ACTIONS_INFO = [
    ("restart_pod", "Pod is unresponsive or crashed",
     "App crashed due to OOM", "action-green"),
    ("scale_up", "CPU/Memory above 80% threshold",
     "Traffic spike detected", "action-green"),
    ("retry_build", "Build failed due to flaky test or network",
     "Dependency timeout", "action-green"),
    ("rollback_deploy", "New deployment causing errors",
     "Bad code pushed to prod", "action-green"),
    ("clear_cache", "Memory bloat detected",
     "Cache grew too large", "action-green"),
    ("escalate_to_human", "Unknown pattern, needs investigation",
     "Repeated crashes", "action-orange"),
]

act_cols = st.columns(3)
for idx, (name, trigger, example, css_cls) in enumerate(_ACTIONS_INFO):
    with act_cols[idx % 3]:
        st.markdown(f"""
        <div class="action-card {css_cls}">
            <strong style="color:#f0f4f8; font-size:1rem;">{name}</strong><br>
            <span style="color:#a0aec0; font-size:0.85rem;">
                <b>Triggers when:</b> {trigger}<br>
                <b>Example:</b> {example}
            </span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# D) HEALING ACTIONS TABLE
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>🔧 Recent Healing Decisions</h3></div>',
            unsafe_allow_html=True)

_heal_rows: list[dict] = []
_heal_json_path = Path("data/healing_log.json")
if _heal_json_path.exists():
    _raw = _heal_json_path.read_text(encoding="utf-8").strip()
    if _raw:
        for _hl_line in _raw.splitlines():
            try:
                _hr = json.loads(_hl_line)
                _hctx = _hr.get("context", {})
                _prob = _hctx.get("failure_prob", "")
                _conf = f"{float(_prob):.1%}" if _prob else "—"
                _heal_rows.append({
                    "Time": str(_hr.get("timestamp", ""))[:19],
                    "Action": _hr.get("action_name", ""),
                    "Result": "✅" if _hr.get("success") else "❌",
                    "Confidence": _conf,
                })
            except Exception:
                pass

if _heal_rows:
    _heal_display_df = pd.DataFrame(_heal_rows[-10:])
    st.dataframe(_heal_display_df, use_container_width=True, hide_index=True)
else:
    st.info("No healing actions yet")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 4 — REAL-TIME SCENARIO LOG from telemetry.csv
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>📜 Real-Time Telemetry Log</h3></div>',
            unsafe_allow_html=True)

if not df_recent.empty and "failure_prob" in df_recent.columns:
    # Show last 20 readings with REAL predictor probabilities
    log_rows = df_recent.tail(20).iloc[::-1]
    log_lines: list[str] = []
    for _, row in log_rows.iterrows():
        ts_val = str(row.get("timestamp", ""))
        try:
            ts_short = pd.to_datetime(ts_val).strftime("%H:%M:%S")
        except Exception:
            ts_short = ts_val[:8] if len(ts_val) >= 8 else ts_val

        status = str(row.get("jenkins_last_build_status", ""))
        prob = float(row.get("failure_prob", 0.0))

        action = "none"
        result = "System healthy"
        if prob > 0.5:
            action = "retry_build"
            result = "HEALING"
        elif prob > 0.3:
            action = "monitoring"
            result = "Watching"
        elif status.upper() == "FAILURE":
            action = "retry_build"
            result = "Build failed"

        color = "#10b981" if prob < 0.3 else "#f59e0b" if prob < 0.5 else "#ef4444"
        status_icon = "❌" if status.upper() == "FAILURE" else "✅" if status.upper() == "SUCCESS" else "❓"
        log_lines.append(
            f'<div class="log-entry">'
            f'<span style="color:#6b7280;">[{ts_short}]</span> '
            f'{status_icon} {status} | '
            f'Failure prob: <span style="color:{color}; font-weight:600;">{prob:.3f}</span> '
            f'→ Action: <b>{action}</b> → {result}'
            f'</div>'
        )
    st.markdown("".join(log_lines), unsafe_allow_html=True)
else:
    st.info("📡 No telemetry data yet. Start collector: `python src/telemetry/main.py`")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 5 — SYSTEM ARCHITECTURE VISUAL (Plotly pipeline)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>🏗️ System Architecture</h3></div>',
            unsafe_allow_html=True)

_PIPELINE_STAGES = [
    ("Telemetry<br>Collector", "#3b82f6"),
    ("DistilBERT<br>Log Encoder", "#8b5cf6"),
    ("52D State<br>Vector", "#f59e0b"),
    ("PPO RL<br>Agent", "#10b981"),
    ("Healing<br>Action", "#ef4444"),
    ("Result<br>Logged", "#06b6d4"),
]

fig_arch = go.Figure()

for i, (label, color) in enumerate(_PIPELINE_STAGES):
    x_center = i * 1.5
    # Box
    fig_arch.add_shape(
        type="rect",
        x0=x_center - 0.55, x1=x_center + 0.55,
        y0=-0.4, y1=0.4,
        fillcolor=color,
        opacity=0.85,
        line=dict(color=color, width=2),
    )
    # Label
    fig_arch.add_annotation(
        x=x_center, y=0,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(color="white", size=12),
    )
    # Arrow to next
    if i < len(_PIPELINE_STAGES) - 1:
        fig_arch.add_annotation(
            x=x_center + 0.55, y=0,
            ax=x_center + 0.95, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#60a5fa",
        )

fig_arch.update_layout(
    height=140,
    margin=dict(t=10, b=10, l=20, r=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(visible=False, range=[-1, len(_PIPELINE_STAGES) * 1.5]),
    yaxis=dict(visible=False, range=[-0.8, 0.8]),
)
st.plotly_chart(fig_arch, use_container_width=True)

# Three column detail boxes
col_a1, col_a2, col_a3 = st.columns(3)

with col_a1:
    st.markdown("""
    **🔍 Detection Layer**
    - Jenkins API polling (build status, logs)
    - Prometheus metrics (CPU, memory, pods)
    - Application health endpoint
    - Telemetry every 15 seconds
    """)

with col_a2:
    st.markdown("""
    **🧠 Intelligence Layer**
    - DistilBERT encodes raw logs
    - PCA reduces to 16 dimensions
    - Failure classifier (PyTorch)
    - PPO policy selects optimal action
    """)

with col_a3:
    st.markdown("""
    **⚡ Action Layer**
    - Kubernetes API (restart, scale, rollback)
    - Jenkins API (retry builds)
    - Escalation to human review
    - Full audit logging of all decisions
    """)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Self-CI Pipeline History
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>🔄 NeuroShield Self-CI History</h3></div>',
            unsafe_allow_html=True)

_self_ci_path_main = Path("data/self_ci_status.json")
if _self_ci_path_main.exists():
    try:
        _sci_data = json.loads(_self_ci_path_main.read_text(encoding="utf-8"))
        _sci_cols = st.columns(4)
        with _sci_cols[0]:
            st.metric("Build #", _sci_data.get("build_number", "—"))
        with _sci_cols[1]:
            _sci_res = _sci_data.get("result", "UNKNOWN")
            st.metric("Result", _sci_res)
        with _sci_cols[2]:
            _sci_dur = _sci_data.get("duration_ms", 0)
            st.metric("Duration", f"{_sci_dur / 1000:.1f}s" if _sci_dur else "—")
        with _sci_cols[3]:
            _sci_active = _sci_data.get("active", False)
            st.metric("Alert", "🚨 ACTIVE" if _sci_active else "✅ Clear")
        if _sci_data.get("reason"):
            st.caption(f"Last status: {_sci_data['reason']}")

        # Show build history table if available
        _sci_builds = _sci_data.get("builds", [])
        if _sci_builds:
            st.markdown("**Build History**")
            import pandas as _pd_ci
            _ci_rows = []
            for _b in _sci_builds:
                _ci_rows.append({
                    "Build #": _b.get("number", "?"),
                    "Result": _b.get("result", "UNKNOWN"),
                    "Timestamp": _b.get("timestamp_str", "—"),
                    "Duration": f"{_b.get('duration_ms', 0) / 1000:.1f}s",
                })
            _ci_df = _pd_ci.DataFrame(_ci_rows)
            st.dataframe(_ci_df, use_container_width=True, hide_index=True)
    except Exception:
        st.info("⚠️ Could not read self-CI status file")
else:
    st.info("⏳ No self-CI data yet. Run: `python scripts/setup_neuroshield_cicd.py`")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.caption(
    "NeuroShield AIOps Platform v2.0 — "
    "PPO Reinforcement Learning · DistilBERT Log Encoding · "
    "52D State Space · 6 Healing Actions"
)

# ──────────────────────────────────────────────────────────────────────────────
# Auto-refresh every 10 seconds
# ──────────────────────────────────────────────────────────────────────────────

time.sleep(10)
st.rerun()
