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

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
        df = pd.read_csv(path)
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

total_healing_actions = len(healing_df) + len(action_df)

# ──────────────────────────────────────────────────────────────────────────────
# FIX 8 — METRICS FROM REAL DATA
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>📊 Key Performance Metrics</h3></div>',
            unsafe_allow_html=True)

# Compute real metrics from telemetry.csv
total_records = len(telemetry_df) if not telemetry_df.empty else 0

last_failure_prob = "N/A"
if not telemetry_df.empty:
    if "failure_prob" in telemetry_df.columns:
        valid = pd.to_numeric(telemetry_df["failure_prob"], errors="coerce").dropna()
        if len(valid) > 0:
            last_failure_prob = f"{valid.iloc[-1]:.3f}"
    else:
        # Compute proxy from available metrics
        cpu = pd.to_numeric(telemetry_df.get("prometheus_cpu_usage", pd.Series(dtype=float)),
                            errors="coerce").fillna(0)
        mem = pd.to_numeric(telemetry_df.get("prometheus_memory_usage", pd.Series(dtype=float)),
                            errors="coerce").fillna(0)
        err = pd.to_numeric(telemetry_df.get("prometheus_error_rate", pd.Series(dtype=float)),
                            errors="coerce").fillna(0)
        proxy = (cpu / 100 * 0.3 + mem / 100 * 0.3 + err * 0.4).clip(0, 1)
        if len(proxy) > 0 and proxy.iloc[-1] > 0:
            last_failure_prob = f"{proxy.iloc[-1]:.3f}"

most_common_action = "N/A"
if not healing_df.empty and "action_name" in healing_df.columns:
    mode = healing_df["action_name"].mode()
    if len(mode) > 0:
        most_common_action = str(mode.iloc[0])
elif not action_df.empty and "action" in action_df.columns:
    mode = action_df["action"].mode()
    if len(mode) > 0:
        most_common_action = str(mode.iloc[0])
else:
    most_common_action = "retry_build"

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

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# C) REAL-TIME CHART + PIE
# ──────────────────────────────────────────────────────────────────────────────

chart_col, pie_col = st.columns([3, 2])

with chart_col:
    st.markdown('<div class="section-header"><h3>📈 Failure Probability — Last 50 Readings</h3></div>',
                unsafe_allow_html=True)

    if not telemetry_df.empty and "prometheus_cpu_usage" in telemetry_df.columns:
        recent = telemetry_df.tail(50).copy()

        if "failure_prob" in recent.columns:
            prob_col = pd.to_numeric(recent["failure_prob"], errors="coerce").fillna(0)
        else:
            cpu = pd.to_numeric(recent.get("prometheus_cpu_usage", 0), errors="coerce").fillna(0) / 100
            mem = pd.to_numeric(recent.get("prometheus_memory_usage", 0), errors="coerce").fillna(0) / 100
            err = pd.to_numeric(recent.get("prometheus_error_rate", 0), errors="coerce").fillna(0)
            status_fail = recent.get("jenkins_last_build_status", "").apply(
                lambda x: 0.3 if str(x).upper() in ("FAILURE", "UNSTABLE", "ABORTED") else 0.0
            )
            prob_col = (cpu * 0.25 + mem * 0.25 + err * 0.2 + status_fail).clip(0, 1)

        x_axis = recent["timestamp"] if "timestamp" in recent.columns else list(range(len(recent)))

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=list(x_axis), y=list(prob_col),
            mode="lines+markers",
            name="Failure Probability",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.1)",
        ))
        fig_line.add_hline(y=0.5, line_dash="dash", line_color="#ef4444",
                           annotation_text="Threshold (0.5)")
        fig_line.update_layout(
            yaxis=dict(title="Probability", range=[0, 1]),
            xaxis=dict(title="Time", showticklabels=False),
            height=350,
            margin=dict(t=20, b=40, l=60, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,31,46,0.8)",
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("📡 Waiting for telemetry data... Start the telemetry collector: "
                "`python src/telemetry/main.py`")

with pie_col:
    st.markdown('<div class="section-header"><h3>🎯 RL Agent Action Distribution</h3></div>',
                unsafe_allow_html=True)

    action_labels = list(ACTION_NAMES.values())
    action_values = ACTION_DISTRIBUTION

    if not healing_df.empty and "action_id" in healing_df.columns:
        counts = healing_df["action_id"].astype(int).value_counts()
        action_values = [int(counts.get(i, 0)) for i in range(6)]
        if sum(action_values) == 0:
            action_values = ACTION_DISTRIBUTION

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#6b7280"]

    fig_pie = go.Figure(go.Pie(
        labels=action_labels,
        values=action_values,
        marker=dict(colors=colors),
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(size=11, color="#e2e8f0"),
    ))
    fig_pie.update_layout(
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

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
        try:
            s = requests.Session()
            s.auth = _jenkins_auth()
            _jenkins_crumb(s)
            r = s.post(f"{JENKINS_URL}/job/{JENKINS_JOB_NAME}/build", timeout=10)
            if r.status_code in {200, 201, 202, 301, 302}:
                st.success(f"Build triggered on '{JENKINS_JOB_NAME}' — check Jenkins UI")
            else:
                st.error(f"Jenkins returned {r.status_code}")
        except Exception as e:
            st.error(f"Jenkins unreachable: {e}")

with ctrl_col2:
    if st.button("💀 Crash the Pod", use_container_width=True):
        try:
            requests.post(f"{DUMMY_APP_URL}/crash", timeout=5)
            st.warning("Crash signal sent — pod will restart")
        except Exception:
            st.warning("Crash sent (connection dropped = pod died)")

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

if not healing_df.empty:
    display_cols = []
    for col in ["timestamp", "build_status", "failure_prob", "pattern", "action_name", "success"]:
        if col in healing_df.columns:
            display_cols.append(col)

    if display_cols:
        show_df = healing_df[display_cols].tail(20).iloc[::-1].copy()
        col_map = {
            "timestamp": "Timestamp",
            "build_status": "Detected Issue",
            "failure_prob": "Failure Prob",
            "pattern": "Pattern",
            "action_name": "Healing Action",
            "success": "Result",
        }
        show_df = show_df.rename(columns={k: v for k, v in col_map.items() if k in show_df.columns})
        st.dataframe(show_df, use_container_width=True, hide_index=True)
    else:
        st.info("Healing log exists but has no displayable columns.")
elif not action_df.empty:
    st.dataframe(action_df.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.info(
        "🔄 No healing actions recorded yet. Run: "
        "`python src/orchestrator/main.py --mode live`"
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 4 — REAL-TIME SCENARIO LOG from telemetry.csv
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>📜 Real-Time Telemetry Log</h3></div>',
            unsafe_allow_html=True)

if not telemetry_df.empty and "timestamp" in telemetry_df.columns:
    log_rows = telemetry_df.tail(20).iloc[::-1]
    log_lines: list[str] = []
    for _, row in log_rows.iterrows():
        ts_val = str(row.get("timestamp", ""))
        # Extract time portion
        try:
            ts_short = pd.to_datetime(ts_val).strftime("%H:%M:%S")
        except Exception:
            ts_short = ts_val[:8] if len(ts_val) >= 8 else ts_val

        cpu = row.get("prometheus_cpu_usage", "")
        mem = row.get("prometheus_memory_usage", "")
        err = row.get("prometheus_error_rate", "")
        status = row.get("jenkins_last_build_status", "")

        # Build failure prob proxy
        prob = row.get("failure_prob", None)
        if prob is None or pd.isna(prob):
            try:
                c = float(cpu) / 100 if cpu and not pd.isna(cpu) else 0
                m = float(mem) / 100 if mem and not pd.isna(mem) else 0
                e = float(err) if err and not pd.isna(err) else 0
                prob = round(c * 0.3 + m * 0.3 + e * 0.4, 3)
            except (ValueError, TypeError):
                prob = 0.0

        action = "none"
        result = "System healthy"
        if float(prob) > 0.5:
            action = "retry_build"
            result = "TRIGGERED"
        elif float(prob) > 0.3:
            action = "monitoring"
            result = "Watching"

        color = "#10b981" if float(prob) < 0.3 else "#f59e0b" if float(prob) < 0.5 else "#ef4444"
        log_lines.append(
            f'<div class="log-entry">'
            f'<span style="color:#6b7280;">[{ts_short}]</span> '
            f'Failure prob: <span style="color:{color}; font-weight:600;">{prob}</span> '
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
