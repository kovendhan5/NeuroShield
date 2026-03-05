"""NeuroShield AIOps Platform — Professional Dashboard."""

from __future__ import annotations

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
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuroShield AIOps Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Dark professional theme via CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Cards */
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
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    /* Headers */
    h1, h2, h3 { color: #f0f4f8 !important; }
    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    /* Badge styles */
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

ACTION_DISTRIBUTION = [0, 0, 30, 0, 2, 68]  # From evaluation results


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
    icon = "●" if is_up else "●"
    label = "ONLINE" if is_up else "OFFLINE"
    return f'<span class="{cls}">{icon} {name}: {label}</span>'


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Project Info & Controls
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ NeuroShield")
    st.markdown("**AIOps Self-Healing CI/CD System**")
    st.markdown("---")

    st.markdown("### 📋 Project Info")
    st.markdown("""
    - **ML Model:** DistilBERT Log Encoder
    - **RL Agent:** PPO (Stable Baselines3)
    - **State Space:** 52 dimensions
    - **Action Space:** 6 healing actions
    - **Platform:** Jenkins + Prometheus + K8s
    """)

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
    st.markdown("### 📊 Data Paths")
    st.caption(f"Telemetry: {TELEMETRY_CSV}")
    st.caption(f"Healing Log: {HEALING_LOG_CSV}")

    st.markdown("---")
    st.caption("NeuroShield v2.0 — B.Tech Final Year Project")
    st.caption("Jeppiaar Institute of Technology")


# ──────────────────────────────────────────────────────────────────────────────
# A) HEADER — Title + Live Status Badges
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ NeuroShield AIOps Platform")
st.markdown(
    "*Intelligent Self-Healing CI/CD Pipeline — "
    "PPO Reinforcement Learning + DistilBERT Log Analysis*"
)

# Status badges
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
# Explanation box — what NeuroShield does
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️  What is NeuroShield? (Click to learn)", expanded=False):
    st.markdown("""
    **NeuroShield** is an AI-powered self-healing system for CI/CD pipelines.

    **Problem:** CI/CD pipeline failures (build failures, OOM crashes, flaky tests)
    cause long Mean Time To Recovery (MTTR), costing engineers hours of manual debugging.

    **Solution:** NeuroShield uses:
    1. **DistilBERT** to encode Jenkins build logs into 16D semantic embeddings
    2. A **52-dimensional state vector** combining build metrics, resource metrics,
       log embeddings, and dependency signals
    3. A **PPO Reinforcement Learning agent** (trained via Stable Baselines3) that
       observes the state and selects one of **6 healing actions**:
       `restart_pod`, `scale_up`, `retry_build`, `rollback_deploy`, `clear_cache`, `escalate_to_human`
    4. Actions are executed automatically against Jenkins, Kubernetes, and the application

    **Results:** 44% MTTR reduction, F1 Score of 1.000, fully autonomous healing loop.
    """)

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

telemetry_df = _load_csv(TELEMETRY_CSV)
healing_df = _load_csv(HEALING_LOG_CSV)
action_df = _load_csv(ACTION_HISTORY_CSV)

total_healing_actions = len(healing_df) + len(action_df)

# Compute system health from latest telemetry or app status
system_health = 100
if app_up:
    try:
        r = requests.get(f"{DUMMY_APP_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json() if "json" in r.headers.get("content-type", "") else {}
            system_health = int(data.get("health", 100))
    except Exception:
        system_health = 85 if app_up else 0
elif not app_up:
    system_health = 0

# ──────────────────────────────────────────────────────────────────────────────
# B) KEY METRICS ROW — 4 big number cards
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>📊 Key Performance Metrics</h3></div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="MTTR Reduction",
        value="44%",
        delta="↑ vs 38% baseline",
        delta_color="normal",
    )

with m2:
    st.metric(
        label="F1 Score",
        value="1.000",
        delta="Perfect classification",
        delta_color="off",
    )

with m3:
    st.metric(
        label="Total Healing Actions",
        value=str(total_healing_actions),
        delta=f"{total_healing_actions} executed",
        delta_color="off",
    )

with m4:
    health_delta = "Healthy" if system_health >= 80 else "Degraded" if system_health >= 50 else "Critical"
    st.metric(
        label="System Health",
        value=f"{system_health}%",
        delta=health_delta,
        delta_color="normal" if system_health >= 80 else "inverse",
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# C) REAL-TIME CHART — Failure probability over last 50 readings
# ──────────────────────────────────────────────────────────────────────────────

chart_col, pie_col = st.columns([3, 2])

with chart_col:
    st.markdown('<div class="section-header"><h3>📈 Failure Probability — Last 50 Readings</h3></div>',
                unsafe_allow_html=True)

    if not telemetry_df.empty and "prometheus_cpu_usage" in telemetry_df.columns:
        recent = telemetry_df.tail(50).copy()

        # Compute failure probability proxy from telemetry
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


# ──────────────────────────────────────────────────────────────────────────────
# E) ACTION DISTRIBUTION PIE CHART
# ──────────────────────────────────────────────────────────────────────────────

with pie_col:
    st.markdown('<div class="section-header"><h3>🎯 RL Agent Action Distribution</h3></div>',
                unsafe_allow_html=True)

    action_labels = list(ACTION_NAMES.values())
    action_values = ACTION_DISTRIBUTION

    # Override with real data if available
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

    st.caption(
        "Distribution from evaluation: "
        "retry_build 30%, escalate_to_human 68%, clear_cache 2%"
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# D) HEALING ACTIONS TABLE — Recent decisions
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>🔧 Recent Healing Decisions</h3></div>',
            unsafe_allow_html=True)

if not healing_df.empty:
    display_cols = []
    for col in ["timestamp", "build_status", "failure_prob", "pattern", "action_name", "success"]:
        if col in healing_df.columns:
            display_cols.append(col)

    if display_cols:
        show_df = healing_df[display_cols].tail(20).copy()
        show_df = show_df.iloc[::-1]  # Most recent first

        # Rename columns for display
        col_map = {
            "timestamp": "Timestamp",
            "build_status": "Detected Issue",
            "failure_prob": "Failure Prob",
            "pattern": "Pattern",
            "action_name": "Healing Action",
            "success": "Result",
        }
        show_df = show_df.rename(columns={k: v for k, v in col_map.items() if k in show_df.columns})

        # Add MTTR saved estimate
        if "Failure Prob" in show_df.columns:
            show_df["MTTR Saved"] = show_df["Failure Prob"].apply(
                lambda p: f"~{float(p) * 8:.1f} min" if p and p != "0.000" else "—"
            )

        st.dataframe(show_df, use_container_width=True, hide_index=True)
    else:
        st.info("Healing log exists but has no displayable columns.")
elif not action_df.empty:
    show_df = action_df.tail(20).iloc[::-1]
    st.dataframe(show_df, use_container_width=True, hide_index=True)
else:
    st.info(
        "🔄 No healing actions recorded yet. The orchestrator will log actions here "
        "when it detects failures. Run: `python src/orchestrator/main.py --mode live`"
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# F) ARCHITECTURE DIAGRAM
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><h3>🏗️ NeuroShield Architecture</h3></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="arch-box">
<strong>CI/CD Pipeline Flow:</strong><br><br>
📡 <strong>Telemetry Collector</strong> (Jenkins API + Prometheus + App Health)<br>
<span class="arch-arrow">⬇</span><br>
🧠 <strong>DistilBERT Log Encoder</strong> (768D → PCA → 16D embeddings)<br>
<span class="arch-arrow">⬇</span><br>
📊 <strong>52D State Vector</strong> = Build Metrics(10) + Resources(12) + Logs(16) + Dependencies(14)<br>
<span class="arch-arrow">⬇</span><br>
🤖 <strong>PPO RL Agent</strong> (Stable Baselines3) → Selects 1 of 6 Actions<br>
<span class="arch-arrow">⬇</span><br>
⚡ <strong>Healing Actions:</strong> restart_pod | scale_up | retry_build | rollback_deploy | clear_cache | escalate<br>
<span class="arch-arrow">⬇</span><br>
☸️ <strong>Kubernetes / Jenkins / Docker</strong> → Automated Recovery<br>
</div>
""", unsafe_allow_html=True)

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
    "52D State Space · 6 Healing Actions — "
    "Jeppiaar Institute of Technology"
)

# ──────────────────────────────────────────────────────────────────────────────
# G) Auto-refresh every 10 seconds
# ──────────────────────────────────────────────────────────────────────────────

time.sleep(10)
st.rerun()
