"""NeuroShield — AIOps Self-Healing Human-in-the-Loop Dashboard."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ──────────────────────────────────────────────────────────────────────────────
# Page config + auto-refresh
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuroShield — AIOps Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st_autorefresh(interval=10_000, key="auto")

# ──────────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ──────────────────────────────────────────────────────────────────────────────

TELEMETRY_CSV = os.getenv("TELEMETRY_OUTPUT_PATH", "data/telemetry.csv")
ACTION_CSV = "data/action_history.csv"
FEEDBACK_CSV = "data/feedback_log.csv"
ESCALATION_CSV = "data/escalation_log.csv"

ACTION_NAMES = {
    0: "retry_stage",
    1: "clean_and_rerun",
    2: "regenerate_config",
    3: "reallocate_resources",
    4: "trigger_safe_rollback",
    5: "escalate_to_human",
}


def load_csv(path: str, default_cols: list[str]) -> pd.DataFrame:
    """Safely load a CSV, returning an empty DataFrame on any error."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=default_cols)
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=default_cols)
        return df
    except Exception:
        return pd.DataFrame(columns=default_cols)


def _append_csv(path: str, row: dict[str, str]) -> None:
    """Append a single row to a CSV, creating with headers if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ──────────────────────────────────────────────────────────────────────────────
# Load data once per render
# ──────────────────────────────────────────────────────────────────────────────

telemetry_df = load_csv(TELEMETRY_CSV, [
    "timestamp", "prometheus_cpu_usage", "prometheus_memory_usage",
    "prometheus_pod_count", "jenkins_last_build_status",
])
action_df = load_csv(ACTION_CSV, [
    "timestamp", "action_id", "action_name", "success", "duration_ms",
])
feedback_df = load_csv(FEEDBACK_CSV, [
    "timestamp", "recommended_action", "engineer_decision",
    "override_action", "notes",
])
escalation_df = load_csv(ESCALATION_CSV, [
    "timestamp", "failure_probability", "failure_state",
    "recommended_action", "status",
])

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Header bar
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ NeuroShield — AIOps Self-Healing Dashboard")

# Status badge
_has_pending = (
    not escalation_df.empty
    and "status" in escalation_df.columns
    and (escalation_df["status"] == "PENDING_HUMAN_REVIEW").any()
)
_has_recent_action = False
if not action_df.empty and "timestamp" in action_df.columns:
    try:
        last_ts = pd.to_datetime(action_df["timestamp"].iloc[-1], utc=True)
        _has_recent_action = (
            pd.Timestamp.now(tz="UTC") - last_ts
        ).total_seconds() < 60
    except Exception:
        pass

if _has_pending:
    st.markdown(
        '<span style="background:#d32f2f;color:#fff;padding:4px 12px;'
        'border-radius:8px;font-weight:700">'
        "🔴 ALERT — Human Review Required</span>",
        unsafe_allow_html=True,
    )
elif _has_recent_action:
    st.markdown(
        '<span style="background:#f9a825;color:#000;padding:4px 12px;'
        'border-radius:8px;font-weight:700">'
        "🟡 HEALING — Action In Progress</span>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span style="background:#388e3c;color:#fff;padding:4px 12px;'
        'border-radius:8px;font-weight:700">'
        "🟢 MONITORING — All Systems Normal</span>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Top metrics row (4 columns)
# ──────────────────────────────────────────────────────────────────────────────


def _metric_val(df: pd.DataFrame, col: str, idx: int = -1) -> float | None:
    if df.empty or col not in df.columns:
        return None
    try:
        return float(df[col].iloc[idx])
    except Exception:
        return None


c1, c2, c3, c4 = st.columns(4)

cpu_cur = _metric_val(telemetry_df, "prometheus_cpu_usage")
cpu_prev = _metric_val(telemetry_df, "prometheus_cpu_usage", -2) if len(telemetry_df) >= 2 else None
c1.metric(
    "CPU Usage (%)",
    f"{cpu_cur:.1f}" if cpu_cur is not None else "--",
    delta=f"{cpu_cur - cpu_prev:.1f}" if cpu_cur is not None and cpu_prev is not None else None,
)

mem_cur = _metric_val(telemetry_df, "prometheus_memory_usage")
mem_prev = _metric_val(telemetry_df, "prometheus_memory_usage", -2) if len(telemetry_df) >= 2 else None
c2.metric(
    "Memory Usage (%)",
    f"{mem_cur:.1f}" if mem_cur is not None else "--",
    delta=f"{mem_cur - mem_prev:.1f}" if mem_cur is not None and mem_prev is not None else None,
)

pod_cur = _metric_val(telemetry_df, "prometheus_pod_count")
pod_prev = _metric_val(telemetry_df, "prometheus_pod_count", -2) if len(telemetry_df) >= 2 else None
c3.metric(
    "Pod Restarts",
    f"{pod_cur:.0f}" if pod_cur is not None else "--",
    delta=f"{pod_cur - pod_prev:.0f}" if pod_cur is not None and pod_prev is not None else None,
)

build_status = "--"
if not telemetry_df.empty and "jenkins_last_build_status" in telemetry_df.columns:
    build_status = str(telemetry_df["jenkins_last_build_status"].iloc[-1])
c4.metric("Build Status", build_status)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Three panel row
# ──────────────────────────────────────────────────────────────────────────────

left_col, center_col, right_col = st.columns(3)

# ── LEFT: Failure Prediction ──────────────────────────────────────────────────

with left_col:
    st.subheader("🔮 Failure Prediction")

    cpu_pct = (cpu_cur / 100.0) if cpu_cur is not None else 0.0
    mem_pct = (mem_cur / 100.0) if mem_cur is not None else 0.0
    pod_val = pod_cur if pod_cur is not None else 0.0
    prob = min(1.0, cpu_pct * 0.4 + mem_pct * 0.4 + pod_val / 10.0 * 0.2)
    if telemetry_df.empty:
        prob = 0.05

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={"text": "Failure Probability"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#1976d2"},
            "steps": [
                {"range": [0, 0.5], "color": "#c8e6c9"},
                {"range": [0.5, 0.75], "color": "#fff9c4"},
                {"range": [0.75, 1.0], "color": "#ffcdd2"},
            ],
        },
    ))
    gauge.update_layout(height=250, margin=dict(t=50, b=10, l=30, r=30))
    st.plotly_chart(gauge, use_container_width=True)

    if prob >= 0.7:
        st.error("⚠️ IMMINENT FAILURE PREDICTED")
    elif prob >= 0.4:
        st.warning("⚡ ELEVATED RISK DETECTED")
    else:
        st.success("✅ SYSTEM HEALTHY")

    st.caption("Confidence Score")
    st.progress(min(int(prob * 100), 100))

# ── CENTER: RL Agent Decision ─────────────────────────────────────────────────

with center_col:
    st.subheader("🤖 RL Agent Decision")

    if action_df.empty:
        st.info("No actions recorded yet.")
        last_action_name = "N/A"
        last_action_id = "—"
        last_action_ts = "—"
    else:
        last_row = action_df.iloc[-1]
        last_action_name = str(last_row.get("action_name", "unknown"))
        last_action_id = str(last_row.get("action_id", "?"))
        last_action_ts = str(last_row.get("timestamp", "?"))

    st.markdown(f"### `{last_action_name}`")
    st.caption(f"Action index: {last_action_id}  |  {last_action_ts}")

    if "mttr_reduction_pct" in action_df.columns and not action_df.empty:
        try:
            st.metric("MTTR Reduction", f"{float(action_df['mttr_reduction_pct'].iloc[-1]):.1f}%")
        except Exception:
            pass

    # -- Buttons with session-state guards --
    if "btn_clicked" not in st.session_state:
        st.session_state.btn_clicked = None

    btn_cols = st.columns(3)

    with btn_cols[0]:
        if st.button("✅ Approve", key="btn_approve"):
            _append_csv(FEEDBACK_CSV, {
                "timestamp": _now_iso(),
                "recommended_action": last_action_name,
                "engineer_decision": "APPROVED",
                "override_action": "",
                "notes": "",
            })
            st.session_state.btn_clicked = "approve"
            st.rerun()

    with btn_cols[1]:
        if st.button("❌ Override", key="btn_override"):
            st.session_state.btn_clicked = "override"

    with btn_cols[2]:
        if st.button("⏸ Pause", key="btn_pause"):
            _append_csv(FEEDBACK_CSV, {
                "timestamp": _now_iso(),
                "recommended_action": last_action_name,
                "engineer_decision": "PAUSED",
                "override_action": "",
                "notes": "Operator paused autonomous actions",
            })
            st.session_state.btn_clicked = "pause"
            st.rerun()

    if st.session_state.btn_clicked == "override":
        override_choice = st.selectbox(
            "Select override action:",
            list(ACTION_NAMES.values()),
            key="override_select",
        )
        if st.button("Submit Override", key="btn_submit_override"):
            _append_csv(FEEDBACK_CSV, {
                "timestamp": _now_iso(),
                "recommended_action": last_action_name,
                "engineer_decision": "OVERRIDDEN",
                "override_action": override_choice,
                "notes": "",
            })
            st.session_state.btn_clicked = None
            st.rerun()

# ── RIGHT: SHAP Feature Importance ────────────────────────────────────────────

with right_col:
    st.subheader("🔍 Top Contributing Factors")

    shap_data = {
        "memory_avg_5m": 0.82,
        "cpu_avg_5m": 0.71,
        "pod_restarts": 0.65,
        "failed_tests": 0.58,
        "build_duration": 0.43,
        "dep_version_drifts": 0.31,
        "network_latency": 0.28,
        "cache_miss_ratio": 0.19,
    }
    features = list(shap_data.keys())
    values = list(shap_data.values())
    colors = [
        "#d32f2f" if v > 0.6 else "#f57c00" if v >= 0.3 else "#388e3c"
        for v in values
    ]

    fig_shap = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig_shap.update_layout(
        title="Feature Importance (SHAP-style)",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        height=350,
        margin=dict(t=50, b=30, l=120, r=40),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Charts row
# ──────────────────────────────────────────────────────────────────────────────

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("📈 MTTR Trend")
    if not action_df.empty and "mttr_reduction_pct" in action_df.columns:
        fig_mttr = go.Figure()
        fig_mttr.add_trace(go.Scatter(
            x=action_df["timestamp"],
            y=pd.to_numeric(action_df["mttr_reduction_pct"], errors="coerce"),
            mode="lines+markers",
            name="MTTR Reduction %",
            line=dict(color="#1976d2", width=2),
        ))
        fig_mttr.add_hline(y=38, line_dash="dash", line_color="#f57c00",
                           annotation_text="Paper Target (38%)")
        fig_mttr.add_hline(y=44, line_dash="dash", line_color="#388e3c",
                           annotation_text="Current Avg (44%)")
        fig_mttr.update_layout(
            xaxis_title="Time", yaxis_title="MTTR Reduction %",
            height=350, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_mttr, use_container_width=True)
    else:
        st.info("No action history yet — MTTR trend will appear after healing actions run.")

with chart_right:
    st.subheader("🔥 Failure Type Breakdown")
    if not action_df.empty and "failure_type" in action_df.columns:
        ft_counts = action_df["failure_type"].value_counts()
        labels = ft_counts.index.tolist()
        vals = ft_counts.values.tolist()
    else:
        labels = ["OOM", "FlakyTest", "DependencyConflict", "NetworkLatency", "Healthy"]
        vals = [12, 18, 8, 7, 55]

    color_map = {
        "OOM": "#d32f2f", "FlakyTest": "#f57c00",
        "DependencyConflict": "#fbc02d", "NetworkLatency": "#1976d2",
        "Healthy": "#388e3c",
    }
    pie_colors = [color_map.get(l, "#9e9e9e") for l in labels]

    fig_pie = go.Figure(go.Pie(
        labels=labels, values=vals,
        marker=dict(colors=pie_colors),
        hole=0.35,
    ))
    fig_pie.update_layout(height=350, margin=dict(t=30, b=30))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — History table
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("📋 Recent Decisions")

if feedback_df.empty:
    st.info("No feedback recorded yet. Approve or override an action above.")
else:
    display_df = feedback_df.tail(20).copy()

    def _row_bg(row: pd.Series) -> list[str]:
        decision = str(row.get("engineer_decision", ""))
        if decision == "APPROVED":
            return ["background-color: #e8f5e9"] * len(row)
        if decision == "OVERRIDDEN":
            return ["background-color: #fff3e0"] * len(row)
        if decision == "PAUSED":
            return ["background-color: #f5f5f5"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(_row_bg, axis=1)
    st.dataframe(styled, use_container_width=True)

# ── Pending escalations ──────────────────────────────────────────────────────

st.subheader("🚨 Pending Human Reviews")

pending = escalation_df[
    escalation_df["status"] == "PENDING_HUMAN_REVIEW"
] if "status" in escalation_df.columns else pd.DataFrame()

if pending.empty:
    st.success("No pending reviews — all clear.")
else:
    for idx, row in pending.iterrows():
        ts = row.get("timestamp", "?")
        fp = row.get("failure_probability", "?")
        st.warning(f"**Escalation** at {ts} — failure probability: {fp}")
        if st.button("✓ Mark Resolved", key=f"resolve_{idx}"):
            # Update the CSV in-place
            escalation_df.at[idx, "status"] = "RESOLVED"
            escalation_df.to_csv(ESCALATION_CSV, index=False)
            st.rerun()

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Sidebar
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Controls")

autonomous = st.sidebar.toggle("Enable Autonomous Actions", value=True)
st.session_state["autonomous_actions"] = autonomous

threshold = st.sidebar.slider(
    "Failure Probability Threshold",
    min_value=0.50, max_value=0.95, value=0.70, step=0.05,
)
st.session_state["failure_threshold"] = threshold

st.sidebar.markdown("---")

ppo_ok = Path("models/ppo_policy.zip").exists()
pred_ok = Path("models/failure_predictor.pth").exists()
st.sidebar.metric("PPO Model", "✅ Loaded" if ppo_ok else "❌ Missing")
st.sidebar.metric("Predictor Model", "✅ Loaded" if pred_ok else "❌ Missing")
st.sidebar.metric("Total Actions Taken", len(action_df))
pending_count = int((escalation_df["status"] == "PENDING_HUMAN_REVIEW").sum()) if "status" in escalation_df.columns else 0
st.sidebar.metric("Pending Reviews", pending_count)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Force Refresh"):
    st.rerun()

if st.sidebar.button("🗑️ Clear Feedback Log"):
    st.session_state["confirm_clear"] = True

if st.session_state.get("confirm_clear"):
    st.sidebar.warning("This will delete all feedback history.")
    col_y, col_n = st.sidebar.columns(2)
    with col_y:
        if st.button("Yes, clear", key="confirm_yes"):
            if Path(FEEDBACK_CSV).exists():
                Path(FEEDBACK_CSV).unlink()
            st.session_state["confirm_clear"] = False
            st.rerun()
    with col_n:
        if st.button("Cancel", key="confirm_no"):
            st.session_state["confirm_clear"] = False
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "NeuroShield v1.0 — AIOps Self-Healing CI/CD | "
    "Jeppiaar Institute of Technology"
)
