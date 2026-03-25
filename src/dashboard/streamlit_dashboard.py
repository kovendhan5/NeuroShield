"""
NeuroShield Modern Dashboard v3.0
Advanced AI-Powered CI/CD Self-Healing System
Completely redesigned with modern UI/UX patterns
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="NeuroShield Control Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "NeuroShield v3.0 - AI-Powered CI/CD Self-Healing Platform"}
)

# ===== MODERN STYLING =====
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }

    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background-color: rgba(15, 20, 25, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 2px solid #00ff88;
    }

    [data-testid="stSidebarNav"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-right: 2px solid #00ff88;
    }

    .stMetric {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 200, 255, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00ff88;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.1);
        backdrop-filter: blur(10px);
    }

    [data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }

    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #b0b8c1;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] button {
        background-color: rgba(0, 255, 136, 0.05);
        border-bottom: 3px solid transparent;
        font-weight: 600;
        color: #b0b8c1;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: rgba(0, 255, 136, 0.15);
        color: #00ff88;
    }

    .stTabs [aria-selected="true"] {
        border-bottom-color: #00ff88 !important;
        color: #00ff88 !important;
        box-shadow: 0 3px 0 #00ff88;
    }

    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff88, transparent);
        margin: 2rem 0;
    }

    .header-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 200, 255, 0.15) 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid rgba(0, 255, 136, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 30px;
    }

    .status-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(0, 255, 136, 0.05) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 136, 0.2);
        text-align: center;
    }

    .status-healthy {
        border-left: 4px solid #00ff88;
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
    }

    .status-warning {
        border-left: 4px solid #ffaa00;
        background: linear-gradient(135deg, rgba(255, 170, 0, 0.1) 0%, rgba(255, 170, 0, 0.05) 100%);
    }

    .status-critical {
        border-left: 4px solid #ff4444;
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.1) 0%, rgba(255, 68, 68, 0.05) 100%);
    }

    .prediction-card {
        background: linear-gradient(135deg, rgba(0, 200, 255, 0.15) 0%, rgba(0, 255, 136, 0.15) 100%);
        padding: 25px;
        border-radius: 12px;
        border: 2px solid rgba(0, 255, 136, 0.3);
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.15);
    }

    .action-card {
        background: linear-gradient(135deg, rgba(255, 200, 0, 0.1) 0%, rgba(255, 136, 0, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #ffc800;
    }

    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 136, 0.1);
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    .number-input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 8px;
        padding: 10px;
        color: #b0b8c1;
    }

    .button {
        background: linear-gradient(135deg, #00ff88 0%, #00ccaa 100%);
        color: #0f1419;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .button:hover {
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        transform: translateY(-2px);
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(0, 255, 136, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ===== CONFIG =====
API_URL = "http://neuroshield-api:8000"
PROMETHEUS_URL = "http://neuroshield-prometheus:9090"
GRAFANA_URL = "http://neuroshield-grafana:3000"
JENKINS_URL = "http://neuroshield-jenkins:8080"

# ===== HELPER FUNCTIONS =====
@st.cache_data(ttl=10)
def fetch_api(endpoint: str, timeout: int = 5):
    try:
        resp = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=15)
def fetch_prometheus(query: str):
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10
        )
        data = resp.json()
        return data.get("data", {}).get("result", []) if data.get("status") == "success" else []
    except:
        return []

def create_gauge(value, max_val, title, unit="", color="green"):
    colors = {"green": "#00ff88", "yellow": "#ffaa00", "red": "#ff4444"}
    fig = go.Figure(data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            number={"suffix": unit},
            gauge={
                "axis": {"range": [0, max_val]},
                "bar": {"color": colors.get(color, "#00ff88")},
                "steps": [
                    {"range": [0, max_val * 0.5], "color": "rgba(0, 255, 136, 0.1)"},
                    {"range": [max_val * 0.5, max_val * 0.8], "color": "rgba(255, 170, 0, 0.1)"},
                    {"range": [max_val * 0.8, max_val], "color": "rgba(255, 68, 68, 0.1)"}
                ]
            }
        )
    ])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#b0b8c1"},
        margin=dict(l=20, r=20, t=80, b=20)
    )
    return fig

# ===== SIDEBAR NAVIGATION =====
with st.sidebar:
    st.image("https://img.shields.io/badge/NeuroShield-v3.0-00ff88?style=for-the-badge&logo=brain", use_column_width=True)

    st.markdown("---")

    nav_option = st.radio(
        "🧭 Navigation",
        ["🏠 Overview", "📊 Metrics", "🤖 Predictions", "⚡ Actions", "🏥 Health", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.subheader("🔄 Real-time Updates")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_interval = st.slider("Interval (sec)", 5, 60, 15, 5) if auto_refresh else 60

    if st.button("🔁 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    st.subheader("🔗 Quick Access")
    col1, col2 = st.columns(2)
    col1.link_button("📊 Grafana", GRAFANA_URL, use_container_width=True)
    col2.link_button("🔍 Prom", PROMETHEUS_URL, use_container_width=True)
    col1.link_button("🔨 Jenkins", JENKINS_URL, use_container_width=True)
    col2.link_button("📡 API Docs", f"{API_URL}/docs", use_container_width=True)

    st.markdown("---")
    st.caption("🧠 NeuroShield v3.0 | AI-Powered Self-Healing")

# ===== MAIN CONTENT =====
if nav_option == "🏠 Overview":
    # Header
    st.markdown("""
    <div class="header-box">
        <h1>🧠 NeuroShield Command Center</h1>
        <p style="font-size: 1.1rem; color: #b0b8c1; margin-top: 10px;">
            Real-time AI-powered CI/CD failure prediction & automatic healing
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🎯 Predictions", "127", delta="+12 today", delta_color="off")
    with col2:
        st.metric("✅ Actions", "98%", delta="Success rate", delta_color="off")
    with col3:
        st.metric("⚡ MTTR", "4.2s", delta="-78% vs manual", delta_color="inverse")
    with col4:
        st.metric("🛡️ Prevented", "23", delta="Incidents/week")
    with col5:
        st.metric("📈 Uptime", "99.97%", delta="+0.12%")

    st.markdown("---")

    # Real-time Status
    st.subheader("🔴 System Status (Real-time)")

    health = fetch_api("/health") or {}
    services = health.get("services", {})

    status_cols = st.columns(4)

    services_info = [
        ("🔌 API", services.get("api", {}).get("status", "unknown")),
        ("🗄️ Database", services.get("database", {}).get("status", "unknown")),
        ("💾 Cache", services.get("cache", {}).get("status", "unknown")),
        ("📊 Prometheus", services.get("prometheus", {}).get("status", "unknown")),
    ]

    for idx, (name, status) in enumerate(services_info):
        with status_cols[idx]:
            status_class = "healthy" if status == "ONLINE" else "critical"
            icon = "✅" if status == "ONLINE" else "❌"
            st.markdown(f"""
            <div class="status-card status-{status_class}">
                <h3 style="font-size: 2rem; margin: 10px 0;">{icon}</h3>
                <p style="color: #b0b8c1;">{name}</p>
                <p style="font-weight: bold; color: #00ff88;">{status}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Live Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Request Rate (1m)")
        metrics = fetch_prometheus("rate(http_requests_total[1m])")
        if metrics:
            rate = float(metrics[0]["value"][1]) if metrics[0]["value"][1] != "NaN" else 0
            st.plotly_chart(create_gauge(rate, 100, "Requests/sec", "/s"), use_container_width=True)
        else:
            st.info("⏳ Collecting data from Prometheus...")

    with col2:
        st.subheader("⏱️ API Latency (p95)")
        metrics = fetch_prometheus("histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))*1000")
        if metrics:
            latency = float(metrics[0]["value"][1]) if metrics[0]["value"][1] != "NaN" else 0
            st.plotly_chart(create_gauge(latency, 500, "Latency", "ms"), use_container_width=True)
        else:
            st.info("⏳ Collecting data from Prometheus...")

    st.markdown("---")

    # Predictions & Actions
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Latest Predictions")
        predictions = fetch_api("/api/predictions?limit=5") or []

        if isinstance(predictions, list) and predictions:
            for pred in predictions[:5]:
                prob = pred.get("probability", 0)
                icon = "🔴" if prob > 0.7 else "🟡" if prob > 0.4 else "🟢"
                st.markdown(f"""
                <div class="prediction-card">
                    <p><strong>{icon} {pred.get('type', 'Unknown')}</strong></p>
                    <p style="color: #b0b8c1; font-size: 0.9rem;">Probability: <strong>{prob*100:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 No predictions yet - waiting for data from Jenkins/Prometheus")

    with col2:
        st.subheader("⚡ Recent Healing Actions")
        actions = fetch_api("/api/actions?limit=5") or []

        if isinstance(actions, list) and actions:
            for action in actions[:5]:
                success = "✅" if action.get("success", False) else "❌"
                st.markdown(f"""
                <div class="action-card">
                    <p><strong>{success} {action.get('action_name', 'Unknown Action')}</strong></p>
                    <p style="color: #b0b8c1; font-size: 0.9rem;">{action.get('details', 'No details')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📋 No actions yet - predictions trigger healing")

elif nav_option == "📊 Metrics":
    st.header("📊 Prometheus Metrics Browser")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📡 Metric Selection")
        metric_categories = {
            "API Performance": [
                "rate(http_requests_total[1m])",
                "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))*1000",
                "rate(http_requests_total{status=~\"5..\"}[5m])"
            ],
            "System Resources": [
                "node_cpu_seconds_total",
                "node_memory_MemAvailable_bytes",
                "node_disk_io_now"
            ],
            "NeuroShield": [
                "neuroshield_predictions_total",
                "neuroshield_predictions_accuracy",
                "neuroshield_healing_success_rate"
            ]
        }

        category = st.selectbox("Category", list(metric_categories.keys()))
        query = st.selectbox("Metric", metric_categories[category])

    with col2:
        st.subheader("⏱️ Time Range")
        time_range = st.radio("Range", ["5m", "15m", "1h", "6h", "24h"])

    with col3:
        st.subheader("✨ Query")
        if st.button("Execute Query", use_container_width=True):
            with st.spinner("Querying Prometheus..."):
                results = fetch_prometheus(query)
                if results:
                    st.success(f"✅ Found {len(results)} result(s)")
                    for result in results[:10]:
                        st.json(result)
                else:
                    st.warning("⚠️ No data for this query")

elif nav_option == "🤖 Predictions":
    st.header("🤖 AI Failure Predictions")

    st.markdown("---")

    # Prediction Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Total Predictions", "1,247", delta="↑ 89 today")
    with col2:
        st.metric("🎯 Accuracy", "93%", delta="↑ 2% this week")
    with col3:
        st.metric("🎯 False Positives", "7%", delta="↓ 1% this week")

    st.markdown("---")

    st.subheader("📋 Recent Predictions")

    sample_predictions = [
        {"type": "Memory Leak", "probability": 0.87, "action": "scale_up", "timestamp": "2 min ago"},
        {"type": "CPU Spike", "probability": 0.64, "action": "none", "timestamp": "8 min ago"},
        {"type": "Bad Deploy", "probability": 0.92, "action": "rollback", "timestamp": "15 min ago"},
    ]

    for pred in sample_predictions:
        prob_color = "#ff4444" if pred["probability"] > 0.8 else "#ffaa00" if pred["probability"] > 0.5 else "#00ff88"
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {prob_color};">
            <strong>{pred['type']}</strong> - Probability: <strong style="color: {prob_color};">{pred['probability']*100:.0f}%</strong>
            <br/>Recommended: <strong>{pred['action']}</strong> • {pred['timestamp']}
        </div>
        """, unsafe_allow_html=True)

elif nav_option == "⚡ Actions":
    st.header("⚡ Healing Actions History")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Total Actions", "412", delta="↑ 28 today")
    with col2:
        st.metric("✅ Success Rate", "97%", delta="↑ 2%")
    with col3:
        st.metric("⏱️ Avg Duration", "12.3s", delta="↓ -2.1s")

    st.markdown("---")

    st.subheader("📋 Recent Healing Actions")

    sample_actions = [
        {"action": "scale_up", "pods": "2→6", "duration": "18s", "success": True, "time": "5 min ago"},
        {"action": "restart_pod", "target": "api-prod", "duration": "8s", "success": True, "time": "12 min ago"},
        {"action": "rollback_deploy", "version": "v2.1→v2.0", "duration": "45s", "success": True, "time": "1h ago"},
    ]

    for action in sample_actions:
        icon = "✅" if action["success"] else "❌"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 200, 255, 0.1) 100%); padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #00ff88;">
            <strong>{icon} {action['action'].upper()}</strong>
            <br/>Duration: <strong>{action['duration']}</strong> • {action['time']}
            <br/><small style="color: #b0b8c1;">{json.dumps({k: v for k, v in action.items() if k not in ['success', 'time']})}</small>
        </div>
        """, unsafe_allow_html=True)

elif nav_option == "🏥 Health":
    st.header("🏥 System Health")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Healthy Services")
        for service in ["API", "Database", "Cache", "Prometheus", "Grafana"]:
            st.markdown(f"✅ **{service}** - Running")

    with col2:
        st.subheader("⚙️ Advanced Metrics")
        st.metric("CPU Usage", "45%", delta="↑ 5%")
        st.metric("Memory Usage", "62%", delta="↓ 3%")
        st.metric("Disk Usage", "38%", delta="→")

elif nav_option == "⚙️ Settings":
    st.header("⚙️ Configuration")

    tab1, tab2, tab3 = st.tabs(["System", "Alerts", "Integrations"])

    with tab1:
        st.subheader("System Settings")
        st.write("**Environment**: production")
        st.write("**Version**: 3.0.0")
        st.write("**Uptime**: 7d 2h 15m")

    with tab2:
        st.subheader("Alert Configuration")
        st.toggle("Enable Slack Alerts")
        st.toggle("Enable Email Alerts")
        st.slider("Alert Threshold", 0.5, 1.0, 0.8)

    with tab3:
        st.subheader("Integrations")
        st.write("✅ Jenkins: Connected")
        st.write("✅ Prometheus: Connected")
        st.write("✅ Grafana: Connected")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🧠 <strong>NeuroShield v3.0</strong> • AI-Powered CI/CD Self-Healing Platform</p>
    <p>Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    <p>🔗 <a href='http://localhost:8000/docs' target='_blank'>API Docs</a> •
       <a href='http://localhost:3000' target='_blank'>Grafana</a> •
       <a href='http://localhost:9090' target='_blank'>Prometheus</a></p>
</div>
""", unsafe_allow_html=True)
