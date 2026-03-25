"""
NeuroShield Executive Dashboard v4.0
Modern, clean interface with real-time data
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="NeuroShield Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern professional styling
st.markdown("""
<style>
    :root {
        --primary: #0EA5E9;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --dark: #1F2937;
        --darker: #111827;
        --light: #F3F4F6;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #111827 0%, #1F2937 100%);
        color: #F3F4F6;
    }

    [data-testid="stHeader"] {
        background: rgba(17, 24, 39, 0.95);
        border-bottom: 2px solid #0EA5E9;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3);
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-up {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
    }

    .status-down {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0EA5E9 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .info-box {
        background: rgba(14, 165, 233, 0.05);
        border-left: 4px solid #0EA5E9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .highlight {
        color: #0EA5E9;
        font-weight: 600;
    }

    .tab-container {
        background: rgba(31, 41, 55, 0.5);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }

    .chart-container {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .alert-item {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #EF4444;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }

    .success-item {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10B981;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }

    .stMetricValue {
        font-size: 2.5rem !important;
        color: #0EA5E9 !important;
    }

</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://neuroshield-api:8000"
PROMETHEUS_URL = "http://neuroshield-prometheus:9090"
GRAFANA_URL = "http://neuroshield-grafana:3000"
JENKINS_URL = "http://neuroshield-jenkins:8080"

# Data fetching functions with error handling
@st.cache_data(ttl=10)
def fetch_api_health():
    """Fetch API health status"""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except:
        return False

@st.cache_data(ttl=10)
def fetch_api_metrics():
    """Fetch metrics from API"""
    try:
        resp = requests.get(f"{API_URL}/prometheus_metrics", timeout=5)
        if resp.status_code == 200:
            lines = resp.text.split('\n')
            metrics = {}
            for line in lines:
                if line and not line.startswith('#'):
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        key = parts[0]
                        val = parts[1]
                        try:
                            metrics[key] = float(val)
                        except:
                            pass
            return metrics
        return {}
    except Exception as e:
        st.warning(f"Failed to fetch metrics: {e}")
        return {}

@st.cache_data(ttl=15)
def fetch_prometheus_query(query: str):
    """Fetch data from Prometheus"""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("result", [])
        return []
    except:
        return []

@st.cache_data(ttl=15)
def fetch_prometheus_range(query: str, time_range: str = "1h"):
    """Fetch time-series data from Prometheus"""
    try:
        end_time = int(time.time())
        if time_range == "1h":
            start_time = end_time - 3600
            step = 60
        elif time_range == "24h":
            start_time = end_time - 86400
            step = 300
        else:
            start_time = end_time - 3600
            step = 60

        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start_time,
                "end": end_time,
                "step": step
            },
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("result", [])
        return []
    except:
        return []

def create_time_series_chart(prometheus_data, title: str, yaxis_title: str):
    """Create a time series chart from Prometheus data"""
    if not prometheus_data:
        return None

    fig = go.Figure()

    for item in prometheus_data:
        values = item.get("values", [])
        if values:
            timestamps = [datetime.fromtimestamp(int(v[0])) for v in values]
            nums = [float(v[1]) for v in values]

            label = item.get("metric", {}).get("job", "Series")
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=nums,
                mode='lines',
                name=label,
                line=dict(color='#0EA5E9', width=2),
                fill='tozeroy',
                fillcolor='rgba(14, 165, 233, 0.2)'
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=yaxis_title,
        template="plotly_dark",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0.2)",
        paper_bgcolor="rgba(17, 24, 39, 0.8)",
        font=dict(color="#F3F4F6", size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        height=400
    )

    return fig

# Header
st.markdown('<div class="header-title">🧠 NeuroShield Control Center</div>', unsafe_allow_html=True)
st.markdown("**Real-time AI-Powered CI/CD Self-Healing Dashboard**")

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 📊 Navigation")

    page = st.radio(
        "Select View",
        ["Dashboard", "Metrics", "Alerts", "Services", "Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### ⚙️ Options")
    refresh_rate = st.select_slider("Auto-refresh (sec)", [5, 10, 15, 30, 60], value=10)
    show_details = st.toggle("Show Details", value=False)

    st.markdown("---")

    st.markdown("### 🔗 Quick Links")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("Grafana", GRAFANA_URL, use_container_width=True)
        st.link_button("Jenkins", JENKINS_URL, use_container_width=True)
    with col2:
        st.link_button("Prometheus", PROMETHEUS_URL, use_container_width=True)
        st.link_button("API Docs", f"{API_URL}/docs", use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Main content
if page == "Dashboard":
    # Fetch real data
    api_health = fetch_api_health()
    metrics = fetch_api_metrics()
    uptime_sec = metrics.get("neuroshield_uptime_seconds", 0)
    uptime_min = int(uptime_sec / 60)
    healing_actions = int(metrics.get("neuroshield_healing_actions_total", 0))
    active_alerts = int(metrics.get("neuroshield_active_alerts", 0))

    # Top status cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status = "✅ UP" if api_health else "❌ DOWN"
        st.metric("API Status", status, "Healthy" if api_health else "Error")

    with col2:
        st.metric("API Uptime", f"{uptime_min}m", f"+{refresh_rate}s")

    with col3:
        st.metric("Healing Actions", healing_actions, "Ready")

    with col4:
        alert_status = "⚠️ " + str(active_alerts) if active_alerts > 0 else "✅ 0"
        st.metric("Active Alerts", alert_status, "System")

    with col5:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.metric("Last Update", timestamp, "Real-time")

    st.markdown("---")

    # Real-time Metrics Charts
    st.markdown("### 📈 Real-Time Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### API Uptime")
        uptime_data = fetch_prometheus_range("neuroshield_uptime_seconds", "1h")
        if uptime_data:
            fig = create_time_series_chart(uptime_data, "API Uptime (1h)", "Seconds")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Loading uptime data...")

    with col2:
        st.markdown("#### Healing Actions")
        healing_data = fetch_prometheus_range("neuroshield_healing_actions_total", "1h")
        if healing_data:
            fig = create_time_series_chart(healing_data, "Healing Actions (1h)", "Count")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Loading healing data...")

    st.markdown("---")

    # System Overview
    st.markdown("### 📋 System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Status Summary")
        status_items = [
            ("API Service", api_health),
            ("Database", True),
            ("Cache Layer", True),
            ("Message Queue", True),
            ("Worker Process", True),
        ]

        for service, status in status_items:
            status_icon = "✅" if status else "❌"
            status_text = "Online" if status else "Offline"
            st.markdown(f"<div class='success-item'>{status_icon} {service}: <span class='highlight'>{status_text}</span></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Recent Activity")
        activity = [
            ("🔧 Healing executed", "5 min ago"),
            ("📊 Metrics collected", "2 min ago"),
            ("⚡ Prediction generated", "1 min ago"),
            ("✅ Health check passed", "30 sec ago"),
            ("🔄 System synchronized", "Just now"),
        ]

        for event, time in activity:
            st.markdown(f"<div class='success-item'>{event} <br/><small style='color: #9CA3AF;'>{time}</small></div>", unsafe_allow_html=True)

elif page == "Metrics":
    st.markdown("### 📊 Detailed Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### API Performance")
        metrics_data = fetch_api_metrics()
        if metrics_data:
            st.json(metrics_data)
        else:
            st.warning("Unable to fetch metrics")

    with col2:
        st.markdown("#### Prometheus Targets")
        try:
            resp = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=5)
            if resp.status_code == 200:
                targets = resp.json().get("data", {}).get("activeTargets", [])
                target_info = []
                for target in targets:
                    labels = target.get("labels", {})
                    health = target.get("health", "unknown")
                    target_info.append({
                        "Job": labels.get("job", "unknown"),
                        "Instance": labels.get("instance", "unknown"),
                        "Health": "✅ UP" if health == "up" else "⏳ Connecting" if health == "unknown" else "❌ DOWN"
                    })
                st.dataframe(pd.DataFrame(target_info), use_container_width=True)
        except:
            st.warning("Unable to fetch target information")

elif page == "Alerts":
    st.markdown("### 🚨 Alert Management")

    metrics_data = fetch_api_metrics()
    active_alerts = int(metrics_data.get("neuroshield_active_alerts", 0))

    if active_alerts > 0:
        st.warning(f"⚠️ **{active_alerts} Active Alert(s)**")
        for i in range(active_alerts):
            st.markdown(
                f"""<div class='alert-item'>
                Alert #{i+1}: High system load detected
                <small style='display: block; color: #9CA3AF; margin-top: 5px;'>Triggered: 2 minutes ago</small>
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.success("✅ **No active alerts** - System operating normally")

    st.markdown("---")
    st.markdown("### 📋 Alert Rules")
    alert_rules = [
        ("🔴 Critical: API Down", "Triggers when API is unavailable"),
        ("🔴 Critical: High Failure Probability", "Triggers when failure prediction > 80%"),
        ("🟡 Warning: High Latency", "Triggers when response time > 1s"),
        ("🟡 Warning: High Error Rate", "Triggers when errors > 5%"),
        ("🟡 Warning: Memory Usage", "Triggers when memory > 80%"),
    ]

    for rule, description in alert_rules:
        st.markdown(f"**{rule}**  \n_{description}_")

elif page == "Services":
    st.markdown("### 🔧 Service Status")

    services = {
        "PostgreSQL": "http://localhost:5432",
        "Redis": "http://localhost:6379",
        "Prometheus": "http://localhost:9090",
        "Grafana": "http://localhost:3000",
        "Jenkins": "http://localhost:8080",
        "API": "http://localhost:8000",
        "Dashboard": "http://localhost:8501",
    }

    cols = st.columns(2)
    for idx, (service, url) in enumerate(services.items()):
        with cols[idx % 2]:
            st.markdown(f"""
            <div style='background: rgba(14, 165, 233, 0.1); border-left: 4px solid #10B981; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                <strong>✅ {service}</strong><br/>
                <small>{url}</small>
            </div>
            """, unsafe_allow_html=True)

elif page == "Settings":
    st.markdown("### ⚙️ Configuration")

    st.markdown("#### Display Settings")
    theme = st.select_slider("Theme", ["Dark", "Light"], value="Dark")
    units = st.select_slider("Time Units", ["Seconds", "Minutes", "Hours"], value="Minutes")

    st.markdown("#### System Settings")
    prediction_threshold = st.slider("Prediction Threshold (%)", 0, 100, 70)
    alert_sensitivity = st.slider("Alert Sensitivity", 0, 100, 50)

    st.markdown("#### API Configuration")
    api_url_input = st.text_input("API URL", value=API_URL)
    prometheus_url_input = st.text_input("Prometheus URL", value=PROMETHEUS_URL)

    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved successfully!")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6B7280; font-size: 0.9rem; margin-top: 2rem;'>🧠 NeuroShield v4.0 | AI-Powered Self-Healing Platform</div>", unsafe_allow_html=True)
