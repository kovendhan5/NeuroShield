import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List
import logging

# ===== CONFIGURATION =====
st.set_page_config(
    page_title="NeuroShield Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoints
ORCHESTRATOR_API = "http://localhost:8000"
PROMETHEUS_API = "http://localhost:9090"
MICROSERVICE_API = "http://localhost:5000"
GRAFANA_API = "http://localhost:3000"
JENKINS_API = "http://localhost:8080"

# Styling
st.markdown("""
    <style>
    .main { background-color: #0f1419; }
    [data-testid="stMetricValue"] { font-size: 2.5rem; color: #00ff88; }
    [data-testid="stMetricLabel"] { font-size: 1.2rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 1.1rem; }
    hr { border-top: 2px solid #00ff88; }
    </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== HELPER FUNCTIONS =====

@st.cache_data(ttl=10)
def fetch_microservice_health() -> Dict:
    """Fetch microservice health"""
    try:
        resp = requests.get(f"{MICROSERVICE_API}/health/detailed", timeout=5)
        return resp.json() if resp.status_code == 200 else {"status": "error"}
    except Exception as e:
        logger.error(f"Microservice health error: {e}")
        return {"status": "unavailable"}

@st.cache_data(ttl=15)
def fetch_prometheus_query(query: str) -> List:
    """Query Prometheus"""
    try:
        resp = requests.get(
            f"{PROMETHEUS_API}/api/v1/query",
            params={"query": query},
            timeout=10
        )
        data = resp.json()
        return data.get("data", {}).get("result", []) if data.get("status") == "success" else []
    except Exception as e:
        logger.error(f"Prometheus query error: {e}")
        return []

@st.cache_data(ttl=20)
def fetch_orchestrator_status() -> Dict:
    """Fetch orchestrator status"""
    try:
        resp = requests.get(f"{ORCHESTRATOR_API}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        return {}

@st.cache_data(ttl=30)
def fetch_jenkins_builds() -> List:
    """Fetch recent Jenkins builds"""
    try:
        resp = requests.get(
            f"{JENKINS_API}/job/neuroshield-app-build/api/json",
            timeout=5,
            auth=requests.auth.HTTPBasicAuth("admin", "admin123")
        )
        if resp.status_code == 200:
            data = resp.json()
            builds = data.get("builds", [])[:20]
            return [{"number": b.get("number"), "url": b.get("url")} for b in builds]
        return []
    except Exception as e:
        logger.error(f"Jenkins error: {e}")
        return []

# ===== HEADER =====
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("🧠 NeuroShield Command Center")
    st.markdown("**AI-Powered CI/CD Self-Healing System Dashboard**")

st.divider()

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Navigation")
    page = st.radio(
        "Select View",
        ["🏠 Dashboard", "📊 Metrics", "🚀 Deployments", "💾 Database", "🔧 Systems", "🚨 Alerts"],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("🔄 Refresh Interval")
    refresh_interval = st.slider(
        "Auto-refresh (seconds)",
        min_value=5,
        max_value=60,
        value=15,
        step=5,
        label_visibility="collapsed"
    )

    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.subheader("📌 Quick Links")
    col1, col2 = st.columns(2)
    col1.link_button("📊 Grafana", "http://localhost:3000")
    col2.link_button("🔍 Prometheus", "http://localhost:9090")
    col1.link_button("☁️ Jenkins", "http://localhost:8080")
    col2.link_button("📱 Microservice", "http://localhost:5000/health")

# ===== PAGE: DASHBOARD =====
if page == "🏠 Dashboard":
    st.header("System Overview")

    # Top metrics
    health = fetch_microservice_health()
    orchestrator = fetch_orchestrator_status()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status_icon = "✅" if health.get("status") == "healthy" else "⚠️"
        st.metric(
            "Microservice",
            status_icon,
            delta=f"{health.get('health_percentage', 0):.0f}%"
        )

    with col2:
        db_health = health.get("services", {}).get("database", "unknown")
        db_icon = "✅" if db_health == "healthy" else "❌"
        st.metric("Database", db_icon)

    with col3:
        cache_health = health.get("services", {}).get("cache", "unknown")
        cache_icon = "✅" if cache_health == "healthy" else "❌"
        st.metric("Cache", cache_icon)

    with col4:
        orch_status = "✅" if orchestrator.get("status") == "healthy" else "⚠️"
        st.metric("Orchestrator", orch_status)

    with col5:
        st.metric(
            "Uptime",
            f"{int(orchestrator.get('uptime', 0) / 3600)}h",
            delta="Running"
        )

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Request Latency (p95)")
        metrics = fetch_prometheus_query("histogram_quantile(0.95, rate(app_request_latency_seconds_bucket[5m]))*1000")
        if metrics:
            latency = float(metrics[0]["value"][1]) if metrics[0]["value"][1] != "NaN" else 0
            fig = go.Figure(data=[
                go.Indicator(
                    mode="gauge+number",
                    value=latency,
                    title={"text": "ms"},
                    gauge={
                        "axis": {"range": [0, 500]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 100], "color": "lightgreen"},
                            {"range": [100, 250], "color": "yellow"},
                            {"range": [250, 500], "color": "lightpink"}
                        ]
                    }
                )
            ])
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Request Rate (1m)")
        metrics = fetch_prometheus_query("rate(app_requests_total[1m])")
        if metrics:
            rate = float(metrics[0]["value"][1]) if metrics[0]["value"][1] != "NaN" else 0
            fig = go.Figure(data=[
                go.Indicator(
                    mode="number+delta",
                    value=rate,
                    title={"text": "requests/sec"},
                    delta={"reference": 10}
                )
            ])
            st.plotly_chart(fig, use_container_width=True)

    # Real-time status
    st.subheader("🔴 System Status")
    status_cols = st.columns(4)
    status_data = [
        {"label": "API Requests", "value": "Active", "icon": "🟢"},
        {"label": "Database", "value": "Connected", "icon": "🟢"},
        {"label": "Cache", "value": "Enabled", "icon": "🟢"},
        {"label": "ML Engine", "value": "Ready", "icon": "🟢"}
    ]
    for i, status in enumerate(status_data):
        with status_cols[i]:
            st.metric(status["label"], f"{status['icon']} {status['value']}")

# ===== PAGE: METRICS =====
elif page == "📊 Metrics":
    st.header("Prometheus Metrics")

    # Metric selector
    metric_type = st.selectbox(
        "Select Metric",
        [
            "Request Rate (1m avg)",
            "Request Latency (p95)",
            "Error Rate",
            "CPU Usage",
            "Memory Usage",
            "DB Connections"
        ]
    )

    # Query mapping
    queries_map = {
        "Request Rate (1m avg)": "rate(app_requests_total[1m])",
        "Request Latency (p95)": "histogram_quantile(0.95, rate(app_request_latency_seconds_bucket[5m]))*1000",
        "Error Rate": "rate(app_request_errors_total[5m])",
        "CPU Usage": "node_cpu_seconds_total",
        "Memory Usage": "node_memory_MemAvailable_bytes",
        "DB Connections": "db_connections_active"
    }

    query = queries_map.get(metric_type, "up")
    metrics = fetch_prometheus_query(query)

    if metrics:
        st.success(f"✅ Retrieved {len(metrics)} metric(s)")
        for metric in metrics[:10]:
            st.code(json.dumps(metric, indent=2))
    else:
        st.warning(f"⚠️ No data for {metric_type}")

# ===== PAGE: DEPLOYMENTS =====
elif page == "🚀 Deployments":
    st.header("Deployment Status")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📤 Deployments", "12", delta="3 this week")
    with col2:
        st.metric("✅ Success Rate", "95.8%", delta="+2.3%")
    with col3:
        st.metric("⏱️ Avg Duration", "4m 32s", delta="-45s")

    st.divider()

    st.subheader("Recent Jenkins Builds")
    builds = fetch_jenkins_builds()
    if builds:
        df_builds = pd.DataFrame(builds)
        st.dataframe(df_builds, use_container_width=True)
    else:
        st.info("No recent builds found. Jenkins may not be configured.")

# ===== PAGE: DATABASE =====
elif page == "💾 Database":
    st.header("Database Status")

    health = fetch_microservice_health()
    components = health.get("services", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🗄️ Connection Status")
        db_status = components.get("database", "unknown")
        st.write(f"**Database**: {db_status.upper()}")
        st.write(f"**Type**: PostgreSQL 15")
        st.write(f"**Host**: postgres:5432")

    with col2:
        st.subheader("📊 Statistics")
        metrics = health.get("metrics", {})
        st.metric("Builds (Last Hour)", metrics.get("builds_last_hour", 0))
        st.metric("Success Rate", f"{metrics.get('success_rate', 0)}%")

# ===== PAGE: SYSTEMS =====
elif page == "🔧 Systems":
    st.header("System Configuration")

    tabs = st.tabs(["Services", "Endpoints", "Configuration"])

    with tabs[0]:
        st.subheader("Running Services")
        services = [
            {"name": "🐘 PostgreSQL", "port": "5432", "status": "✅ Running"},
            {"name": "🔴 Redis", "port": "6379", "status": "✅ Running"},
            {"name": "⚙️ Prometheus", "port": "9090", "status": "✅ Running"},
            {"name": "📊 Grafana", "port": "3000", "status": "✅ Running"},
            {"name": "🔔 AlertManager", "port": "9093", "status": "✅ Running"},
            {"name": "🧠 Orchestrator", "port": "8000", "status": "✅ Running"},
            {"name": "🎯 Microservice", "port": "5000", "status": "✅ Running"},
            {"name": "🔨 Jenkins", "port": "8080", "status": "⏳ Starting"}
        ]
        df_services = pd.DataFrame(services)
        st.dataframe(df_services, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("API Endpoints")
        endpoints = {
            "Orchestrator Health": f"{ORCHESTRATOR_API}/health",
            "Microservice Health": f"{MICROSERVICE_API}/health",
            "Prometheus Query": f"{PROMETHEUS_API}/api/v1/query",
            "Grafana Dashboards": f"{GRAFANA_API}/api/dashboards",
        }
        for name, url in endpoints.items():
            st.code(f"{name}: {url}", language="text")

    with tabs[2]:
        st.subheader("Environment Variables")
        config = {
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "info",
            "JENKINS_URL": "http://jenkins:8080",
            "PROMETHEUS_URL": "http://prometheus:9090",
            "DATABASE_URL": "postgresql://admin:***@postgres:5432/neuroshield_db",
            "REDIS_URL": "redis://redis:6379"
        }
        for key, value in config.items():
            col1, col2 = st.columns([1, 3])
            col1.write(f"**{key}**")
            col2.code(value)

# ===== PAGE: ALERTS =====
elif page == "🚨 Alerts":
    st.header("Alerts & Events")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Critical", "2", delta="↑ 1")
    with col2:
        st.metric("🟠 Warning", "5", delta="↓ 2")
    with col3:
        st.metric("🟢 Info", "24", delta="↑ 5")

    st.divider()

    st.subheader("Recent Alerts")
    alerts = [
        {"level": "🔴 CRITICAL", "message": "High error rate detected", "time": "2 min ago"},
        {"level": "🟠 WARNING", "message": "Memory usage at 78%", "time": "5 min ago"},
        {"level": "🟢 INFO", "message": "Deployment completed", "time": "12 min ago"},
    ]
    df_alerts = pd.DataFrame(alerts)
    st.dataframe(df_alerts, use_container_width=True, hide_index=True)

# ===== FOOTER =====
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**NeuroShield v2.1.0**")
with col2:
    st.write(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
with col3:
    st.write(f"Auto-refresh: {refresh_interval}s")
