import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
import logging

# ===== Configuration =====
st.set_page_config(
    page_title="NeuroShield Control Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove default Streamlit styling
st.markdown("""
    <style>
    .main { padding: 0; }
    [data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== API Endpoints =====
ORCHESTRATOR_URL = "http://orchestrator:8000"
PROMETHEUS_URL = "http://prometheus:9090"
JENKINS_URL = "http://jenkins:8080"
BACKEND_URL = "http://backend-app:5000"

# ===== Cache Functions =====
@st.cache_data(ttl=30)
def fetch_pipeline_status():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/pipeline-status", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.error(f"Pipeline status error: {e}")
        return None

@st.cache_data(ttl=30)
def fetch_deployments():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/deployments", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.error(f"Deployments error: {e}")
        return None

@st.cache_data(ttl=30)
def fetch_app_health():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/health/detailed", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_prometheus_metrics():
    try:
        query = 'node_cpu_seconds_total{mode="idle"}'
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.error(f"Prometheus error: {e}")
        return None

@st.cache_data(ttl=30)
def fetch_healing_log():
    try:
        resp = requests.get(f"{ORCHESTRATOR_URL}/api/healing-log", timeout=5)
        return resp.json() if resp.status_code == 200 else []
    except Exception as e:
        logger.error(f"Healing log error: {e}")
        return []

# ===== Header =====
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("🧠 NeuroShield Control Center")
    st.markdown("**AI-Powered CI/CD Self-Healing System**")

# ===== Sidebar =====
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select View",
        ["Dashboard", "Pipeline", "Deployments", "Health", "AI Actions", "Prometheus", "Settings"],
        label_visibility="collapsed"
    )

    st.divider()
    st.subheader("System Status")

    # Quick health indicators
    health = fetch_app_health()
    if health:
        cols = st.columns(3)
        cols[0].metric("System", "✅ Healthy" if health.get("status") == "healthy" else "⚠️ Degraded")
        cols[1].metric("DB", "✅" if health.get("components", {}).get("database") == "healthy" else "❌")
        cols[2].metric("Cache", "✅" if health.get("components", {}).get("cache") == "healthy" else "❌")

    st.divider()
    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()

# ===== PAGE: DASHBOARD =====
if page == "Dashboard":
    st.header("System Overview")

    # Top metrics
    pipeline = fetch_pipeline_status()
    deployments = fetch_deployments()
    health = fetch_app_health()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if pipeline:
            st.metric("📊 Total Builds", pipeline.get("total", 0))
        else:
            st.metric("📊 Total Builds", "N/A")

    with col2:
        if pipeline:
            pass_rate = (pipeline.get("passing", 0) / max(pipeline.get("total", 1), 1)) * 100
            st.metric("✅ Pass Rate", f"{pass_rate:.1f}%")
        else:
            st.metric("✅ Pass Rate", "N/A")

    with col3:
        if deployments:
            st.metric("🚀 Active Deployments", deployments.get("active", 0))
        else:
            st.metric("🚀 Active Deployments", "N/A")

    with col4:
        if health:
            success_rate = health.get("metrics", {}).get("success_rate", 0)
            st.metric("💚 Success Rate", f"{success_rate}%")
        else:
            st.metric("💚 Success Rate", "N/A")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent Builds")
        if pipeline and pipeline.get("builds"):
            df_builds = pd.DataFrame(pipeline["builds"])
            if "created_at" in df_builds.columns:
                df_builds['created_at'] = pd.to_datetime(df_builds['created_at'])
                status_counts = df_builds['status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Build Status Distribution",
                    color_discrete_map={"SUCCESS": "#00ff88", "FAILED": "#ff0055"}
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Deployment Timeline")
        if deployments and deployments.get("deployments"):
            df_deploy = pd.DataFrame(deployments["deployments"])
            if "started_at" in df_deploy.columns:
                df_deploy['started_at'] = pd.to_datetime(df_deploy['started_at'])
                status_counts = df_deploy['status'].value_counts()
                fig = px.bar(
                    x=status_counts.index,
                    y=status_counts.values,
                    title="Deployments by Status",
                    color=status_counts.index,
                    color_discrete_map={"SUCCESS": "#00ff88", "FAILED": "#ff0055", "IN_PROGRESS": "#ffaa00"}
                )
                st.plotly_chart(fig, use_container_width=True)

# ===== PAGE: PIPELINE =====
elif page == "Pipeline":
    st.header("Pipeline Status")
    pipeline = fetch_pipeline_status()

    if pipeline:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Builds", pipeline.get("total", 0), delta=None)
        col2.metric("Successful", pipeline.get("passing", 0))
        col3.metric("Failed", pipeline.get("failing", 0))

        st.subheader("Recent Builds")
        if pipeline.get("builds"):
            df = pd.DataFrame(pipeline["builds"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent builds found")
    else:
        st.error("Could not fetch pipeline status")

# ===== PAGE: DEPLOYMENTS =====
elif page == "Deployments":
    st.header("Deployment Status")
    deployments = fetch_deployments()

    if deployments:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", deployments.get("total", 0))
        col2.metric("Successful", deployments.get("successful", 0))
        col3.metric("Failed", deployments.get("failed", 0))
        col4.metric("Active", deployments.get("active", 0))

        st.subheader("Deployment History")
        if deployments.get("deployments"):
            df = pd.DataFrame(deployments["deployments"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No deployments found")
    else:
        st.error("Could not fetch deployment status")

# ===== PAGE: HEALTH =====
elif page == "Health":
    st.header("System Health")
    health = fetch_app_health()

    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("System Status", health.get("status", "unknown").upper())
        col2.metric("Database", health.get("components", {}).get("database", "unknown"))
        col3.metric("Cache (Redis)", health.get("components", {}).get("cache", "unknown"))

        st.divider()

        metrics = health.get("metrics", {})
        col1, col2 = st.columns(2)
        col1.metric("Builds (Last Hour)", metrics.get("builds_last_hour", 0))
        col2.metric("Success Rate (Last Hour)", f"{metrics.get('success_rate', 0)}%")
    else:
        st.error("Could not fetch health information")

# ===== PAGE: AI ACTIONS =====
elif page == "AI Actions":
    st.header("🤖 NeuroShield AI Actions")

    healing_log = fetch_healing_log()
    if healing_log:
        st.subheader("Recent Healing Actions")
        df_healing = pd.DataFrame(healing_log)
        st.dataframe(df_healing, use_container_width=True)
    else:
        st.info("No healing actions recorded yet")

# ===== PAGE: PROMETHEUS =====
elif page == "Prometheus":
    st.header("Prometheus Metrics")
    st.markdown(f"**Endpoint:** {PROMETHEUS_URL}")

    st.subheader("Query Expression")
    query = st.text_input("Enter PromQL query", value="up")

    if st.button("Execute Query"):
        try:
            resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query},
                timeout=10
            )
            if resp.status_code == 200:
                result = resp.json()
                st.json(result)
            else:
                st.error(f"Query failed: {resp.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

# ===== PAGE: SETTINGS =====
elif page == "Settings":
    st.header("Settings & Configuration")

    st.subheader("Service Endpoints")
    st.code(f"""
ORCHESTRATOR_URL = {ORCHESTRATOR_URL}
PROMETHEUS_URL = {PROMETHEUS_URL}
JENKINS_URL = {JENKINS_URL}
BACKEND_URL = {BACKEND_URL}
""")

    st.subheader("System Information")
    st.write(f"**Current Time:** {datetime.now().isoformat()}")
    st.write(f"**Page Refresh Interval:** 30 seconds (cached)")

    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# ===== Footer =====
st.divider()
st.markdown("""
<center>

**NeuroShield v2.1.0** | AI-Powered CI/CD Self-Healing System
[Documentation](https://github.com/) | [GitHub](https://github.com/)

</center>
""", unsafe_allow_html=True)
