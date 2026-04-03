"""Status endpoints: /, /health, /metrics."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests as http_requests
from fastapi import APIRouter

from src.api.models import (
    APIInfo,
    HealthResponse,
    MetricsResponse,
    ModelStatus,
    ServiceStatus,
)

router = APIRouter()

JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
DUMMY_APP_URL = os.getenv("DUMMY_APP_URL", "http://localhost:5000")
DUMMY_APP_REQUIRED = os.getenv("DUMMY_APP_REQUIRED", "false").lower() == "true"
TELEMETRY_CSV = Path("data/telemetry.csv")


def _service_status(url: str) -> str:
    try:
        r = http_requests.get(url, timeout=3)
        return "ONLINE" if r.status_code < 500 else "DEGRADED"
    except Exception:
        return "OFFLINE"


@router.get("/", response_model=APIInfo)
def root():
    return APIInfo()


@router.get("/health", response_model=HealthResponse)
def health():
    services = {
        "jenkins": ServiceStatus(status=_service_status(JENKINS_URL), url=JENKINS_URL),
        "prometheus": ServiceStatus(status=_service_status(f"{PROMETHEUS_URL}/-/healthy"), url=PROMETHEUS_URL),
    }
    dummy_status = _service_status(f"{DUMMY_APP_URL}/health")
    services["dummy_app"] = ServiceStatus(status=dummy_status, url=DUMMY_APP_URL)

    models = ModelStatus(
        failure_predictor="loaded" if Path("models/failure_predictor.pth").exists() else "missing",
        ppo_policy="loaded" if Path("models/ppo_policy.zip").exists() else "missing",
        pca_encoder="loaded" if Path("models/log_pca.joblib").exists() else "missing",
    )

    required_services = {
        name: svc
        for name, svc in services.items()
        if name != "dummy_app" or DUMMY_APP_REQUIRED
    }
    all_online = all(s.status == "ONLINE" for s in required_services.values())
    all_models = all(v == "loaded" for v in models.model_dump().values())
    overall = "HEALTHY" if (all_online and all_models) else "DEGRADED"

    return HealthResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
        models=models,
        overall=overall,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    if not TELEMETRY_CSV.exists():
        return MetricsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_usage=0.0, memory_usage=0.0, failure_probability=0.0,
            build_status="UNKNOWN", pod_count=0, total_telemetry_rows=0,
        )

    df = pd.read_csv(TELEMETRY_CSV, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    if df.empty:
        return MetricsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_usage=0.0, memory_usage=0.0, failure_probability=0.0,
            build_status="UNKNOWN", pod_count=0, total_telemetry_rows=0,
        )

    row = df.iloc[-1]

    def sf(val, default=0.0):
        try:
            v = float(val)
            return default if v != v else v
        except (TypeError, ValueError):
            return default

    return MetricsResponse(
        timestamp=str(row.get("timestamp", "")),
        cpu_usage=sf(row.get("prometheus_cpu_usage")),
        memory_usage=sf(row.get("prometheus_memory_usage")),
        failure_probability=sf(row.get("failure_probability")),
        build_status=str(row.get("jenkins_last_build_status", "UNKNOWN") or "UNKNOWN"),
        pod_count=sf(row.get("prometheus_pod_count")),
        total_telemetry_rows=len(df),
    )
