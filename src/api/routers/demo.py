"""Demo/injection endpoints: /demo/*."""

from __future__ import annotations

import logging
import os

import requests as http_requests
from fastapi import APIRouter

from src.api.models import DemoTriggerResponse

router = APIRouter(prefix="/demo", tags=["demo"])
logger = logging.getLogger(__name__)

JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080")
JENKINS_USER = os.getenv("JENKINS_USERNAME", "admin")
JENKINS_TOKEN = os.getenv("JENKINS_TOKEN", "")
JENKINS_JOB = os.getenv("JENKINS_JOB", "build-pipeline")
APP_URL = os.getenv("DUMMY_APP_URL", "http://localhost:5000")


def _get_jenkins_crumb() -> dict:
    """Fetch Jenkins CSRF crumb."""
    try:
        r = http_requests.get(
            f"{JENKINS_URL}/crumbIssuer/api/json",
            auth=(JENKINS_USER, JENKINS_TOKEN),
            timeout=5,
        )
        if r.ok:
            data = r.json()
            return {data["crumbRequestField"]: data["crumb"]}
    except Exception:
        pass
    return {}


@router.post("/trigger-build-failure", response_model=DemoTriggerResponse)
def trigger_build_failure():
    """Trigger a Jenkins build (randomly fails ~35% of the time)."""
    try:
        crumb = _get_jenkins_crumb()
        r = http_requests.post(
            f"{JENKINS_URL}/job/{JENKINS_JOB}/build",
            auth=(JENKINS_USER, JENKINS_TOKEN),
            headers=crumb,
            timeout=10,
        )
        build_url = f"{JENKINS_URL}/job/{JENKINS_JOB}/"
        return DemoTriggerResponse(triggered=True, build_url=build_url, detail="Build triggered")
    except Exception as exc:
        logger.warning("Build trigger failed: %s", exc)
        return DemoTriggerResponse(triggered=False, detail=str(exc))


@router.post("/crash-pod", response_model=DemoTriggerResponse)
def crash_pod():
    """Send POST /crash to the dummy app."""
    try:
        http_requests.post(f"{APP_URL}/crash", timeout=5)
        return DemoTriggerResponse(triggered=True, detail="Crash signal sent")
    except Exception as exc:
        logger.warning("Pod crash failed: %s", exc)
        return DemoTriggerResponse(triggered=False, detail=str(exc))


@router.post("/stress-memory", response_model=DemoTriggerResponse)
def stress_memory():
    """Send GET /stress to the dummy app to allocate ~200 MB for 30s."""
    try:
        # Get current memory before stress
        mem_before = None
        try:
            r = http_requests.get(f"{APP_URL}/health", timeout=3)
            if r.ok:
                mem_before = r.json().get("memory_mb")
        except Exception:
            pass

        http_requests.get(f"{APP_URL}/stress", timeout=10)
        return DemoTriggerResponse(
            triggered=True,
            detail="Memory stress triggered (~200 MB for 30s)",
            memory_before_mb=mem_before,
        )
    except Exception as exc:
        logger.warning("Memory stress failed: %s", exc)
        return DemoTriggerResponse(triggered=False, detail=str(exc))
