"""Advanced health check endpoints for NeuroShield.

Provides:
- Liveness: Is service running?
- Readiness: Is service ready for traffic?
- Detailed: Full dependency health
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass, asdict
import asyncio

import psutil
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@dataclass
class HealthStatus:
    """Health status of a dependency."""
    status: str  # ok, degraded, critical
    latency_ms: float
    details: Dict[str, Any]


async def check_database() -> HealthStatus:
    """Check PostgreSQL database connectivity."""
    try:
        from src.database import Session

        start = asyncio.get_event_loop().time()
        session = Session()
        session.execute("SELECT 1")
        session.close()
        latency = (asyncio.get_event_loop().time() - start) * 1000

        return HealthStatus(
            status="ok",
            latency_ms=latency,
            details={"type": "postgresql"}
        )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return HealthStatus(
            status="critical",
            latency_ms=-1,
            details={"error": str(e)}
        )


async def check_redis() -> HealthStatus:
    """Check Redis cache connectivity."""
    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)

        start = asyncio.get_event_loop().time()
        r.ping()
        latency = (asyncio.get_event_loop().time() - start) * 1000

        return HealthStatus(
            status="ok",
            latency_ms=latency,
            details={"type": "redis"}
        )
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return HealthStatus(
            status="degraded",  # Cache failure is non-critical
            latency_ms=-1,
            details={"error": str(e), "note": "Cache unavailable - system will use fallbacks"}
        )


async def check_jenkins() -> HealthStatus:
    """Check Jenkins API connectivity."""
    try:
        from src.integrations.jenkins import JenkinsClient

        jenkins = JenkinsClient()
        start = asyncio.get_event_loop().time()
        jenkins.get_queue_length()
        latency = (asyncio.get_event_loop().time() - start) * 1000

        return HealthStatus(
            status="ok",
            latency_ms=latency,
            details={"type": "jenkins", "url": jenkins.url}
        )
    except Exception as e:
        logger.error(f"Jenkins health check failed: {e}")
        return HealthStatus(
            status="critical",
            latency_ms=-1,
            details={"error": str(e)}
        )


async def check_prometheus() -> HealthStatus:
    """Check Prometheus connectivity."""
    try:
        from src.integrations.prometheus import PrometheusClient

        prom = PrometheusClient()
        start = asyncio.get_event_loop().time()
        prom.query("up")
        latency = (asyncio.get_event_loop().time() - start) * 1000

        return HealthStatus(
            status="ok",
            latency_ms=latency,
            details={"type": "prometheus", "url": prom.url}
        )
    except Exception as e:
        logger.error(f"Prometheus health check failed: {e}")
        return HealthStatus(
            status="degraded",  # Prometheus failure is non-critical for immediate healing
            latency_ms=-1,
            details={"error": str(e), "note": "Monitoring unavailable - using cached metrics"}
        )


async def check_system_resources() -> HealthStatus:
    """Check system resource usage."""
    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        status = "ok"
        if cpu_pct > 90 or memory.percent > 90 or disk.percent > 95:
            status = "critical"
        elif cpu_pct > 75 or memory.percent > 75 or disk.percent > 85:
            status = "degraded"

        return HealthStatus(
            status=status,
            latency_ms=0,
            details={
                "cpu_percent": cpu_pct,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            }
        )
    except Exception as e:
        logger.error(f"System resource check failed: {e}")
        return HealthStatus(
            status="degraded",
            latency_ms=-1,
            details={"error": str(e)}
        )


@router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe: Is the service running?"""
    return {"alive": True}


@router.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe: Is the service ready for traffic?

    Returns 200 if all critical services are healthy.
    Returns 503 if any critical service is DOWN.

    Critical: database, jenkins
    Non-critical: redis, prometheus (graceful degradation available)
    """
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "jenkins": await check_jenkins(),
        "prometheus": await check_prometheus(),
        "system": await check_system_resources(),
    }

    # Convert to dict
    checks_dict = {name: asdict(check) for name, check in checks.items()}

    # Determine status
    critical_healthy = all(
        checks[name].status in ["ok", "degraded"]
        for name in ["database", "jenkins"]
    )

    if not critical_healthy:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "checks": checks_dict}
        )

    return JSONResponse(
        status_code=200,
        content={"ready": True, "checks": checks_dict}
    )


@router.get("/detailed")
async def detailed_health():
    """Detailed health report with all metrics.

    Returns comprehensive health information for monitoring dashboards.
    """
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "jenkins": await check_jenkins(),
        "prometheus": await check_prometheus(),
        "system": await check_system_resources(),
    }

    checks_dict = {name: asdict(check) for name, check in checks.items()}

    overall_status = "ok"
    if any(c["status"] == "critical" for c in checks_dict.values()):
        overall_status = "critical"
    elif any(c["status"] == "degraded" for c in checks_dict.values()):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "checks": checks_dict,
        "timestamp": datetime.utcnow().isoformat()
    }


# For backward compatibility
@router.get("/")
async def health():
    """Default health check (Kubernetes uses /health/ready)."""
    ready = await readiness_probe()
    return ready


import os
from datetime import datetime

