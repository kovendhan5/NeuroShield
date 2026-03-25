"""NeuroShield AIOps REST API.

FastAPI application that exposes NeuroShield's capabilities to external
tools, scripts, and monitoring systems.

Run:
    python scripts/start_api.py          # http://localhost:8502
    uvicorn src.api.main:app --port 8502 # same, manual
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

load_dotenv()

from src.api.routers import demo, healing, mttr, prediction, report, status, telemetry  # noqa: E402
from src.api.models import AlertManagerWebhookPayload  # noqa: E402
from src.events.webhook_server import get_event_queue  # noqa: E402

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="NeuroShield AIOps API",
    description=(
        "REST API for the NeuroShield self-healing CI/CD platform. "
        "Provides telemetry, prediction, healing actions, and MTTR analytics."
    ),
    version="2.0",
    docs_url="/docs" if os.getenv("NEUROSHIELD_ENV", "development") != "production" else None,
    redoc_url="/redoc" if os.getenv("NEUROSHIELD_ENV", "development") != "production" else None,
)

# CORS — restrict origins; defaults safe for local dev, override for production
_allowed_origins = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:8501,http://localhost:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(status.router)
app.include_router(telemetry.router)
app.include_router(healing.router)
app.include_router(prediction.router)
app.include_router(mttr.router)
app.include_router(demo.router)
app.include_router(report.router)

_app_start_time = time.time()
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_TELEMETRY_CSV = _DATA_DIR / "telemetry.csv"
_ACTIVE_ALERT_JSON = _DATA_DIR / "active_alert.json"
_RECENT_FIX_REDIS_KEY = "neuroshield:telemetry:recent_fix"
_REDIS_CLIENT = None

_last_ws_payload: Dict[str, Any] = {
    "cpu": 0.0,
    "memory": 0.0,
    "health_score": 100.0,
    "active_alerts": 0,
    "recent_fix": None,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}


class ConnectionManager:
    """Thread-safe manager for active websocket clients."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self.active_connections)

        stale: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)

        if stale:
            async with self._lock:
                for stale_connection in stale:
                    if stale_connection in self.active_connections:
                        self.active_connections.remove(stale_connection)


_ws_manager = ConnectionManager()
_ws_broadcast_task: Optional[asyncio.Task] = None


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        number = float(value)
        return default if number != number else number
    except (TypeError, ValueError):
        return default


def _get_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None

    try:
        import redis  # type: ignore

        _REDIS_CLIENT = redis.from_url(redis_url, decode_responses=True)
        _REDIS_CLIENT.ping()
        return _REDIS_CLIENT
    except Exception as exc:
        logging.warning("Telemetry redis queue unavailable: %s", exc)
        _REDIS_CLIENT = None
        return None


def _read_latest_telemetry_values() -> Dict[str, Optional[float]]:
    if not _TELEMETRY_CSV.exists():
        return {"cpu": None, "memory": None}

    try:
        with open(_TELEMETRY_CSV, "r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
            if not rows:
                return {"cpu": None, "memory": None}
            latest = rows[-1]
            return {
                "cpu": _safe_float(latest.get("prometheus_cpu_usage")),
                "memory": _safe_float(latest.get("prometheus_memory_usage")),
            }
    except Exception as exc:
        logging.warning("Failed reading telemetry csv: %s", exc)
        return {"cpu": None, "memory": None}


def _compute_health_score(cpu: float, memory: float, active_alerts: int) -> float:
    penalty = 0.0
    if cpu > 70.0:
        penalty += (cpu - 70.0) * 0.7
    if memory > 75.0:
        penalty += (memory - 75.0) * 0.5
    penalty += float(active_alerts) * 10.0
    score = max(0.0, min(100.0, 100.0 - penalty))
    return round(score, 1)


def _drain_recent_fix_event() -> Optional[Dict[str, Any]]:
    client = _get_redis_client()
    if client is None:
        return None

    latest_event: Optional[Dict[str, Any]] = None
    try:
        while True:
            raw = client.lpop(_RECENT_FIX_REDIS_KEY)
            if raw is None:
                break
            payload = json.loads(raw)
            if isinstance(payload, dict):
                latest_event = payload
    except Exception as exc:
        logging.warning("Failed draining recent_fix queue: %s", exc)

    return latest_event


def _parse_active_alerts_from_prometheus(metrics_text: str) -> Optional[int]:
    for line in metrics_text.splitlines():
        line = line.strip()
        if line.startswith("neuroshield_active_alerts "):
            parts = line.split()
            if len(parts) == 2:
                value = _safe_float(parts[1])
                if value is not None:
                    return int(value)
    return None


async def _build_ws_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(_last_ws_payload)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    telemetry = _read_latest_telemetry_values()
    if telemetry["cpu"] is not None:
        payload["cpu"] = round(float(telemetry["cpu"]), 2)
    if telemetry["memory"] is not None:
        payload["memory"] = round(float(telemetry["memory"]), 2)

    active_alerts: Optional[int] = None
    try:
        metrics_text = await prometheus_metrics()
        active_alerts = _parse_active_alerts_from_prometheus(metrics_text)
    except Exception as exc:
        logging.warning("Unable to refresh prometheus metrics for websocket payload: %s", exc)

    if active_alerts is None:
        try:
            if _ACTIVE_ALERT_JSON.exists():
                with open(_ACTIVE_ALERT_JSON, "r", encoding="utf-8") as active_file:
                    active_doc = json.load(active_file)
                    active_alerts = 1 if active_doc.get("active") else 0
        except Exception as exc:
            logging.warning("Unable to read active alert fallback: %s", exc)

    if active_alerts is not None:
        payload["active_alerts"] = int(active_alerts)

    payload["health_score"] = _compute_health_score(
        cpu=float(payload["cpu"]),
        memory=float(payload["memory"]),
        active_alerts=int(payload["active_alerts"]),
    )

    recent_fix = _drain_recent_fix_event()
    if recent_fix is not None:
        payload["recent_fix"] = recent_fix

    _last_ws_payload.update(payload)
    return payload


async def _telemetry_broadcast_loop() -> None:
    while True:
        try:
            payload = await _build_ws_payload()
            await _ws_manager.broadcast(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.exception("Websocket broadcast loop error: %s", exc)
        await asyncio.sleep(2)


@app.on_event("startup")
async def start_telemetry_broadcaster() -> None:
    global _ws_broadcast_task
    if _ws_broadcast_task is None or _ws_broadcast_task.done():
        _ws_broadcast_task = asyncio.create_task(_telemetry_broadcast_loop())


@app.on_event("shutdown")
async def stop_telemetry_broadcaster() -> None:
    global _ws_broadcast_task
    if _ws_broadcast_task is not None:
        _ws_broadcast_task.cancel()
        with suppress(asyncio.CancelledError):
            await _ws_broadcast_task
        _ws_broadcast_task = None


@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    await _ws_manager.connect(websocket)
    try:
        await websocket.send_json(await _build_ws_payload())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await _ws_manager.disconnect(websocket)
    except Exception:
        await _ws_manager.disconnect(websocket)
        logging.exception("Unexpected websocket error on /ws/telemetry")


@app.post("/alerts")
async def receive_alertmanager_webhook(payload: AlertManagerWebhookPayload):
    """Receive Alertmanager webhooks and forward alerts into orchestrator queue."""
    try:
        event_queue = get_event_queue()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to access event queue: {exc}") from exc

    forwarded = 0
    for alert in payload.alerts:
        labels = alert.labels
        summary = ""
        description = ""
        if alert.annotations is not None:
            summary = alert.annotations.summary or ""
            description = alert.annotations.description or ""

        event_type = f"prometheus_alert_{alert.status.lower()}"
        event_data = {
            "receiver": payload.receiver,
            "group_status": payload.status,
            "alert_status": alert.status,
            "alertname": labels.alertname,
            "severity": labels.severity,
            "instance": labels.instance,
            "job": labels.job,
            "service": labels.service,
            "namespace": labels.namespace,
            "pod": labels.pod,
            "summary": summary,
            "description": description,
            "startsAt": alert.startsAt,
            "endsAt": alert.endsAt,
            "generatorURL": alert.generatorURL,
            "fingerprint": alert.fingerprint,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "raw_common_labels": payload.commonLabels,
            "raw_common_annotations": payload.commonAnnotations,
            "raw_group_labels": payload.groupLabels,
        }
        event_queue.put_event(event_type=event_type, data=event_data, source="alertmanager")
        forwarded += 1

    logging.info(
        "Alertmanager webhook received: receiver=%s status=%s alerts=%d forwarded=%d",
        payload.receiver,
        payload.status,
        len(payload.alerts),
        forwarded,
    )
    return {"received": True, "forwarded": forwarded, "status": payload.status}


@app.get("/prometheus_metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    import json, os

    # Count healing actions from healing_log.json (NDJSON: one JSON object per line)
    healing_count = 0
    action_counts = {}
    try:
        path = os.path.join(os.path.dirname(__file__), "../../data/healing_log.json")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    healing_count += 1
                    action = entry.get("action_name", entry.get("action", "unknown"))
                    action_counts[action] = action_counts.get(action, 0) + 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    # Count open incidents from active_alert.json
    active_alerts = 0
    try:
        path = os.path.join(os.path.dirname(__file__), "../../data/active_alert.json")
        with open(path) as f:
            alert = json.load(f)
            active_alerts = 1 if alert.get("active") else 0
    except Exception:
        pass

    uptime = time.time() - _app_start_time

    lines = [
        "# HELP neuroshield_healing_actions_total Total healing actions taken",
        "# TYPE neuroshield_healing_actions_total counter",
        f"neuroshield_healing_actions_total {healing_count}",
        "",
        "# HELP neuroshield_healing_by_action Healing actions broken down by type",
        "# TYPE neuroshield_healing_by_action counter",
    ]
    for action, count in action_counts.items():
        lines.append(f'neuroshield_healing_by_action{{action="{action}"}} {count}')

    lines += [
        "",
        "# HELP neuroshield_uptime_seconds API uptime in seconds",
        "# TYPE neuroshield_uptime_seconds gauge",
        f"neuroshield_uptime_seconds {uptime:.1f}",
        "",
        "# HELP neuroshield_api_up API health (1=up)",
        "# TYPE neuroshield_api_up gauge",
        "neuroshield_api_up 1",
        "",
        "# HELP neuroshield_active_alerts Current active alerts",
        "# TYPE neuroshield_active_alerts gauge",
        f"neuroshield_active_alerts {active_alerts}",
    ]
    return "\n".join(lines)
