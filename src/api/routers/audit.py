"""Audit log endpoints for NeuroShield.

Provides:
- REST API to query audit logs
- WebSocket streaming for real-time audit log display
- Prometheus metrics for audit events
"""

import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audit", tags=["audit"])

# In-memory audit log buffer for real-time streaming
_audit_buffer: deque = deque(maxlen=1000)
_audit_ws_clients: List[WebSocket] = []
_audit_lock = asyncio.Lock()

# Audit metrics for Prometheus
_audit_metrics = {
    "total_events": 0,
    "by_category": {},
    "by_result": {},
    "by_actor": {},
}


class AuditLogEntry(BaseModel):
    """Audit log entry model."""
    timestamp: str
    category: str
    action: str
    actor: str
    resource: str
    result: str
    details: Dict[str, Any]
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    ip_address: Optional[str] = None


class AuditLogResponse(BaseModel):
    """Response model for audit log queries."""
    logs: List[AuditLogEntry]
    total: int
    page: int
    page_size: int
    has_more: bool


def _get_audit_log_path() -> Path:
    """Get path to audit log file."""
    data_dir = Path(__file__).resolve().parents[3] / "data" / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "audit.jsonl"


def _load_audit_logs_from_file(
    limit: int = 100,
    offset: int = 0,
    category: Optional[str] = None,
    actor: Optional[str] = None,
    result: Optional[str] = None,
    since: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], int]:
    """Load audit logs from file with filtering."""
    log_path = _get_audit_log_path()

    if not log_path.exists():
        return [], 0

    logs = []
    total = 0

    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)

                    # Apply filters
                    if category and entry.get("category") != category:
                        continue
                    if actor and entry.get("actor") != actor:
                        continue
                    if result and entry.get("result") != result:
                        continue
                    if since:
                        entry_time = entry.get("timestamp", "")
                        if entry_time < since:
                            continue

                    total += 1
                    logs.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Error reading audit log file: {e}")
        return [], 0

    # Sort by timestamp descending (newest first)
    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    # Apply pagination
    paginated = logs[offset:offset + limit]

    return paginated, total


async def push_audit_event(event: Dict[str, Any]) -> None:
    """Push audit event to buffer and broadcast to WebSocket clients."""
    global _audit_metrics

    # Add to buffer
    async with _audit_lock:
        _audit_buffer.append(event)

        # Update metrics
        _audit_metrics["total_events"] += 1

        category = event.get("category", "UNKNOWN")
        _audit_metrics["by_category"][category] = _audit_metrics["by_category"].get(category, 0) + 1

        result = event.get("result", "UNKNOWN")
        _audit_metrics["by_result"][result] = _audit_metrics["by_result"].get(result, 0) + 1

        actor = event.get("actor", "unknown")
        _audit_metrics["by_actor"][actor] = _audit_metrics["by_actor"].get(actor, 0) + 1

    # Persist to file
    try:
        log_path = _get_audit_log_path()
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Error writing audit log: {e}")

    # Broadcast to WebSocket clients
    await broadcast_audit_event(event)


async def broadcast_audit_event(event: Dict[str, Any]) -> None:
    """Broadcast audit event to all connected WebSocket clients."""
    stale_clients = []

    for ws in _audit_ws_clients:
        try:
            await ws.send_json({
                "type": "audit_event",
                "data": event,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception:
            stale_clients.append(ws)

    # Remove stale clients
    for ws in stale_clients:
        if ws in _audit_ws_clients:
            _audit_ws_clients.remove(ws)


@router.get("/logs", response_model=AuditLogResponse)
async def get_audit_logs(
    limit: int = Query(50, ge=1, le=500, description="Number of logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    category: Optional[str] = Query(None, description="Filter by category"),
    actor: Optional[str] = Query(None, description="Filter by actor"),
    result: Optional[str] = Query(None, description="Filter by result"),
    since: Optional[str] = Query(None, description="Filter by timestamp (ISO 8601)"),
):
    """Get audit logs with optional filtering and pagination."""
    logs, total = _load_audit_logs_from_file(
        limit=limit,
        offset=offset,
        category=category,
        actor=actor,
        result=result,
        since=since,
    )

    return AuditLogResponse(
        logs=[AuditLogEntry(**log) for log in logs],
        total=total,
        page=offset // limit + 1,
        page_size=limit,
        has_more=offset + limit < total,
    )


@router.get("/logs/recent")
async def get_recent_audit_logs(
    limit: int = Query(20, ge=1, le=100, description="Number of recent logs"),
):
    """Get most recent audit logs from in-memory buffer (fastest)."""
    async with _audit_lock:
        recent = list(_audit_buffer)[-limit:]

    return {
        "logs": recent[::-1],  # Newest first
        "count": len(recent),
        "buffer_size": len(_audit_buffer),
    }


@router.get("/stats")
async def get_audit_stats():
    """Get audit log statistics."""
    return {
        "total_events": _audit_metrics["total_events"],
        "by_category": _audit_metrics["by_category"],
        "by_result": _audit_metrics["by_result"],
        "top_actors": dict(
            sorted(
                _audit_metrics["by_actor"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        ),
        "buffer_size": len(_audit_buffer),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/categories")
async def get_audit_categories():
    """Get available audit categories."""
    return {
        "categories": [
            {"name": "USER_ACTION", "description": "User-initiated actions via API or CLI"},
            {"name": "SYSTEM_EVENT", "description": "System events like restarts and config changes"},
            {"name": "CONFIG_CHANGE", "description": "Configuration modifications"},
            {"name": "SECURITY_EVENT", "description": "Security-related events"},
            {"name": "HEALING_ACTION", "description": "Auto-healing actions taken by orchestrator"},
            {"name": "DATA_ACCESS", "description": "Database and data access events"},
        ]
    }


@router.websocket("/ws")
async def audit_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time audit log streaming."""
    await websocket.accept()
    _audit_ws_clients.append(websocket)

    logger.info(f"Audit WebSocket client connected. Total clients: {len(_audit_ws_clients)}")

    try:
        # Send recent logs on connect
        async with _audit_lock:
            recent = list(_audit_buffer)[-20:]

        await websocket.send_json({
            "type": "initial",
            "data": recent[::-1],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30)

                # Handle ping/pong
                if message == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Audit WebSocket error: {e}")
    finally:
        if websocket in _audit_ws_clients:
            _audit_ws_clients.remove(websocket)
        logger.info(f"Audit WebSocket client disconnected. Total clients: {len(_audit_ws_clients)}")


def get_audit_prometheus_metrics() -> str:
    """Get Prometheus metrics for audit logs."""
    lines = [
        "# HELP neuroshield_audit_events_total Total audit events logged",
        "# TYPE neuroshield_audit_events_total counter",
        f"neuroshield_audit_events_total {_audit_metrics['total_events']}",
        "",
    ]

    # Events by category
    lines.append("# HELP neuroshield_audit_by_category Audit events by category")
    lines.append("# TYPE neuroshield_audit_by_category counter")
    for category, count in _audit_metrics["by_category"].items():
        lines.append(f'neuroshield_audit_by_category{{category="{category}"}} {count}')
    for baseline_category in (
        "USER_ACTION",
        "HEALING_ACTION",
        "SECURITY_EVENT",
        "SYSTEM_EVENT",
        "CONFIG_CHANGE",
        "DATA_ACCESS",
    ):
        if baseline_category not in _audit_metrics["by_category"]:
            lines.append(f'neuroshield_audit_by_category{{category="{baseline_category}"}} 0')
    lines.append("")

    # Events by result
    lines.append("# HELP neuroshield_audit_by_result Audit events by result")
    lines.append("# TYPE neuroshield_audit_by_result counter")
    for result, count in _audit_metrics["by_result"].items():
        lines.append(f'neuroshield_audit_by_result{{result="{result}"}} {count}')
    for baseline_result in ("SUCCESS", "FAILURE", "DENIED"):
        if baseline_result not in _audit_metrics["by_result"]:
            lines.append(f'neuroshield_audit_by_result{{result="{baseline_result}"}} 0')
    lines.append("")

    # WebSocket clients
    lines.append("# HELP neuroshield_audit_ws_clients Active audit WebSocket clients")
    lines.append("# TYPE neuroshield_audit_ws_clients gauge")
    lines.append(f"neuroshield_audit_ws_clients {len(_audit_ws_clients)}")

    return "\n".join(lines)
