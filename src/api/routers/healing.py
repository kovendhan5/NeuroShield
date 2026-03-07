"""Healing endpoints: /healing/history, /healing/stats, /healing/trigger."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query

from src.api.models import HealingEntry, HealingStats, HealingTriggerRequest, HealingTriggerResponse

router = APIRouter(prefix="/healing", tags=["healing"])
logger = logging.getLogger(__name__)

HEALING_LOG_JSON = Path("data/healing_log.json")
HEALING_LOG_CSV = Path("data/healing_log.csv")
MTTR_LOG_CSV = Path("data/mttr_log.csv")

ALLOWED_ACTIONS = {
    "restart_pod": 0,
    "scale_up": 1,
    "retry_build": 2,
    "rollback_deploy": 3,
    "clear_cache": 4,
    "escalate_to_human": 5,
}


def _load_healing_json() -> list:
    if not HEALING_LOG_JSON.exists():
        return []
    entries = []
    with open(HEALING_LOG_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


@router.get("/history", response_model=List[HealingEntry])
def healing_history(limit: int = Query(default=20, ge=1, le=500)):
    entries = _load_healing_json()
    tail = entries[-limit:]
    results = []
    for e in reversed(tail):
        ctx = e.get("context", {})
        results.append(HealingEntry(
            timestamp=e.get("timestamp", ""),
            action=e.get("action_name", ""),
            reason=ctx.get("escalation_reason", ctx.get("failure_pattern", "")),
            result="success" if e.get("success") else "failed",
            failure_probability=float(ctx.get("failure_prob", 0)),
        ))
    return results


@router.get("/stats", response_model=HealingStats)
def healing_stats():
    entries = _load_healing_json()
    if not entries:
        return HealingStats(
            total_actions=0, action_distribution={},
            success_rate=0.0, avg_mttr_reduction=0.0,
        )

    total = len(entries)
    dist: dict[str, int] = {}
    successes = 0
    for e in entries:
        name = e.get("action_name", "unknown")
        dist[name] = dist.get(name, 0) + 1
        if e.get("success"):
            successes += 1

    # MTTR reduction from mttr_log.csv
    avg_mttr = 0.0
    if MTTR_LOG_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(MTTR_LOG_CSV)
            if "reduction_pct" in df.columns and not df.empty:
                vals = pd.to_numeric(df["reduction_pct"], errors="coerce").dropna()
                if len(vals) > 0:
                    avg_mttr = round(float(vals.mean()), 1)
        except Exception:
            pass

    return HealingStats(
        total_actions=total,
        action_distribution=dist,
        success_rate=round(successes / total, 2) if total else 0.0,
        avg_mttr_reduction=avg_mttr,
    )


@router.post("/trigger", response_model=HealingTriggerResponse)
def trigger_healing(req: HealingTriggerRequest):
    if req.action not in ALLOWED_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{req.action}'. Allowed: {list(ALLOWED_ACTIONS.keys())}",
        )

    action_id = ALLOWED_ACTIONS[req.action]
    try:
        from src.orchestrator.main import execute_healing_action
        context = {
            "build_number": "api",
            "affected_service": "dummy-app",
            "failure_prob": "0.0",
            "failure_pattern": "ManualAPI",
            "escalation_reason": req.reason,
            "prometheus_cpu_usage": "0",
            "prometheus_memory_usage": "0",
            "jenkins_last_build_status": "UNKNOWN",
        }
        execute_healing_action(action_id, context)
    except Exception as exc:
        logger.warning("Healing trigger failed: %s", exc)

    return HealingTriggerResponse(
        triggered=True,
        action=req.action,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
