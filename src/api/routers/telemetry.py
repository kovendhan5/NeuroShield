"""Telemetry endpoints: /telemetry, /telemetry/summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Query

from src.api.models import TelemetrySummary

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

TELEMETRY_CSV = Path("data/telemetry.csv")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def _load_telemetry() -> pd.DataFrame:
    if not TELEMETRY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(TELEMETRY_CSV)
    return df


@router.get("", response_model=List[Dict[str, Any]])
def get_telemetry(limit: int = Query(default=50, ge=1, le=500)):
    """Return the last *limit* telemetry rows as JSON."""
    df = _load_telemetry()
    if df.empty:
        return []
    tail = df.tail(limit)
    # Replace NaN with None for clean JSON
    return tail.where(tail.notna(), None).to_dict(orient="records")


@router.get("/summary", response_model=TelemetrySummary)
def telemetry_summary():
    df = _load_telemetry()
    if df.empty:
        return TelemetrySummary(
            total_records=0, uptime_hours=0.0, avg_cpu=0.0,
            avg_memory=0.0, failure_rate=0.0,
            most_common_build_status="UNKNOWN",
        )

    total = len(df)

    # Uptime: difference between first and last timestamp
    uptime_hours = 0.0
    if "timestamp" in df.columns and total > 1:
        try:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            delta = ts.iloc[-1] - ts.iloc[0]
            uptime_hours = round(delta.total_seconds() / 3600, 1)
        except Exception:
            pass

    avg_cpu = round(_safe_float(df.get("prometheus_cpu_usage", pd.Series(dtype=float)).mean()), 1)
    avg_mem = round(_safe_float(df.get("prometheus_memory_usage", pd.Series(dtype=float)).mean()), 1)

    # Failure rate: fraction of builds that are not SUCCESS
    status_col = df.get("jenkins_last_build_status", pd.Series(dtype=str))
    non_empty = status_col.dropna().astype(str)
    non_empty = non_empty[non_empty.str.strip() != ""]
    if len(non_empty) > 0:
        failure_rate = round(1.0 - (non_empty == "SUCCESS").sum() / len(non_empty), 3)
        most_common = non_empty.mode().iloc[0] if not non_empty.mode().empty else "UNKNOWN"
    else:
        failure_rate = 0.0
        most_common = "UNKNOWN"

    return TelemetrySummary(
        total_records=total,
        uptime_hours=uptime_hours,
        avg_cpu=avg_cpu,
        avg_memory=avg_mem,
        failure_rate=failure_rate,
        most_common_build_status=most_common,
    )
