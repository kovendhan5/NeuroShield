"""MTTR endpoint: GET /mttr."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter

from src.api.models import MTTRIncident, MTTRResponse

router = APIRouter(tags=["mttr"])

MTTR_CSV = Path("data/mttr_log.csv")


@router.get("/mttr", response_model=MTTRResponse)
def mttr():
    if not MTTR_CSV.exists():
        return MTTRResponse(
            incidents=[], avg_reduction_pct=0.0,
            avg_actual_mttr_seconds=0.0, best_reduction_pct=0.0,
            total_incidents=0,
        )

    df = pd.read_csv(MTTR_CSV)
    if df.empty:
        return MTTRResponse(
            incidents=[], avg_reduction_pct=0.0,
            avg_actual_mttr_seconds=0.0, best_reduction_pct=0.0,
            total_incidents=0,
        )

    incidents: List[MTTRIncident] = []
    for _, row in df.iterrows():
        incidents.append(MTTRIncident(
            timestamp=str(row.get("timestamp", "")),
            failure_type=str(row.get("failure_type", "Unknown")),
            action=str(row.get("action", "")),
            actual_mttr_s=float(row.get("actual_mttr_s", 0)),
            baseline_mttr_s=float(row.get("baseline_mttr_s", 0)),
            reduction_pct=float(row.get("reduction_pct", 0)),
        ))

    reductions = pd.to_numeric(df["reduction_pct"], errors="coerce").dropna()
    actual = pd.to_numeric(df["actual_mttr_s"], errors="coerce").dropna()

    return MTTRResponse(
        incidents=incidents,
        avg_reduction_pct=round(float(reductions.mean()), 1) if len(reductions) else 0.0,
        avg_actual_mttr_seconds=round(float(actual.mean()), 1) if len(actual) else 0.0,
        best_reduction_pct=round(float(reductions.max()), 1) if len(reductions) else 0.0,
        total_incidents=len(df),
    )
