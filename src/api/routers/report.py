"""Report router — model performance report endpoints."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["report"])

_SUMMARY_PATH = Path("data/model_report_summary.json")
_REPORT_PATH = Path("data/model_report.html")


@router.get("/report/summary")
def report_summary():
    """Return the latest model performance report metrics."""
    if not _SUMMARY_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "No report generated yet. POST /report/generate first."},
        )
    data = json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))
    return data


@router.post("/report/generate")
def report_generate():
    """Trigger model report regeneration in the background."""
    subprocess.Popen(
        [sys.executable, "scripts/generate_model_report.py"],
        cwd=str(Path.cwd()),
    )
    return {
        "status": "generating",
        "estimated_time_seconds": 30,
        "report_path": "data/model_report.html",
        "summary_path": "data/model_report_summary.json",
    }
