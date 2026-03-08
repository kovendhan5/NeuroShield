"""NeuroShield AIOps REST API.

FastAPI application that exposes NeuroShield's capabilities to external
tools, scripts, and monitoring systems.

Run:
    python scripts/start_api.py          # http://localhost:8502
    uvicorn src.api.main:app --port 8502 # same, manual
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from src.api.routers import demo, healing, mttr, prediction, report, status, telemetry  # noqa: E402

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
