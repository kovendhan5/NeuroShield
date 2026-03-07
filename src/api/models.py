"""Pydantic request/response models for the NeuroShield REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Status ────────────────────────────────────────────────────────────────────

class APIInfo(BaseModel):
    name: str = "NeuroShield AIOps API"
    version: str = "2.0"
    status: str = "running"
    docs: str = "/docs"


class ServiceStatus(BaseModel):
    status: str
    url: str


class ModelStatus(BaseModel):
    failure_predictor: str
    ppo_policy: str
    pca_encoder: str


class HealthResponse(BaseModel):
    timestamp: str
    services: Dict[str, ServiceStatus]
    models: ModelStatus
    overall: str


class MetricsResponse(BaseModel):
    timestamp: str
    cpu_usage: float
    memory_usage: float
    failure_probability: float
    build_status: str
    pod_count: float
    total_telemetry_rows: int


# ── Telemetry ─────────────────────────────────────────────────────────────────

class TelemetrySummary(BaseModel):
    total_records: int
    uptime_hours: float
    avg_cpu: float
    avg_memory: float
    failure_rate: float
    most_common_build_status: str


# ── Healing ───────────────────────────────────────────────────────────────────

class HealingEntry(BaseModel):
    timestamp: str
    action: str
    reason: str = ""
    result: str = ""
    failure_probability: float = 0.0


class HealingStats(BaseModel):
    total_actions: int
    action_distribution: Dict[str, int]
    success_rate: float
    avg_mttr_reduction: float


class HealingTriggerRequest(BaseModel):
    action: str = Field(
        ...,
        description="One of: restart_pod, scale_up, retry_build, rollback_deploy, clear_cache, escalate_to_human",
    )
    reason: str = Field(default="manual trigger", description="Reason for manual trigger")


class HealingTriggerResponse(BaseModel):
    triggered: bool
    action: str
    timestamp: str


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    log_text: str = Field(default="", description="Jenkins log text")
    cpu: float = Field(default=0.0, ge=0)
    memory: float = Field(default=0.0, ge=0)
    error_rate: float = Field(default=0.0, ge=0)
    build_status: str = Field(default="SUCCESS")


class PredictionResponse(BaseModel):
    failure_probability: float
    risk_level: str
    recommended_action: str
    explanation: str


# ── MTTR ──────────────────────────────────────────────────────────────────────

class MTTRIncident(BaseModel):
    timestamp: str
    failure_type: str
    action: str
    actual_mttr_s: float
    baseline_mttr_s: float
    reduction_pct: float


class MTTRResponse(BaseModel):
    incidents: List[MTTRIncident]
    avg_reduction_pct: float
    avg_actual_mttr_seconds: float
    best_reduction_pct: float
    total_incidents: int


# ── Demo ──────────────────────────────────────────────────────────────────────

class DemoTriggerResponse(BaseModel):
    triggered: bool
    detail: str = ""
    build_url: Optional[str] = None
    memory_before_mb: Optional[float] = None
