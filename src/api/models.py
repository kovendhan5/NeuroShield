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


# ── Alertmanager Webhook ───────────────────────────────────────────────────────

class AlertManagerLabelSet(BaseModel):
    alertname: Optional[str] = None
    severity: Optional[str] = None
    instance: Optional[str] = None
    job: Optional[str] = None
    service: Optional[str] = None
    namespace: Optional[str] = None
    pod: Optional[str] = None


class AlertManagerAnnotations(BaseModel):
    summary: Optional[str] = None
    description: Optional[str] = None
    runbook_url: Optional[str] = None


class AlertManagerAlert(BaseModel):
    status: str
    labels: AlertManagerLabelSet
    annotations: Optional[AlertManagerAnnotations] = None
    startsAt: str
    endsAt: Optional[str] = None
    generatorURL: Optional[str] = None
    fingerprint: Optional[str] = None


class AlertManagerWebhookPayload(BaseModel):
    receiver: str
    status: str
    alerts: List[AlertManagerAlert] = Field(default_factory=list)
    groupLabels: Dict[str, str] = Field(default_factory=dict)
    commonLabels: Dict[str, str] = Field(default_factory=dict)
    commonAnnotations: Dict[str, str] = Field(default_factory=dict)
    externalURL: Optional[str] = None
    version: Optional[str] = None
    groupKey: Optional[str] = None
    truncatedAlerts: Optional[int] = None


class PipelineEventPayload(BaseModel):
    pipeline_id: str
    project: str
    use_case: str
    environment: str = "production"
    deploy_target: str = "kubernetes"
    status: str
    success: bool
    k8s_namespace: str
    k8s_deployment: str
    deployment_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    build_number: Optional[str] = None
    failed_pods: Optional[int] = None
    pod_restarts_total: Optional[int] = None
    error_message: Optional[str] = None
    stage: Optional[str] = None
    incident_kind: Optional[str] = None
    healed_by: Optional[str] = None
    heal_action: Optional[str] = None
    build_url: Optional[str] = None
    timestamp: Optional[str] = None
