"""Prediction endpoint: POST /predict."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.api.models import PredictionRequest, PredictionResponse

router = APIRouter(tags=["prediction"])
logger = logging.getLogger(__name__)

# Lazy-loaded predictor (heavy model — only load once)
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from src.prediction.predictor import FailurePredictor
        _predictor = FailurePredictor()
        logger.info("FailurePredictor loaded for API")
    return _predictor


ACTION_MAP = {
    (0.0, 0.3): "no_action",
    (0.3, 0.5): "clear_cache",
    (0.5, 0.7): "retry_build",
    (0.7, 0.85): "rollback_deploy",
    (0.85, 1.01): "escalate_to_human",
}


def _recommend_action(prob: float) -> str:
    for (lo, hi), action in ACTION_MAP.items():
        if lo <= prob < hi:
            return action
    return "escalate_to_human"


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    if prob < 0.6:
        return "MEDIUM"
    return "HIGH"


def _explain(req: PredictionRequest, prob: float) -> str:
    parts = []
    status = req.build_status.upper()
    if status in ("FAILURE", "UNSTABLE", "ABORTED"):
        parts.append(f"Build status {status} detected")
    if req.cpu > 80:
        parts.append(f"High CPU usage ({req.cpu}%)")
    if req.memory > 85:
        parts.append(f"High memory usage ({req.memory}%)")
    if req.error_rate > 0.3:
        parts.append(f"Elevated error rate ({req.error_rate})")
    if "FAILURE" in req.log_text.upper():
        parts.append("Failure keyword in log text")
    if not parts:
        parts.append("System metrics within normal range")
    return "; ".join(parts)


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    predictor = _get_predictor()

    telemetry = {
        "prometheus_cpu_usage": req.cpu,
        "prometheus_memory_usage": req.memory,
        "prometheus_error_rate": req.error_rate,
        "jenkins_last_build_status": req.build_status,
        "prometheus_pod_count": 1,
        "jenkins_last_build_duration": 0,
        "jenkins_queue_length": 0,
    }

    prob = predictor.predict(req.log_text or "No log", telemetry)

    return PredictionResponse(
        failure_probability=round(prob, 4),
        risk_level=_risk_level(prob),
        recommended_action=_recommend_action(prob),
        explanation=_explain(req, prob),
    )
