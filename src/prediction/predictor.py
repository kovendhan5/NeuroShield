"""Inference utilities for NeuroShield failure prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
import numpy as np
import pandas as pd
from joblib import load

from .log_encoder import LogEncoder

TelemetryInput = Union[Mapping[str, Any], str, Path, None]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert values to float safely."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def telemetry_to_vector(telemetry: Mapping[str, Any]) -> np.ndarray:
    """Convert telemetry metrics into a fixed-length feature vector.

    Args:
        telemetry: Telemetry mapping (dict-like).

    Returns:
        NumPy array with 8 telemetry features.
    """
    status = str(telemetry.get("jenkins_last_build_status") or "").upper()
    status_success = 1.0 if status == "SUCCESS" else 0.0
    status_failure = 1.0 if status in {"FAILURE", "UNSTABLE", "ABORTED"} else 0.0

    duration_ms = _safe_float(telemetry.get("jenkins_last_build_duration"))
    queue_len = _safe_float(telemetry.get("jenkins_queue_length"))
    cpu = _safe_float(telemetry.get("prometheus_cpu_usage"))
    mem = _safe_float(telemetry.get("prometheus_memory_usage"))
    pods = _safe_float(telemetry.get("prometheus_pod_count"))
    error_rate = _safe_float(telemetry.get("prometheus_error_rate"))

    features = np.array(
        [
            np.log1p(duration_ms),
            queue_len,
            cpu,
            mem,
            pods,
            error_rate,
            status_success,
            status_failure,
        ],
        dtype=np.float32,
    )
    return features


def _load_latest_telemetry(csv_path: Path) -> Dict[str, Any]:
    """Load latest telemetry row from CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def resolve_telemetry(telemetry_input: TelemetryInput) -> Dict[str, Any]:
    """Resolve telemetry input to a dictionary."""
    if telemetry_input is None:
        return {}
    if isinstance(telemetry_input, (str, Path)):
        return _load_latest_telemetry(Path(telemetry_input))
    return dict(telemetry_input)


class FailurePredictor:
    """Predicts CI/CD failure probability and builds state vectors."""

    def __init__(
        self,
        model_dir: str | Path = "models",
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
    ) -> None:
        """Initialize predictor by loading PCA and classifier.

        Args:
            model_dir: Directory with saved PCA and classifier.
            model_name: Hugging Face model for log encoding.
            max_length: Tokenization max length.
        """
        self.model_dir = Path(model_dir)
        self.encoder = LogEncoder(model_name=model_name, max_length=max_length)
        self.encoder.load_pca(self.model_dir / "log_pca.joblib")
        self.classifier = load(self.model_dir / "failure_classifier.joblib")

    def build_state_vector(
        self,
        log_text: str,
        telemetry: TelemetryInput = None,
    ) -> np.ndarray:
        """Build a state vector from logs and telemetry.

        Args:
            log_text: Jenkins log text.
            telemetry: Telemetry dict or CSV path.

        Returns:
            State vector of length 24 (16D log + 8 telemetry).
        """
        telemetry_dict = resolve_telemetry(telemetry)
        log_embed = self.encoder.encode_logs([log_text])[0]
        telemetry_vec = telemetry_to_vector(telemetry_dict)
        return np.concatenate([log_embed.astype(np.float32), telemetry_vec], axis=0)

    def predict(
        self,
        log_text: str,
        telemetry: TelemetryInput = None,
    ) -> tuple[float, np.ndarray]:
        """Predict failure probability.

        Args:
            log_text: Jenkins log text.
            telemetry: Telemetry dict or CSV path.

        Returns:
            Tuple of (failure_prob, state_vector).
        """
        state_vector = self.build_state_vector(log_text, telemetry)
        prob = float(self.classifier.predict_proba([state_vector])[0][1])
        return prob, state_vector
