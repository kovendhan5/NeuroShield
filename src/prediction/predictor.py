"""Inference utilities for NeuroShield failure prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
import numpy as np
import pandas as pd
import torch

from .log_encoder import LogEncoder
from .model import FailureClassifier

TelemetryInput = Union[Mapping[str, Any], str, Path, None]
_DEFAULT_PREDICTOR: FailurePredictor | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert values to float safely, treating NaN as *default*."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    try:
        result = float(value)
        if result != result:  # NaN check (works for float & numpy NaN)
            return default
        return result
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = FailureClassifier(input_dim=24)
        # weights_only=True prevents arbitrary code execution from untrusted model files
        state = torch.load(
            self.model_dir / "failure_predictor.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.classifier.load_state_dict(state)
        self.classifier.to(self.device)
        self.classifier.eval()

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

    def predict(self, log_text: str, telemetry: TelemetryInput = None) -> float:
        """Predict failure probability from log text and telemetry.

        Args:
            log_text: Jenkins log text.
            telemetry: Telemetry dict or CSV path.

        Returns:
            Failure probability between 0 and 1.

        Raises:
            ValueError: If log_text is not a string.
            TypeError: If model inference fails.
        """
        if not isinstance(log_text, str):
            raise ValueError(
                f"log_text must be a str, got {type(log_text).__name__}. "
                "Use predict_from_state() for numpy arrays."
            )

        try:
            state_vector = self.build_state_vector(log_text, telemetry)
            return self.predict_from_state(state_vector, telemetry)
        except Exception as e:
            raise TypeError(f"Prediction failed: {str(e)}") from e

    def predict_from_state(
        self,
        state_vector: np.ndarray,
        telemetry: TelemetryInput = None
    ) -> float:
        """Predict failure probability from a pre-built state vector.

        This method is used when you already have a 52D or 24D state vector
        and want to predict without re-encoding logs.

        Args:
            state_vector: NumPy array of shape (24,) or (52,).
            telemetry: Telemetry dict for status boost (optional).

        Returns:
            Failure probability between 0 and 1.

        Raises:
            ValueError: If state_vector has wrong shape.
            RuntimeError: If model inference fails.
        """
        if not isinstance(state_vector, np.ndarray):
            raise ValueError(
                f"state_vector must be np.ndarray, got {type(state_vector).__name__}"
            )

        # Handle both 24D and 52D state vectors
        if state_vector.shape[0] == 52:
            # Take first 24 dimensions (log embedding + telemetry)
            state_vector = state_vector[:24]
        elif state_vector.shape[0] != 24:
            raise ValueError(
                f"state_vector must be 24D or 52D, got shape {state_vector.shape}"
            )

        try:
            tensor = torch.tensor(
                state_vector,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                logits = self.classifier(tensor).squeeze(0)
                prob = torch.sigmoid(logits).item()

            prob = self._apply_status_boost(float(prob), telemetry)
            return prob
        except Exception as e:
            raise RuntimeError(
                f"Model inference failed: {str(e)}"
            ) from e

    def _apply_status_boost(self, prob: float, telemetry: TelemetryInput) -> float:
        """Boost probability for FAILURE builds when model signal is too weak."""
        tdict = resolve_telemetry(telemetry)
        status = str(tdict.get("jenkins_last_build_status", "")).upper()
        if status in ("FAILURE", "UNSTABLE", "ABORTED") and prob < 0.5:
            return max(prob, 0.65)
        return prob

    def predict_with_state(
        self,
        log_text: str,
        telemetry: TelemetryInput = None,
    ) -> tuple[float, np.ndarray]:
        """Predict failure probability and return state vector."""
        state_vector = self.build_state_vector(log_text, telemetry)
        tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.classifier(tensor).squeeze(0)
            prob = torch.sigmoid(logits).item()
        prob = self._apply_status_boost(float(prob), telemetry)
        return prob, state_vector


def predict(log_text: str, telemetry: TelemetryInput = None) -> float:
    """Module-level helper for quick prediction.

    Args:
        log_text: Jenkins log text.
        telemetry: Telemetry dict or CSV path.

    Returns:
        Failure probability between 0 and 1.
    """
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = FailurePredictor()
    return _DEFAULT_PREDICTOR.predict(log_text, telemetry)


# ---------------------------------------------------------------------------
# 52-dimensional state builder (matches RL env observation layout)
# ---------------------------------------------------------------------------
# [ 0:10]  Build Metrics          (10D)
# [10:22]  Resource Metrics        (12D)
# [22:38]  Log Embeddings          (16D)  PCA-reduced DistilBERT
# [38:52]  Dependency Signals      (14D)
# ---------------------------------------------------------------------------

_BUILD_METRIC_KEYS = [
    "build_duration", "passed_tests", "failed_tests", "retry_count",
    "stage_failure_rate", "build_number", "queue_time", "artifact_size",
    "test_coverage", "change_set_size",
]

_RESOURCE_METRIC_KEYS = [
    "cpu_avg_5m", "memory_avg_5m", "memory_max", "pod_restarts",
    "throttle_events", "network_latency", "disk_io", "cpu_limit_pct",
    "memory_limit_pct", "node_count", "pending_pods", "evicted_pods",
]

_DEPENDENCY_SIGNAL_KEYS = [
    "dep_version_drifts", "cache_hit_ratio", "cache_miss_ratio",
    "new_deps_count", "outdated_deps", "pkg_manager_npm",
    "pkg_manager_maven", "pkg_manager_pip", "dep_resolution_time",
    "lock_file_changed", "transitive_dep_count", "dep_conflict_count",
    "registry_latency", "dep_download_failures",
]


def build_52d_state(
    jenkins_data: Dict[str, Any],
    prometheus_data: Dict[str, Any],
    log_text: str,
    encoder: Optional[LogEncoder] = None,
) -> np.ndarray:
    """Build a 52-dimensional state vector for the PPO policy.

    Args:
        jenkins_data: Dict with build metrics and dependency signals.
        prometheus_data: Dict with resource / cluster metrics.
        log_text: Raw Jenkins console log for DistilBERT encoding.
        encoder: Pre-initialised LogEncoder with PCA loaded.
                 If *None*, log embeddings default to zeros(16).

    Returns:
        np.ndarray of shape (52,) with values clipped to [-10, 10].
    """
    # -- Build Metrics (10D) ------------------------------------------------
    build_vec = [_safe_float(jenkins_data.get(k)) for k in _BUILD_METRIC_KEYS]

    # -- Resource Metrics (12D) ---------------------------------------------
    resource_vec = [_safe_float(prometheus_data.get(k)) for k in _RESOURCE_METRIC_KEYS]

    # -- Log Embeddings (16D) -----------------------------------------------
    if log_text and encoder is not None:
        try:
            log_embed = encoder.encode_logs([log_text])[0].astype(np.float32)
        except Exception:
            log_embed = np.zeros(16, dtype=np.float32)
    else:
        log_embed = np.zeros(16, dtype=np.float32)

    # -- Dependency Signals (14D) -------------------------------------------
    dep_vec = [_safe_float(jenkins_data.get(k)) for k in _DEPENDENCY_SIGNAL_KEYS]

    # -- Concatenate and clip -----------------------------------------------
    state = np.concatenate(
        [
            np.array(build_vec, dtype=np.float32),    # 10D
            np.array(resource_vec, dtype=np.float32),  # 12D
            log_embed,                                  # 16D
            np.array(dep_vec, dtype=np.float32),       # 14D
        ],
        axis=0,
    )
    assert state.shape == (52,), f"Expected 52D state, got {state.shape}"
    return np.clip(state, -10.0, 10.0)
