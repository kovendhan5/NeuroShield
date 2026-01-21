"""Synthetic data generator for NeuroShield failure prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np
import pandas as pd

FAILURE_TYPES: List[str] = [
    "oom",
    "flaky_test",
    "network_timeout",
    "config_error",
    "disk_full",
]

HEALTHY_TYPE = "healthy"


@dataclass
class SyntheticSample:
    """Container for a single synthetic sample."""

    log_text: str
    label: int
    failure_type: str
    telemetry: Dict[str, float | int | str | None]


def _generate_log_text(failure_type: str, rng: random.Random) -> str:
    """Generate a synthetic Jenkins-style log snippet.

    Args:
        failure_type: Failure category or "healthy".
        rng: Random generator.

    Returns:
        Synthetic log text.
    """
    if failure_type == "oom":
        return (
            "[INFO] Running tests...\n"
            "[ERROR] Java heap space OutOfMemoryError\n"
            "[WARN] Container killed with exit code 137\n"
            "[INFO] Build failed due to OOM at step compile"
        )
    if failure_type == "flaky_test":
        return (
            "[INFO] Running unit tests...\n"
            "[ERROR] TestLoginFlow failed intermittently\n"
            "[INFO] Retrying test suite...\n"
            "[ERROR] Flaky test detected in module auth"
        )
    if failure_type == "network_timeout":
        return (
            "[INFO] Pulling dependencies...\n"
            "[ERROR] npm ERR! network timeout while fetching package\n"
            "[WARN] Retry exceeded for registry.npmjs.org\n"
            "[INFO] Build failed due to network timeout"
        )
    if failure_type == "config_error":
        return (
            "[INFO] Loading pipeline configuration...\n"
            "[ERROR] YAML parse error: unexpected token at line 42\n"
            "[INFO] Aborting pipeline due to invalid config"
        )
    if failure_type == "disk_full":
        return (
            "[INFO] Writing artifacts...\n"
            "[ERROR] No space left on device\n"
            "[WARN] Artifact upload failed due to disk full\n"
            "[INFO] Build failed during archive step"
        )

    build_id = rng.randint(1000, 9999)
    duration = rng.randint(30, 240)
    return (
        f"[INFO] Build #{build_id} started\n"
        "[INFO] Running tests...\n"
        "[INFO] All checks passed\n"
        f"[INFO] Build finished successfully in {duration}s"
    )


def _generate_telemetry(label: int, rng: random.Random) -> Dict[str, float | int | str | None]:
    """Generate telemetry metrics aligned with a label.

    Args:
        label: 1 for failure, 0 for healthy.
        rng: Random generator.

    Returns:
        Telemetry dictionary.
    """
    if label == 1:
        cpu = rng.uniform(70, 98)
        mem = rng.uniform(75, 99)
        error_rate = rng.uniform(0.05, 0.25)
        queue_len = rng.randint(5, 20)
        duration_ms = rng.uniform(150_000, 600_000)
        pods = rng.randint(15, 40)
        status = rng.choice(["FAILURE", "UNSTABLE", "ABORTED"])
    else:
        cpu = rng.uniform(15, 55)
        mem = rng.uniform(20, 60)
        error_rate = rng.uniform(0.0, 0.03)
        queue_len = rng.randint(0, 4)
        duration_ms = rng.uniform(30_000, 120_000)
        pods = rng.randint(3, 12)
        status = "SUCCESS"

    return {
        "jenkins_last_build_status": status,
        "jenkins_last_build_duration": duration_ms,
        "jenkins_queue_length": queue_len,
        "prometheus_cpu_usage": cpu,
        "prometheus_memory_usage": mem,
        "prometheus_pod_count": pods,
        "prometheus_error_rate": error_rate,
    }


def generate_sample(rng: random.Random) -> SyntheticSample:
    """Generate a single synthetic sample.

    Args:
        rng: Random generator.

    Returns:
        SyntheticSample instance.
    """
    if rng.random() < 0.55:
        failure_type = rng.choice(FAILURE_TYPES)
        label = 1
    else:
        failure_type = HEALTHY_TYPE
        label = 0

    log_text = _generate_log_text(failure_type, rng)
    telemetry = _generate_telemetry(label, rng)
    return SyntheticSample(log_text=log_text, label=label, failure_type=failure_type, telemetry=telemetry)


def generate_dataset(num_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset for training.

    Args:
        num_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        DataFrame with log_text, label, failure_type, and telemetry columns.
    """
    rng = random.Random(seed)
    records: List[Dict[str, object]] = []

    for _ in range(num_samples):
        sample = generate_sample(rng)
        record = {
            "log_text": sample.log_text,
            "label": sample.label,
            "failure_type": sample.failure_type,
        }
        record.update(sample.telemetry)
        records.append(record)

    return pd.DataFrame(records)


def split_logs_and_telemetry(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, object]], np.ndarray]:
    """Split a dataset into logs, telemetry, and labels.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (logs, telemetry dicts, labels array).
    """
    logs = df["log_text"].tolist()
    telemetry = df[
        [
            "jenkins_last_build_status",
            "jenkins_last_build_duration",
            "jenkins_queue_length",
            "prometheus_cpu_usage",
            "prometheus_memory_usage",
            "prometheus_pod_count",
            "prometheus_error_rate",
        ]
    ].to_dict(orient="records")
    labels = df["label"].astype(int).to_numpy()
    return logs, telemetry, labels
