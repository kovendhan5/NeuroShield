"""Data generator for NeuroShield failure prediction — real + synthetic patterns."""

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

# ---------------------------------------------------------------------------
# Real Jenkins log patterns (from actual freestyle builds)
# ---------------------------------------------------------------------------

REAL_FAILURE_LOGS: List[str] = [
    "ERROR: script returned exit code 1\nBuild step 'Execute shell' marked build as failure\nFinished: FAILURE",
    "Build step 'Execute shell' marked build as failure\nFinished: FAILURE",
    "Finished: FAILURE\nProcess leaked file descriptors",
    "java.lang.OutOfMemoryError: Java heap space\nFinished: FAILURE",
    "Connection refused\nERROR: script returned exit code 1\nFinished: FAILURE",
    "fatal: unable to access 'https://github.com/...': Could not resolve host\nFinished: FAILURE",
    "pip: command not found\nERROR: script returned exit code 127\nFinished: FAILURE",
    "ModuleNotFoundError: No module named 'flask'\nFinished: FAILURE",
    "TimeoutError: timed out after 120 seconds\nFinished: FAILURE",
    "FAILED (failures=1)\nERROR: script returned exit code 1\nFinished: FAILURE",
    "No space left on device\nERROR: script returned exit code 1\nFinished: FAILURE",
    "npm ERR! code ELIFECYCLE\nnpm ERR! errno 1\nFinished: FAILURE",
    "FATAL: command execution failed\njava.io.IOException: error=2, No such file or directory\nFinished: FAILURE",
    "ERROR: Maven build failed\n[ERROR] Failed to execute goal\nFinished: FAILURE",
    "pytest: error: unrecognized arguments\nERROR: script returned exit code 2\nFinished: FAILURE",
    "docker: Error response from daemon: pull access denied\nFinished: FAILURE",
    "ERROR: script returned exit code 137\nFinished: FAILURE",  # OOM killed
    "ssl.SSLCertVerificationError: certificate verify failed\nFinished: FAILURE",
    "PermissionError: [Errno 13] Permission denied\nFinished: FAILURE",
    "SyntaxError: invalid syntax\nFinished: FAILURE",
]

REAL_SUCCESS_LOGS: List[str] = [
    "Finished: SUCCESS",
    "Build step 'Execute shell' marked build as success\nFinished: SUCCESS",
    "All tests passed\nFinished: SUCCESS",
    "Successfully built\nFinished: SUCCESS",
    "Deployment successful\nFinished: SUCCESS",
    "100% tests passed, 0 failures\nFinished: SUCCESS",
    "[INFO] BUILD SUCCESS\n[INFO] Total time: 12.345 s\nFinished: SUCCESS",
    "pytest: 42 passed in 8.31s\nFinished: SUCCESS",
    "docker build completed successfully\nFinished: SUCCESS",
    "All checks passed\nFinished: SUCCESS",
]


@dataclass
class SyntheticSample:
    """Container for a single synthetic sample."""

    log_text: str
    label: int
    failure_type: str
    telemetry: Dict[str, float | int | str | None]


def _generate_log_text(failure_type: str, rng: random.Random) -> str:
    """Generate a Jenkins-style log snippet — mix of real and synthetic."""
    if failure_type == "oom":
        return rng.choice([
            "[ERROR] Java heap space OutOfMemoryError\n[WARN] Container killed with exit code 137\nFinished: FAILURE",
            "java.lang.OutOfMemoryError: Java heap space\nFinished: FAILURE",
            "ERROR: script returned exit code 137\nFinished: FAILURE",
        ])
    if failure_type == "flaky_test":
        return rng.choice([
            "FAILED (failures=1)\nERROR: script returned exit code 1\nFinished: FAILURE",
            "[ERROR] TestLoginFlow failed intermittently\n[INFO] Retrying test suite...\nFinished: FAILURE",
            "pytest: 41 passed, 1 failed in 8.31s\nERROR: script returned exit code 1\nFinished: FAILURE",
        ])
    if failure_type == "network_timeout":
        return rng.choice([
            "Connection refused\nERROR: script returned exit code 1\nFinished: FAILURE",
            "TimeoutError: timed out after 120 seconds\nFinished: FAILURE",
            "fatal: unable to access 'https://github.com/...': Could not resolve host\nFinished: FAILURE",
        ])
    if failure_type == "config_error":
        return rng.choice([
            "ModuleNotFoundError: No module named 'flask'\nFinished: FAILURE",
            "pip: command not found\nERROR: script returned exit code 127\nFinished: FAILURE",
            "SyntaxError: invalid syntax\nFinished: FAILURE",
        ])
    if failure_type == "disk_full":
        return rng.choice([
            "No space left on device\nERROR: script returned exit code 1\nFinished: FAILURE",
            "[ERROR] No space left on device\n[WARN] Artifact upload failed\nFinished: FAILURE",
        ])

    # Healthy / success
    return rng.choice(REAL_SUCCESS_LOGS)


def _generate_real_sample(rng: random.Random) -> SyntheticSample:
    """Generate a sample using real Jenkins log patterns with realistic telemetry."""
    if rng.random() < 0.45:
        # Failure sample from real patterns
        log_text = rng.choice(REAL_FAILURE_LOGS)
        label = 1
        failure_type = rng.choice(FAILURE_TYPES)
        cpu = rng.uniform(40, 98)
        mem = rng.uniform(50, 99)
        error_rate = rng.uniform(0.02, 0.3)
        queue_len = rng.randint(2, 20)
        duration_ms = rng.uniform(100_000, 600_000)
        pods = rng.randint(8, 40)
        status = rng.choice(["FAILURE", "UNSTABLE", "ABORTED"])
    else:
        # Success sample from real patterns
        log_text = rng.choice(REAL_SUCCESS_LOGS)
        label = 0
        failure_type = HEALTHY_TYPE
        cpu = rng.uniform(10, 55)
        mem = rng.uniform(15, 60)
        error_rate = rng.uniform(0.0, 0.02)
        queue_len = rng.randint(0, 3)
        duration_ms = rng.uniform(20_000, 120_000)
        pods = rng.randint(2, 12)
        status = "SUCCESS"

    telemetry = {
        "jenkins_last_build_status": status,
        "jenkins_last_build_duration": duration_ms,
        "jenkins_queue_length": queue_len,
        "prometheus_cpu_usage": cpu,
        "prometheus_memory_usage": mem,
        "prometheus_pod_count": pods,
        "prometheus_error_rate": error_rate,
    }
    return SyntheticSample(log_text=log_text, label=label, failure_type=failure_type, telemetry=telemetry)


def _generate_telemetry(label: int, rng: random.Random) -> Dict[str, float | int | str | None]:
    """Generate telemetry metrics aligned with a label."""
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
    """Generate a single sample (synthetic path)."""
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
    """Generate a mixed dataset: 60% real patterns + 40% synthetic.

    Args:
        num_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        DataFrame with log_text, label, failure_type, and telemetry columns.
    """
    rng = random.Random(seed)
    records: List[Dict[str, object]] = []

    num_real = int(num_samples * 0.6)
    num_synthetic = num_samples - num_real

    # 60% real Jenkins log patterns
    for _ in range(num_real):
        sample = _generate_real_sample(rng)
        record = {
            "log_text": sample.log_text,
            "label": sample.label,
            "failure_type": sample.failure_type,
        }
        record.update(sample.telemetry)
        records.append(record)

    # 40% synthetic patterns
    for _ in range(num_synthetic):
        sample = generate_sample(rng)
        record = {
            "log_text": sample.log_text,
            "label": sample.label,
            "failure_type": sample.failure_type,
        }
        record.update(sample.telemetry)
        records.append(record)

    # Shuffle
    rng.shuffle(records)

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
