"""NeuroShield Orchestrator - Real-time CI/CD Monitoring."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, TypeVar

import numpy as np
import requests
from dotenv import load_dotenv
from stable_baselines3 import PPO

from src.prediction.predictor import FailurePredictor

T = TypeVar("T")


def retry_call(fn: Callable[[], T], max_attempts: int = 3, delay: int = 2) -> T:
    """Retry *fn* with exponential back-off.

    Args:
        fn: Zero-argument callable to retry.
        max_attempts: Maximum number of attempts.
        delay: Base delay in seconds (doubled each retry).
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay * (2 ** attempt))
    # unreachable, but keeps mypy happy
    raise RuntimeError("retry_call exhausted")

ACTION_NAMES: Dict[int, str] = {
    0: "Retry",
    1: "Scale Pods",
    2: "Rollback",
    3: "No-op",
}


def _namespace() -> str:
    return os.getenv("K8S_NAMESPACE", "neuroshield")


def _affected_service() -> str:
    return os.getenv("AFFECTED_SERVICE", "dummy-app")


def _scale_replicas() -> int:
    return int(os.getenv("SCALE_REPLICAS", "3"))


@dataclass
class BuildInfo:
    number: int
    timestamp_ms: int
    duration_ms: int
    result: str
    url: str

    @property
    def end_time_ms(self) -> int:
        return int(self.timestamp_ms + self.duration_ms)


def _setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "orchestrator_audit.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def _load_env() -> None:
    load_dotenv(override=False)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_latest_build_info(jenkins_url: str, job_name: str, username: str, token: str) -> Optional[BuildInfo]:
    """Get the latest build information from Jenkins (with retry)."""
    url = f"{jenkins_url}/job/{job_name}/lastBuild/api/json"
    auth = (username, token)

    def _call() -> Optional[BuildInfo]:
        response = requests.get(url, auth=auth, timeout=15)
        if response.status_code == 200:
            build_data = response.json()
            result = build_data.get("result") or "RUNNING"
            return BuildInfo(
                number=int(build_data["number"]),
                timestamp_ms=int(build_data["timestamp"]),
                duration_ms=int(build_data.get("duration") or 0),
                result=str(result),
                url=str(build_data.get("url") or ""),
            )
        logging.warning("Error fetching build info: %s", response.status_code)
        return None

    try:
        return retry_call(_call)
    except Exception as exc:
        logging.exception("Exception in get_latest_build_info: %s", exc)
        return None


def get_build_log(jenkins_url: str, job_name: str, build_number: int, username: str, token: str) -> str:
    """Get the console log for a specific build (with retry)."""
    url = f"{jenkins_url}/job/{job_name}/{build_number}/consoleText"
    auth = (username, token)

    def _call() -> str:
        response = requests.get(url, auth=auth, timeout=30)
        if response.status_code == 200:
            return response.text
        logging.warning("Error fetching build log: %s", response.status_code)
        return ""

    try:
        return retry_call(_call)
    except Exception as exc:
        logging.exception("Exception in get_build_log: %s", exc)
        return ""


def detect_failure_pattern(log_text: str) -> Tuple[str, Optional[int]]:
    """Detect known failure patterns to bias healing actions."""
    text = log_text.lower()
    if "outofmemoryerror" in text or "oom" in text or "java heap space" in text:
        return "OOM", 1
    if "flaky" in text or "intermittent" in text or "retry" in text:
        return "FlakyTest", 0
    if "dependency" in text or "resolution failed" in text or "rollback" in text:
        return "Dependency", 2
    if "timeout" in text or "timed out" in text:
        return "Timeout", 0
    return "Unknown", None


def _parse_kubectl_top_nodes(output: str) -> Tuple[float, float]:
    """Parse `kubectl top nodes` output and return avg CPU/memory usage.

    Returns:
        (cpu_millicores_avg, memory_mib_avg)
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0, 0.0
    cpu_vals: list[float] = []
    mem_vals: list[float] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        cpu_raw = parts[1]
        mem_raw = parts[3]
        if cpu_raw.endswith("m"):
            cpu_vals.append(float(cpu_raw[:-1]))
        elif cpu_raw.isdigit():
            cpu_vals.append(float(cpu_raw) * 1000.0)
        if mem_raw.endswith("Mi"):
            mem_vals.append(float(mem_raw[:-2]))
        elif mem_raw.endswith("Gi"):
            mem_vals.append(float(mem_raw[:-2]) * 1024.0)
    if not cpu_vals or not mem_vals:
        return 0.0, 0.0
    return float(np.mean(cpu_vals)), float(np.mean(mem_vals))


def _get_k8s_resource_metrics(namespace: str) -> Dict[str, float]:
    """Fetch basic resource metrics from Kubernetes via kubectl (with retry).

    Returns defaults on failure.
    """
    metrics = {
        "cpu_millicores_avg": 0.0,
        "memory_mib_avg": 0.0,
        "pod_count": 0.0,
        "error_rate": 0.0,
    }

    def _top_nodes() -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", "top", "nodes"],
            capture_output=True, text=True, check=True,
        )

    try:
        top_nodes = retry_call(_top_nodes)
        cpu_avg, mem_avg = _parse_kubectl_top_nodes(top_nodes.stdout)
        metrics["cpu_millicores_avg"] = cpu_avg
        metrics["memory_mib_avg"] = mem_avg
    except Exception as exc:
        logging.debug("kubectl top nodes failed: %s", exc)

    def _get_pods() -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "--no-headers"],
            capture_output=True, text=True, check=True,
        )

    try:
        pods = retry_call(_get_pods)
        metrics["pod_count"] = float(len([line for line in pods.stdout.splitlines() if line.strip()]))
    except Exception as exc:
        logging.debug("kubectl get pods failed: %s", exc)

    return metrics


def execute_healing_action(action_id: int, context: Dict[str, str]) -> bool:
    """Execute the healing action based on PPO decision (with retry)."""
    try:
        namespace = _namespace()
        if action_id == 0:  # Retry
            logging.info("Executing action: Retry build #%s", context.get("build_number", "unknown"))
            jenkins_url = _required_env("JENKINS_URL")
            job_name = _required_env("JENKINS_JOB")
            username = _required_env("JENKINS_USERNAME")
            token = _required_env("JENKINS_TOKEN")
            retry_url = f"{jenkins_url}/job/{job_name}/build"

            def _trigger_build() -> bool:
                response = requests.post(retry_url, auth=(username, token), timeout=15)
                if response.status_code not in {200, 201, 202}:
                    raise RuntimeError(f"Jenkins trigger failed: {response.status_code}")
                return True

            return retry_call(_trigger_build)

        if action_id == 1:  # Scale pods
            service = context.get("affected_service", _affected_service())
            replicas = _scale_replicas()
            logging.info("Executing action: Scale pods for %s to %d", service, replicas)
            cmd = ["kubectl", "scale", f"deploy/{service}", f"--replicas={replicas}", "-n", namespace]

            def _scale() -> bool:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True

            return retry_call(_scale)

        if action_id == 2:  # Rollback
            service = context.get("affected_service", _affected_service())
            logging.info("Executing action: Rollback deployment for %s", service)
            cmd = ["kubectl", "rollout", "undo", f"deploy/{service}", "-n", namespace]

            def _rollback() -> bool:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True

            return retry_call(_rollback)

        if action_id == 3:  # No-op
            logging.info("Executing action: No operation")
            return True

        logging.warning("Unknown action id: %s", action_id)
        return False
    except Exception as exc:
        logging.exception("Error executing action %s: %s", action_id, exc)
        return False


def _is_failure(result: str) -> bool:
    return result.upper() in {"FAILURE", "UNSTABLE", "ABORTED"}


def main() -> None:
    _setup_logging()
    _load_env()
    logging.info("Starting NeuroShield Real-Time Orchestrator")

    jenkins_url = _required_env("JENKINS_URL")
    job_name = _required_env("JENKINS_JOB")
    username = _required_env("JENKINS_USERNAME")
    token = _required_env("JENKINS_TOKEN")
    poll_interval = int(os.getenv("POLL_INTERVAL", "10"))
    namespace = _namespace()

    logging.info("Connecting to Jenkins: %s", jenkins_url)
    logging.info("Monitoring job: %s", job_name)

    predictor = FailurePredictor(model_dir="models")
    try:
        policy = PPO.load("models/ppo_policy.zip")
    except Exception as exc:
        policy = None
        logging.warning("PPO policy not loaded, falling back to Retry: %s", exc)

    baseline_mttr_min = 0.0
    neuro_mttr_min = 0.0
    total_failures = 0
    successful_interventions = 0

    last_build_number: Optional[int] = None
    pending_intervention_start_ms: Optional[int] = None

    try:
        while True:
            build = get_latest_build_info(jenkins_url, job_name, username, token)

            if build and build.number != last_build_number:
                logging.info("New build detected: #%s (%s)", build.number, build.result)
                log_text = get_build_log(jenkins_url, job_name, build.number, username, token)

                if log_text:
                    resource_metrics = _get_k8s_resource_metrics(namespace)
                    telemetry_dict = {
                        "jenkins_last_build_status": build.result,
                        "jenkins_last_build_duration": build.duration_ms,
                        "jenkins_queue_length": 0,
                        "prometheus_cpu_usage": resource_metrics["cpu_millicores_avg"],
                        "prometheus_memory_usage": resource_metrics["memory_mib_avg"],
                        "prometheus_pod_count": resource_metrics["pod_count"],
                        "prometheus_error_rate": resource_metrics["error_rate"],
                    }

                    failure_prob, state_vector = predictor.predict_with_state(log_text, telemetry_dict)
                    pattern, pattern_action = detect_failure_pattern(log_text)
                    logging.info(
                        "Failure probability: %.3f | pattern=%s | build_url=%s",
                        failure_prob,
                        pattern,
                        build.url,
                    )

                    if failure_prob > 0.5:
                        if policy is None:
                            action_id = 0
                        else:
                            action, _ = policy.predict(np.asarray(state_vector, dtype=np.float32), deterministic=True)
                            action_id = int(action)
                        if pattern_action is not None:
                            action_id = pattern_action
                        logging.info("NeuroShield decision: %s", ACTION_NAMES.get(action_id, str(action_id)))

                        success = execute_healing_action(
                            action_id,
                            {
                                "build_number": str(build.number),
                                "affected_service": _affected_service(),
                            },
                        )
                        if success:
                            successful_interventions += 1
                            pending_intervention_start_ms = build.end_time_ms
                            logging.info("Action executed successfully")
                        else:
                            logging.warning("Action execution failed")
                    else:
                        logging.info("Build looks healthy; no intervention")

                if _is_failure(build.result):
                    total_failures += 1
                    baseline_mttr_min += build.duration_ms / 60000.0
                    if pending_intervention_start_ms is None:
                        neuro_mttr_min += build.duration_ms / 60000.0
                elif build.result.upper() == "SUCCESS" and pending_intervention_start_ms is not None:
                    recovery_mttr = max(build.end_time_ms - pending_intervention_start_ms, 0) / 60000.0
                    neuro_mttr_min += recovery_mttr
                    pending_intervention_start_ms = None

                if total_failures > 0:
                    mttr_reduction = (
                        (baseline_mttr_min - neuro_mttr_min) / baseline_mttr_min * 100.0
                        if baseline_mttr_min > 0
                        else 0.0
                    )
                    logging.info(
                        "MTTR baseline=%.2f min | neuro=%.2f min | reduction=%.1f%% | interventions=%s",
                        baseline_mttr_min,
                        neuro_mttr_min,
                        mttr_reduction,
                        successful_interventions,
                    )

                last_build_number = build.number

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logging.info("Orchestrator stopped by user")
    except Exception as exc:
        logging.exception("Error in orchestrator: %s", exc)


if __name__ == "__main__":
    main()
