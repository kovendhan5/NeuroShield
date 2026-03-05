"""NeuroShield Orchestrator - Real-time CI/CD Monitoring."""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, TypeVar

# Ensure project root is on sys.path for direct invocation
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import requests
from dotenv import load_dotenv
from stable_baselines3 import PPO

from src.prediction.predictor import FailurePredictor, build_52d_state

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
    0: "retry_stage",
    1: "clean_and_rerun",
    2: "regenerate_config",
    3: "reallocate_resources",
    4: "trigger_safe_rollback",
    5: "escalate_to_human",
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


def _append_csv(path: str, row: Dict[str, str]) -> None:
    """Append a row to a CSV file, creating with headers if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _log_action_history(action_id: int, success: bool, duration_ms: float) -> None:
    """Record every healing action to data/action_history.csv."""
    _append_csv("data/action_history.csv", {
        "timestamp": datetime.utcnow().isoformat(),
        "action_id": str(action_id),
        "action_name": ACTION_NAMES.get(action_id, "unknown"),
        "success": str(success),
        "duration_ms": f"{duration_ms:.0f}",
    })


def execute_healing_action(action_id: int, context: Dict[str, str]) -> bool:
    """Execute one of 6 healing actions based on PPO decision (with retry).

    Actions:
        0 — retry_stage:          Re-trigger Jenkins build
        1 — clean_and_rerun:      Trigger clean build (CLEAN_WORKSPACE=true)
        2 — regenerate_config:    Flag for manual config review
        3 — reallocate_resources: Patch K8s deployment resource limits
        4 — trigger_safe_rollback: kubectl rollout undo + wait for status
        5 — escalate_to_human:    Write alert to escalation log
    """
    start_ms = time.time() * 1000.0
    success = False
    try:
        namespace = _namespace()
        service = context.get("affected_service", _affected_service())

        if action_id == 0:  # retry_stage
            logging.info("[ACTION] retry_stage — build #%s", context.get("build_number", "?"))
            jenkins_url = _required_env("JENKINS_URL")
            job_name = _required_env("JENKINS_JOB")
            username = _required_env("JENKINS_USERNAME")
            token = _required_env("JENKINS_TOKEN")
            build_url = f"{jenkins_url}/job/{job_name}/build"

            def _trigger() -> bool:
                r = requests.post(build_url, auth=(username, token), timeout=15)
                if r.status_code not in {200, 201, 202}:
                    raise RuntimeError(f"Jenkins trigger failed: {r.status_code}")
                return True

            success = retry_call(_trigger)

        elif action_id == 1:  # clean_and_rerun
            logging.info("[ACTION] clean_and_rerun — build #%s", context.get("build_number", "?"))
            jenkins_url = _required_env("JENKINS_URL")
            job_name = _required_env("JENKINS_JOB")
            username = _required_env("JENKINS_USERNAME")
            token = _required_env("JENKINS_TOKEN")
            build_url = f"{jenkins_url}/job/{job_name}/buildWithParameters"

            def _clean_build() -> bool:
                r = requests.post(
                    build_url,
                    auth=(username, token),
                    params={"CLEAN_WORKSPACE": "true"},
                    timeout=15,
                )
                if r.status_code not in {200, 201, 202}:
                    # Fallback: plain build if parameterised trigger unsupported
                    r2 = requests.post(
                        f"{jenkins_url}/job/{job_name}/build",
                        auth=(username, token),
                        timeout=15,
                    )
                    if r2.status_code not in {200, 201, 202}:
                        raise RuntimeError(f"Clean build trigger failed: {r2.status_code}")
                return True

            success = retry_call(_clean_build)
            logging.info("[ACTION] Clean & Re-run triggered")

        elif action_id == 2:  # regenerate_config
            logging.info("[ACTION] Config regeneration flagged — manual review needed")
            _append_csv("data/config_regen_log.csv", {
                "timestamp": datetime.utcnow().isoformat(),
                "job_name": context.get("build_number", "unknown"),
                "reason": context.get("failure_pattern", "unknown"),
                "action_taken": "regenerate_config — manual review required",
            })
            success = True

        elif action_id == 3:  # reallocate_resources
            logging.info("[ACTION] reallocate_resources — %s in %s", service, namespace)
            patch_json = json.dumps({
                "spec": {"template": {"spec": {"containers": [{
                    "name": service,
                    "resources": {"limits": {"cpu": "500m", "memory": "512Mi"}},
                }]}}}
            })
            cmd = [
                "kubectl", "patch", "deployment", service,
                "-n", namespace,
                "--type=strategic",
                f"--patch={patch_json}",
            ]

            def _patch() -> bool:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True

            success = retry_call(_patch)

        elif action_id == 4:  # trigger_safe_rollback
            logging.info("[ACTION] trigger_safe_rollback — %s in %s", service, namespace)
            undo_cmd = ["kubectl", "rollout", "undo", f"deploy/{service}", "-n", namespace]
            status_cmd = ["kubectl", "rollout", "status", f"deploy/{service}", "-n", namespace, "--timeout=60s"]

            def _rollback_and_wait() -> bool:
                subprocess.run(undo_cmd, capture_output=True, text=True, check=True)
                subprocess.run(status_cmd, capture_output=True, text=True, check=True)
                return True

            success = retry_call(_rollback_and_wait)

        elif action_id == 5:  # escalate_to_human
            logging.info("[ACTION] Escalated to human review — check data/escalation_log.csv")
            _append_csv("data/escalation_log.csv", {
                "timestamp": datetime.utcnow().isoformat(),
                "failure_probability": context.get("failure_prob", "?"),
                "failure_state": context.get("failure_pattern", "unknown"),
                "recommended_action": "escalate_to_human",
                "status": "PENDING_HUMAN_REVIEW",
            })
            success = True

        else:
            logging.warning("Unknown action id: %s", action_id)

    except Exception as exc:
        logging.exception("Error executing action %s: %s", action_id, exc)

    duration_ms = time.time() * 1000.0 - start_ms
    _log_action_history(action_id, success, duration_ms)
    return success


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

                    # --- 24D telemetry dict for the failure classifier ---
                    telemetry_dict = {
                        "jenkins_last_build_status": build.result,
                        "jenkins_last_build_duration": build.duration_ms,
                        "jenkins_queue_length": 0,
                        "prometheus_cpu_usage": resource_metrics["cpu_millicores_avg"],
                        "prometheus_memory_usage": resource_metrics["memory_mib_avg"],
                        "prometheus_pod_count": resource_metrics["pod_count"],
                        "prometheus_error_rate": resource_metrics["error_rate"],
                    }
                    failure_prob = predictor.predict(log_text, telemetry_dict)

                    # --- 52D state vector for the PPO policy ---
                    jenkins_data = {
                        "build_duration": build.duration_ms,
                        "build_number": build.number,
                        "retry_count": 0,
                    }
                    prometheus_data = {
                        "cpu_avg_5m": resource_metrics["cpu_millicores_avg"],
                        "memory_avg_5m": resource_metrics["memory_mib_avg"],
                        "pod_restarts": 0,
                        "node_count": resource_metrics["pod_count"],
                    }
                    state_52d = build_52d_state(
                        jenkins_data, prometheus_data, log_text, predictor.encoder,
                    )

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
                            action, _ = policy.predict(state_52d, deterministic=True)
                            action_id = int(action)
                        if pattern_action is not None:
                            action_id = pattern_action
                        logging.info("NeuroShield decision: %s", ACTION_NAMES.get(action_id, str(action_id)))

                        success = execute_healing_action(
                            action_id,
                            {
                                "build_number": str(build.number),
                                "affected_service": _affected_service(),
                                "failure_prob": f"{failure_prob:.3f}",
                                "failure_pattern": pattern or "none",
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


# ---------------------------------------------------------------------------
# Simulate mode – one-shot simulated decision (no Jenkins / K8s needed)
# ---------------------------------------------------------------------------

def run_once(model_dir: str = "models") -> None:
    """Run one end-to-end simulated decision (no live infra required)."""
    import random
    from src.prediction.data_generator import generate_sample
    from src.rl_agent.simulator import simulate_action

    _setup_logging()
    model_path = Path(model_dir)
    predictor = FailurePredictor(model_dir=model_path)

    try:
        policy = PPO.load(str(model_path / "ppo_policy.zip"))
    except Exception:
        policy = None
        logging.warning("PPO policy not found; falling back to retry_stage")

    sample = generate_sample(random.Random(42))
    log_text = sample.log_text
    failure_prob = predictor.predict(log_text, sample.telemetry)

    jenkins_data = {"build_duration": 60_000, "build_number": 0}
    prometheus_data = {"cpu_avg_5m": 0.5, "memory_avg_5m": 256}
    state_52d = build_52d_state(jenkins_data, prometheus_data, log_text, predictor.encoder)

    if policy is None:
        action = 0
    else:
        action, _ = policy.predict(state_52d, deterministic=True)
        action = int(action)

    result = simulate_action(sample.failure_type, action, random.Random(123))
    baseline = simulate_action(sample.failure_type, 0, random.Random(456))
    mttr_reduction = max(0.0, (baseline.mttr - result.mttr) / max(baseline.mttr, 1.0))

    logging.info("Failure probability: %.3f", failure_prob)
    logging.info("Chosen action: %s", ACTION_NAMES.get(action, str(action)))
    logging.info("Simulated MTTR: %.1fs (baseline retry %.1fs)", result.mttr, baseline.mttr)
    logging.info("MTTR reduction: %.1f%%", mttr_reduction * 100.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuroShield Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["live", "simulate"],
        default="live",
        help="'live' monitors Jenkins in real-time; 'simulate' runs one offline decision",
    )
    cli_args = parser.parse_args()

    if cli_args.mode == "simulate":
        run_once()
    else:
        main()
