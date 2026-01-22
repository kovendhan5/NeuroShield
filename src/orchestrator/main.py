"""NeuroShield Orchestrator - Real-time CI/CD Monitoring."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from stable_baselines3 import PPO

from src.prediction.predictor import FailurePredictor

ACTION_NAMES: Dict[int, str] = {
    0: "Retry",
    1: "Scale Pods",
    2: "Rollback",
    3: "No-op",
}


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
    """Get the latest build information from Jenkins."""
    url = f"{jenkins_url}/job/{job_name}/lastBuild/api/json"
    auth = (username, token)

    try:
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
    except Exception as exc:
        logging.exception("Exception in get_latest_build_info: %s", exc)
        return None


def get_build_log(jenkins_url: str, job_name: str, build_number: int, username: str, token: str) -> str:
    """Get the console log for a specific build."""
    url = f"{jenkins_url}/job/{job_name}/{build_number}/consoleText"
    auth = (username, token)

    try:
        response = requests.get(url, auth=auth, timeout=30)
        if response.status_code == 200:
            return response.text
        logging.warning("Error fetching build log: %s", response.status_code)
        return ""
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


def execute_healing_action(action_id: int, context: Dict[str, str]) -> bool:
    """Execute the healing action based on PPO decision."""
    try:
        if action_id == 0:  # Retry
            logging.info("Executing action: Retry build #%s", context.get("build_number", "unknown"))
            jenkins_url = _required_env("JENKINS_URL")
            job_name = _required_env("JENKINS_JOB")
            username = _required_env("JENKINS_USERNAME")
            token = _required_env("JENKINS_TOKEN")
            retry_url = f"{jenkins_url}/job/{job_name}/build"
            response = requests.post(retry_url, auth=(username, token), timeout=15)
            return response.status_code in {200, 201, 202}

        if action_id == 1:  # Scale pods
            service = context.get("affected_service", "carts")
            logging.info("Executing action: Scale pods for %s", service)
            cmd = ["kubectl", "scale", f"deploy/{service}", "--replicas=3", "-n", "sock-shop"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logging.error("kubectl scale failed: %s", result.stderr.strip())
            return result.returncode == 0

        if action_id == 2:  # Rollback
            service = context.get("affected_service", "carts")
            logging.info("Executing action: Rollback deployment for %s", service)
            cmd = ["kubectl", "rollout", "undo", f"deploy/{service}", "-n", "sock-shop"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logging.error("kubectl rollback failed: %s", result.stderr.strip())
            return result.returncode == 0

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
    telemetry_path = os.getenv("TELEMETRY_OUTPUT")

    logging.info("Connecting to Jenkins: %s", jenkins_url)
    logging.info("Monitoring job: %s", job_name)

    predictor = FailurePredictor(model_dir="models")
    policy = PPO.load("models/ppo_policy.zip")

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
                    failure_prob, state_vector = predictor.predict_with_state(log_text, telemetry_path)
                    pattern, pattern_action = detect_failure_pattern(log_text)
                    logging.info(
                        "Failure probability: %.3f | pattern=%s | build_url=%s",
                        failure_prob,
                        pattern,
                        build.url,
                    )

                    if failure_prob > 0.5:
                        action, _ = policy.predict(np.asarray(state_vector, dtype=np.float32), deterministic=True)
                        action_id = int(action)
                        if pattern_action is not None:
                            action_id = pattern_action
                        logging.info("NeuroShield decision: %s", ACTION_NAMES.get(action_id, str(action_id)))

                        success = execute_healing_action(
                            action_id,
                            {
                                "build_number": str(build.number),
                                "affected_service": os.getenv("AFFECTED_SERVICE", "carts"),
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
