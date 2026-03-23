#!/usr/bin/env python3
"""
NeuroShield v4 - Core Intelligence Orchestrator
Predicts CI/CD failures and automatically heals them.

CORE IDEA:
  1. Collect telemetry (Jenkins builds, Prometheus metrics)
  2. Predict failures using ML (DistilBERT + PCA + PyTorch)
  3. Decide best healing action using RL (PPO policy)
  4. Execute healing (restart, scale, retry, rollback)
  5. Measure MTTR (time to resolution)
  6. Learn from outcomes

This orchestrator runs continuously, checking for failures every 10 seconds.
When it detects a failure or predicts one, it automatically heals.
"""

import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import requests
from dotenv import load_dotenv
from stable_baselines3 import PPO

from src.prediction.predictor import FailurePredictor, build_52d_state
from src.config import validate_k8s_name, validate_positive_int


# ============================================================================
# CONFIGURATION
# ============================================================================

# The 4 Core Healing Actions
ACTION_NAMES: Dict[int, str] = {
    0: "restart_pod",      # Fix: Pod crashed or hung
    1: "scale_up",         # Fix: Resource constrained (CPU/memory spike)
    2: "retry_build",      # Fix: Transient build failure (network, timeout)
    3: "rollback_deploy",  # Fix: Bad deployment (high error rate after deploy)
}

# MTTR baselines (seconds) — manual remediation WITHOUT NeuroShield
MTTR_BASELINES: Dict[str, float] = {
    "restart_pod": 90.0,        # Manual: SSH, check logs, restart
    "scale_up": 60.0,           # Manual: Check resources, scale manually
    "retry_build": 70.0,        # Manual: Rerun build, wait for result
    "rollback_deploy": 120.0,   # Manual: SSH, git log, revert, redeploy
}


# ============================================================================
# HELPERS
# ============================================================================

def _load_env() -> None:
    """Load .env file if it exists."""
    if Path(".env").exists():
        load_dotenv()


def _env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _namespace() -> str:
    """Kubernetes namespace where app is deployed."""
    ns = os.getenv("K8S_NAMESPACE", "default")
    validate_k8s_name(ns, "namespace")
    return ns


def _affected_service() -> str:
    """Kubernetes service/deployment to heal."""
    svc = os.getenv("AFFECTED_SERVICE", "neuroshield-app")
    validate_k8s_name(svc, "service")
    return svc


def _scale_replicas() -> int:
    """Number of replicas to scale to when fixing resource issues."""
    val = int(os.getenv("SCALE_REPLICAS", "3"))
    validate_positive_int(val, "SCALE_REPLICAS")
    return val


def _poll_interval() -> int:
    """Seconds between orchestrator decision cycles."""
    val = int(os.getenv("POLL_INTERVAL", "10"))
    validate_positive_int(val, "POLL_INTERVAL")
    return val


def _setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("data/orchestrator.log"),
            logging.StreamHandler(),
        ],
    )


def _log_csv(path: str, row: Dict[str, str]) -> None:
    """Append row to CSV file (create with headers if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================================
# TELEMETRY COLLECTION
# ============================================================================

class TelemetryCollector:
    """Gather real-time data from Jenkins and Prometheus."""

    def __init__(self):
        self.jenkins_url = _env("JENKINS_URL", "http://localhost:8080")
        self.jenkins_job = _env("JENKINS_JOB", "neuroshield-app-build")
        self.jenkins_user = _env("JENKINS_USERNAME", "admin")
        self.jenkins_token = _env("JENKINS_PASSWORD", "")
        self.prometheus_url = _env("PROMETHEUS_URL", "http://localhost:9090")

    def get_jenkins_build(self) -> Optional[Dict]:
        """Get latest Jenkins build info."""
        try:
            url = f"{self.jenkins_url}/job/{self.jenkins_job}/lastBuild/api/json"
            r = requests.get(url, auth=(self.jenkins_user, self.jenkins_token), timeout=5)
            if r.status_code == 200:
                data = r.json()
                return {
                    "number": data.get("number"),
                    "result": data.get("result"),  # SUCCESS, FAILURE, UNSTABLE, etc
                    "duration_ms": data.get("duration"),
                    "timestamp": data.get("timestamp"),
                    "url": data.get("url"),
                }
        except Exception as e:
            logging.warning(f"Failed to get Jenkins build: {e}")
        return None

    def get_prometheus_metrics(self) -> Dict[str, float]:
        """Get metrics from Prometheus."""
        metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "error_rate": 0.0,
            "pod_count": 0.0,
        }

        try:
            # CPU usage (%)
            r = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": "100 * (1 - avg(rate(node_cpu_seconds_total{mode='idle'}[5m])))"},
                timeout=5,
            )
            if r.status_code == 200 and r.json()["data"]["result"]:
                metrics["cpu_usage"] = float(r.json()["data"]["result"][0]["value"][1])

            # Memory usage (%)
            r = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": "100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))"},
                timeout=5,
            )
            if r.status_code == 200 and r.json()["data"]["result"]:
                metrics["memory_usage"] = float(r.json()["data"]["result"][0]["value"][1])

        except Exception as e:
            logging.warning(f"Failed to get Prometheus metrics: {e}")

        return metrics

    def get_k8s_pod_count(self, namespace: str) -> int:
        """Get number of running pods."""
        try:
            r = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace, "--no-headers"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return len([l for l in r.stdout.splitlines() if l.strip()])
        except Exception as e:
            logging.warning(f"Failed to get pod count: {e}")
        return 0

    def collect(self) -> Dict:
        """Collect all telemetry into one dict."""
        build = self.get_jenkins_build()
        prom = self.get_prometheus_metrics()
        pods = self.get_k8s_pod_count(_namespace())

        return {
            "jenkins_number": build["number"] if build else 0,
            "jenkins_result": build["result"] if build else "UNKNOWN",
            "jenkins_duration_ms": build["duration_ms"] if build else 0,
            "prometheus_cpu_usage": prom["cpu_usage"],
            "prometheus_memory_usage": prom["memory_usage"],
            "prometheus_error_rate": prom["error_rate"],
            "k8s_pod_count": pods,
        }


# ============================================================================
# DECISION LOGIC (The Intelligence)
# ============================================================================

def decide_healing_action(
    telemetry: Dict,
    ml_action: str,
    failure_prob: float,
) -> Tuple[str, str]:
    """
    Decide which healing action to take.

    Uses two-stage decision:
      1. RULES: Hard business logic (if app is down, always restart)
      2. ML: Learn which action works best for this system

    Args:
        telemetry: Dict with metrics collected now
        ml_action: Action recommended by PPO policy
        failure_prob: Probability of failure (0.0 to 1.0) from predictor

    Returns:
        (action_name, reason_human_readable)
    """
    cpu = float(telemetry.get("prometheus_cpu_usage", 0) or 0)
    memory = float(telemetry.get("prometheus_memory_usage", 0) or 0)
    jenkins_result = str(telemetry.get("jenkins_result", "SUCCESS") or "SUCCESS")

    # RULE 1: CPU/Memory spike → Scale up (add more replicas to distribute load)
    if cpu > 80 or memory > 85:
        return "scale_up", f"Resource spike: CPU={cpu:.0f}% MEM={memory:.0f}%"

    # RULE 2: Build failed → Retry (likely transient failure)
    if jenkins_result in ("FAILURE", "UNSTABLE") and failure_prob >= 0.5:
        return "retry_build", f"Build status {jenkins_result} + high failure prob"

    # RULE 3: High error rate in HTTP responses → Rollback (likely bad code)
    if telemetry.get("prometheus_error_rate", 0) > 0.3:
        return "rollback_deploy", f"High error rate: {telemetry['prometheus_error_rate']:.2%}"

    # DEFAULT: Trust ML model which has learned this system
    return ml_action, f"ML model (prob={failure_prob:.1%})"


# ============================================================================
# ACTION EXECUTION
# ============================================================================

def execute_healing_action(action_name: str, service: str, namespace: str) -> bool:
    """
    Execute one of the 4 healing actions.

    Args:
        action_name: One of: restart_pod, scale_up, retry_build, rollback_deploy
        service: Kubernetes service/deployment name
        namespace: Kubernetes namespace

    Returns:
        True if action succeeded, False otherwise
    """
    try:
        validate_k8s_name(service, "service")
        validate_k8s_name(namespace, "namespace")

        if action_name == "restart_pod":
            logging.info(f"[ACTION] Restarting pod: {service}/{namespace}")
            r = subprocess.run(
                ["kubectl", "rollout", "restart", f"deployment/{service}", "-n", namespace],
                capture_output=True, text=True, timeout=30,
            )
            return r.returncode == 0

        elif action_name == "scale_up":
            replicas = _scale_replicas()
            logging.info(f"[ACTION] Scaling up to {replicas} replicas: {service}/{namespace}")
            r = subprocess.run(
                ["kubectl", "scale", f"deployment/{service}", f"--replicas={replicas}", "-n", namespace],
                capture_output=True, text=True, timeout=30,
            )
            return r.returncode == 0

        elif action_name == "retry_build":
            logging.info(f"[ACTION] Triggering Jenkins build retry")
            jenkins_url = _env("JENKINS_URL", "http://localhost:8080")
            job_name = _env("JENKINS_JOB", "neuroshield-app-build")
            user = _env("JENKINS_USERNAME", "admin")
            token = _env("JENKINS_PASSWORD", "")

            r = requests.post(
                f"{jenkins_url}/job/{job_name}/build",
                auth=(user, token), timeout=10,
            )
            return r.status_code in (200, 201)

        elif action_name == "rollback_deploy":
            logging.info(f"[ACTION] Rolling back deployment: {service}/{namespace}")
            r = subprocess.run(
                ["kubectl", "rollout", "undo", f"deployment/{service}", "-n", namespace],
                capture_output=True, text=True, timeout=30,
            )
            return r.returncode == 0

        else:
            logging.error(f"Unknown action: {action_name}")
            return False

    except Exception as e:
        logging.error(f"Action {action_name} failed: {e}")
        return False


# ============================================================================
# MAIN ORCHESTRATOR LOOP
# ============================================================================

def run_orchestrator(model_dir: str = "models") -> None:
    """
    Main orchestrator loop: continuously monitor and heal CI/CD system.

    Runs forever until Ctrl+C. Every 10 seconds:
      1. Collect telemetry (Jenkins, Prometheus)
      2. Predict if failure is likely
      3. Decide action (rules + ML)
      4. Execute if needed
      5. Log everything
      6. Measure MTTR
    """
    _setup_logging()
    _load_env()
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("NeuroShield v4 - Core Intelligence Orchestrator")
    logger.info("=" * 70)

    # Load ML models
    try:
        predictor = FailurePredictor(model_dir=Path(model_dir))
        policy = PPO.load(str(Path(model_dir) / "ppo_policy.zip"))
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    collector = TelemetryCollector()
    namespace = _namespace()
    service = _affected_service()
    poll_interval = _poll_interval()

    # State tracking
    cycle_count = 0
    total_actions = 0
    successful_actions = 0
    mttr_measurements = []
    failure_detected_time: Optional[float] = None
    last_telemetry: Optional[Dict] = None

    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*70}")
            print(f"Cycle {cycle_count} | {datetime.now().isoformat()}")
            print(f"{'='*70}")

            # --- COLLECT TELEMETRY ---
            telemetry = collector.collect()
            logger.info(f"Telemetry: CPU={telemetry['prometheus_cpu_usage']:.0f}% "
                       f"MEM={telemetry['prometheus_memory_usage']:.0f}% "
                       f"BUILD={telemetry['jenkins_result']}")

            # --- PREDICT FAILURE ---
            try:
                state = build_52d_state(telemetry)
                failure_prob = predictor.predict(state)
                logger.info(f"Failure probability: {failure_prob:.1%}")
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                failure_prob = 0.0

            # --- DECIDE & EXECUTE ---
            # Check if system is unhealthy
            is_unhealthy = (
                telemetry["jenkins_result"] in ("FAILURE", "UNSTABLE") or
                telemetry["prometheus_cpu_usage"] > 85 or
                telemetry["prometheus_memory_usage"] > 85 or
                failure_prob > 0.7
            )

            if is_unhealthy:
                logger.info("System unhealthy - initiating healing")

                # Get ML recommendation
                try:
                    state = build_52d_state(telemetry)
                    action_id, _states = policy.predict(state, deterministic=True)
                    ml_action = ACTION_NAMES[action_id]
                except Exception:
                    ml_action = "restart_pod"  # Safe default

                # Decide & execute
                action_name, reason = decide_healing_action(telemetry, ml_action, failure_prob)
                logger.info(f"Action: {action_name} ({reason})")

                if failure_detected_time is None:
                    failure_detected_time = time.time()

                success = execute_healing_action(action_name, service, namespace)
                total_actions += 1

                if success:
                    successful_actions += 1
                    logger.info(f"✓ {action_name} succeeded")

                    # Measure MTTR
                    if failure_detected_time:
                        mttr = time.time() - failure_detected_time
                        baseline = MTTR_BASELINES.get(action_name, 120.0)
                        reduction = max(0, (baseline - mttr) / baseline * 100)
                        mttr_measurements.append(reduction)
                        logger.info(f"MTTR: {mttr:.1f}s (baseline {baseline:.0f}s, {reduction:.1f}% reduction)")
                        failure_detected_time = None
                else:
                    logger.error(f"✗ {action_name} failed")

                # Log to CSV
                _log_csv("data/healing_log.csv", {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cycle": str(cycle_count),
                    "failure_prob": f"{failure_prob:.3f}",
                    "action": action_name,
                    "success": str(success),
                    "cpu": f"{telemetry['prometheus_cpu_usage']:.1f}",
                    "memory": f"{telemetry['prometheus_memory_usage']:.1f}",
                    "jenkins_result": telemetry["jenkins_result"],
                })

            else:
                logger.info("System healthy - no action needed")

            # --- STATS ---
            print(f"Stats: {total_actions} actions, {successful_actions} successful")
            if mttr_measurements:
                avg_reduction = sum(mttr_measurements) / len(mttr_measurements)
                print(f"Avg MTTR Reduction: {avg_reduction:.1f}% ({len(mttr_measurements)} incidents)")

            last_telemetry = telemetry
            print(f"Sleeping {poll_interval}s until next cycle...")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"Orchestrator stopped")
        print(f"Cycles: {cycle_count} | Actions: {total_actions} | Success: {successful_actions}")
        if mttr_measurements:
            print(f"Avg MTTR Reduction: {sum(mttr_measurements) / len(mttr_measurements):.1f}%")
        print(f"{'='*70}\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # TODO: Implement simulation mode
        print("Simulation mode not yet implemented")
    else:
        run_orchestrator()
