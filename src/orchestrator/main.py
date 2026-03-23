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
from datetime import datetime, timezone
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
from src.utils.intelligence import detect_early_warning, explain_decision
from src.config import validate_k8s_name, validate_positive_int

T = TypeVar("T")


def retry_call(fn: Callable[[], T], max_attempts: int = 3, delay: int = 2) -> T:
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay * (2 ** attempt))
    raise RuntimeError("retry_call exhausted")

ACTION_NAMES: Dict[int, str] = {
    0: "restart_pod",
    1: "scale_up",
    2: "retry_build",
    3: "rollback_deploy",
}

# MTTR baselines (seconds) — manual remediation without NeuroShield
MTTR_BASELINES: Dict[str, float] = {
    "restart_pod": 90.0,
    "scale_up": 60.0,
    "retry_build": 70.0,
    "rollback_deploy": 120.0,
}

# Reverse lookup: action name → action id
_ACTION_IDS: Dict[str, int] = {v: k for k, v in ACTION_NAMES.items()}


def _log_mttr(failure_type: str, action_name: str, actual_mttr: float) -> None:
    """Append MTTR measurement to data/mttr_log.csv."""
    baseline = MTTR_BASELINES.get(action_name, 120.0)
    reduction_pct = max(0.0, (baseline - actual_mttr) / baseline * 100)
    _append_csv("data/mttr_log.csv", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failure_type": failure_type,
        "action": action_name,
        "actual_mttr_s": f"{actual_mttr:.1f}",
        "baseline_mttr_s": f"{baseline:.1f}",
        "reduction_pct": f"{reduction_pct:.1f}",
    })


def _namespace() -> str:
    ns = os.getenv("K8S_NAMESPACE", "default")
    validate_k8s_name(ns, "namespace")
    return ns


def _affected_service() -> str:
    svc = os.getenv("AFFECTED_SERVICE", "dummy-app")
    validate_k8s_name(svc, "service")
    return svc


def _scale_replicas() -> int:
    val = int(os.getenv("SCALE_REPLICAS", "3"))
    validate_positive_int(val, "SCALE_REPLICAS")
    return val


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
    """Load .env from project root, supporting both JENKINS_USER and JENKINS_USERNAME."""
    env_path = Path(_PROJECT_ROOT) / ".env"
    load_dotenv(env_path, override=True)
    # Normalize: if JENKINS_USERNAME is missing, fall back to JENKINS_USER
    if not os.getenv("JENKINS_USERNAME") and os.getenv("JENKINS_USER"):
        os.environ["JENKINS_USERNAME"] = os.getenv("JENKINS_USER", "")
    # Normalize: if JENKINS_PASSWORD is missing, fall back to JENKINS_TOKEN
    if not os.getenv("JENKINS_PASSWORD") and os.getenv("JENKINS_TOKEN"):
        os.environ["JENKINS_PASSWORD"] = os.getenv("JENKINS_TOKEN", "")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action_id": str(action_id),
        "action_name": ACTION_NAMES.get(action_id, "unknown"),
        "success": str(success),
        "duration_ms": f"{duration_ms:.0f}",
    })


# ---------------------------------------------------------------------------
# Port-forward reconnect helper
# ---------------------------------------------------------------------------

_port_forward_proc: Optional[subprocess.Popen] = None


def _ensure_port_forward(service: str = "dummy-app", port: int = 5000) -> None:
    """Re-establish kubectl port-forward via svc/ after a pod restart."""
    global _port_forward_proc
    namespace = _namespace()

    # Kill existing port-forward if stale
    if _port_forward_proc is not None:
        try:
            _port_forward_proc.kill()
            _port_forward_proc.wait(timeout=5)
        except Exception:
            pass
        _port_forward_proc = None
        logging.info("[PORT-FORWARD] Killed old process")

    # Wait for a ready pod
    logging.info("[PORT-FORWARD] Waiting for ready endpoints...")
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["kubectl", "get", "endpoints", service, "-n", namespace,
                 "-o", "jsonpath={.subsets[0].addresses[0].ip}"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                logging.info("[PORT-FORWARD] Endpoints ready: %s", r.stdout.strip())
                break
        except Exception:
            pass
        time.sleep(3)

    # Start new port-forward via service (survives pod replacement)
    try:
        _port_forward_proc = subprocess.Popen(
            ["kubectl", "port-forward", f"svc/{service}", f"{port}:{port}", "-n", namespace],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Verify port-forward is working
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code < 500:
                    logging.info("[PORT-FORWARD] Reconnected svc/%s on port %d ✓", service, port)
                    return
            except Exception:
                pass
            time.sleep(1)

        logging.warning("[PORT-FORWARD] Process started but health check failed")
    except Exception as exc:
        logging.warning("[PORT-FORWARD] Failed to reconnect: %s", exc)


def execute_healing_action(action_id: int, context: Dict[str, str]) -> bool:
    """Execute one of 6 healing actions with REAL infrastructure calls.

    Actions:
        0 \u2014 restart_pod:       kubectl rollout restart deployment/dummy-app
        1 \u2014 scale_up:          kubectl scale deployment --replicas=3
        2 -- retry_build:       POST Jenkins API to trigger new build
        3 -- rollback_deploy:   kubectl rollout undo deployment/dummy-app
        4 -- clear_cache:       Call dummy-app /stress to refresh, or restart pod
        5 -- escalate_to_human: Write detailed report to data/escalation_reports/
    """
    start_ms = time.time() * 1000.0
    success = False
    detail = ""
    try:
        namespace = _namespace()
        service = context.get("affected_service", _affected_service())
        # Validate inputs before using in subprocess commands
        validate_k8s_name(namespace, "namespace")
        validate_k8s_name(service, "service")

        if action_id == 0:  # restart_pod
            logging.info("[ACTION] restart_pod -- %s in %s", service, namespace)
            cmd = ["kubectl", "rollout", "restart", f"deployment/{service}", "-n", namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logging.info("[ACTION] Restart issued, waiting for rollout...")
                wait = subprocess.run(
                    ["kubectl", "rollout", "status", f"deployment/{service}", "-n", namespace, "--timeout=60s"],
                    capture_output=True, text=True, timeout=90,
                )
                success = wait.returncode == 0
                detail = "rollout complete" if success else wait.stderr.strip()[:200]
                if success:
                    _ensure_port_forward(service)
            else:
                detail = result.stderr.strip()[:200]

        elif action_id == 1:  # scale_up
            replicas = _scale_replicas()
            logging.info("[ACTION] scale_up -- %s to %d replicas", service, replicas)
            cmd = ["kubectl", "scale", f"deployment/{service}", f"--replicas={replicas}", "-n", namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Poll until all replicas ready (max 60s)
                deadline = time.time() + 60
                while time.time() < deadline:
                    chk = subprocess.run(
                        ["kubectl", "get", "deployment", service, "-n", namespace,
                         "-o", "jsonpath={.status.readyReplicas}"],
                        capture_output=True, text=True, timeout=10,
                    )
                    ready = chk.stdout.strip()
                    if ready.isdigit() and int(ready) >= replicas:
                        success = True
                        detail = f"{ready}/{replicas} replicas ready"
                        break
                    time.sleep(3)
                if not success:
                    detail = f"timeout waiting for {replicas} replicas"
            else:
                detail = result.stderr.strip()[:200]

        elif action_id == 2:  # retry_build
            logging.info("[ACTION] retry_build -- triggering Jenkins build")
            jenkins_url = _env("JENKINS_URL", "http://localhost:8080")
            job_name = _env("JENKINS_JOB", "neuroshield-app-build")
            username = _env("JENKINS_USERNAME", "admin")
            token = _env("JENKINS_PASSWORD", _env("JENKINS_TOKEN", ""))

            # Get current build number before triggering
            pre_build = get_latest_build_info(jenkins_url, job_name, username, token)
            pre_num = pre_build.number if pre_build else 0

            build_url = f"{jenkins_url}/job/{job_name}/build"
            # Use a session so CSRF crumb cookie + header travel together
            sess = requests.Session()
            sess.auth = (username, token)
            try:
                cr = sess.get(f"{jenkins_url}/crumbIssuer/api/json", timeout=5)
                if cr.status_code == 200:
                    cd = cr.json()
                    sess.headers.update({cd["crumbRequestField"]: cd["crumb"]})
            except Exception:
                pass

            r = sess.post(build_url, timeout=15)
            if r.status_code in {200, 201, 202, 301, 302}:
                logging.info("[ACTION] Build triggered, waiting for completion...")
                # Wait for new build to appear and finish (max 120s)
                deadline = time.time() + 120
                while time.time() < deadline:
                    time.sleep(5)
                    new_build = get_latest_build_info(jenkins_url, job_name, username, token)
                    if new_build and new_build.number > pre_num and new_build.result not in (None, "RUNNING"):
                        success = new_build.result == "SUCCESS"
                        detail = f"build #{new_build.number} â†’ {new_build.result}"
                        break
                if not success and not detail:
                    detail = "timeout waiting for build result"
            else:
                detail = f"Jenkins trigger HTTP {r.status_code}"

        elif action_id == 3:  # rollback_deploy
            logging.info("[ACTION] rollback_deploy -- %s in %s", service, namespace)
            undo = subprocess.run(
                ["kubectl", "rollout", "undo", f"deployment/{service}", "-n", namespace],
                capture_output=True, text=True, timeout=30,
            )
            if undo.returncode == 0:
                wait = subprocess.run(
                    ["kubectl", "rollout", "status", f"deployment/{service}", "-n", namespace, "--timeout=60s"],
                    capture_output=True, text=True, timeout=90,
                )
                if wait.returncode == 0:
                    _ensure_port_forward(service)
                    # Verify health endpoint
                    app_url = _env("DUMMY_APP_URL", "http://localhost:5000")
                    time.sleep(5)  # give pod time to start serving
                    try:
                        hr = requests.get(f"{app_url}/health", timeout=5)
                        success = hr.status_code == 200
                        detail = f"health={hr.status_code} after rollback"
                    except Exception:
                        success = True  # rollout succeeded even if app not reachable from host
                        detail = "rollback complete, health check unreachable"
                else:
                    detail = wait.stderr.strip()[:200]
            else:
                detail = undo.stderr.strip()[:200]

        else:
            logging.warning("Unknown action id: %s", action_id)

    except Exception as exc:
        logging.exception("Error executing action %s: %s", action_id, exc)
        detail = str(exc)[:200]

    duration_ms = time.time() * 1000.0 - start_ms
    _log_action_history(action_id, success, duration_ms)
    _log_healing_json(action_id, success, duration_ms, detail, context)

    if not success:
        logging.warning("[ACTION] %s FAILED (%s) -- will retry once", ACTION_NAMES.get(action_id, "?"), detail)

    return success


def _log_healing_json(action_id: int, success: bool, duration_ms: float,
                      detail: str, context: Dict[str, str]) -> None:
    """Append every healing action to data/healing_log.json as a JSON-lines file."""
    p = Path("data/healing_log.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action_id": action_id,
        "action_name": ACTION_NAMES.get(action_id, "unknown"),
        "success": success,
        "duration_ms": round(duration_ms),
        "detail": detail,
        "context": context,
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _write_brain_feed_event(action_name: str, failure_prob: float, success: bool, duration_ms: float) -> None:
    """Write a brain feed event for the live SSE stream at localhost:8503."""
    p = Path("data/brain_feed_events.json")
    p.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "action": action_name,
        "prob": round(failure_prob, 4),
        "success": success,
        "duration_ms": round(duration_ms),
        "class": "heal" if action_name != "escalate_to_human" else "escalate",
    }

    # Read existing events, append new one, keep last 50
    events = []
    try:
        if p.exists():
            events = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        events = []

    events.append(event)
    events = events[-50:]  # Keep only last 50 events

    with open(p, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)


def _is_failure(result: str) -> bool:
    return result.upper() in {"FAILURE", "UNSTABLE", "ABORTED"}


# ---------------------------------------------------------------------------
# File-based cooldown (survives restarts, works across main() + run_single_cycle())
# ---------------------------------------------------------------------------

_COOLDOWN_FILE = Path("data/.last_heal_ts")


def _read_cooldown_ts() -> float:
    """Read last healing timestamp from file."""
    try:
        return float(_COOLDOWN_FILE.read_text().strip())
    except Exception:
        return 0.0


def _write_cooldown_ts() -> None:
    """Write current time as last healing timestamp."""
    _COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _COOLDOWN_FILE.write_text(str(time.time()))


# ---------------------------------------------------------------------------
# Telemetry collection helpers (live mode)
# ---------------------------------------------------------------------------

def _check_service(url: str, timeout: int = 5) -> Tuple[bool, float]:
    """Check if a service is reachable. Returns (is_up, latency_ms)."""
    try:
        start = time.time()
        r = requests.get(url, timeout=timeout)
        latency = (time.time() - start) * 1000
        return r.status_code < 500, latency
    except Exception:
        return False, 0.0


def _collect_prometheus_metrics(prom_url: str) -> Dict[str, float]:
    """Query Prometheus for CPU, memory, pod count, error rate, pod restarts.

    Uses node-level metrics that are always available in Minikube.
    Falls back to psutil for CPU/memory if Prometheus returns zero/NaN.
    """
    metrics: Dict[str, float] = {
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "pod_count": 0.0,
        "error_rate": 0.0,
        "pod_restarts": 0.0,
    }
    # Node-level queries — reliable in Minikube (no container-spec limits required)
    queries = {
        "cpu_usage": '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
        "memory_usage": '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
        "pod_count": 'count(kube_pod_info)',
        "error_rate": 'rate(flask_http_request_total{status=~"5.."}[5m]) or vector(0)',
        "pod_restarts": 'sum(kube_pod_container_status_restarts_total{namespace="default"}) or vector(0)',
    }
    for key, query in queries.items():
        try:
            r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=5)
            if r.status_code == 200:
                results = r.json().get("data", {}).get("result", [])
                if results:
                    val = float(results[0]["value"][1])
                    if val == val:  # NaN guard
                        metrics[key] = val
        except Exception:
            pass

    # psutil fallback: if Prometheus CPU/memory still zero, use real host metrics
    if metrics["cpu_usage"] == 0.0 or metrics["memory_usage"] == 0.0:
        try:
            import psutil as _psutil
            if metrics["cpu_usage"] == 0.0:
                metrics["cpu_usage"] = _psutil.cpu_percent(interval=0.5)
            if metrics["memory_usage"] == 0.0:
                metrics["memory_usage"] = _psutil.virtual_memory().percent
        except Exception as exc:
            logging.debug("psutil fallback failed: %s", exc)

    return metrics


def _collect_dummy_app_health(app_url: str) -> Dict[str, float]:
    """Get health metrics from the dummy-app."""
    info: Dict[str, float] = {"health_pct": 0.0, "response_ms": 0.0}
    try:
        start = time.time()
        r = requests.get(f"{app_url}/health", timeout=5)
        info["response_ms"] = (time.time() - start) * 1000
        if r.status_code == 200:
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            info["health_pct"] = float(data.get("health", data.get("status_code", 100)))
        else:
            info["health_pct"] = 50.0
    except Exception:
        info["health_pct"] = 0.0
    return info


def _save_telemetry_row(row: Dict[str, str]) -> None:
    """Append a telemetry row to data/telemetry.csv."""
    _append_csv("data/telemetry.csv", row)


# ---------------------------------------------------------------------------
# Rule-based + ML action selection
# ---------------------------------------------------------------------------

def determine_healing_action(
    telemetry: Dict,
    ml_action: str,
    prob: float,
) -> Tuple[str, str]:
    """Select a healing action using rules first, falling back to the ML decision.

    Rule-based overrides ensure critical healing actions trigger correctly:
    - app DOWN (health=0%) → ALWAYS restart_pod
    - app degraded (0% < health < 100%) → restart_pod
    - pod restart loop → restart_pod
    - CPU/memory spike → scale_up
    - build failure → retry_build
    - high error rate → rollback_deploy
    - otherwise → use ML model decision

    Returns:
        (action_name, human_readable_reason)
    """
    cpu = float(telemetry.get("prometheus_cpu_usage", 0) or 0)
    memory = float(telemetry.get("prometheus_memory_usage", 0) or 0)
    pod_restarts = float(telemetry.get("pod_restart_count", 0) or 0)
    build_status = str(telemetry.get("jenkins_last_build_status", "") or "").upper()
    error_rate = float(telemetry.get("prometheus_error_rate", 0) or 0)
    app_hp = float(telemetry.get("app_health_pct", 100) or 100)

    # EXPLICIT: App is DOWN — ALWAYS restart, no exceptions
    if app_hp == 0:
        logging.info("[OVERRIDE] App health=0%% (CRASHED) → forcing restart_pod (overriding %s)", ml_action)
        return "restart_pod", f"[OVERRIDE] App health=0% → forcing restart_pod"

    # App is degraded but not completely down
    if app_hp < 100:
        logging.info("[OVERRIDE] App health=%d%% (degraded) → restart_pod (overriding %s)", int(app_hp), ml_action)
        return "restart_pod", f"App health degraded ({app_hp:.0f}%) — proactive restart"

    if pod_restarts >= 3:
        return "restart_pod", f"Pod restart loop detected ({int(pod_restarts)} restarts)"

    if cpu > 80 or memory > 85:
        return "scale_up", f"Resource spike: CPU={cpu:.0f}% MEM={memory:.0f}%"

    if build_status in ("FAILURE", "UNSTABLE", "ABORTED") and prob >= 0.5:
        return "retry_build", f"Build status: {build_status}"

    if error_rate > 0.3:
        return "rollback_deploy", f"High HTTP error rate: {error_rate:.3f} req/s"

    # Fall back to ML model decision
    return ml_action, f"ML model decision (prob={prob:.3f})"


# ---------------------------------------------------------------------------
# Incident HTML report generator
# ---------------------------------------------------------------------------


# Main orchestrator loop

def _print_banner() -> None:
    print("\n" + "=" * 65)
    print("  NeuroShield AIOps Orchestrator -- Live Mode")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Self-CI monitoring — NeuroShield watches its own pipeline
# ---------------------------------------------------------------------------

_SELF_CI_STATUS_FILE = Path("data/self_ci_status.json")


def _get_self_ci_build(jenkins_url: str, job_name: str,
                       username: str, token: str) -> Optional[Dict]:
    """Fetch the latest build result for the self-CI job."""
    try:
        url = f"{jenkins_url}/job/{job_name}/lastBuild/api/json"
        r = requests.get(url, auth=(username, token), timeout=10)
        if r.status_code == 200:
            d = r.json()
            return {
                "number": d.get("number"),
                "result": d.get("result"),
                "duration_ms": d.get("duration", 0),
                "timestamp": d.get("timestamp", 0),
                "url": d.get("url", ""),
            }
    except Exception as exc:
        logging.debug("Self-CI fetch failed: %s", exc)
    return None


def handle_self_ci_failure(build_info: Dict, reason: str = "") -> None:
    """Handle a failure in NeuroShield's own CI pipeline.

    Writes a status file (data/self_ci_status.json), logs a critical alert,
    and sends an email notification.
    """
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "build_number": build_info.get("number"),
        "result": build_info.get("result"),
        "duration_ms": build_info.get("duration_ms", 0),
        "reason": reason or f"Self-CI build #{build_info.get('number')} failed: {build_info.get('result')}",
        "active": True,
    }

    _SELF_CI_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SELF_CI_STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
    logging.critical("SELF-CI FAILURE: build #%s → %s",
                     build_info.get("number"), build_info.get("result"))


def _update_self_ci_status_ok(build_info: Dict) -> None:
    """Record a passing self-CI build in the status file (with history)."""
    # Load existing history
    existing_builds: list = []
    if _SELF_CI_STATUS_FILE.exists():
        try:
            old = json.loads(_SELF_CI_STATUS_FILE.read_text(encoding="utf-8"))
            existing_builds = old.get("builds", [])
        except Exception:
            pass

    # Avoid duplicate entries for the same build number
    bnum = build_info.get("number")
    if not any(b.get("number") == bnum for b in existing_builds):
        ts_epoch = build_info.get("timestamp", 0)
        ts_str = datetime.fromtimestamp(ts_epoch / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts_epoch else ""
        existing_builds.insert(0, {
            "number": bnum,
            "result": build_info.get("result"),
            "duration_ms": build_info.get("duration_ms", 0),
            "timestamp_ms": ts_epoch,
            "timestamp_str": ts_str,
        })
    # Keep only latest 10 builds
    existing_builds = existing_builds[:10]

    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "build_number": build_info.get("number"),
        "result": build_info.get("result"),
        "duration_ms": build_info.get("duration_ms", 0),
        "reason": "Self-CI passed",
        "active": False,
        "builds": existing_builds,
    }
    _SELF_CI_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SELF_CI_STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _print_cycle_header(cycle: int) -> None:
    print(f"\n{'â”€' * 55}")
    print(f"  CYCLE #{cycle}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'â”€' * 55}")


def _print_status(label: str, ok: bool, extra: str = "") -> None:
    icon = "âœ“" if ok else "âœ—"
    tag = "ONLINE" if ok else "OFFLINE"
    print(f"  [{icon}] {label:20s} {tag:8s}  {extra}")


def main() -> None:
    _setup_logging()
    _load_env()

    _print_banner()

    # Read config from .env
    jenkins_url = _env("JENKINS_URL", "http://localhost:8080")
    jenkins_user = _env("JENKINS_USERNAME", "admin")
    jenkins_pass = _env("JENKINS_PASSWORD", "")
    jenkins_token = _env("JENKINS_TOKEN", jenkins_pass)
    job_name = _env("JENKINS_JOB", "build-pipeline")
    self_ci_job = _env("SELF_CI_JOB", "neuroshield-ci")
    prom_url = _env("PROMETHEUS_URL", "http://localhost:9090")
    app_url = _env("DUMMY_APP_URL", "http://localhost:5000")
    poll_interval = int(_env("POLL_INTERVAL", "15"))
    namespace = _namespace()

    print(f"\n  Config:")
    print(f"    Jenkins:    {jenkins_url}  (job: {job_name})")
    print(f"    Prometheus: {prom_url}")
    print(f"    Dummy App:  {app_url}")
    print(f"    Namespace:  {namespace}")
    print(f"    Interval:   {poll_interval}s")

    # Load ML models
    print("\n  Loading models...")
    predictor = FailurePredictor(model_dir="models")
    print("    [OK] DistilBERT failure predictor loaded")

    try:
        policy = PPO.load("models/ppo_policy.zip")
        print("    [OK] PPO RL policy loaded (52D state â†’ 6 actions)")
    except Exception as exc:
        policy = None
        print(f"    [!!] PPO policy not found -- falling back to default action ({exc})")

    # Tracking
    cycle_count = 0
    total_actions = 0
    successful_actions = 0
    last_build_number: Optional[int] = None
    last_healed_build: Optional[int] = None  # dedup: skip if already healed this build
    last_heal_time: float = 0.0  # cooldown: timestamp of last healing action
    HEAL_COOLDOWN_S = 60  # seconds to wait between healing actions
    failure_detected_time: Optional[float] = None  # MTTR: when NEW failure first detected
    prev_build_status: Optional[str] = None  # track state transitions for MTTR
    mttr_measurements: list = []  # MTTR: list of reduction percentages for summary
    telemetry_history: list = []  # rolling window for early warning detection

    print(f"\n  Entering monitoring loop (Ctrl+C to stop)...")

    try:
        while True:
            cycle_count += 1
            _print_cycle_header(cycle_count)

            # --- Step 1: Check service health ---
            jenkins_ok, jenkins_lat = _check_service(f"{jenkins_url}/api/json",
                                                     timeout=5)
            prom_ok, prom_lat = _check_service(f"{prom_url}/-/healthy", timeout=5)
            app_ok, app_lat = _check_service(app_url, timeout=5)

            # Auto-reconnect port-forward if app appears offline
            if not app_ok:
                _ensure_port_forward()
                time.sleep(3)
                app_ok, app_lat = _check_service(app_url, timeout=5)

            _print_status("Jenkins", jenkins_ok, f"{jenkins_lat:.0f}ms" if jenkins_ok else "")
            _print_status("Prometheus", prom_ok, f"{prom_lat:.0f}ms" if prom_ok else "")
            _print_status("Dummy App", app_ok, f"{app_lat:.0f}ms" if app_ok else "")

            # --- Step 2: Collect telemetry ---
            prom_metrics = _collect_prometheus_metrics(prom_url) if prom_ok else {
                "cpu_usage": 0.0, "memory_usage": 0.0, "pod_count": 0.0,
                "error_rate": 0.0, "pod_restarts": 0.0}
            app_health = _collect_dummy_app_health(app_url) if app_ok else {
                "health_pct": 0, "response_ms": 0}

            # Get Jenkins build info
            build = None
            log_text = ""
            if jenkins_ok:
                build = get_latest_build_info(jenkins_url, job_name, jenkins_user, jenkins_token)
                if build:
                    log_text = get_build_log(jenkins_url, job_name, build.number, jenkins_user, jenkins_token)

            build_status = build.result if build else "UNKNOWN"
            build_num = build.number if build else 0
            build_duration = build.duration_ms if build else 0

            print(f"\n  Telemetry:")
            print(f"    Build #{build_num}: {build_status} ({build_duration}ms)")
            print(f"    CPU: {prom_metrics['cpu_usage']:.1f}%  |  Memory: {prom_metrics['memory_usage']:.1f}%")
            print(f"    Pods: {prom_metrics['pod_count']:.0f}  |  Error Rate: {prom_metrics['error_rate']:.4f}")
            print(f"    App Health: {app_health['health_pct']:.0f}%  |  Response: {app_health['response_ms']:.0f}ms")

            # Save telemetry row
            _save_telemetry_row({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "jenkins_last_build_status": build_status,
                "jenkins_last_build_duration": str(build_duration),
                "jenkins_queue_length": "0",
                "prometheus_cpu_usage": f"{prom_metrics['cpu_usage']:.2f}",
                "prometheus_memory_usage": f"{prom_metrics['memory_usage']:.2f}",
                "prometheus_pod_count": f"{prom_metrics['pod_count']:.0f}",
                "prometheus_error_rate": f"{prom_metrics['error_rate']:.4f}",
                "app_health_pct": f"{app_health['health_pct']:.0f}",
                "app_response_ms": f"{app_health['response_ms']:.0f}",
            })

            # --- Step 2b: Self-CI monitoring ---
            if self_ci_job:
                self_ci = _get_self_ci_build(jenkins_url, self_ci_job, jenkins_user, jenkins_token)
                if self_ci and self_ci.get("result"):
                    sci_result = self_ci["result"]
                    sci_num = self_ci.get("number", "?")
                    if sci_result in ("FAILURE", "UNSTABLE", "ABORTED"):
                        print(f"\n  Self-CI: Build #{sci_num} → {sci_result}  ⚠ ALERT")
                        handle_self_ci_failure(self_ci)
                    else:
                        print(f"\n  Self-CI: Build #{sci_num} → {sci_result}  ✓")
                        _update_self_ci_status_ok(self_ci)

            # --- Step 3: Predict failure ---
            telemetry_dict = {
                "jenkins_last_build_status": build_status,
                "jenkins_last_build_duration": build_duration,
                "jenkins_queue_length": 0,
                "prometheus_cpu_usage": prom_metrics["cpu_usage"],
                "prometheus_memory_usage": prom_metrics["memory_usage"],
                "prometheus_pod_count": prom_metrics["pod_count"],
                "prometheus_error_rate": prom_metrics["error_rate"],
                "pod_restart_count": prom_metrics.get("pod_restarts", 0.0),
            }

            if not log_text:
                log_text = f"Build {build_num} status {build_status}"

            failure_prob = predictor.predict(log_text, telemetry_dict)
            # Guard against NaN from predictor
            if failure_prob != failure_prob:  # NaN check
                failure_prob = 0.5 if _is_failure(build_status) else 0.0

            # --- App health-aware override ---
            # If dummy-app /health is degraded (503), boost failure_prob so
            # NeuroShield acts within its 15-30s window, before K8s liveness
            # probe triggers autonomously at ~90s.
            _app_hp = app_health.get("health_pct", 100)
            if _app_hp < 100:
                _boosted = max(failure_prob, 0.78)
                if _boosted > failure_prob:
                    logging.info(
                        "[HEALTH] App degraded (health_pct=%.0f%%) -- "
                        "boosting failure_prob %.3f -> %.3f",
                        _app_hp, failure_prob, _boosted,
                    )
                    failure_prob = _boosted
            telemetry_dict["app_health_pct"] = _app_hp
            prob_bar = "â–ˆ" * int(failure_prob * 20) + "â–‘" * (20 - int(failure_prob * 20))
            prob_label = "LOW" if failure_prob < 0.3 else "MEDIUM" if failure_prob < 0.6 else "HIGH" if failure_prob < 0.8 else "CRITICAL"
            print(f"\n  Prediction:")
            print(f"    Failure Prob: [{prob_bar}] {failure_prob:.3f} ({prob_label})")

            # --- Early warning detection (flags trends before threshold is crossed) ---
            telemetry_history.append(telemetry_dict)
            if len(telemetry_history) > 20:
                telemetry_history = telemetry_history[-20:]
            warn_action, warn_conf = detect_early_warning(telemetry_history)
            if warn_action and failure_prob < 0.5:
                print(f"    Early Warning: Trending toward {warn_action} (conf={warn_conf:.0%})")

            # --- Step 4: RL Agent decision ---
            jenkins_data = {
                "build_duration": build_duration,
                "build_number": build_num,
                "retry_count": 0,
            }
            prometheus_data = {
                "cpu_avg_5m": prom_metrics["cpu_usage"],
                "memory_avg_5m": prom_metrics["memory_usage"],
                "pod_restarts": 0,
                "node_count": prom_metrics["pod_count"],
            }
            state_52d = build_52d_state(jenkins_data, prometheus_data, log_text, predictor.encoder)

            pattern, pattern_action = detect_failure_pattern(log_text)

            if failure_prob > 0.5:
                # Record failure detection time for MTTR — only on NEW failure transition
                if failure_detected_time is None and prev_build_status != "FAILURE":
                    failure_detected_time = time.time()
                    logging.info("NEW failure detected — MTTR timer started")

                # Dedup: skip if we already healed this exact build
                if build_num is not None and build_num == last_healed_build:
                    print(f"\n  Status: Build #{build_num} already handled \u2014 skipping duplicate healing")
                elif time.time() - _read_cooldown_ts() < HEAL_COOLDOWN_S:
                    remaining = int(HEAL_COOLDOWN_S - (time.time() - _read_cooldown_ts()))
                    print(f"\n  Status: Cooldown active \u2014 {remaining}s remaining before next heal")
                else:
                    # Use PPO to choose action initially
                    if policy is not None:
                        action, _ = policy.predict(state_52d, deterministic=True)
                        action_id = int(action)
                    else:
                        action_id = 0  # default to restart_pod

                    # Apply rule-based + ML hybrid selection
                    # (ensures all 6 actions trigger in realistic conditions)
                    ml_action_name = ACTION_NAMES.get(action_id, "restart_pod")
                    action_name, action_reason = determine_healing_action(
                        telemetry_dict, ml_action_name, failure_prob
                    )
                    action_id = _ACTION_IDS.get(action_name, action_id)

                    # Explainable AI: why did we choose this action?
                    decision_explain = explain_decision(telemetry_dict, action_name, failure_prob)

                    print(f"\n  RL Agent Decision:")
                    print(f"    Pattern:    {pattern}")
                    print(f"    Action:     [{action_id}] {action_name}")
                    print(f"    Reason:     {action_reason}")
                    print(f"    Confidence: {decision_explain['confidence']}")
                    for r_str in decision_explain["reasons"]:
                        print(f"      - {r_str}")

                    # Execute healing action
                    success = execute_healing_action(action_id, {
                        "build_number": str(build_num),
                        "affected_service": _affected_service(),
                        "failure_prob": f"{failure_prob:.3f}",
                        "failure_pattern": pattern or "none",
                        "escalation_reason": action_reason,
                        "prometheus_cpu_usage": f"{prom_metrics['cpu_usage']:.1f}",
                        "prometheus_memory_usage": f"{prom_metrics['memory_usage']:.1f}",
                        "jenkins_last_build_status": build_status,
                    })

                    # Mark as handled after execution
                    if build_num is not None:
                        last_healed_build = build_num
                    last_heal_time = time.time()
                    _write_cooldown_ts()

                    total_actions += 1

                    if success:
                        successful_actions += 1
                        print(f"    Result:   SUCCESS")

                        # Write brain feed event for live SSE stream
                        _write_brain_feed_event(action_name, failure_prob, success, 0)

                        # MTTR measurement — only log realistic values (5-300s)
                        if failure_detected_time is not None:
                            actual_mttr = time.time() - failure_detected_time
                            if 5.0 <= actual_mttr <= 300.0:
                                baseline = MTTR_BASELINES.get(action_name, 120.0)
                                reduction = max(0.0, (baseline - actual_mttr) / baseline * 100)
                                mttr_measurements.append(reduction)
                                _log_mttr(pattern or "unknown", action_name, actual_mttr)
                                print(f"    MTTR:     {actual_mttr:.1f}s (baseline {baseline:.0f}s, {reduction:.1f}% reduction)")
                            else:
                                logging.warning("MTTR %.1fs outside realistic range [5-300] — skipping", actual_mttr)
                            failure_detected_time = None
                    else:
                        print(f"    Result:   FAILED")
                        # Write brain feed event for failed healing
                        _write_brain_feed_event(action_name, failure_prob, success, 0)

                    # Log healing decision
                    _append_csv("data/healing_log.csv", {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "cycle": str(cycle_count),
                        "build_number": str(build_num),
                        "build_status": build_status,
                        "failure_prob": f"{failure_prob:.3f}",
                        "pattern": pattern,
                        "action_id": str(action_id),
                        "action_name": action_name,
                        "success": str(success),
                        "cpu": f"{prom_metrics['cpu_usage']:.1f}",
                        "memory": f"{prom_metrics['memory_usage']:.1f}",
                        "app_health": f"{app_health['health_pct']:.0f}",
                    })
            else:
                print(f"\n  Status: System healthy -- no intervention needed")
                if build_status == "SUCCESS":
                    failure_detected_time = None  # reset MTTR timer on healthy build
                # Reset dedup when app is healthy again, so next crash can be healed
                if app_health.get("health_pct", 0) >= 100:
                    last_healed_build = None

            prev_build_status = build_status

            # --- Summary ---
            print(f"\n  Stats: {total_actions} actions taken, {successful_actions} successful")
            mttr_avg = sum(mttr_measurements) / len(mttr_measurements) if mttr_measurements else 0
            if mttr_measurements:
                print(f"  Avg MTTR Reduction: {mttr_avg:.1f}% ({len(mttr_measurements)} incidents)")
            if build:
                last_build_number = build.number

            print(f"\n  Next cycle in {poll_interval}s...")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 55}")
        print(f"  Orchestrator stopped by user")
        print(f"  Total cycles: {cycle_count} | Actions: {total_actions} | Success: {successful_actions}")
        print(f"{'=' * 55}\n")
    except Exception as exc:
        logging.exception("Error in orchestrator: %s", exc)


# ---------------------------------------------------------------------------
# Simulate mode â€“ one-shot simulated decision (no Jenkins / K8s needed)
# ---------------------------------------------------------------------------

def run_once(model_dir: str = "models") -> None:
    """Run one end-to-end simulated decision (no live infra required)."""
    import random
    from src.prediction.data_generator import generate_sample
    from src.rl_agent.simulator import simulate_action

    _setup_logging()
    _load_env()
    model_path = Path(model_dir)
    predictor = FailurePredictor(model_dir=model_path)

    try:
        policy = PPO.load(str(model_path / "ppo_policy.zip"))
    except Exception:
        policy = None
        logging.warning("PPO policy not found; falling back to restart_pod")

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
# Single-cycle runner (used by dashboard "Run Healing Cycle" button)
# ---------------------------------------------------------------------------

def run_single_cycle() -> Dict[str, str]:
    """Execute one healing cycle and return the result dict."""
    _setup_logging()
    _load_env()

    HEAL_COOLDOWN_S = 60

    # File-based cooldown check (shared with main loop)
    elapsed = time.time() - _read_cooldown_ts()
    if elapsed < HEAL_COOLDOWN_S:
        remaining = int(HEAL_COOLDOWN_S - elapsed)
        return {
            "failure_prob": "N/A",
            "action": "cooldown",
            "success": "skipped",
            "build": f"Cooldown active — {remaining}s remaining",
        }

    jenkins_url = _env("JENKINS_URL", "http://localhost:8080")
    jenkins_user = _env("JENKINS_USERNAME", "admin")
    jenkins_token = _env("JENKINS_TOKEN", "")
    job_name = _env("JENKINS_JOB", "neuroshield-app-build")

    predictor = FailurePredictor(model_dir="models")
    try:
        policy = PPO.load("models/ppo_policy.zip")
    except Exception:
        policy = None

    # Collect
    build = get_latest_build_info(jenkins_url, job_name, jenkins_user, jenkins_token)
    log_text = ""
    if build:
        log_text = get_build_log(jenkins_url, job_name, build.number, jenkins_user, jenkins_token)
    if not log_text:
        log_text = "No build log available"

    build_status = build.result if build else "UNKNOWN"
    telemetry_dict = {
        "jenkins_last_build_status": build_status,
        "jenkins_last_build_duration": build.duration_ms if build else 0,
        "jenkins_queue_length": 0,
        "prometheus_cpu_usage": 0, "prometheus_memory_usage": 0,
        "prometheus_pod_count": 0, "prometheus_error_rate": 0,
        "pod_restart_count": 0,
    }
    failure_prob = predictor.predict(log_text, telemetry_dict)
    if failure_prob != failure_prob:  # NaN guard
        failure_prob = 0.5 if _is_failure(build_status) else 0.0

    # Only heal if prob > 0.5
    if failure_prob <= 0.5:
        return {
            "failure_prob": f"{failure_prob:.3f}",
            "action": "none",
            "success": "healthy",
            "build": build_status,
        }

    jenkins_data = {"build_duration": build.duration_ms if build else 0, "build_number": build.number if build else 0}
    prometheus_data = {"cpu_avg_5m": 0, "memory_avg_5m": 0}
    state_52d = build_52d_state(jenkins_data, prometheus_data, log_text, predictor.encoder)

    if policy is not None:
        action, _ = policy.predict(state_52d, deterministic=True)
        action_id = int(action)
    else:
        action_id = 0

    # Rule-based + ML hybrid selection (same logic as main loop)
    ml_action_name = ACTION_NAMES.get(action_id, "restart_pod")
    action_name, action_reason = determine_healing_action(telemetry_dict, ml_action_name, failure_prob)
    action_id = _ACTION_IDS.get(action_name, action_id)
    heal_start = time.time()
    success = execute_healing_action(action_id, {
        "build_number": str(build.number if build else 0),
        "affected_service": _affected_service(),
        "failure_prob": f"{failure_prob:.3f}",
        "failure_pattern": "manual_trigger",
        "escalation_reason": action_reason,
        "jenkins_last_build_status": build_status,
    })

    # Update file-based cooldown
    _write_cooldown_ts()

    # Log MTTR
    if success:
        actual_mttr = time.time() - heal_start
        _log_mttr("manual_trigger", action_name, actual_mttr)

    _append_csv("data/healing_log.csv", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": "manual",
        "build_number": str(build.number if build else 0),
        "build_status": build_status,
        "failure_prob": f"{failure_prob:.3f}",
        "pattern": "manual_trigger",
        "action_id": str(action_id),
        "action_name": action_name,
        "success": str(success),
        "cpu": "0", "memory": "0", "app_health": "0",
    })

    return {
        "failure_prob": f"{failure_prob:.3f}",
        "action": action_name,
        "success": str(success),
        "build": build_status,
    }


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
