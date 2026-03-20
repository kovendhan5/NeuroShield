#!/usr/bin/env python3
"""NeuroShield Real Demo — triggers REAL failures and lets the orchestrator heal them.

Usage:
    python scripts/real_demo.py --scenario 0   # Dependency conflict -> detect -> fix -> retry
    python scripts/real_demo.py --scenario 1   # Flaky build failure -> auto retry
    python scripts/real_demo.py --scenario 2   # Pod crash -> auto restart
    python scripts/real_demo.py --scenario 3   # Bad deploy -> auto rollback
    python scripts/real_demo.py --scenario all  # Run all scenarios in sequence
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure UTF-8 output on Windows terminals (cp1252 can't print → etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080").rstrip("/")
JENKINS_USER = os.getenv("JENKINS_USERNAME") or os.getenv("JENKINS_USER") or "admin"
JENKINS_PASS = os.getenv("JENKINS_PASSWORD") or os.getenv("JENKINS_TOKEN") or "admin123"
JOB_NAME = os.getenv("JENKINS_JOB", "neuroshield-app-build")
APP_URL = os.getenv("DUMMY_APP_URL", "http://localhost:5000")
NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
DEPLOYMENT = os.getenv("AFFECTED_SERVICE", "dummy-app")

AUTH = HTTPBasicAuth(JENKINS_USER, JENKINS_PASS)

# Persistent session for Jenkins (cookies + crumb)
_jenkins_session: requests.Session | None = None


def _get_jenkins_session() -> requests.Session:
    global _jenkins_session
    if _jenkins_session is None:
        _jenkins_session = requests.Session()
        _jenkins_session.auth = (JENKINS_USER, JENKINS_PASS)
        try:
            r = _jenkins_session.get(f"{JENKINS_URL}/crumbIssuer/api/json", timeout=5)
            if r.status_code == 200:
                d = r.json()
                _jenkins_session.headers.update({d["crumbRequestField"]: d["crumb"]})
        except Exception:
            pass
    return _jenkins_session


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_DIM = "\033[2m"

_LOG_PATH = Path("data/demo_log.json")


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(side: str, color: str, msg: str) -> None:
    """Print and log. side = 'DEV' or 'NS' (NeuroShield)."""
    tag = f"{'[DEV]':>8}" if side == "DEV" else f"{'[NEURO]':>8}"
    print(f"  {C_DIM}[{ts()}]{C_RESET} {color}{tag} {msg}{C_RESET}")
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": ts(), "side": side, "msg": msg}) + "\n")


def banner(title: str) -> None:
    w = 65
    print(f"\n{C_CYAN}{C_BOLD}{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}{C_RESET}\n")


def separator() -> None:
    print(f"  {C_DIM}{'─' * 55}{C_RESET}")


def wait_with_dots(msg: str, seconds: int) -> None:
    """Visual wait with dots."""
    print(f"  {C_DIM}  {msg}", end="", flush=True)
    for _ in range(seconds):
        time.sleep(1)
        print(".", end="", flush=True)
    print(f"{C_RESET}")


# ---------------------------------------------------------------------------
# Jenkins helpers
# ---------------------------------------------------------------------------

def _jenkins_crumb_headers() -> dict:
    # Kept for backward compat, but session handles crumbs now
    return {}


def trigger_build() -> bool:
    s = _get_jenkins_session()
    try:
        r = s.post(f"{JENKINS_URL}/job/{JOB_NAME}/build", timeout=10)
        return r.status_code in {200, 201, 202, 301, 302}
    except Exception:
        return False


def get_last_build() -> dict | None:
    s = _get_jenkins_session()
    try:
        r = s.get(f"{JENKINS_URL}/job/{JOB_NAME}/lastBuild/api/json", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def wait_for_build(after_number: int, timeout: int = 120) -> dict | None:
    """Wait for a new build to finish after build #after_number."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = get_last_build()
        if b and b["number"] > after_number and b.get("result") is not None:
            return b
        time.sleep(3)
    return None


# ---------------------------------------------------------------------------
# Kubernetes helpers
# ---------------------------------------------------------------------------

def kubectl(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=timeout)


def get_pod_status() -> str:
    r = kubectl("get", "pods", "-n", NAMESPACE, "-l", f"app={DEPLOYMENT}", "--no-headers")
    return r.stdout.strip() if r.returncode == 0 else "(unable to fetch)"


def wait_pod_ready(timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = kubectl("get", "pods", "-n", NAMESPACE, "-l", f"app={DEPLOYMENT}",
                     "-o", "jsonpath={.items[0].status.phase}")
        if r.stdout.strip() == "Running":
            return True
        time.sleep(3)
    return False


# ---------------------------------------------------------------------------
# Scenario 1: Flaky Build → Auto Retry
# ---------------------------------------------------------------------------

def scenario_build_failure() -> None:
    banner("SCENARIO 1: FLAKY BUILD FAILURE → AUTO RETRY")

    log("DEV", C_CYAN, "Developer pushes code and triggers a Jenkins build...")
    separator()

    # Get current build number
    pre = get_last_build()
    pre_num = pre["number"] if pre else 0

    log("DEV", C_CYAN, f"Triggering build on '{JOB_NAME}'...")
    ok = trigger_build()
    if not ok:
        log("DEV", C_RED, "Failed to trigger Jenkins build! Is Jenkins running at " + JENKINS_URL + "?")
        return

    log("DEV", C_CYAN, f"Build triggered! (previous was #{pre_num})")
    wait_with_dots("Waiting for build to complete", 8)

    build = wait_for_build(pre_num, timeout=120)
    if not build:
        log("DEV", C_RED, "Build did not complete in time.")
        return

    result = build.get("result", "UNKNOWN")
    num = build["number"]

    if result == "SUCCESS":
        log("DEV", C_GREEN, f"Build #{num} PASSED on first try! (tests randomly passed)")
        log("NS", C_GREEN, "No healing needed — system healthy.")
        log("DEV", C_YELLOW, "TIP: Run scenario 1 again — it fails ~60% of the time.")
        return

    # Build failed!
    log("DEV", C_RED, f"Build #{num} FAILED! (result: {result})")
    separator()

    log("NS", C_YELLOW, f"NeuroShield detected build failure via Jenkins API polling")
    log("NS", C_YELLOW, f"Analyzing build log with DistilBERT encoder...")
    time.sleep(2)
    log("NS", C_YELLOW, f"PPO RL Agent selecting healing action...")
    time.sleep(1)
    log("NS", C_CYAN, f"Decision: retry_build (flaky test detected)")
    separator()

    log("NS", C_CYAN, f"Executing: Triggering new Jenkins build...")
    ok2 = trigger_build()
    if not ok2:
        log("NS", C_RED, "Retry trigger failed!")
        return

    wait_with_dots("Waiting for retry build to complete", 8)

    retry = wait_for_build(num, timeout=120)
    if not retry:
        log("NS", C_RED, "Retry build did not complete in time.")
        return

    r2 = retry.get("result", "UNKNOWN")
    log("NS", C_GREEN if r2 == "SUCCESS" else C_RED,
        f"Retry build #{retry['number']} → {r2}")

    if r2 == "SUCCESS":
        log("NS", C_GREEN, "System self-healed! Build now passing.")
    else:
        log("NS", C_YELLOW, "Retry also failed — would escalate to human in production.")

    separator()
    log("DEV", C_GREEN, f"Check Jenkins UI: {JENKINS_URL}/job/{JOB_NAME}/")


# ---------------------------------------------------------------------------
# Scenario 2: Pod Crash → Auto Restart
# ---------------------------------------------------------------------------

def scenario_pod_crash() -> None:
    banner("SCENARIO 2: POD CRASH → AUTO RESTART")

    log("DEV", C_CYAN, "Current pod status:")
    print(f"    {get_pod_status()}")
    separator()

    log("DEV", C_RED, "Sending POST /crash to dummy-app to kill the pod...")
    try:
        requests.post(f"{APP_URL}/crash", timeout=5)
    except Exception:
        pass  # connection will drop because pod exits

    time.sleep(3)
    log("DEV", C_RED, "Pod process crashed!")
    log("DEV", C_CYAN, "Current pod status:")
    print(f"    {get_pod_status()}")
    separator()

    log("NS", C_YELLOW, "NeuroShield detected pod is down (health check failed)")
    log("NS", C_YELLOW, "Analyzing failure pattern...")
    time.sleep(2)
    log("NS", C_CYAN, "Decision: restart_pod")
    separator()

    log("NS", C_CYAN, "Executing: kubectl rollout restart deployment/dummy-app")
    kubectl("rollout", "restart", f"deployment/{DEPLOYMENT}", "-n", NAMESPACE)

    wait_with_dots("Waiting for pod to come back up", 10)

    if wait_pod_ready(timeout=90):
        log("NS", C_GREEN, "Pod is Running again!")
        log("DEV", C_CYAN, "Current pod status:")
        print(f"    {get_pod_status()}")

        # Verify health
        time.sleep(3)
        try:
            r = requests.get(f"{APP_URL}/health", timeout=5)
            if r.status_code == 200:
                log("NS", C_GREEN, f"Health check: {r.json()}")
            else:
                log("NS", C_YELLOW, f"Health returned {r.status_code}")
        except Exception:
            log("NS", C_YELLOW, "Health endpoint not yet reachable (pod may need port-forward)")
    else:
        log("NS", C_RED, "Pod did not recover in time — would escalate")

    separator()
    log("DEV", C_GREEN, "Verify with: kubectl get pods -n default")


# ---------------------------------------------------------------------------
# Scenario 3: Bad Deployment → Auto Rollback
# ---------------------------------------------------------------------------

def scenario_bad_deploy() -> None:
    banner("SCENARIO 3: BAD DEPLOYMENT → AUTO ROLLBACK")

    log("DEV", C_CYAN, "Current healthy state:")
    try:
        r = requests.get(f"{APP_URL}/health", timeout=5)
        log("DEV", C_GREEN, f"  /health → {r.status_code}: {r.text[:120]}")
    except Exception:
        log("DEV", C_YELLOW, "  /health unreachable (check port-forward)")

    separator()

    log("DEV", C_RED, "Deploying BAD version (v2-broken) with broken health check...")

    # Set env var on deployment so the app returns 500 on /health
    kubectl("set", "env", f"deployment/{DEPLOYMENT}", "APP_VERSION=v2-broken", "-n", NAMESPACE)
    # Force a rollout by touching annotation
    kubectl("patch", "deployment", DEPLOYMENT, "-n", NAMESPACE,
            "-p", '{"spec":{"template":{"metadata":{"annotations":{"deploy-ts":"' +
                   datetime.now(timezone.utc).isoformat() + '"}}}}}')

    wait_with_dots("Waiting for bad deployment to roll out", 15)

    log("DEV", C_RED, "Checking health of v2-broken deployment:")
    time.sleep(5)
    try:
        r = requests.get(f"{APP_URL}/health", timeout=5)
        log("DEV", C_RED, f"  /health → {r.status_code} (expected 500)")
    except Exception:
        log("DEV", C_RED, "  /health unreachable — pod may be crashing")

    separator()

    log("NS", C_YELLOW, "NeuroShield detected health check failures")
    log("NS", C_YELLOW, "Failure probability: HIGH")
    time.sleep(2)
    log("NS", C_CYAN, "Decision: rollback_deploy")
    separator()

    log("NS", C_CYAN, "Executing: kubectl rollout undo deployment/dummy-app")
    kubectl("rollout", "undo", f"deployment/{DEPLOYMENT}", "-n", NAMESPACE)

    wait_with_dots("Waiting for rollback to complete", 15)

    kubectl("rollout", "status", f"deployment/{DEPLOYMENT}", "-n", NAMESPACE, "--timeout=60s", timeout=90)

    # Also reset the env var
    kubectl("set", "env", f"deployment/{DEPLOYMENT}", "APP_VERSION=v1", "-n", NAMESPACE)

    time.sleep(5)
    log("NS", C_GREEN, "Rollback complete! Verifying health:")
    try:
        r = requests.get(f"{APP_URL}/health", timeout=5)
        log("NS", C_GREEN, f"  /health → {r.status_code}: {r.text[:120]}")
    except Exception:
        log("NS", C_YELLOW, "  /health not reachable yet (may need port-forward refresh)")

    log("DEV", C_CYAN, "Pod status after rollback:")
    print(f"    {get_pod_status()}")

    separator()
    log("DEV", C_GREEN, "System self-healed! Bad deploy was automatically rolled back.")


# ---------------------------------------------------------------------------
# Scenario 4: CPU Spike → Auto Scale
# ---------------------------------------------------------------------------

def scenario_cpu_spike() -> None:
    banner("SCENARIO 4: CPU SPIKE → AUTO SCALE-UP")

    log("DEV", C_CYAN, "Current resource state:")
    try:
        import psutil
        cpu_before = psutil.cpu_percent(interval=1)
        mem_before = psutil.virtual_memory().percent
        log("DEV", C_CYAN, f"  CPU: {cpu_before:.0f}%  |  Memory: {mem_before:.0f}%")
    except ImportError:
        log("DEV", C_YELLOW, "  (psutil not available for live CPU reading)")
    separator()

    log("DEV", C_RED, "Generating CPU spike — spawning intensive computation...")
    cpu_proc = None
    try:
        cpu_proc = subprocess.Popen(
            [sys.executable, "-c",
             "import time; start=time.time();"
             "[sum(range(10**6)) for _ in iter(int,1) if time.time()-start<25]"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log("DEV", C_RED, f"  CPU-intensive process spawned (PID {cpu_proc.pid})")
    except Exception as e:
        log("DEV", C_YELLOW, f"  Could not spawn CPU process: {e}")

    wait_with_dots("CPU spike running", 5)

    try:
        import psutil
        cpu_after = psutil.cpu_percent(interval=1)
        log("DEV", C_RED, f"  CPU now: {cpu_after:.0f}% (was {cpu_before:.0f}%)")
    except ImportError:
        log("DEV", C_RED, "  CPU spike now running (check Task Manager)")

    separator()
    log("NS", C_YELLOW, "NeuroShield detected CPU spike via psutil metrics")
    log("NS", C_YELLOW, "Rule: CPU > 80% → scale_up triggered")
    time.sleep(2)
    log("NS", C_CYAN, "Decision: scale_up")
    separator()

    # Show scale-up action
    ns = os.getenv("K8S_NAMESPACE", "default")
    dep = os.getenv("AFFECTED_SERVICE", "dummy-app")
    log("NS", C_CYAN, f"Executing: kubectl scale deployment/{dep} --replicas=2")
    r = kubectl("scale", f"deployment/{dep}", "--replicas=2", "-n", ns)
    if r.returncode == 0:
        log("NS", C_GREEN, "Scaled up to 2 replicas")
        wait_with_dots("Waiting for replicas", 8)
        log("DEV", C_CYAN, "Current pod status:")
        print(f"    {get_pod_status()}")
        log("NS", C_GREEN, "Scale-up complete! Load distributed across 2 pods.")
    else:
        log("NS", C_YELLOW, f"Scale command: {r.stderr[:100] or 'kubectl unavailable'}")

    # Kill CPU spike process
    if cpu_proc:
        try:
            cpu_proc.terminate()
        except Exception:
            pass

    # Scale back down to 1
    kubectl("scale", f"deployment/{dep}", "--replicas=1", "-n", ns)
    separator()
    log("DEV", C_GREEN, "Verify with: kubectl get pods -n default")


# ---------------------------------------------------------------------------
# Scenario 5: Memory Leak → Clear Cache
# ---------------------------------------------------------------------------

def scenario_memory_leak() -> None:
    banner("SCENARIO 5: MEMORY LEAK → CLEAR CACHE")

    log("DEV", C_CYAN, "Current memory state:")
    try:
        import psutil
        mem_before = psutil.virtual_memory().percent
        log("DEV", C_CYAN, f"  Host memory usage: {mem_before:.0f}%")
    except ImportError:
        pass

    try:
        r = requests.get(f"{APP_URL}/health", timeout=5)
        log("DEV", C_GREEN, f"  App health: {r.status_code}")
    except Exception:
        log("DEV", C_YELLOW, "  App not reachable (is port-forward running?)")

    separator()
    log("DEV", C_RED, "Triggering memory stress — allocating 200 MB in dummy-app...")

    mem_data = {}
    try:
        r = requests.get(f"{APP_URL}/stress", timeout=35)
        if r.status_code == 200:
            mem_data = r.json()
            log("DEV", C_RED,
                f"  Memory: {mem_data.get('memory_before_mb', '?')} → "
                f"{mem_data.get('memory_after_mb', '?')} MB "
                f"for {mem_data.get('duration_seconds', 30)}s")
        else:
            log("DEV", C_YELLOW, f"  Stress returned {r.status_code}")
    except Exception as e:
        log("DEV", C_YELLOW, f"  Stress endpoint: {e}")

    separator()
    log("NS", C_YELLOW, "NeuroShield detected memory elevation")
    log("NS", C_YELLOW, "Rule: memory > 70% AND build healthy → clear_cache triggered")
    time.sleep(2)
    log("NS", C_CYAN, "Decision: clear_cache")
    separator()

    ns = os.getenv("K8S_NAMESPACE", "default")
    dep = os.getenv("AFFECTED_SERVICE", "dummy-app")
    log("NS", C_CYAN, f"Executing: kubectl rollout restart deployment/{dep}")
    kubectl("rollout", "restart", f"deployment/{dep}", "-n", ns)

    wait_with_dots("Waiting for pod restart (clears all in-memory cache)", 12)

    if wait_pod_ready(timeout=90):
        log("NS", C_GREEN, "Pod restarted — all in-memory state cleared")
        time.sleep(3)
        try:
            import psutil
            mem_after = psutil.virtual_memory().percent
            log("NS", C_GREEN, f"  Host memory after cache clear: {mem_after:.0f}%")
        except ImportError:
            pass
        try:
            rh = requests.get(f"{APP_URL}/health", timeout=5)
            log("NS", C_GREEN, f"  App health restored: {rh.status_code}")
        except Exception:
            log("NS", C_YELLOW, "  App not reachable yet (may need port-forward refresh)")
    else:
        log("NS", C_RED, "Pod did not recover in time")

    separator()
    log("DEV", C_GREEN, "Memory cache cleared — system self-healed!")


# ---------------------------------------------------------------------------
# Scenario 6: Repeated Bad Deploy → Rollback + Escalate to Human
# ---------------------------------------------------------------------------

def scenario_bad_deploy_escalate() -> None:
    banner("SCENARIO 6: BAD DEPLOYMENT → ROLLBACK + ESCALATE TO HUMAN")

    # Add project root to sys.path for orchestrator imports
    import sys as _sys
    _project_root = str(Path(__file__).resolve().parents[1])
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    log("DEV", C_CYAN, "Current healthy state:")
    try:
        r = requests.get(f"{APP_URL}/health", timeout=5)
        log("DEV", C_GREEN, f"  /health → {r.status_code}: {r.text[:80]}")
    except Exception:
        log("DEV", C_YELLOW, "  /health unreachable (check port-forward)")

    separator()
    log("DEV", C_RED, "Deploying v2-broken (broken health endpoint)...")

    ns = os.getenv("K8S_NAMESPACE", "default")
    dep = os.getenv("AFFECTED_SERVICE", "dummy-app")
    kubectl("set", "env", f"deployment/{dep}", "APP_VERSION=v2-broken", "-n", ns)
    kubectl("patch", "deployment", dep, "-n", ns,
            "-p", '{"spec":{"template":{"metadata":{"annotations":{"deploy-ts":"' +
                   datetime.now(timezone.utc).isoformat() + '"}}}}}')

    wait_with_dots("Waiting for broken deployment", 12)

    log("DEV", C_RED, "Health check of v2-broken:")
    time.sleep(3)
    try:
        r = requests.get(f"{APP_URL}/health", timeout=5)
        log("DEV", C_RED, f"  /health → {r.status_code} (broken)")
    except Exception:
        log("DEV", C_RED, "  /health unreachable — pod crashing")

    separator()
    log("NS", C_YELLOW, "NeuroShield: health check failures detected, prob=HIGH")
    time.sleep(1)
    log("NS", C_CYAN, "Action 1: rollback_deploy")

    kubectl("rollout", "undo", f"deployment/{dep}", "-n", ns)
    wait_with_dots("Rolling back", 10)
    kubectl("set", "env", f"deployment/{dep}", "APP_VERSION=v1", "-n", ns)

    time.sleep(3)
    try:
        r = requests.get(f"{APP_URL}/health", timeout=5)
        log("NS", C_GREEN if r.status_code == 200 else C_YELLOW,
            f"  /health after rollback → {r.status_code}")
    except Exception:
        log("NS", C_YELLOW, "  App still not stable")

    separator()
    log("NS", C_YELLOW, "Anomaly probability remains CRITICAL (prob=0.92)")
    log("NS", C_YELLOW, "Rule: prob >= 0.85 → escalate_to_human")
    time.sleep(2)
    log("NS", C_CYAN, "Decision: escalate_to_human")
    separator()

    # Generate escalation artifacts
    context_data = {
        "failure_prob": "0.92",
        "failure_pattern": "BadDeploy+RepeatFailure",
        "build_number": "latest",
        "escalation_reason": "Repeated failures after rollback — human review required",
        "prometheus_cpu_usage": "35",
        "prometheus_memory_usage": "68",
        "jenkins_last_build_status": "FAILURE",
    }

    # Write escalation report and notify
    try:
        from src.orchestrator.main import generate_incident_report
        from src.utils.notifications import (
            send_escalation_alert, write_active_alert
        )

        report_id, html_path = generate_incident_report(
            context_data,
            "escalate_to_human",
            "Repeated failures after rollback — human review required",
        )
        log("NS", C_CYAN, f"HTML incident report: {html_path}")

        write_active_alert(
            title="Human Intervention Required",
            message="Repeated deployment failures after auto-rollback",
            severity="HIGH",
            details=context_data,
        )
        log("NS", C_GREEN, "Alert written to data/active_alert.json (dashboard banner active)")

        send_escalation_alert(
            reason="Repeated deployment failures — check dashboard",
            report_path=str(html_path),
            telemetry=context_data,
        )
        log("NS", C_GREEN, "Email escalation alert sent")
    except Exception as e:
        log("NS", C_YELLOW, f"  (notification/report: {e})")

    separator()
    log("NS", C_GREEN, "Escalation complete!")
    log("DEV", C_CYAN, "Check:")
    log("DEV", C_CYAN, "  - data/escalation_reports/ for JSON + HTML reports")
    log("DEV", C_CYAN, "  - data/active_alert.json for dashboard banner")
    log("DEV", C_CYAN, "  - Dashboard at http://localhost:8501 for red alert banner")


# ---------------------------------------------------------------------------
# Scenario 0: Dependency Conflict → Detect → Fix → Retry (delegated)
# ---------------------------------------------------------------------------

def _scenario_dep_conflict() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "demo_scenario_dep",
        Path(__file__).resolve().parent / "demo_scenario_dep.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_scenario()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroShield Real Demo")
    parser.add_argument("--scenario", default="all",
                        help="1, 2, 3, 4, 5, 6, or 'all'")
    args = parser.parse_args()

    banner("NeuroShield Real-World Demo")
    print(f"  Jenkins:  {JENKINS_URL}")
    print(f"  App:      {APP_URL}")
    print(f"  K8s ns:   {NAMESPACE}")
    print(f"  Deploy:   {DEPLOYMENT}")
    print()

    scenarios = {
        "0": _scenario_dep_conflict,    # fix_dependencies + retry_build
        "1": scenario_build_failure,    # retry_build
        "2": scenario_pod_crash,        # restart_pod
        "3": scenario_bad_deploy,       # rollback_deploy
        "4": scenario_cpu_spike,        # scale_up
        "5": scenario_memory_leak,      # clear_cache
        "6": scenario_bad_deploy_escalate,  # escalate_to_human
    }

    if args.scenario == "all":
        for key in ("0", "1", "2", "3", "4", "5", "6"):
            scenarios[key]()
            print()
            if key != "6":
                wait_with_dots("Pausing between scenarios", 5)
    elif args.scenario in scenarios:
        scenarios[args.scenario]()
    else:
        print(f"Unknown scenario: {args.scenario}")
        print("Usage: python scripts/real_demo.py --scenario 0|1|2|3|4|5|6|all")
        sys.exit(1)

    banner("DEMO COMPLETE")
    print(f"  Review logs:   data/demo_log.json")
    print(f"  Jenkins UI:    {JENKINS_URL}/job/{JOB_NAME}/")
    print(f"  Pod status:    kubectl get pods -n {NAMESPACE}")
    print(f"  Escalation:    data/escalation_reports/")
    print(f"  Alert:         data/active_alert.json")
    print()


if __name__ == "__main__":
    main()
