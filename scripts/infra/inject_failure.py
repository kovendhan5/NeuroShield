#!/usr/bin/env python3
"""NeuroShield Failure Injector — randomly injects REAL failures on a schedule.

Runs in a loop, every ~2 minutes picks a random failure type and triggers it
against the real infrastructure.  NeuroShield orchestrator (running separately)
detects and heals these failures naturally.

Usage:
    python scripts/inject_failure.py                   # default 120s interval
    python scripts/inject_failure.py --interval 60     # every 60s
    python scripts/inject_failure.py --type build_fail # inject specific type once
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
LOG_PATH = Path("data/injection_log.json")

FAILURE_TYPES = ["build_fail", "pod_crash", "memory_stress", "bad_deploy",
                 "cpu_spike", "memory_pressure"]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = ts()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  [{ts()}] {entry.get('type', '?')} → {entry.get('result', '?')}")


def _jenkins_crumb() -> dict:
    try:
        r = requests.get(f"{JENKINS_URL}/crumbIssuer/api/json", auth=AUTH, timeout=5)
        if r.status_code == 200:
            d = r.json()
            return {d["crumbRequestField"]: d["crumb"]}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Failure injectors
# ---------------------------------------------------------------------------

def inject_build_fail() -> dict:
    """Trigger the Jenkins job (which randomly fails 60% of the time)."""
    headers = _jenkins_crumb()
    r = requests.post(f"{JENKINS_URL}/job/{JOB_NAME}/build", auth=AUTH, headers=headers, timeout=10)
    ok = r.status_code in {200, 201, 202, 301, 302}
    return {"type": "build_fail", "result": "triggered" if ok else f"HTTP {r.status_code}"}


def inject_pod_crash() -> dict:
    """Send POST /crash to dummy-app to kill the pod."""
    try:
        requests.post(f"{APP_URL}/crash", timeout=5)
        return {"type": "pod_crash", "result": "crash sent"}
    except Exception:
        return {"type": "pod_crash", "result": "crash sent (connection dropped)"}


def inject_memory_stress() -> dict:
    """Send GET /stress to dummy-app to allocate 200 MB for 30s."""
    try:
        r = requests.get(f"{APP_URL}/stress", timeout=10)
        return {"type": "memory_stress", "result": f"HTTP {r.status_code}",
                "data": r.json() if r.status_code == 200 else r.text[:200]}
    except Exception as e:
        return {"type": "memory_stress", "result": str(e)[:200]}


def inject_bad_deploy() -> dict:
    """Set APP_VERSION=v2-broken so /health returns 500."""
    r1 = subprocess.run(
        ["kubectl", "set", "env", f"deployment/{DEPLOYMENT}",
         "APP_VERSION=v2-broken", "-n", NAMESPACE],
        capture_output=True, text=True, timeout=30,
    )
    # Touch annotation to trigger rollout
    subprocess.run(
        ["kubectl", "patch", "deployment", DEPLOYMENT, "-n", NAMESPACE,
         "-p", '{"spec":{"template":{"metadata":{"annotations":{"inject-ts":"' +
               datetime.now(timezone.utc).isoformat() + '"}}}}}'],
        capture_output=True, text=True, timeout=30,
    )
    return {"type": "bad_deploy", "result": "ok" if r1.returncode == 0 else r1.stderr[:200]}


def inject_cpu_spike() -> dict:
    """Spawn a CPU-intensive Python process for 30 seconds.

    Triggers scale_up action: CPU will exceed 80% threshold on most machines.
    """
    import sys as _sys
    try:
        proc = subprocess.Popen(
            [_sys.executable, "-c",
             "import time; start=time.time(); [sum(range(10**6)) for _ in iter(int,1) "
             "if time.time()-start < 30]"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("CPU spike injected — scale_up should trigger (PID: %d)" % proc.pid)
        return {"type": "cpu_spike", "result": f"spawned pid={proc.pid}"}
    except Exception as e:
        return {"type": "cpu_spike", "result": str(e)[:200]}


def inject_memory_pressure() -> dict:
    """Send GET /stress to dummy-app to allocate 200 MB for 30s.

    Equivalent to memory_stress but explicitly named for the demo scenario.
    Triggers clear_cache action when memory > 70% and build is healthy.
    """
    try:
        r = requests.get(f"{APP_URL}/stress", timeout=10)
        return {"type": "memory_pressure", "result": f"HTTP {r.status_code}",
                "data": r.json() if r.status_code == 200 else r.text[:200]}
    except Exception as e:
        return {"type": "memory_pressure", "result": str(e)[:200]}


INJECTORS = {
    "build_fail": inject_build_fail,
    "pod_crash": inject_pod_crash,
    "memory_stress": inject_memory_stress,
    "bad_deploy": inject_bad_deploy,
    "cpu_spike": inject_cpu_spike,
    "memory_pressure": inject_memory_pressure,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroShield Failure Injector")
    parser.add_argument("--interval", type=int, default=120,
                        help="Seconds between injections (default: 120)")
    parser.add_argument("--type", choices=FAILURE_TYPES, default=None,
                        help="Inject a specific failure once and exit")
    args = parser.parse_args()

    print(f"\n{'=' * 55}")
    print(f"  NeuroShield Failure Injector")
    print(f"{'=' * 55}")
    print(f"  Jenkins:  {JENKINS_URL}")
    print(f"  App:      {APP_URL}")
    print(f"  Interval: {args.interval}s")
    print(f"  Log:      {LOG_PATH}")
    print(f"{'=' * 55}\n")

    if args.type:
        result = INJECTORS[args.type]()
        _log(result)
        return

    print("  Starting continuous injection loop (Ctrl+C to stop)...\n")
    try:
        while True:
            failure_type = random.choice(FAILURE_TYPES)
            print(f"\n  [{ts()}] Injecting: {failure_type}")
            result = INJECTORS[failure_type]()
            _log(result)
            print(f"  [{ts()}] Next injection in {args.interval}s...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n  Injector stopped. Log at: {LOG_PATH}\n")


if __name__ == "__main__":
    main()
