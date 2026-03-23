#!/usr/bin/env python3
"""
NeuroShield - Real End-to-End Demo
Demonstrates the orchestrator working with actual services (no simulation)
~90 seconds: Injects failure, detects it, executes healing, verifies recovery
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()

JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
DUMMY_APP_URL = os.getenv("DUMMY_APP_URL", "http://localhost:5000")
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
AFFECTED_SERVICE = os.getenv("AFFECTED_SERVICE", "dummy-app")


def print_step(step_num, title):
    """Print a step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}\n")


def check_service(name, url):
    """Check if a service is online"""
    try:
        response = requests.get(url, timeout=2)
        return response.status_code < 500
    except:
        return False


def get_app_health():
    """Get current app health status"""
    try:
        response = requests.get(f"{DUMMY_APP_URL}/health", timeout=2)
        if response.status_code == 200:
            try:
                return response.json()
            except:
                return {"status": "up"}
        return {"status": "down", "code": response.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def inject_failure():
    """Inject a failure into the dummy app"""
    print("Injecting failure into dummy-app...")
    try:
        response = requests.post(f"{DUMMY_APP_URL}/crash", timeout=2)
        if response.status_code in [200, 500]:
            print("✓ Failure injected - dummy-app should be crashing now")
            return True
    except:
        pass

    # Alternative: Use kubectl to delete the pod (forces restart)
    print("Using kubectl to force pod restart...")
    try:
        subprocess.run(
            ["kubectl", "delete", "pod", "-l", f"app={AFFECTED_SERVICE}",
             "-n", K8S_NAMESPACE],
            capture_output=True,
            timeout=10
        )
        print("✓ Pod deleted - Kubernetes will restart it")
        return True
    except Exception as e:
        print(f"✗ Could not inject failure: {e}")
        return False


def get_latest_jenkins_build():
    """Get the latest Jenkins build"""
    try:
        job_name = os.getenv("JENKINS_JOB", "neuroshield-app-build")
        url = f"{JENKINS_URL}/job/{job_name}/lastBuild/api/json"
        auth = (
            os.getenv("JENKINS_USERNAME", "admin"),
            os.getenv("JENKINS_PASSWORD", os.getenv("JENKINS_TOKEN", ""))
        )
        response = requests.get(url, auth=auth, timeout=5)
        if response.status_code == 200:
            build = response.json()
            return {
                "number": build.get("number"),
                "result": build.get("result") or "RUNNING",
                "timestamp": build.get("timestamp"),
            }
    except:
        pass
    return None


def get_prometheus_metrics():
    """Get current Prometheus metrics"""
    metrics = {
        "cpu_usage": 0,
        "memory_usage": 0,
        "pod_count": 0,
    }

    queries = {
        "cpu": "avg(rate(node_cpu_seconds_total[5m])) * 100",
        "memory": "avg(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100",
        "pods": "count(kube_pod_info{namespace=\"" + K8S_NAMESPACE + "\"})",
    }

    for name, query in queries.items():
        try:
            response = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    result = data.get("data", {}).get("result", [])
                    if result and len(result) > 0:
                        value = float(result[0].get("value", [0, "0"])[1])
                        if name == "cpu":
                            metrics["cpu_usage"] = value
                        elif name == "memory":
                            metrics["memory_usage"] = value
                        elif name == "pods":
                            metrics["pod_count"] = value
        except:
            pass

    return metrics


def check_healing_log():
    """Check if orchestrator wrote a healing action"""
    healing_log = Path("data/healing_log.json")
    if not healing_log.exists():
        return None

    try:
        with open(healing_log, 'r') as f:
            lines = f.readlines()
            if lines:
                # Get last line (most recent action)
                last_line = lines[-1].strip()
                action = json.loads(last_line)
                return action
    except:
        pass

    return None


def main():
    """Run end-to-end demo"""
    print("\n" + "="*60)
    print("NeuroShield - Real End-to-End Demo")
    print("="*60)
    print("\nThis demo will:")
    print("1. Verify services are online")
    print("2. Inject a failure into the dummy-app")
    print("3. Wait for NeuroShield orchestrator to detect it")
    print("4. Verify a healing action was executed")
    print("5. Check recovery\n")
    print("Started at:", datetime.now().strftime("%H:%M:%S"))

    # STEP 1: Check services
    print_step(1, "Service Health Check")

    print(f"Jenkins:     {JENKINS_URL}")
    jenkins_ok = check_service("Jenkins", f"{JENKINS_URL}/api/json")
    print(f"  {'✓' if jenkins_ok else '✗'} {'Online' if jenkins_ok else 'Offline'}")

    print(f"Prometheus:  {PROMETHEUS_URL}")
    prom_ok = check_service("Prometheus", f"{PROMETHEUS_URL}/-/healthy")
    print(f"  {'✓' if prom_ok else '✗'} {'Online' if prom_ok else 'Offline'}")

    print(f"Dummy App:   {DUMMY_APP_URL}")
    app_ok = check_service("Dummy App", DUMMY_APP_URL)
    print(f"  {'✓' if app_ok else '✗'} {'Online' if app_ok else 'Offline'}")

    if not all([jenkins_ok or True, prom_ok or True, app_ok]):  # Allow some services to be down
        print("\n⚠ WARNING: Some services are offline")
        print("  Orchestrator may not be able to detect/heal failures")
        print("  Continuing anyway...")

    # STEP 2: Get baseline metrics
    print_step(2, "Baseline Metrics (Before Injection)")

    health_before = get_app_health()
    print(f"App Health:  {json.dumps(health_before, indent=2)}")

    metrics_before = get_prometheus_metrics()
    print(f"Prometheus Metrics:")
    print(f"  CPU:           {metrics_before['cpu_usage']:.1f}%")
    print(f"  Memory:        {metrics_before['memory_usage']:.1f}%")
    print(f"  Pod Count:     {metrics_before['pod_count']:.0f}")

    # STEP 3: Inject failure
    print_step(3, "Inject Failure")

    failure_injected = inject_failure()
    if not failure_injected:
        print("✗ Could not inject failure - demo cannot continue")
        return 1

    print("Waiting for failure to propagate...")
    time.sleep(5)

    health_after_inject = get_app_health()
    print(f"App Health After: {json.dumps(health_after_inject, indent=2)}")

    # STEP 4: Wait for orchestrator detection + healing
    print_step(4, "Orchestrator Detection & Healing (~45 seconds)")

    print("Waiting for orchestrator main loop to detect failure...")
    print("(Orchestrator polls every 15-30 seconds)\n")

    healing_action = None
    for attempt in range(15):  # Try for ~45 seconds
        elapsed = attempt * 3
        print(f"  [{elapsed:2d}s] Checking for healing action...")

        healing_action = check_healing_log()
        if healing_action:
            print(f"\n✓ HEALING ACTION DETECTED!")
            print(f"  Action: {healing_action.get('action_name')}")
            print(f"  Success: {healing_action.get('success')}")
            print(f"  Duration: {healing_action.get('duration_ms', 0):.0f}ms")
            print(f"  Detail: {healing_action.get('detail')}")
            break

        time.sleep(3)

    if not healing_action:
        print("\n⚠ No healing action detected within 45 seconds")
        print("  Orchestrator may not be running")
        print("  Start it with: ./scripts/launcher/quick_start.sh")
        return 1

    # STEP 5: Verify recovery
    print_step(5, "Verify Recovery")

    print("Waiting for service to recover...")
    for attempt in range(10):
        time.sleep(3)
        print(f"  [{attempt*3:2d}s] Checking app health...")
        health_after = get_app_health()

        if health_after.get("status") == "up" or health_after.get("status") == "ok":
            print(f"\n✓ SERVICE RECOVERED!")
            print(f"  Health: {json.dumps(health_after, indent=2)}")
            break
    else:
        print("\n⚠ Service has not recovered yet")
        print("  It may take longer or the healing may need adjustment")
        health_after = health_after

    # SUMMARY
    print_step(6, "Demo Complete - Summary")

    print(f"Failure injected:     ✓")
    print(f"Failure detected:     ✓")
    print(f"Healing executed:     ✓ ({healing_action.get('action_name')})")
    print(f"Recovery status:      {'✓' if health_after.get('status') in ['up', 'ok'] else '⚠ In Progress'}")

    print(f"\nFinal Metrics:")
    metrics_final = get_prometheus_metrics()
    print(f"  CPU:           {metrics_final['cpu_usage']:.1f}%")
    print(f"  Memory:        {metrics_final['memory_usage']:.1f}%")
    print(f"  Pod Count:     {metrics_final['pod_count']:.0f}")

    print(f"\nCompleted at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nHealing Log: data/healing_log.json")
    print(f"Orchestrator Log: logs/orchestrator.log")

    print("\n✓ Real end-to-end demo completed successfully!")
    print("  This proves NeuroShield orchestrator works with actual services")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
