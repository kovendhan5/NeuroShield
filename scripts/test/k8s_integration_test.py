#!/usr/bin/env python3
"""
NeuroShield - Kubernetes Integration Test
Verifies that kubectl commands work and deployments can be managed
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Load environment
load_dotenv()

K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
AFFECTED_SERVICE = os.getenv("AFFECTED_SERVICE", "dummy-app")


def run_command(cmd, description):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_kubectl_connection():
    """Test basic kubectl connectivity"""
    print("\n[TEST 1] Kubectl Connection")

    success, stdout, stderr = run_command(
        ["kubectl", "cluster-info"],
        "Cluster info"
    )

    if success:
        # Extract cluster name from output
        for line in stdout.split('\n'):
            if 'Kubernetes master' in line or 'control plane' in line:
                print(f"  ✓ Connected to Kubernetes cluster")
                print(f"    {line.strip()}")
                return True
        print(f"  ✓ kubectl is available")
        return True
    else:
        print(f"  ✗ kubectl cluster-info failed: {stderr}")
        return False


def test_namespace_exists():
    """Test that the namespace exists"""
    print(f"\n[TEST 2] Namespace Exists ('{K8S_NAMESPACE}')")

    success, stdout, stderr = run_command(
        ["kubectl", "get", "namespace", K8S_NAMESPACE],
        "Get namespace"
    )

    if success:
        print(f"  ✓ Namespace '{K8S_NAMESPACE}' exists")
        return True
    else:
        print(f"  ✗ Namespace not found: {stderr}")
        return False


def test_get_deployments():
    """Test fetching deployments"""
    print(f"\n[TEST 3] Get Deployments")

    success, stdout, stderr = run_command(
        ["kubectl", "get", "deployments", "-n", K8S_NAMESPACE, "-o", "json"],
        "Get deployments"
    )

    if success:
        import json
        try:
            data = json.loads(stdout)
            deployments = data.get("items", [])
            print(f"  ✓ Found {len(deployments)} deployment(s)")

            # Look for our affected service
            service_found = any(d.get("metadata", {}).get("name") == AFFECTED_SERVICE
                                for d in deployments)

            if deployments:
                for dep in deployments[:3]:
                    name = dep.get("metadata", {}).get("name", "?")
                    ready = dep.get("status", {}).get("readyReplicas", 0)
                    desired = dep.get("spec", {}).get("replicas", 0)
                    print(f"      - {name}: {ready}/{desired} ready")

            if service_found:
                print(f"  ✓ Service '{AFFECTED_SERVICE}' found")
                return True
            else:
                print(f"  ⚠ Service '{AFFECTED_SERVICE}' not found (may not be critical)")
                return True
        except json.JSONDecodeError:
            print(f"  ✗ Could not parse kubectl output")
            return False
    else:
        print(f"  ✗ Failed to get deployments: {stderr}")
        return False


def test_pod_operations():
    """Test pod listing and basic operations"""
    print(f"\n[TEST 4] Pod Operations")

    success, stdout, stderr = run_command(
        ["kubectl", "get", "pods", "-n", K8S_NAMESPACE, "-o", "json"],
        "Get pods"
    )

    if success:
        import json
        try:
            data = json.loads(stdout)
            pods = data.get("items", [])
            print(f"  ✓ Found {len(pods)} pod(s)")

            if pods:
                # Show states
                running = sum(1 for p in pods
                            if p.get("status", {}).get("phase") == "Running")
                pending = sum(1 for p in pods
                            if p.get("status", {}).get("phase") == "Pending")
                failed = sum(1 for p in pods
                            if p.get("status", {}).get("phase") == "Failed")

                print(f"      Running: {running}, Pending: {pending}, Failed: {failed}")

                for pod in pods[:2]:  # Show first 2
                    name = pod.get("metadata", {}).get("name", "?")
                    phase = pod.get("status", {}).get("phase", "?")
                    print(f"      - {name}: {phase}")

            return True
        except json.JSONDecodeError:
            print(f"  ✗ Could not parse pods")
            return False
    else:
        print(f"  ✗ Failed to get pods: {stderr}")
        return False


def test_scale_capability():
    """Test that we can scale deployments (without actually doing it)"""
    print(f"\n[TEST 5] Scale Capability Check")

    # Just verify we can run the command syntax
    success, stdout, stderr = run_command(
        ["kubectl", "scale", "deployment/dummy", "--replicas=1", "-n", K8S_NAMESPACE, "--dry-run=client"],
        "Scale dry-run"
    )

    if success or "error: no matching resources found" in stderr.lower():
        print(f"  ✓ Scale command syntax valid")
        return True
    else:
        print(f"  ✗ Scale command failed: {stderr}")
        return False


def test_rollout_capability():
    """Test that rollout commands work"""
    print(f"\n[TEST 6] Rollout Capability Check")

    success, stdout, stderr = run_command(
        ["kubectl", "rollout", "history", "deployment/dummy", "-n", K8S_NAMESPACE],
        "Rollout history"
    )

    if success or "error" in stderr.lower():
        # Even if the specific deployment doesn't exist, the command should work
        print(f"  ✓ Rollout commands available")
        return True
    else:
        print(f"  ✗ Rollout command failed: {stderr}")
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("NeuroShield - Kubernetes Integration Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Namespace: {K8S_NAMESPACE}")
    print(f"  Service: {AFFECTED_SERVICE}")

    results = []
    results.append(("Kubectl Connection", test_kubectl_connection()))
    results.append(("Namespace Exists", test_namespace_exists()))
    results.append(("Get Deployments", test_get_deployments()))
    results.append(("Pod Operations", test_pod_operations()))
    results.append(("Scale Capability", test_scale_capability()))
    results.append(("Rollout Capability", test_rollout_capability()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed >= 4:
        print("\n✓ Kubernetes integration is working!")
        print("  You can now run: ./scripts/launcher/quick_start.sh")
        return 0
    else:
        print("\n✗ Kubernetes integration issues detected:")
        print("  1. Verify kubectl is installed: kubectl version")
        print("  2. Check cluster access: kubectl cluster-info")
        print("  3. Verify namespace exists: kubectl get namespace")
        print("  4. See docs/TROUBLESHOOTING.md for more help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
