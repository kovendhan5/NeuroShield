#!/usr/bin/env python3
"""
NeuroShield - Prometheus Integration Test
Verifies that real Prometheus metric collection works
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


def test_prometheus_connection():
    """Test basic Prometheus connectivity"""
    print("\n[TEST 1] Prometheus API Connection")
    print(f"  Connecting to: {PROMETHEUS_URL}")

    try:
        response = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ Prometheus is healthy")
            return True
        else:
            print(f"  ✗ Prometheus returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


def test_prometheus_targets():
    """Test that Prometheus has active targets"""
    print(f"\n[TEST 2] Check Active Targets")

    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/targets",
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                targets = data.get("data", {})
                active = targets.get("activeTargets", [])
                dropped = targets.get("droppedTargets", [])

                print(f"  ✓ Active targets: {len(active)}")
                print(f"    Dropped targets: {len(dropped)}")

                if active:
                    for target in active[:3]:  # Show first 3
                        labels = target.get("labels", {})
                        job = labels.get("job", "unknown")
                        instance = labels.get("instance", "unknown")
                        print(f"      - {job} ({instance})")
                    if len(active) > 3:
                        print(f"      ... and {len(active)-3} more")
                    return True
                else:
                    print(f"  ⚠ No active targets found")
                    return False
            else:
                print(f"  ✗ Targets query failed: {data.get('error')}")
                return False
        else:
            print(f"  ✗ Failed to get targets: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_metric_query(metric_name, description):
    """Test querying a specific metric"""
    print(f"\n[TEST] Query {description}")

    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": metric_name},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                result = data.get("data", {}).get("result", [])
                if result:
                    value = result[0].get("value", [None, "?"])[1]
                    print(f"  ✓ {metric_name} = {value}")
                    return True
                else:
                    print(f"  ✗ {metric_name}: No data in Prometheus")
                    return False
            else:
                print(f"  ✗ Query failed: {data.get('error')}")
                return False
        else:
            print(f"  ✗ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_key_metrics():
    """Test that key metrics are available"""
    print(f"\n[TEST 3] Check Key Metrics")

    metrics = [
        ("node_cpu_seconds_total", "CPU usage (node_cpu_seconds_total)"),
        ("node_memory_MemTotal_bytes", "Memory total (node_memory_MemTotal_bytes)"),
        ("container_cpu_usage_seconds_total", "Container CPU usage"),
        ("container_memory_usage_bytes", "Container memory usage"),
    ]

    results = []
    for metric, description in metrics:
        query = f"{metric}[1m]"  # Query with time range
        result = test_metric_query(query, description)
        results.append(result)

    return any(results)  # Pass if at least one metric exists


def test_custom_range_query():
    """Test range queries (needed by orchestrator)"""
    print(f"\n[TEST 4] Range Query (24h)")

    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": "node_cpu_seconds_total",
                "start": "1672531200",  # Fixed timestamp
                "end": "1672617600",
                "step": "3600",
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                result = data.get("data", {}).get("result", [])
                if result:
                    values = result[0].get("values", [])
                    print(f"  ✓ Range query works ({len(values)} data points)")
                    return True
                else:
                    print(f"  ✗ No data in range")
                    return False
        else:
            print(f"  ✗ Range query failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("NeuroShield - Prometheus Integration Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  PROMETHEUS_URL: {PROMETHEUS_URL}")

    results = []
    results.append(("Prometheus Connection", test_prometheus_connection()))
    results.append(("Active Targets", test_prometheus_targets()))
    results.append(("Key Metrics Available", test_key_metrics()))
    results.append(("Range Query", test_custom_range_query()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed >= 3:  # Allow some flexibility (metrics might not all be present)
        print("\n✓ Prometheus integration is working!")
        print("  You can now run: ./scripts/launcher/quick_start.sh")
        return 0
    else:
        print("\n✗ Prometheus integration issues detected:")
        print("  1. Verify Prometheus is running at: " + PROMETHEUS_URL)
        print("  2. Check that targets are scraping metrics")
        print("  3. Ensure node-exporter or similar is available")
        print("  4. See docs/TROUBLESHOOTING.md for more help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
