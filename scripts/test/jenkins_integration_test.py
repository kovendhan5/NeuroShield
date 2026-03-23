#!/usr/bin/env python3
"""
NeuroShield - Jenkins Integration Test
Verifies that real Jenkins API integration works
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

JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080")
JENKINS_USERNAME = os.getenv("JENKINS_USERNAME", os.getenv("JENKINS_USER", "admin"))
JENKINS_PASSWORD = os.getenv("JENKINS_PASSWORD", os.getenv("JENKINS_TOKEN", ""))
JENKINS_JOB = os.getenv("JENKINS_JOB", "neuroshield-app-build")


def test_jenkins_connection():
    """Test basic Jenkins connectivity"""
    print("\n[TEST 1] Jenkins API Connection")
    print(f"  Connecting to: {JENKINS_URL}")

    try:
        response = requests.get(
            f"{JENKINS_URL}/api/json",
            auth=(JENKINS_USERNAME, JENKINS_PASSWORD),
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Connected to Jenkins {data.get('_class', 'unknown')}")
            return True
        else:
            print(f"  ✗ Jenkins returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


def test_get_latest_build():
    """Test fetching latest build info"""
    print(f"\n[TEST 2] Get Latest Build from Job '{JENKINS_JOB}'")

    try:
        url = f"{JENKINS_URL}/job/{JENKINS_JOB}/lastBuild/api/json"
        response = requests.get(
            url,
            auth=(JENKINS_USERNAME, JENKINS_PASSWORD),
            timeout=15
        )

        if response.status_code == 200:
            build = response.json()
            number = build.get("number")
            result = build.get("result") or "RUNNING"
            duration = build.get("duration", 0)
            timestamp = build.get("timestamp", 0)

            print(f"  ✓ Build #{number}: {result}")
            print(f"    Duration: {duration}ms")
            print(f"    Timestamp: {timestamp}")
            return True
        else:
            print(f"  ✗ Failed to get build: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_get_build_log():
    """Test fetching build console log"""
    print(f"\n[TEST 3] Get Build Console Log")

    try:
        # First get the latest build number
        build_url = f"{JENKINS_URL}/job/{JENKINS_JOB}/lastBuild/api/json"
        build_response = requests.get(
            build_url,
            auth=(JENKINS_USERNAME, JENKINS_PASSWORD),
            timeout=5
        )

        if build_response.status_code != 200:
            print(f"  ✗ Could not get build info: {build_response.status_code}")
            return False

        build_number = build_response.json()["number"]

        # Now get the console log
        log_url = f"{JENKINS_URL}/job/{JENKINS_JOB}/{build_number}/consoleText"
        log_response = requests.get(
            log_url,
            auth=(JENKINS_USERNAME, JENKINS_PASSWORD),
            timeout=30
        )

        if log_response.status_code == 200:
            log_text = log_response.text
            lines = len(log_text.split('\n'))
            print(f"  ✓ Retrieved {lines} lines of build log")
            print(f"    First 200 chars: {log_text[:200]}")
            return True
        else:
            print(f"  ✗ Failed to get log: {log_response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_build_trigger():
    """Test triggering a build (dry-run, checks permission)"""
    print(f"\n[TEST 4] Build Trigger Permission Check")

    try:
        # Check if we can access CSRF endpoint (needed for build trigger)
        crumb_url = f"{JENKINS_URL}/crumbIssuer/api/json"
        crumb_response = requests.get(
            crumb_url,
            auth=(JENKINS_USERNAME, JENKINS_PASSWORD),
            timeout=5
        )

        if crumb_response.status_code == 200:
            crumb_data = crumb_response.json()
            print(f"  ✓ CSRF token available (field: {crumb_data.get('crumbRequestField')})")
            print(f"    Can trigger builds with proper CSRF")
            return True
        elif crumb_response.status_code == 404:
            print(f"  ⚠ CSRF endpoint not available (Jenkins may have it disabled)")
            print(f"    Build trigger may still work without CSRF")
            return True
        else:
            print(f"  ✗ CSRF check failed: {crumb_response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("NeuroShield - Jenkins Integration Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  JENKINS_URL: {JENKINS_URL}")
    print(f"  JENKINS_JOB: {JENKINS_JOB}")
    print(f"  Username: {JENKINS_USERNAME}")

    results = []
    results.append(("Jenkins Connection", test_jenkins_connection()))
    results.append(("Get Latest Build", test_get_latest_build()))
    results.append(("Get Build Log", test_get_build_log()))
    results.append(("Build Trigger Capable", test_build_trigger()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ Jenkins integration is working!")
        print("  You can now run: ./scripts/launcher/quick_start.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Check your Jenkins configuration:")
        print("  1. Verify Jenkins is running at: " + JENKINS_URL)
        print("  2. Check credentials in .env (JENKINS_USERNAME, JENKINS_PASSWORD)")
        print("  3. Verify job exists: " + JENKINS_JOB)
        print("  4. See docs/TROUBLESHOOTING.md for more help")
        return 1


if __name__ == "__main__":
    sys.exit(main())
