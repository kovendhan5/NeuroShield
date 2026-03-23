#!/usr/bin/env python3
"""
NeuroShield v3 - Full Verification Script
Checks that everything works end-to-end
"""

import subprocess
import time
import requests
import json
import sys
import sqlite3

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def verify_prerequisites():
    """Check Docker and dependencies"""
    print_header("1. VERIFYING PREREQUISITES")

    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Docker: {result.stdout.strip()}")
        else:
            print("[FAIL] Docker not found")
            return False
    except Exception as e:
        print(f"[FAIL] Docker not found: {e}")
        return False

    # Check Docker daemon
    try:
        subprocess.run(['docker', 'ps'], capture_output=True, check=True)
        print("[OK] Docker daemon running")
    except Exception as e:
        print(f"[FAIL] Docker daemon not running: {e}")
        return False

    # Check Python
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True)
        print(f"[OK] Python: {result.stdout.strip()}")
    except:
        print("[FAIL] Python3 not found")
        return False

    return True

def verify_containers():
    """Check containers are running"""
    print_header("2. VERIFYING CONTAINERS")

    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=neuroshield'],
            capture_output=True, text=True, check=True
        )

        if 'neuroshield' in result.stdout:
            print("[OK] Orchestrator container running")
            print(result.stdout)
            return True
        else:
            print("[FAIL] Orchestrator container not running")
            return False
    except Exception as e:
        print(f"[FAIL] Error checking containers: {e}")
        return False

def verify_api():
    """Check API is responding"""
    print_header("3. VERIFYING API")

    try:
        # Health check
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("[OK] /health endpoint: OK")
            print(f"  Response: {response.json()}")
        else:
            print(f"[FAIL] /health endpoint: {response.status_code}")
            return False

        # Status endpoint
        response = requests.get('http://localhost:8000/api/status', timeout=5)
        if response.status_code == 200:
            print("[OK] /api/status endpoint: OK")
            data = response.json()
            print(f"  State: {data.get('state')}")
            print(f"  Anomaly Score: {data.get('anomaly_score')}")
        else:
            print(f"[FAIL] /api/status endpoint: {response.status_code}")
            return False

        # History endpoint
        response = requests.get('http://localhost:8000/api/history?limit=5', timeout=5)
        if response.status_code == 200:
            print("[OK] /api/history endpoint: OK")
            data = response.json()
            print(f"  Stored actions: {data.get('count', 0)}")
        else:
            print(f"[FAIL] /api/history endpoint: {response.status_code}")
            return False

        return True
    except Exception as e:
        print(f"[FAIL] API not responding: {e}")
        return False

def verify_dashboard():
    """Check dashboard is served"""
    print_header("4. VERIFYING DASHBOARD")

    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        if response.status_code == 200 and 'NeuroShield' in response.text:
            print("[OK] Dashboard HTML: OK")
            lines = len(response.text.split('\n'))
            print(f"  Lines of HTML: {lines}")
            return True
        else:
            print(f"[FAIL] Dashboard not found: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Dashboard not accessible: {e}")
        return False

def verify_database():
    """Check database and data"""
    print_header("5. VERIFYING DATABASE")

    try:
        import os
        db_path = 'data/neuroshield.db'

        if not os.path.exists(db_path):
            print(f"[FAIL] Database not found at {db_path}")
            return False

        size = os.path.getsize(db_path)
        print(f"[OK] Database exists: {size} bytes")

        # Connect and check tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table';")
        table_count = cursor.fetchone()[0]
        print(f"[OK] Tables: {table_count}")

        # Check data
        cursor.execute("SELECT count(*) FROM events;")
        event_count = cursor.fetchone()[0]
        print(f"  Events logged: {event_count}")

        cursor.execute("SELECT count(*) FROM actions;")
        action_count = cursor.fetchone()[0]
        print(f"  Actions logged: {action_count}")

        cursor.execute("SELECT count(*) FROM metrics;")
        metric_count = cursor.fetchone()[0]
        print(f"  Metrics logged: {metric_count}")

        conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] Database error: {e}")
        return False

def verify_logs():
    """Check log file"""
    print_header("6. VERIFYING LOGS")

    try:
        import os
        log_path = 'logs/neuroshield.log'

        if not os.path.exists(log_path):
            print(f"[!] Log file not found (will be created)")
            return True

        size = os.path.getsize(log_path)
        print(f"[OK] Log file exists: {size} bytes")

        # Show last 3 lines
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if lines:
                print("  Last 3 entries:")
                for line in lines[-3:]:
                    print(f"    {line.strip()}")

        return True
    except Exception as e:
        print(f"[FAIL] Log file error: {e}")
        return False

def verify_demo_ready():
    """Check if system is ready for demo"""
    print_header("7. DEMO READINESS")

    try:
        # Trigger a cycle
        response = requests.post('http://localhost:8000/api/cycle/trigger', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("[OK] Orchestration cycle: OK")
            print(f"  Cycle time: {data.get('duration_ms')}ms")
            print(f"  Anomalies detected: {len(data.get('anomalies', []))}")
            print(f"  Actions taken: {len(data.get('actions_taken', []))}")
            return True
        else:
            print(f"[FAIL] Cycle trigger failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Demo test failed: {e}")
        return False

def main():
    """Run all verifications"""
    print("\n")
    print("="*62)
    print("| Verification: NeuroShield v3 - Full System Check            |")
    print("="*62)

    checks = [
        ("Prerequisites", verify_prerequisites),
        ("Containers", verify_containers),
        ("API", verify_api),
        ("Dashboard", verify_dashboard),
        ("Database", verify_database),
        ("Logs", verify_logs),
        ("Demo Readiness", verify_demo_ready),
    ]

    results = []
    for name, check_func in checks:
        try:
            results.append((name, check_func()))
        except Exception as e:
            print(f"\n[FAIL] {name} check crashed: {e}")
            results.append((name, False))

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status:8} {name}")

    print(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        print("\n[OK] SYSTEM IS READY FOR USE")
        print("\nNext steps:")
        print("  1. Open dashboard: http://localhost:8000")
        print("  2. Run demo: python demo.py")
        print("  3. Check status: bash scripts/status.sh")
        sys.exit(0)
    else:
        print("\n[FAIL] SOME CHECKS FAILED")
        print("Fix issues and try again")
        sys.exit(1)

if __name__ == '__main__':
    main()
