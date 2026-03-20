"""Final verification of all fixes."""
import json, requests

print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

# CHECK 1: Prometheus Targets
print("\nCHECK 1: Prometheus Targets (localhost:9090/targets)")
r1 = "FAIL"
try:
    r = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
    data = r.json()
    all_up = True
    for t in data["data"]["activeTargets"]:
        job = t["labels"]["job"]
        health = t["health"]
        err = t.get("lastError", "")
        mark = "PASS" if health == "up" else "FAIL"
        if health != "up":
            all_up = False
        print(f"  [{mark}] {job}: {health}" + (f" ERROR: {err}" if err else ""))
    r1 = "PASS" if all_up else "FAIL"
except Exception as e:
    print(f"  ERROR: {e}")

# CHECK 2: incident_board_open_incidents
print("\nCHECK 2: Query incident_board_open_incidents")
r2 = "FAIL"
try:
    r = requests.get("http://localhost:9090/api/v1/query",
                     params={"query": "incident_board_open_incidents"}, timeout=5)
    results = r.json()["data"]["result"]
    if results:
        val = results[0]["value"][1]
        print(f"  PASS - Value: {val}")
        r2 = "PASS"
    else:
        print(f"  FAIL - No data")
except Exception as e:
    print(f"  ERROR: {e}")

# CHECK 3: neuroshield_healing_actions_total
print("\nCHECK 3: Query neuroshield_healing_actions_total")
r3 = "FAIL"
try:
    r = requests.get("http://localhost:9090/api/v1/query",
                     params={"query": "neuroshield_healing_actions_total"}, timeout=5)
    results = r.json()["data"]["result"]
    if results:
        val = results[0]["value"][1]
        print(f"  PASS - Value: {val}")
        r3 = "PASS"
    else:
        print(f"  FAIL - No data")
except Exception as e:
    print(f"  ERROR: {e}")

# CHECK 4: Self-CI Status
print("\nCHECK 4: Self-CI History (data/self_ci_status.json)")
r4 = "FAIL"
try:
    with open("data/self_ci_status.json") as f:
        ci = json.load(f)
    builds = ci.get("builds", [])
    if len(builds) >= 3:
        print(f"  PASS - {len(builds)} builds:")
        for b in builds:
            print(f"    #{b['number']}: {b['result']} at {b['timestamp_str']} ({b['duration_ms']}ms)")
        r4 = "PASS"
    else:
        print(f"  FAIL - Only {len(builds)} builds")
except FileNotFoundError:
    print(f"  FAIL - File not found")
except Exception as e:
    print(f"  ERROR: {e}")

# CHECK 5: All services
print("\nCHECK 5: All services reachable")
services = {
    "API /prometheus_metrics": "http://localhost:8502/prometheus_metrics",
    "Dashboard": "http://localhost:8503",
    "IncidentBoard": "http://localhost:5000/health",
    "Jenkins": "http://localhost:8080/api/json",
    "Prometheus": "http://localhost:9090/-/healthy",
}
r5 = "PASS"
for name, url in services.items():
    try:
        auth = ("admin", "admin123") if "8080" in url else None
        r = requests.get(url, timeout=5, auth=auth)
        mark = "PASS" if r.status_code == 200 else "FAIL"
        if r.status_code != 200:
            r5 = "FAIL"
        print(f"  [{mark}] {name}: HTTP {r.status_code}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        r5 = "FAIL"

# API metrics content
print("\nCHECK 6: API /prometheus_metrics content")
r6 = "FAIL"
try:
    r = requests.get("http://localhost:8502/prometheus_metrics", timeout=5)
    if r.status_code == 200 and "neuroshield_healing_actions_total" in r.text:
        print(f"  PASS - Prometheus text format:")
        for line in r.text.strip().split("\n"):
            if not line.startswith("#"):
                print(f"    {line}")
        r6 = "PASS"
    else:
        print(f"  FAIL - HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"  ERROR: {e}")

# SUMMARY
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
checks = [
    ("1. Prometheus targets all UP", r1),
    ("2. incident_board_open_incidents", r2),
    ("3. neuroshield_healing_actions_total", r3),
    ("4. Self-CI >= 3 builds", r4),
    ("5. All services reachable", r5),
    ("6. API /prometheus_metrics content", r6),
]
for label, result in checks:
    print(f"  {label}: {result}")
overall = "ALL PASS" if all(r == "PASS" for _, r in checks) else "SOME FAILED"
print(f"\n  OVERALL: {overall}")
