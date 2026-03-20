"""Write self_ci_status.json with build history from Jenkins."""
import json, time, requests

JENKINS = "http://localhost:8080"
s = requests.Session()
s.auth = ("admin", "11e8637529db35ae8f56900be49b5cb376")

r = s.get(f"{JENKINS}/job/neuroshield-ci/api/json?tree=builds[number,result,timestamp,duration]", timeout=10)
if r.status_code != 200:
    print(f"ERROR: HTTP {r.status_code}")
    raise SystemExit(1)

all_builds = r.json().get("builds", [])
print(f"Found {len(all_builds)} builds in Jenkins")

builds_arr = []
for b in all_builds:
    ts = b.get("timestamp", 0)
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts / 1000)) if ts else ""
    builds_arr.append({
        "number": b["number"],
        "result": b.get("result", "UNKNOWN"),
        "duration_ms": b.get("duration", 0),
        "timestamp_ms": ts,
        "timestamp_str": ts_str,
    })

latest = builds_arr[0] if builds_arr else {}
status = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "build_number": latest.get("number", 0),
    "result": latest.get("result", "UNKNOWN"),
    "duration_ms": latest.get("duration_ms", 0),
    "reason": "Self-CI passed" if latest.get("result") == "SUCCESS" else f"Build #{latest.get('number')}: {latest.get('result')}",
    "active": latest.get("result") != "SUCCESS",
    "builds": builds_arr,
}

with open("data/self_ci_status.json", "w") as f:
    json.dump(status, f, indent=2)

print(f"Wrote data/self_ci_status.json with {len(builds_arr)} builds:")
for b in builds_arr:
    print(f"  #{b['number']}: {b['result']} at {b['timestamp_str']} ({b['duration_ms']}ms)")
