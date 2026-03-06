"""Trigger 5 Jenkins builds and print results."""
import requests
import time

s = requests.Session()
s.auth = ("admin", "admin123")
r = s.get("http://localhost:8080/crumbIssuer/api/json", timeout=5)
d = r.json()
s.headers[d["crumbRequestField"]] = d["crumb"]

for i in range(5):
    s.post("http://localhost:8080/job/neuroshield-app-build/build", timeout=10)
    print(f"Build {i+1} triggered, waiting...")
    time.sleep(7)

time.sleep(5)
jr = s.get(
    "http://localhost:8080/job/neuroshield-app-build/api/json?tree=builds[number,result,duration]",
    timeout=5,
)
builds = jr.json()["builds"][:8]
print("\nBuild results:")
for b in builds:
    icon = "PASS" if b.get("result") == "SUCCESS" else "FAIL"
    print(f"  #{b['number']}: {b.get('result', 'RUNNING')} ({icon}) - {b.get('duration', 0)}ms")

successes = sum(1 for b in builds if b.get("result") == "SUCCESS")
print(f"\nPass rate: {successes}/{len(builds)} = {successes/len(builds)*100:.0f}%")
