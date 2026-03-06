#!/usr/bin/env python3
"""Create the real 'neuroshield-app-build' freestyle Jenkins job via REST API.

Usage:
    python scripts/create_real_jenkins_job.py

The job has three shell build steps:
  1. Build stage   — echo compile
  2. Test stage    — randomly fails ~60 % of the time
  3. Deploy stage  — echo deploy (only reached if tests pass)
"""

import os
import sys
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

JOB_NAME = "neuroshield-app-build"

# Freestyle job config.xml -------------------------------------------------
# Shell steps run sequentially; if one exits non-zero the build fails.
CONFIG_XML = """\
<?xml version='1.1' encoding='UTF-8'?>
<project>
  <description>NeuroShield demo build — test stage randomly fails 60%%</description>
  <keepDependencies>false</keepDependencies>
  <properties/>
  <scm class="hudson.scm.NullSCM"/>
  <canRoam>true</canRoam>
  <disabled>false</disabled>
  <blockBuildWhenDownstreamBuilding>false</blockBuildWhenDownstreamBuilding>
  <blockBuildWhenUpstreamBuilding>false</blockBuildWhenUpstreamBuilding>
  <triggers/>
  <concurrentBuild>false</concurrentBuild>
  <builders>
    <hudson.tasks.Shell>
      <command>echo "=== Stage: Build ==="
echo "Compiling application..."
sleep 2
echo "Build complete."</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=== Stage: Test ==="
python3 -c "
import random, sys
score = random.random()
print(f'Running tests... score: {score:.2f}')
if score &lt; 0.6:
    print('TESTS FAILED — flaky test detected')
    sys.exit(1)
else:
    print('All tests passed.')
"</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=== Stage: Deploy ==="
echo "Deploying to Kubernetes..."
sleep 1
echo "Deployment successful."</command>
    </hudson.tasks.Shell>
  </builders>
  <publishers/>
  <buildWrappers/>
</project>
"""


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    jenkins_url = (os.getenv("JENKINS_URL") or "http://localhost:8080").rstrip("/")
    username = os.getenv("JENKINS_USERNAME") or os.getenv("JENKINS_USER") or "admin"
    password = os.getenv("JENKINS_PASSWORD") or os.getenv("JENKINS_TOKEN") or "admin123"

    auth = HTTPBasicAuth(username, password)
    session = requests.Session()
    session.auth = auth

    # --- CSRF crumb --------------------------------------------------------
    try:
        crumb_resp = session.get(f"{jenkins_url}/crumbIssuer/api/json", timeout=10)
        if crumb_resp.status_code == 200:
            crumb = crumb_resp.json()
            session.headers[crumb["crumbRequestField"]] = crumb["crumb"]
            print(f"[OK] CSRF crumb acquired")
        else:
            print(f"[WARN] No crumb (HTTP {crumb_resp.status_code})")
    except requests.RequestException as exc:
        print(f"[WARN] Crumb request failed: {exc}")

    # --- Delete old job if it exists (so we always use latest config) ------
    del_resp = session.post(
        f"{jenkins_url}/job/{JOB_NAME}/doDelete",
        timeout=10,
    )
    if del_resp.status_code == 200:
        print(f"[OK] Deleted old job '{JOB_NAME}'")
    elif del_resp.status_code == 404:
        print(f"[OK] No previous job to delete")
    else:
        # 302 redirect is also success for doDelete
        if del_resp.status_code in (302, 301):
            print(f"[OK] Deleted old job '{JOB_NAME}'")

    # --- Create job --------------------------------------------------------
    create_url = f"{jenkins_url}/createItem?name={JOB_NAME}"
    resp = session.post(
        create_url,
        headers={"Content-Type": "application/xml"},
        data=CONFIG_XML.encode("utf-8"),
        timeout=20,
    )

    if resp.status_code == 200:
        print(f"[OK] Job created: {JOB_NAME}")
        print(f"     View at: {jenkins_url}/job/{JOB_NAME}/")
        return 0

    if resp.status_code == 400 and "already exists" in resp.text.lower():
        # Update existing
        update_url = f"{jenkins_url}/job/{JOB_NAME}/config.xml"
        up = session.post(
            update_url,
            headers={"Content-Type": "application/xml"},
            data=CONFIG_XML.encode("utf-8"),
            timeout=20,
        )
        if up.status_code == 200:
            print(f"[OK] Job updated: {JOB_NAME}")
            return 0
        print(f"[FAIL] Update HTTP {up.status_code}: {up.text[:300]}")
        return 1

    print(f"[FAIL] HTTP {resp.status_code}: {resp.text[:500]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
