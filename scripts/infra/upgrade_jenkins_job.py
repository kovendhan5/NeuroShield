#!/usr/bin/env python3
"""Upgrade the 'neuroshield-app-build' Jenkins job to a 4-stage pipeline.

Stages:
  1. Dependency Install — checks for demo_requirements_broken.txt; if present
     the install fails (simulates a dependency conflict).
  2. Build / Compile   — echoes compile steps, exits 0.
  3. Test Suite        — random pass/fail (~35 % fail rate).
  4. Deploy to K8s     — echoes deploy success.

Usage:
    python scripts/upgrade_jenkins_job.py
"""

import os
import sys
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

JENKINS_URL = (os.getenv("JENKINS_URL") or "http://localhost:8080").rstrip("/")
JENKINS_USER = os.getenv("JENKINS_USERNAME") or os.getenv("JENKINS_USER") or "admin"
JENKINS_PASS = os.getenv("JENKINS_PASSWORD") or os.getenv("JENKINS_TOKEN") or "admin123"
JOB_NAME = os.getenv("JENKINS_JOB", "neuroshield-app-build")

CONFIG_XML = """\
<?xml version='1.1' encoding='UTF-8'?>
<project>
  <description>NeuroShield 4-stage demo build — dependency install, build, test, deploy</description>
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
      <command>echo "=========================================="
echo "  Stage 1 / 4 : Dependency Install"
echo "=========================================="
echo "Checking for dependency manifest..."
sleep 1
if [ -f /tmp/demo_requirements_broken.txt ]; then
    echo ""
    echo "ERROR: Conflicting dependency versions detected!"
    cat /tmp/demo_requirements_broken.txt
    echo ""
    echo "pip install would fail — cannot satisfy constraints."
    exit 1
fi
echo "Dependencies OK — no conflicts found."
echo "Installing packages..."
sleep 2
echo "All dependencies installed successfully."</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=========================================="
echo "  Stage 2 / 4 : Build / Compile"
echo "=========================================="
echo "Compiling application modules..."
sleep 2
echo "Linking shared objects..."
sleep 1
echo "Build artefacts generated."
echo "Build stage PASSED."</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=========================================="
echo "  Stage 3 / 4 : Test Suite"
echo "=========================================="
echo "Running unit + integration tests..."
sleep 2
score=$(awk 'BEGIN{srand(); printf "%d", rand()*100}')
echo "Test score: $score / 100"
if [ "$score" -gt 35 ]; then
    echo "All tests PASSED ($score%)"
else
    echo "Tests FAILED — score $score% below threshold"
    exit 1
fi</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=========================================="
echo "  Stage 4 / 4 : Deploy to Kubernetes"
echo "=========================================="
echo "Pushing image to registry..."
sleep 1
echo "Applying Kubernetes manifests..."
sleep 1
echo "Rolling out deployment/dummy-app..."
sleep 1
echo "Deployment successful — pods healthy."</command>
    </hudson.tasks.Shell>
  </builders>
  <publishers/>
  <buildWrappers/>
</project>
"""


def main() -> int:
    auth = HTTPBasicAuth(JENKINS_USER, JENKINS_PASS)
    session = requests.Session()
    session.auth = auth

    # CSRF crumb
    try:
        r = session.get(f"{JENKINS_URL}/crumbIssuer/api/json", timeout=10)
        if r.status_code == 200:
            crumb = r.json()
            session.headers[crumb["crumbRequestField"]] = crumb["crumb"]
            print("[OK] CSRF crumb acquired")
        else:
            print(f"[WARN] No crumb (HTTP {r.status_code})")
    except requests.RequestException as exc:
        print(f"[WARN] Crumb request failed: {exc}")

    # Delete old job
    del_resp = session.post(f"{JENKINS_URL}/job/{JOB_NAME}/doDelete", timeout=10)
    if del_resp.status_code in (200, 302, 301):
        print(f"[OK] Deleted old job '{JOB_NAME}'")
    elif del_resp.status_code == 404:
        print("[OK] No previous job to delete")

    # Create job
    resp = session.post(
        f"{JENKINS_URL}/createItem?name={JOB_NAME}",
        headers={"Content-Type": "application/xml"},
        data=CONFIG_XML.encode("utf-8"),
        timeout=20,
    )

    if resp.status_code == 200:
        print(f"[OK] 4-stage job created: {JOB_NAME}")
        print(f"     Stages: Dep Install → Build → Test → Deploy")
        print(f"     View: {JENKINS_URL}/job/{JOB_NAME}/")
        return 0

    if resp.status_code == 400 and "already exists" in resp.text.lower():
        up = session.post(
            f"{JENKINS_URL}/job/{JOB_NAME}/config.xml",
            headers={"Content-Type": "application/xml"},
            data=CONFIG_XML.encode("utf-8"),
            timeout=20,
        )
        if up.status_code == 200:
            print(f"[OK] 4-stage job updated: {JOB_NAME}")
            return 0
        print(f"[FAIL] Update HTTP {up.status_code}: {up.text[:300]}")
        return 1

    print(f"[FAIL] HTTP {resp.status_code}: {resp.text[:500]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
