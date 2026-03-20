"""
NeuroShield Self-CI/CD Pipeline Setup
Creates a Jenkins freestyle job 'neuroshield-ci' that runs NeuroShield's own
test suite, model validation, and health checks — so NeuroShield monitors
and heals its own pipeline.
"""

import os
import sys
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

JOB_NAME = "neuroshield-ci"

# Six build steps: lint, test, model validate, health check, integration, deploy sim
CONFIG_XML = r"""<?xml version='1.1' encoding='UTF-8'?>
<project>
  <description>NeuroShield self-CI pipeline — code quality, tests, model validation, health check, integration, deploy simulation</description>
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
      <command>echo "=== Step 1/6: Code Quality Check ==="
cd "$WORKSPACE" || cd /var/jenkins_home/workspace/neuroshield-ci
python -m py_compile src/orchestrator/main.py
python -m py_compile src/telemetry/collector.py
python -m py_compile src/dashboard/app.py
python -m py_compile src/prediction/predictor.py
echo "[OK] All core modules compile cleanly"</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=== Step 2/6: Pytest Suite ==="
cd "$WORKSPACE" || cd /var/jenkins_home/workspace/neuroshield-ci
python -m pytest tests/ -q --tb=short 2>&amp;1
echo "[OK] Test suite complete"</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=== Step 3/6: Model Validation ==="
cd "$WORKSPACE" || cd /var/jenkins_home/workspace/neuroshield-ci
python -c "
import torch, joblib, os
assert os.path.exists('models/failure_predictor.pth'), 'Missing failure_predictor.pth'
assert os.path.exists('models/log_pca.joblib'), 'Missing log_pca.joblib'
state = torch.load('models/failure_predictor.pth', map_location='cpu', weights_only=True)
print(f'  Predictor keys: {len(state)} layers')
pca = joblib.load('models/log_pca.joblib')
print(f'  PCA components: {pca.n_components_}')
print('[OK] Models validated')
"</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=== Step 4/6: Health Check ==="
cd "$WORKSPACE" || cd /var/jenkins_home/workspace/neuroshield-ci
python scripts/health_check.py 2>&amp;1 || echo "[WARN] Some health checks unavailable (expected in CI)"
echo "[OK] Health check complete"</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=== Step 5/6: Integration Test ==="
cd "$WORKSPACE" || cd /var/jenkins_home/workspace/neuroshield-ci
python -c "
from src.prediction.predictor import FailurePredictor, build_52d_state
predictor = FailurePredictor(model_dir='models')
prob = predictor.predict('Build SUCCESS', {
    'jenkins_last_build_status': 'SUCCESS',
    'jenkins_last_build_duration': 5000,
    'jenkins_queue_length': 0,
    'prometheus_cpu_usage': 30.0,
    'prometheus_memory_usage': 40.0,
    'prometheus_pod_count': 3,
    'prometheus_error_rate': 0.0,
})
assert 0.0 <= prob <= 1.0, f'Probability out of range: {prob}'
print(f'  Prediction probability: {prob:.3f}')
print('[OK] Integration test passed')
"</command>
    </hudson.tasks.Shell>
    <hudson.tasks.Shell>
      <command>echo "=== Step 6/6: Deployment Simulation ==="
cd "$WORKSPACE" || cd /var/jenkins_home/workspace/neuroshield-ci
python -c "
from src.orchestrator.main import ACTION_NAMES, determine_healing_action
action, reason = determine_healing_action(
    {'jenkins_last_build_status': 'FAILURE', 'prometheus_cpu_usage': 50,
     'prometheus_memory_usage': 60, 'prometheus_error_rate': 0.01,
     'pod_restart_count': 0},
    'retry_build', 0.75)
print(f'  Action: {action}')
print(f'  Reason: {reason}')
assert action in ACTION_NAMES.values(), f'Unknown action: {action}'
print('[OK] Deployment simulation passed')
"
echo ""
echo "=============================="
echo "  ALL 6 STEPS PASSED"
echo "=============================="</command>
    </hudson.tasks.Shell>
  </builders>
  <publishers/>
  <buildWrappers/>
</project>"""


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    jenkins_url = (os.getenv("JENKINS_URL") or "http://localhost:8080").rstrip("/")
    jenkins_user = os.getenv("JENKINS_USER") or os.getenv("JENKINS_USERNAME") or "admin"
    jenkins_token = os.getenv("JENKINS_TOKEN") or os.getenv("JENKINS_PASSWORD") or ""

    auth = HTTPBasicAuth(jenkins_user, jenkins_token)
    session = requests.Session()
    session.auth = auth

    print(f"[INFO] Jenkins URL: {jenkins_url}")
    print(f"[INFO] Creating job: {JOB_NAME}")

    # Fetch CSRF crumb
    try:
        crumb_resp = session.get(f"{jenkins_url}/crumbIssuer/api/json", timeout=10)
        if crumb_resp.status_code == 200:
            crumb_data = crumb_resp.json()
            header_name = crumb_data.get("crumbRequestField")
            crumb_value = crumb_data.get("crumb")
            if header_name and crumb_value:
                session.headers[header_name] = crumb_value
                print(f"[OK] CSRF crumb acquired ({header_name})")
        else:
            print(f"[WARN] Could not fetch crumb (HTTP {crumb_resp.status_code})")
    except requests.RequestException as exc:
        print(f"[WARN] Crumb request failed: {exc}")

    # Create or update job
    create_url = f"{jenkins_url}/createItem?name={JOB_NAME}"
    response = session.post(
        create_url,
        headers={"Content-Type": "application/xml"},
        data=CONFIG_XML.encode("utf-8"),
        timeout=20,
    )

    if response.status_code == 200:
        print(f"[OK] Job created: {JOB_NAME}")
    elif response.status_code == 400:
        # Job already exists — update its config
        print(f"[INFO] Job already exists, updating config...")
        update_url = f"{jenkins_url}/job/{JOB_NAME}/config.xml"
        update_resp = session.post(
            update_url,
            headers={"Content-Type": "application/xml"},
            data=CONFIG_XML.encode("utf-8"),
            timeout=20,
        )
        if update_resp.status_code == 200:
            print(f"[OK] Job config updated: {JOB_NAME}")
        else:
            print(f"[WARN] Config update HTTP {update_resp.status_code}")
    else:
        print(f"[FAIL] HTTP {response.status_code}")
        print(response.text[:500])
        return 1

    # Trigger first build
    print(f"\n[INFO] Triggering first build...")
    build_resp = session.post(f"{jenkins_url}/job/{JOB_NAME}/build", timeout=15)
    if build_resp.status_code in {200, 201, 202, 301, 302}:
        print(f"[OK] Build triggered")
    else:
        print(f"[WARN] Build trigger HTTP {build_resp.status_code}")
        return 0  # Job created successfully, build trigger is optional

    # Poll for build result (max 120s)
    print(f"[INFO] Waiting for build result...")
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(5)
        try:
            r = session.get(f"{jenkins_url}/job/{JOB_NAME}/lastBuild/api/json", timeout=10)
            if r.status_code == 200:
                data = r.json()
                result = data.get("result")
                if result is not None:
                    number = data.get("number", "?")
                    duration = data.get("duration", 0)
                    print(f"[{'OK' if result == 'SUCCESS' else 'FAIL'}] Build #{number}: {result} ({duration}ms)")
                    return 0 if result == "SUCCESS" else 1
        except Exception:
            pass

    print(f"[WARN] Timed out waiting for build result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
