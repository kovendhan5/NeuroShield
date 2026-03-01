import os
import random
import sys
from pathlib import Path

import requests


def load_env(env_path: Path) -> None:
    # Minimal .env parser to avoid extra dependencies.
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_crumb(session: requests.Session, base_url: str) -> dict:
    crumb_url = f"{base_url}/crumbIssuer/api/json"
    resp = session.get(crumb_url, timeout=10)
    if resp.status_code != 200:
        return {}
    data = resp.json()
    return {data.get("crumbRequestField", ""): data.get("crumb", "")}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    load_env(env_path)

    jenkins_url = os.environ.get("JENKINS_URL", "").rstrip("/")
    jenkins_user = os.environ.get("JENKINS_USERNAME", "")
    jenkins_token = os.environ.get("JENKINS_TOKEN", "")

    if not (jenkins_url and jenkins_user and jenkins_token):
        print("Missing JENKINS_URL, JENKINS_USERNAME, or JENKINS_TOKEN in .env")
        return 1

    pipeline_script = """pipeline {
  agent any
  stages {
    stage('Deploy Dummy App') {
      steps {
        sh 'echo "Deploying dummy app..." && sleep 5'
      }
    }
    stage('Inject Random Failure') {
      steps {
        script {
          def rand = new Random().nextInt(100)
          if (rand < 30) {
            echo 'Injecting failure: deleting dummy-app pod'
            sh 'kubectl delete pod dummy-app -n default --ignore-not-found'
          } else {
            echo 'No failure injected'
          }
        }
      }
    }
    stage('Monitor Health') {
      steps {
        sh 'kubectl get pods -n default | grep dummy-app'
      }
    }
  }
}
"""

    config_xml = f"""<?xml version='1.1' encoding='UTF-8'?>
<flow-definition plugin=\"workflow-job\">
  <description>NeuroShield test pipeline job</description>
  <keepDependencies>false</keepDependencies>
  <properties/>
  <definition class=\"org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition\" plugin=\"workflow-cps\">
    <script>{pipeline_script}</script>
    <sandbox>true</sandbox>
  </definition>
  <triggers/>
  <disabled>false</disabled>
</flow-definition>
"""

    session = requests.Session()
    session.auth = (jenkins_user, jenkins_token)

    headers = {"Content-Type": "application/xml"}
    headers.update(get_crumb(session, jenkins_url))

    job_name = "neuroshield-test-job"
    create_url = f"{jenkins_url}/createItem?name={job_name}"

    resp = session.post(create_url, headers=headers, data=config_xml.encode("utf-8"), timeout=20)
    if resp.status_code in (200, 201):
        print(f"Job created: {job_name}")
        return 0
    if resp.status_code == 400 and "already exists" in resp.text.lower():
        print(f"Job already exists: {job_name}")
        return 0

    print(f"Failed to create job. Status: {resp.status_code}")
    print(resp.text)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())




