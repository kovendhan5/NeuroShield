import os
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    jenkins_url = (os.getenv("JENKINS_URL") or "").rstrip("/")
    jenkins_user = os.getenv("JENKINS_USERNAME") or ""
    jenkins_token = os.getenv("JENKINS_TOKEN") or ""

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
        sh ''' if [ $((RANDOM % 10)) -lt 3 ]; then echo "Injecting Failure: Deleting dummy-app pod"; kubectl delete pod dummy-app -n default --ignore-not-found; else echo "No failure injected this time"; fi '''
      }
    }
    stage('Monitor Health') {
      steps {
        sh 'echo "Checking dummy-app health..."; kubectl get pods -n default | grep dummy-app || echo "dummy-app might not be present or healthy"'
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

    create_url = (
      f"{jenkins_url}/createItem?name=neuroshield-test-job"
      "&mode=org.jenkinsci.plugins.workflow.job.WorkflowJob"
    )
    auth = HTTPBasicAuth(jenkins_user, jenkins_token)

    headers = {"Content-Type": "application/xml"}
    crumb_resp = requests.get(
      f"{jenkins_url}/crumbIssuer/api/json",
      auth=auth,
      timeout=10,
    )
    if crumb_resp.status_code == 200:
      crumb_data = crumb_resp.json()
      header_name = crumb_data.get("crumbRequestField")
      crumb_value = crumb_data.get("crumb")
      if header_name and crumb_value:
        headers[header_name] = crumb_value

    response = requests.post(
      create_url,
      auth=auth,
      headers=headers,
      data=config_xml.encode("utf-8"),
      timeout=20,
    )

    if response.status_code in (200, 201):
        print("Job created: neuroshield-test-job")
        return 0

    # Job already exists — treat as success
    if response.status_code in (400, 409) and "already exists" in response.text.lower():
        print("[SKIP] Job already exists \u2014 continuing")
        return 0

    print(f"Failed to create job. Status: {response.status_code}")
    print(response.text)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
