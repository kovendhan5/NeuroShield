import os
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

JOB_NAME = "neuroshield-test-job"

CONFIG_XML = """<?xml version='1.1' encoding='UTF-8'?>
<project>
  <description>NeuroShield CI/CD Test Job</description>
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
      <command>echo "NeuroShield build triggered"; exit 0</command>
    </hudson.tasks.Shell>
  </builders>
  <publishers/>
  <buildWrappers/>
</project>"""


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")

    jenkins_url = (os.getenv("JENKINS_URL") or "http://localhost:8080").rstrip("/")
    jenkins_user = os.getenv("JENKINS_USER") or os.getenv("JENKINS_USERNAME") or "admin"
    jenkins_token = os.getenv("JENKINS_TOKEN") or "admin123"

    auth = HTTPBasicAuth(jenkins_user, jenkins_token)
    session = requests.Session()
    session.auth = auth

    # Fetch CSRF crumb
    try:
        crumb_resp = session.get(
            f"{jenkins_url}/crumbIssuer/api/json",
            timeout=10,
        )
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

    # Create job
    create_url = f"{jenkins_url}/createItem?name={JOB_NAME}"
    response = session.post(
        create_url,
        headers={"Content-Type": "application/xml"},
        data=CONFIG_XML.encode("utf-8"),
        timeout=20,
    )

    if response.status_code == 200:
        print(f"[OK] Job created: {JOB_NAME}")
        return 0

    if response.status_code == 400:
        print(f"[OK] Job already exists: {JOB_NAME}")
        return 0

    print(f"[FAIL] HTTP {response.status_code}")
    print(response.text[:500])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
