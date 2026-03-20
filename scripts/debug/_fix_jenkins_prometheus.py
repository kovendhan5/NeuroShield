"""Install Prometheus plugin in Jenkins and create neuroshield-ci job."""
import requests
import time
import json
import sys

JENKINS = "http://localhost:8080"
USER = "admin"
PASS = "admin123"


def get_session():
    s = requests.Session()
    s.auth = (USER, PASS)
    cr = s.get(f"{JENKINS}/crumbIssuer/api/json", timeout=10).json()
    s.headers[cr["crumbRequestField"]] = cr["crumb"]
    print(f"[OK] CSRF crumb acquired")
    return s


def install_prometheus_plugin(s):
    print("\n=== Installing Prometheus Plugin ===")
    r = s.post(
        f"{JENKINS}/pluginManager/installNecessaryPlugins",
        headers={"Content-Type": "application/xml"},
        data='<install plugin="prometheus@2.5.3" />',
        timeout=30,
    )
    print(f"Install response: HTTP {r.status_code}")
    if r.status_code in (200, 302):
        print("[OK] Plugin install triggered")
    else:
        print(f"[WARN] Response: {r.text[:300]}")

    # Wait for installation
    print("Waiting for plugin to install...")
    for i in range(12):
        time.sleep(5)
        try:
            pr = s.get(
                f"{JENKINS}/pluginManager/api/json?tree=plugins[shortName]",
                timeout=10,
            )
            if pr.status_code == 200:
                plugins = [p["shortName"] for p in pr.json().get("plugins", [])]
                if "prometheus" in plugins:
                    print(f"[OK] Prometheus plugin installed! ({(i+1)*5}s)")
                    return True
                print(f"  ...not yet ({(i+1)*5}s, {len(plugins)} plugins loaded)")
        except Exception as e:
            print(f"  ...checking ({e})")
    print("[WARN] Plugin may need Jenkins restart")
    return False


def restart_jenkins(s):
    print("\n=== Restarting Jenkins (safe restart) ===")
    try:
        r = s.post(f"{JENKINS}/safeRestart", timeout=10)
        print(f"Restart response: HTTP {r.status_code}")
    except Exception:
        print("Restart request sent (connection closed as expected)")

    print("Waiting for Jenkins to come back...")
    time.sleep(15)
    for i in range(24):
        time.sleep(5)
        try:
            r = requests.get(f"{JENKINS}/api/json", auth=(USER, PASS), timeout=5)
            if r.status_code == 200:
                print(f"[OK] Jenkins is back! ({15 + (i+1)*5}s)")
                return True
        except Exception:
            print(f"  ...waiting ({15 + (i+1)*5}s)")
    print("[FAIL] Jenkins didn't come back in time")
    return False


def verify_prometheus_endpoint(s):
    print("\n=== Verifying /prometheus endpoint ===")
    try:
        r = requests.get(f"{JENKINS}/prometheus/", auth=(USER, PASS), timeout=10)
        print(f"HTTP {r.status_code}, content length: {len(r.text)} bytes")
        if r.status_code == 200 and len(r.text) > 100:
            # Show first few lines
            for line in r.text.split("\n")[:5]:
                print(f"  {line}")
            print("  ...")
            print("[OK] Jenkins /prometheus endpoint working!")
            return True
        else:
            print(f"[FAIL] Unexpected response")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


if __name__ == "__main__":
    s = get_session()
    installed = install_prometheus_plugin(s)
    if not installed:
        # Need restart
        restart_jenkins(s)
        s = get_session()

    ok = verify_prometheus_endpoint(s)
    if not ok:
        print("\nPlugin installed but needs restart. Restarting...")
        restart_jenkins(s)
        s = get_session()
        verify_prometheus_endpoint(s)

    print("\n=== Done ===")
