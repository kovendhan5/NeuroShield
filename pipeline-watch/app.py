"""
PipelineWatch — Real-time CI/CD Monitoring Dashboard
Monitors Jenkins builds and Kubernetes pod health in real-time.
Shows live data from actual systems, no fake generation.
"""

from flask import Flask, jsonify, request, render_template_string, make_response
import threading
import time
import requests
import subprocess
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import base64

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

JENKINS_URL = os.environ.get("JENKINS_URL", "http://localhost:8080")
JENKINS_USER = os.environ.get("JENKINS_USERNAME", "admin")
JENKINS_TOKEN = os.environ.get("JENKINS_TOKEN", "11e8637529db35ae8f56900be49b5cb376")
JENKINS_JOB = "neuroshield-app-build"

# ── State ──────────────────────────────────────────────────────────────────────

state_lock = threading.Lock()
app_healthy = True
app_start_time = time.time()
request_count = 0

# Alert state: {id: {severity, title, source, created_at, auto_healed, healed_by, healed_in, build_number, pod_name, acknowledged_at}}
alerts = {}
alert_id_counter = 1000
last_jenkins_builds = {}  # {build_number: result}
resolved_alerts = []

# ── Jenkins API Helpers ────────────────────────────────────────────────────────

def _jenkins_auth():
    """Return Jenkins authentication header."""
    creds = f"{JENKINS_USER}:{JENKINS_TOKEN}"
    encoded = base64.b64encode(creds.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}

def _get_jenkins_builds(limit=8):
    """Get recent builds from Jenkins API."""
    try:
        url = f"{JENKINS_URL}/job/{JENKINS_JOB}/api/json"
        response = requests.get(url, headers=_jenkins_auth(), timeout=5)
        response.raise_for_status()
        data = response.json()

        builds = []
        for build in data.get("builds", [])[:limit]:
            builds.append({
                "number": build["number"],
                "result": build.get("result", "UNKNOWN"),  # SUCCESS, FAILURE, ABORTED
                "duration": build.get("duration", 0),
                "timestamp": build.get("timestamp", 0),
                "url": build.get("url", ""),
            })
        return sorted(builds, key=lambda x: x["number"], reverse=True)
    except Exception as e:
        return []

def _get_kubectl_pods():
    """Get pod status from Kubernetes."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        pods = []
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})

            # Get status condition
            pod_status = "Unknown"
            for condition in status.get("conditions", []):
                if condition.get("type") == "Ready":
                    pod_status = "Running" if condition.get("status") == "True" else "NotReady"
                    break

            # Count restarts
            restarts = 0
            for container_status in status.get("containerStatuses", []):
                restarts += container_status.get("restartCount", 0)

            # Age
            created = status.get("startTime", "")

            pods.append({
                "name": metadata.get("name", "unknown"),
                "status": pod_status,
                "restarts": restarts,
                "created": created,
            })
        return pods
    except Exception as e:
        return []

# ── Alert Management ───────────────────────────────────────────────────────────

def _create_alert(severity, title, source, build_number=None, pod_name=None):
    """Create a new alert if one doesn't already exist for this event."""
    global alert_id_counter

    with state_lock:
        # Check if alert already exists
        for alert_id, alert in alerts.items():
            if alert["source"] == source:
                if source == "jenkins" and alert.get("build_number") == build_number:
                    return  # Already exists
                if source == "k8s" and alert.get("pod_name") == pod_name:
                    return  # Already exists

        # Create new alert
        alert_id = alert_id_counter
        alert_id_counter += 1

        alerts[alert_id] = {
            "id": alert_id,
            "severity": severity,
            "title": title,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "auto_healed": False,
            "healed_by": None,
            "healed_in": None,
            "build_number": build_number,
            "pod_name": pod_name,
            "acknowledged_at": None,
        }

def _acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    with state_lock:
        if alert_id in alerts:
            alert = alerts.pop(alert_id)
            alert["acknowledged_at"] = datetime.now().isoformat()
            resolved_alerts.insert(0, alert)
            if len(resolved_alerts) > 5:
                resolved_alerts.pop()
            return True
    return False

def _mark_alert_healed(action, mttr, build_number=None, pod_name=None):
    """Mark an alert as auto-healed by NeuroShield."""
    with state_lock:
        for alert_id, alert in alerts.items():
            if (build_number and alert.get("build_number") == build_number) or \
               (pod_name and alert.get("pod_name") == pod_name):
                alert["auto_healed"] = True
                alert["healed_by"] = action
                alert["healed_in"] = mttr
                return

# ── Background Health Monitor ──────────────────────────────────────────────────

def _monitor_jenkins_and_k8s():
    """Background thread: monitor Jenkins builds and K8s pod status."""
    while True:
        time.sleep(15)  # Check every 15 seconds

        try:
            # Check Jenkins builds
            builds = _get_jenkins_builds(limit=1)
            if builds:
                latest = builds[0]
                if latest["result"] == "FAILURE":
                    build_num = latest["number"]
                    if build_num not in last_jenkins_builds or last_jenkins_builds[build_num] != "FAILURE":
                        _create_alert(
                            severity="CRITICAL",
                            title=f"Build #{build_num} failed",
                            source="jenkins",
                            build_number=build_num
                        )
                    last_jenkins_builds[build_num] = "FAILURE"
        except Exception as e:
            pass

        try:
            # Check K8s pod status
            pods = _get_kubectl_pods()
            for pod in pods:
                if pod["status"] != "Running":
                    _create_alert(
                        severity="CRITICAL",
                        title=f"Pod {pod['name']} is {pod['status']}",
                        source="k8s",
                        pod_name=pod["name"]
                    )
        except Exception as e:
            pass

# Start monitoring thread
threading.Thread(target=_monitor_jenkins_and_k8s, daemon=True).start()

# ── API Routes ─────────────────────────────────────────────────────────────────

@app.before_request
def _count_requests():
    global request_count
    request_count += 1
    if not app_healthy and request.path not in ['/health', '/crash', '/api/neuroshield/healed']:
        return jsonify({"error": "service unhealthy"}), 503

@app.route("/health", methods=["GET"])
def health():
    uptime = round(time.time() - app_start_time, 1)
    if app_healthy:
        return jsonify({"status": "healthy", "uptime": uptime}), 200
    return jsonify({"status": "unhealthy"}), 503

@app.route("/crash", methods=["POST"])
def crash():
    """Simulate a crash for demo purposes."""
    global app_healthy
    app_healthy = False

    def _recover():
        global app_healthy
        time.sleep(25)
        app_healthy = True

    threading.Thread(target=_recover, daemon=True).start()
    return jsonify({"message": "Crash simulated"}), 200

@app.route("/api/status", methods=["GET"])
def api_status():
    """Return live Jenkins and K8s status."""
    builds = _get_jenkins_builds(limit=1)
    pods = _get_kubectl_pods()

    pipeline_status = "HEALTHY"
    if builds and builds[0]["result"] == "FAILURE":
        pipeline_status = "CRITICAL"
    if any(p["status"] != "Running" for p in pods):
        pipeline_status = "CRITICAL"

    last_build = None
    if builds:
        b = builds[0]
        last_build = {
            "number": b["number"],
            "result": b["result"],
            "duration": b["duration"],
        }

    pod_status = None
    if pods:
        p = pods[0]
        pod_status = {
            "name": p["name"],
            "status": p["status"],
            "restarts": p["restarts"],
        }

    return jsonify({
        "pipeline_status": pipeline_status,
        "last_build": last_build,
        "pod_status": pod_status,
    })

@app.route("/api/builds", methods=["GET"])
def api_builds():
    """Return recent Jenkins builds."""
    builds = _get_jenkins_builds(limit=8)
    return jsonify(builds)

@app.route("/api/pods", methods=["GET"])
def api_pods():
    """Return Kubernetes pod status."""
    pods = _get_kubectl_pods()
    return jsonify(pods)

@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    """Return all active alerts."""
    with state_lock:
        active = list(alerts.values())
        resolved = resolved_alerts.copy()
    return jsonify({"active": active, "resolved": resolved})

@app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    if _acknowledge_alert(alert_id):
        return jsonify({"success": True}), 200
    return jsonify({"success": False, "error": "not found"}), 404

@app.route("/api/neuroshield/healed", methods=["POST"])
def neuroshield_healed():
    """NeuroShield calls this after healing an action."""
    payload = request.get_json() or {}
    action = payload.get("action", "unknown")
    mttr = payload.get("mttr", 0)
    build_number = payload.get("build_number")
    pod_name = payload.get("pod_name")

    _mark_alert_healed(action, mttr, build_number, pod_name)
    return jsonify({"success": True}), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint."""
    with state_lock:
        active_count = len(alerts)
        resolved_count = len(resolved_alerts)

    uptime = round(time.time() - app_start_time, 1)
    lines = [
        f'pipelinewatch_alerts_active {active_count}',
        f'pipelinewatch_alerts_resolved {resolved_count}',
        f'pipelinewatch_uptime_seconds {uptime}',
        f'pipelinewatch_requests_total {request_count}',
    ]
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; charset=utf-8"}

# ── HTML UI ────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PipelineWatch — Live CI/CD Monitoring</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d1a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
a{color:#00ff88;text-decoration:none}

/* Header */
.header{display:flex;justify-content:space-between;align-items:center;padding:18px 32px;border-bottom:1px solid #2a2a4a;background:linear-gradient(90deg,rgba(0,255,136,0.05),transparent)}
.header h1{font-size:1.3rem;color:#00ff88;display:flex;align-items:center;gap:8px}
.live-badge{display:flex;align-items:center;gap:6px;font-size:.85rem;font-weight:600}
.pulse{width:8px;height:8px;border-radius:50%;background:#00ff88;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,255,136,.5)}50%{box-shadow:0 0 0 8px rgba(0,255,136,0)}}
.clock{font-family:'Courier New',monospace;font-size:.95rem;color:#00ff88}

/* Stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;padding:24px 32px}
.stat-card{background:#1a1a2e;border:1px solid #2a3a4a;border-radius:8px;padding:16px;text-align:center;transition:all .3s}
.stat-card:hover{border-color:#00ff88;box-shadow:0 0 12px rgba(0,255,136,.2)}
.stat-value{font-size:1.8rem;font-weight:700;color:#00ff88;margin:8px 0}
.stat-label{font-size:.7rem;color:#888;text-transform:uppercase;letter-spacing:.5px}

/* Main content */
.container{display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:24px 32px;max-width:1600px;margin:0 auto}
.column{display:flex;flex-direction:column;gap:16px}

.section-title{font-size:.95rem;color:#888;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px}

/* Build cards */
.build-card{background:#1a1a2e;border-left:4px solid #2a2a4a;border-radius:6px;padding:14px;display:flex;justify-content:space-between;align-items:center;transition:all .3s}
.build-card.success{border-left-color:#00ff88}
.build-card.failure{border-left-color:#ff4444}
.build-info{flex:1}
.build-number{font-weight:600;color:#f0f0f0;font-size:.95rem}
.build-meta{font-size:.8rem;color:#888;margin-top:4px}
.build-status{font-weight:700;padding:4px 12px;border-radius:4px;font-size:.8rem}
.status-success{background:rgba(0,255,136,.2);color:#00ff88}
.status-failure{background:rgba(255,68,68,.2);color:#ff6666}
.build-duration{font-size:.8rem;color:#aaa}

/* Pod cards */
.pod-card{background:#1a1a2e;border:1px solid #2a3a4a;border-radius:6px;padding:14px;transition:all .3s}
.pod-card.healthy{border-color:#00ff88;background:rgba(0,255,136,.05)}
.pod-card.unhealthy{border-color:#ff4444;background:rgba(255,68,68,.05)}
.pod-name{font-weight:600;color:#f0f0f0}
.pod-details{display:flex;gap:16px;margin-top:8px;font-size:.85rem}
.pod-detail-item{display:flex;align-items:center;gap:6px}
.status-badge{padding:2px 10px;border-radius:4px;font-size:.75rem;font-weight:600}
.badge-running{background:rgba(0,255,136,.2);color:#00ff88}
.badge-notready{background:rgba(255,136,0,.2);color:#ffaa44}

/* Alerts */
.alerts-section{grid-column:1/-1}
.alert-card{background:#1a1a2e;border:1.5px solid #2a3a4a;border-radius:8px;padding:16px;margin-bottom:12px;transition:all .3s}
.alert-card.critical{border-color:#ff4444;box-shadow:0 0 8px rgba(255,68,68,.2)}
.alert-card.warning{border-color:#ff8800;box-shadow:0 0 8px rgba(255,136,0,.2)}
.alert-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px}
.alert-title{font-weight:600;color:#f0f0f0;flex:1}
.alert-badge{font-size:.65rem;font-weight:700;padding:4px 10px;border-radius:4px;text-transform:uppercase;margin-right:8px}
.badge-critical{background:#ff4444;color:#fff}
.badge-warning{background:#ff8800;color:#fff}
.alert-meta{font-size:.8rem;color:#888;margin-bottom:12px}
.alert-healed{background:rgba(0,255,136,.15);border-left:3px solid #00ff88;padding:10px;border-radius:4px;margin-bottom:12px;color:#00ff88;font-size:.85rem}
.alert-buttons{display:flex;gap:10px}
.btn-ack{background:transparent;border:1px solid #00ff88;color:#00ff88;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.8rem;font-weight:600;transition:all .2s}
.btn-ack:hover{background:#00ff88;color:#0d0d1a}

.resolved-section{opacity:.6;margin-top:20px;padding-top:20px;border-top:1px solid #2a2a4a}

/* Footer */
.footer{text-align:center;padding:18px;border-top:1px solid #2a2a4a;font-size:.75rem;color:#555;margin-top:40px}
</style>
</head>
<body>

<div class="header">
  <h1><span style="font-size:1.5rem">⚡</span> PipelineWatch</h1>
  <div class="live-badge"><div class="pulse"></div><span id="status-text">LIVE</span></div>
  <div class="clock" id="clock">--:--:--</div>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="stat-label">Pipeline Status</div>
    <div class="stat-value" id="stat-status">?</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Last Build</div>
    <div class="stat-value" id="stat-build">?</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Pod Status</div>
    <div class="stat-value" id="stat-pod">?</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Active Alerts</div>
    <div class="stat-value" id="stat-alerts">0</div>
  </div>
</div>

<div class="container">
  <div class="column">
    <div class="section-title">Recent Jenkins Builds</div>
    <div id="builds"></div>
  </div>

  <div class="column">
    <div class="section-title">Kubernetes Pods</div>
    <div id="pods"></div>
  </div>

  <div class="column alerts-section">
    <div class="section-title">Active Alerts</div>
    <div id="alerts"></div>
    <div class="resolved-section">
      <div class="section-title">Resolved (Last 5)</div>
      <div id="resolved"></div>
    </div>
  </div>
</div>

<div class="footer">PipelineWatch — Real-time CI/CD monitoring • Live data from Jenkins & Kubernetes</div>

<script>
// Update clock
function updateClock(){
  const now = new Date();
  const h = String(now.getHours()).padStart(2,'0');
  const m = String(now.getMinutes()).padStart(2,'0');
  const s = String(now.getSeconds()).padStart(2,'0');
  document.getElementById('clock').textContent = h + ':' + m + ':' + s;
}
updateClock();
setInterval(updateClock, 1000);

// Fetch and render data
async function refreshData(){
  try {
    const status = await fetch('/api/status').then(r => r.json());
    const builds = await fetch('/api/builds').then(r => r.json());
    const pods = await fetch('/api/pods').then(r => r.json());
    const alerts_data = await fetch('/api/alerts').then(r => r.json());

    // Update stats
    document.getElementById('stat-status').textContent = status.pipeline_status;
    document.getElementById('stat-status').style.color = status.pipeline_status === 'HEALTHY' ? '#00ff88' : '#ff4444';

    if (status.last_build) {
      document.getElementById('stat-build').textContent = '#' + status.last_build.number;
      document.getElementById('stat-build').style.color = status.last_build.result === 'SUCCESS' ? '#00ff88' : '#ff4444';
    }

    if (status.pod_status) {
      document.getElementById('stat-pod').textContent = status.pod_status.status;
      document.getElementById('stat-pod').style.color = status.pod_status.status === 'Running' ? '#00ff88' : '#ff4444';
    }

    document.getElementById('stat-alerts').textContent = alerts_data.active.length;
    document.getElementById('stat-alerts').style.color = alerts_data.active.length > 0 ? '#ff4444' : '#00ff88';

    // Render builds
    const builds_html = builds.map(b => `
      <div class="build-card ${b.result === 'SUCCESS' ? 'success' : 'failure'}">
        <div class="build-info">
          <div class="build-number">Build #${b.number}</div>
          <div class="build-meta">${new Date(b.timestamp).toLocaleString()}</div>
        </div>
        <div style="display:flex;align-items:center;gap:12px">
          <span class="build-status ${b.result === 'SUCCESS' ? 'status-success' : 'status-failure'}">${b.result}</span>
          <span class="build-duration">${(b.duration/1000).toFixed(1)}s</span>
        </div>
      </div>
    `).join('');
    document.getElementById('builds').innerHTML = builds_html || '<div style="color:#888">No builds found</div>';

    // Render pods
    const pods_html = pods.map(p => `
      <div class="pod-card ${p.status === 'Running' ? 'healthy' : 'unhealthy'}">
        <div class="pod-name">${p.name}</div>
        <div class="pod-details">
          <div class="pod-detail-item">
            <span class="status-badge ${p.status === 'Running' ? 'badge-running' : 'badge-notready'}">${p.status}</span>
          </div>
          <div class="pod-detail-item">Restarts: ${p.restarts}</div>
          <div class="pod-detail-item">Age: ${p.created ? new Date(p.created).toLocaleString() : 'unknown'}</div>
        </div>
      </div>
    `).join('');
    document.getElementById('pods').innerHTML = pods_html || '<div style="color:#888">No pods found</div>';

    // Render alerts
    const alerts_html = alerts_data.active.map(a => `
      <div class="alert-card ${a.severity === 'CRITICAL' ? 'critical' : 'warning'}">
        <div class="alert-header">
          <div style="flex:1">
            <span class="alert-badge ${a.severity === 'CRITICAL' ? 'badge-critical' : 'badge-warning'}">${a.severity}</span>
            <span class="alert-title">${a.title}</span>
          </div>
        </div>
        <div class="alert-meta">${a.source} • ${new Date(a.created_at).toLocaleString()}</div>
        ${a.auto_healed ? `<div class="alert-healed">✅ Auto-healed by NeuroShield (${a.healed_by}, ${a.healed_in}s)</div>` : ''}
        <div class="alert-buttons">
          <button class="btn-ack" onclick="ackAlert(${a.id})">Acknowledge</button>
        </div>
      </div>
    `).join('');
    document.getElementById('alerts').innerHTML = alerts_html || '<div style="color:#888">No active alerts</div>';

    // Render resolved
    const resolved_html = alerts_data.resolved.map(a => `
      <div class="alert-card" style="background:#111a11;border-color:#1a3a1a;opacity:.5">
        <div class="alert-title" style="color:#888">${a.title}</div>
        <div class="alert-meta">${new Date(a.acknowledged_at).toLocaleString()}</div>
      </div>
    `).join('');
    document.getElementById('resolved').innerHTML = resolved_html || '<div style="color:#666">No resolved alerts</div>';

  } catch (e) {
    console.error('Refresh error:', e);
  }
}

async function ackAlert(id){
  await fetch(`/api/alerts/${id}/acknowledge`, {method: 'POST'});
  refreshData();
}

// Initial load and refresh every 10 seconds
refreshData();
setInterval(refreshData, 10000);
</script>

</body>
</html>"""

@app.route("/", methods=["GET"])
def index():
    response = make_response(render_template_string(HTML))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
