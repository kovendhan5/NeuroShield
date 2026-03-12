"""
IncidentBoard — Real-Time DevOps Incident Feed
Monitored by NeuroShield AIOps Platform
"""

from flask import Flask, jsonify, request, render_template_string
import threading
import time
import random
import os
import signal
import psutil
from datetime import datetime, timedelta

app = Flask(__name__)

# ── In-memory incident store ─────────────────────────────────────────────────

_now = datetime.now()

INCIDENTS = [
    {"id": 1, "severity": "CRITICAL", "title": "Jenkins build pipeline timeout",       "service": "jenkins",    "status": "open",         "ts": _now - timedelta(minutes=10)},
    {"id": 2, "severity": "WARNING",  "title": "Prometheus scrape interval exceeded",  "service": "prometheus", "status": "acknowledged", "ts": _now - timedelta(minutes=18)},
    {"id": 3, "severity": "INFO",     "title": "Kubernetes node heartbeat received",   "service": "k8s",       "status": "resolved",     "ts": _now - timedelta(minutes=25)},
    {"id": 4, "severity": "CRITICAL", "title": "Pod OOMKilled — memory limit reached", "service": "k8s",       "status": "open",         "ts": _now - timedelta(minutes=32)},
    {"id": 5, "severity": "WARNING",  "title": "CPU usage 78% on worker node",         "service": "k8s",       "status": "open",         "ts": _now - timedelta(minutes=40)},
    {"id": 6, "severity": "INFO",     "title": "New deployment rollout started",        "service": "jenkins",   "status": "resolved",     "ts": _now - timedelta(minutes=55)},
    {"id": 7, "severity": "CRITICAL", "title": "Database connection pool exhausted",    "service": "db",        "status": "open",         "ts": _now - timedelta(hours=1)},
    {"id": 8, "severity": "WARNING",  "title": "SSL certificate expires in 14 days",   "service": "nginx",     "status": "acknowledged", "ts": _now - timedelta(hours=2)},
]

incident_id_counter = 9
incidents_lock = threading.Lock()
app_healthy = True
stress_active = False
start_time = time.time()
request_count = 0

INCIDENT_TEMPLATES = [
    ("CRITICAL", "Deployment rollback triggered automatically",  "k8s"),
    ("WARNING",  "High memory usage detected on node",           "k8s"),
    ("INFO",     "NeuroShield healing action executed",          "neuroshield"),
    ("CRITICAL", "Build stage failed — dependency not found",    "jenkins"),
    ("WARNING",  "Slow response time on /health endpoint",       "nginx"),
    ("INFO",     "Prometheus alert rule evaluated successfully", "prometheus"),
    ("CRITICAL", "Pod crash loop detected — restarting",         "k8s"),
    ("WARNING",  "Disk usage exceeded 70% threshold",            "storage"),
]


# ── Helper ────────────────────────────────────────────────────────────────────

def _time_ago(ts):
    """Return human-readable relative time string."""
    delta = datetime.now() - ts
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    mins = secs // 60
    if mins < 60:
        return f"{mins} min{'s' if mins != 1 else ''} ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''} ago"


def _incident_json(inc):
    return {
        "id": inc["id"],
        "severity": inc["severity"],
        "title": inc["title"],
        "service": inc["service"],
        "status": inc["status"],
        "time_ago": _time_ago(inc["ts"]),
        "ts_iso": inc["ts"].isoformat(),
    }


# ── Background thread — new incident every 30s ───────────────────────────────

def _incident_generator():
    global incident_id_counter
    while True:
        time.sleep(30)
        severity, title, service = random.choice(INCIDENT_TEMPLATES)
        with incidents_lock:
            new = {
                "id": incident_id_counter,
                "severity": severity,
                "title": title,
                "service": service,
                "status": "open",
                "ts": datetime.now(),
            }
            INCIDENTS.insert(0, new)
            incident_id_counter += 1

threading.Thread(target=_incident_generator, daemon=True).start()


# ── API ROUTES ────────────────────────────────────────────────────────────────

@app.before_request
def _count_requests():
    global request_count
    request_count += 1


@app.route("/api/incidents", methods=["GET"])
def get_incidents():
    with incidents_lock:
        sorted_inc = sorted(INCIDENTS, key=lambda i: i["ts"], reverse=True)
        return jsonify([_incident_json(i) for i in sorted_inc])


@app.route("/api/incidents/<int:inc_id>/acknowledge", methods=["POST"])
def acknowledge_incident(inc_id):
    with incidents_lock:
        for inc in INCIDENTS:
            if inc["id"] == inc_id:
                inc["status"] = "acknowledged"
                return jsonify({"success": True, "id": inc_id})
    return jsonify({"success": False, "error": "not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    if app_healthy:
        open_count = sum(1 for i in INCIDENTS if i["status"] == "open")
        uptime = round(time.time() - start_time, 1)
        return jsonify({"status": "healthy", "uptime": uptime, "incidents_open": open_count}), 200
    return jsonify({"status": "unhealthy", "error": "app crashed"}), 503


@app.route("/crash", methods=["POST"])
def crash():
    global app_healthy
    app_healthy = False

    def _recover():
        global app_healthy
        time.sleep(120)
        app_healthy = True

    threading.Thread(target=_recover, daemon=True).start()
    return jsonify({"message": "App crash simulated", "recovery_in": "120s"})


@app.route("/stress", methods=["POST"])
def stress():
    global stress_active
    if stress_active:
        return jsonify({"message": "Stress test already running"}), 409
    stress_active = True

    def _cpu_burn():
        global stress_active
        end = time.time() + 30
        while time.time() < end:
            _ = sum(i * i for i in range(10000))
        stress_active = False

    threading.Thread(target=_cpu_burn, daemon=True).start()
    return jsonify({"message": "Stress test started", "duration": "30s"})


@app.route("/metrics", methods=["GET"])
def metrics():
    with incidents_lock:
        crit = sum(1 for i in INCIDENTS if i["severity"] == "CRITICAL")
        warn = sum(1 for i in INCIDENTS if i["severity"] == "WARNING")
        info = sum(1 for i in INCIDENTS if i["severity"] == "INFO")
        open_count = sum(1 for i in INCIDENTS if i["status"] == "open")
    uptime = round(time.time() - start_time, 1)
    lines = [
        f'incident_board_incidents_total{{severity="CRITICAL"}} {crit}',
        f'incident_board_incidents_total{{severity="WARNING"}} {warn}',
        f'incident_board_incidents_total{{severity="INFO"}} {info}',
        f'incident_board_open_incidents {open_count}',
        f'incident_board_uptime_seconds {uptime}',
        f'incident_board_requests_total {request_count}',
    ]
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; charset=utf-8"}


# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IncidentBoard — Live DevOps Monitoring</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d1a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;display:flex;flex-direction:column}
a{color:#00ff88;text-decoration:none}

/* Header */
.header{display:flex;justify-content:space-between;align-items:center;padding:18px 32px;border-bottom:1px solid #2a2a4a}
.header-left{display:flex;align-items:center;gap:12px}
.logo{font-size:1.5rem;font-weight:700;color:#00ff88}
.tagline{font-size:.85rem;color:#888;margin-top:2px}
.live-badge{display:flex;align-items:center;gap:8px;font-size:.85rem;color:#00ff88;font-weight:600}
.pulse-dot{width:10px;height:10px;border-radius:50%;background:#00ff88;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,255,136,.5)}50%{box-shadow:0 0 0 8px rgba(0,255,136,0)}}

/* Stats bar */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;padding:24px 32px}
.stat-card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:18px 20px;text-align:center}
.stat-value{font-size:1.8rem;font-weight:700;color:#00ff88}
.stat-label{font-size:.78rem;color:#888;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}

/* Feed */
.feed{flex:1;padding:8px 32px 32px;display:flex;flex-direction:column;gap:12px}
.feed-title{font-size:1rem;color:#888;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;font-weight:600}

.incident-card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:16px 20px;display:flex;justify-content:space-between;align-items:center;transition:border-color .2s}
.incident-card:hover{border-color:#3a3a5a}
.card-left{display:flex;flex-direction:column;gap:6px}
.card-top{display:flex;align-items:center;gap:10px}
.badge{font-size:.7rem;font-weight:700;padding:3px 10px;border-radius:6px;text-transform:uppercase;letter-spacing:.5px}
.badge-CRITICAL{background:rgba(255,68,68,.13);border:1px solid #ff4444;color:#ff4444}
.badge-WARNING{background:rgba(255,136,0,.13);border:1px solid #ff8800;color:#ff8800}
.badge-INFO{background:rgba(0,136,255,.13);border:1px solid #0088ff;color:#0088ff}
.card-title{font-size:.95rem;font-weight:500}
.card-meta{font-size:.78rem;color:#666}
.card-right{flex-shrink:0;margin-left:16px}
.btn-ack{background:transparent;border:1px solid #00ff88;color:#00ff88;padding:7px 18px;border-radius:7px;cursor:pointer;font-size:.78rem;font-weight:600;transition:all .2s}
.btn-ack:hover{background:#00ff88;color:#0d0d1a}
.status-badge{font-size:.75rem;padding:5px 14px;border-radius:7px;font-weight:600}
.status-acknowledged{background:rgba(136,136,136,.15);border:1px solid #666;color:#999}
.status-resolved{background:rgba(0,255,136,.1);border:1px solid #00ff88;color:#00ff88}

/* Footer */
.footer{text-align:center;padding:18px 32px;border-top:1px solid #2a2a4a;font-size:.78rem;color:#555}

/* Responsive */
@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}.header,.stats,.feed{padding-left:16px;padding-right:16px}}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div><span class="logo">⚡ IncidentBoard</span><div class="tagline">Live DevOps Monitoring</div></div>
  </div>
  <div class="live-badge"><div class="pulse-dot"></div> LIVE</div>
</div>

<div class="stats">
  <div class="stat-card"><div class="stat-value" id="stat-open">—</div><div class="stat-label">Open Incidents</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-resolved">—</div><div class="stat-label">Resolved Today</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-uptime">—</div><div class="stat-label">System Uptime</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-alerts">—</div><div class="stat-label">Active Alerts</div></div>
</div>

<div class="feed">
  <div class="feed-title">Incident Feed</div>
  <div id="incidents"></div>
</div>

<div class="footer">Monitored by NeuroShield AIOps Platform &bull; Auto-healing enabled</div>

<script>
const startTs = Date.now();

function fmtUptime(ms){
  let s=Math.floor(ms/1000), h=Math.floor(s/3600), m=Math.floor((s%3600)/60); s=s%60;
  return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
}

function renderIncidents(data){
  let openCount=0, resolvedCount=0, alertCount=0;
  data.forEach(i=>{
    if(i.status==='open'){openCount++;alertCount++;}
    if(i.status==='resolved') resolvedCount++;
    if(i.status==='acknowledged') alertCount++;
  });
  document.getElementById('stat-open').textContent=openCount;
  document.getElementById('stat-resolved').textContent=resolvedCount;
  document.getElementById('stat-alerts').textContent=alertCount;

  const container=document.getElementById('incidents');
  container.innerHTML=data.map(i=>{
    let actionHTML='';
    if(i.status==='open'){
      actionHTML=`<button class="btn-ack" onclick="ackIncident(${i.id})">Acknowledge</button>`;
    } else if(i.status==='acknowledged'){
      actionHTML=`<span class="status-badge status-acknowledged">Acknowledged ✓</span>`;
    } else {
      actionHTML=`<span class="status-badge status-resolved">Resolved ✓</span>`;
    }
    return `<div class="incident-card">
      <div class="card-left">
        <div class="card-top"><span class="badge badge-${i.severity}">${i.severity}</span><span class="card-title">${i.title}</span></div>
        <div class="card-meta">${i.service} &bull; ${i.time_ago}</div>
      </div>
      <div class="card-right">${actionHTML}</div>
    </div>`;
  }).join('');
}

async function fetchIncidents(){
  try{
    const res=await fetch('/api/incidents');
    const data=await res.json();
    renderIncidents(data);
  }catch(e){console.error('Fetch error',e);}
}

async function ackIncident(id){
  await fetch(`/api/incidents/${id}/acknowledge`,{method:'POST'});
  fetchIncidents();
}

// Uptime counter
setInterval(()=>{
  document.getElementById('stat-uptime').textContent=fmtUptime(Date.now()-startTs);
},1000);

// Live refresh every 3 seconds
fetchIncidents();
setInterval(fetchIncidents,3000);
</script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
