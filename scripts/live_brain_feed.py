#!/usr/bin/env python3
"""NeuroShield Live Brain Feed — real-time SSE dashboard at :8503.

Serves a single-page dark-themed UI with three columns:
  1. Architecture pipeline — animated flow diagram
  2. Live AI feed — streams healing_log.json entries every 3 s
  3. Performance metrics — cards showing model stats

SSE endpoint: GET /events  (JSON lines with msg + cls fields)

Usage:
    python scripts/live_brain_feed.py
    Open http://localhost:8503
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI(title="NeuroShield Brain Feed", docs_url=None, redoc_url=None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALING_LOG = PROJECT_ROOT / "data" / "healing_log.json"
MODEL_REPORT = PROJECT_ROOT / "data" / "model_report_summary.json"

# ──────────────────────────────────────────────────────────────────────────
# SSE generator
# ──────────────────────────────────────────────────────────────────────────

def _read_log_tail(n: int = 40) -> list[dict]:
    """Return the last *n* entries from healing_log.json (one JSON per line)."""
    if not HEALING_LOG.exists():
        return []
    lines = HEALING_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    entries = []
    for raw in lines[-n:]:
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return entries


_last_count = 0


async def _event_stream():
    """Yield SSE events: new log entries + heartbeat every 3 s."""
    global _last_count
    _last_count = 0

    while True:
        entries = _read_log_tail(200)
        current = len(entries)

        if current > _last_count:
            for e in entries[_last_count:]:
                ok = e.get("success", False)
                cls = "ok" if ok else "fail"
                action = e.get("action_name", "unknown")
                detail = e.get("detail", "")[:120]
                ts = e.get("timestamp", "")
                msg = f"[{ts}] {action} → {'✓' if ok else '✗'} {detail}"
                payload = json.dumps({"msg": msg, "cls": cls})
                yield f"data: {payload}\n\n"
            _last_count = current
        else:
            # heartbeat
            payload = json.dumps({
                "msg": f"♥ heartbeat {datetime.now().strftime('%H:%M:%S')}",
                "cls": "hb",
            })
            yield f"data: {payload}\n\n"

        await asyncio.sleep(3)


@app.get("/events")
async def sse(request: Request):
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────────────────────
# Metrics endpoint (JSON)
# ──────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """Return model + healing stats as JSON for the dashboard cards."""
    # model report
    model = {}
    if MODEL_REPORT.exists():
        try:
            model = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
        except Exception:
            pass

    entries = _read_log_tail(500)
    total = len(entries)
    ok = sum(1 for e in entries if e.get("success"))
    actions = {}
    for e in entries:
        a = e.get("action_name", "?")
        actions[a] = actions.get(a, 0) + 1

    return {
        "f1": model.get("f1_score", "N/A"),
        "auc": model.get("auc_roc", "N/A"),
        "mttr_reduction": model.get("mttr_reduction_pct", "N/A"),
        "total_heals": total,
        "success_rate": f"{ok/total*100:.0f}%" if total else "N/A",
        "top_actions": dict(sorted(actions.items(), key=lambda x: -x[1])[:5]),
    }


# ──────────────────────────────────────────────────────────────────────────
# HTML dashboard
# ──────────────────────────────────────────────────────────────────────────

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NeuroShield — Live Brain Feed</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d0d1a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;overflow-x:hidden}
.header{text-align:center;padding:18px 0 10px;border-bottom:1px solid #1f1f3a}
.header h1{font-size:1.4rem;letter-spacing:.05em;color:#00e5ff}
.header .dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#0f0;
  margin-right:8px;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 6px #0f0}50%{opacity:.4;box-shadow:none}}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding:16px;height:calc(100vh - 70px)}
.col{background:#12122a;border:1px solid #1f1f3a;border-radius:10px;padding:14px;overflow:hidden;
  display:flex;flex-direction:column}
.col h2{font-size:.95rem;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
/* Pipeline column */
.pipeline{display:flex;flex-direction:column;gap:8px;flex:1;justify-content:center}
.stage{background:#1a1a35;border-radius:6px;padding:10px 14px;font-size:.85rem;
  border-left:3px solid #00e5ff;position:relative;transition:background .3s}
.stage.active{background:#1a2a3a;border-left-color:#0f0;animation:glow 2s infinite alternate}
@keyframes glow{from{box-shadow:0 0 4px rgba(0,229,255,.15)}to{box-shadow:0 0 12px rgba(0,229,255,.35)}}
.stage .num{color:#00e5ff;font-weight:700;margin-right:6px}
.arrow{text-align:center;color:#333;font-size:1.2rem}
/* Feed column */
#feed{flex:1;overflow-y:auto;font-family:'Cascadia Code','Fira Code',monospace;font-size:.78rem;line-height:1.6}
#feed::-webkit-scrollbar{width:5px}
#feed::-webkit-scrollbar-thumb{background:#333;border-radius:4px}
.ev{padding:3px 6px;border-radius:3px;margin-bottom:2px;white-space:pre-wrap;word-break:break-all}
.ev.ok{background:#0a2a0a;color:#4caf50}
.ev.fail{background:#2a0a0a;color:#ef5350}
.ev.hb{color:#444;font-style:italic}
/* Metrics column */
.cards{display:flex;flex-direction:column;gap:10px;flex:1}
.card{background:#1a1a35;border-radius:8px;padding:14px;text-align:center}
.card .label{font-size:.75rem;color:#666;text-transform:uppercase;margin-bottom:4px}
.card .value{font-size:1.6rem;font-weight:700;color:#00e5ff}
.card .value.green{color:#4caf50}
.card .value.amber{color:#ffb74d}
.actions-list{text-align:left;margin-top:8px;font-size:.8rem;line-height:1.8}
.actions-list span{color:#00e5ff}
</style>
</head>
<body>
<div class="header">
  <h1><span class="dot"></span>NEUROSHIELD — LIVE BRAIN FEED</h1>
</div>
<div class="grid">
  <!-- Column 1: Architecture Pipeline -->
  <div class="col">
    <h2>Architecture Pipeline</h2>
    <div class="pipeline" id="pipeline">
      <div class="stage active"><span class="num">1</span>Telemetry Collector<br><small>Jenkins + Prometheus + K8s</small></div>
      <div class="arrow">▼</div>
      <div class="stage"><span class="num">2</span>DistilBERT Log Encoder<br><small>Tokenise → embed → feature vector</small></div>
      <div class="arrow">▼</div>
      <div class="stage"><span class="num">3</span>Failure Predictor (PyTorch)<br><small>MLP ⇒ P(failure) 0-1</small></div>
      <div class="arrow">▼</div>
      <div class="stage"><span class="num">4</span>PPO RL Agent<br><small>Choose action: retry / restart / rollback / scale / clear / escalate</small></div>
      <div class="arrow">▼</div>
      <div class="stage"><span class="num">5</span>Action Executor<br><small>kubectl · Jenkins API · alerts</small></div>
      <div class="arrow">▼</div>
      <div class="stage"><span class="num">6</span>Feedback Loop<br><small>Reward → retrain agent</small></div>
    </div>
  </div>
  <!-- Column 2: Live AI Feed -->
  <div class="col">
    <h2>Live AI Feed</h2>
    <div id="feed"></div>
  </div>
  <!-- Column 3: Performance Metrics -->
  <div class="col">
    <h2>Performance Metrics</h2>
    <div class="cards" id="metrics">
      <div class="card"><div class="label">F1 Score</div><div class="value green" id="m-f1">—</div></div>
      <div class="card"><div class="label">AUC-ROC</div><div class="value green" id="m-auc">—</div></div>
      <div class="card"><div class="label">MTTR Reduction</div><div class="value amber" id="m-mttr">—</div></div>
      <div class="card"><div class="label">Total Heals</div><div class="value" id="m-heals">—</div></div>
      <div class="card"><div class="label">Success Rate</div><div class="value green" id="m-rate">—</div></div>
      <div class="card">
        <div class="label">Top Actions</div>
        <div class="actions-list" id="m-actions">—</div>
      </div>
    </div>
  </div>
</div>
<script>
// SSE feed
const feed = document.getElementById('feed');
const es = new EventSource('/events');
es.onmessage = e => {
  const d = JSON.parse(e.data);
  const div = document.createElement('div');
  div.className = 'ev ' + d.cls;
  div.textContent = d.msg;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  // limit to 300 entries
  while (feed.children.length > 300) feed.removeChild(feed.firstChild);
  // animate pipeline
  animatePipeline(d.cls);
};

// Animate pipeline stages sequentially
let animIdx = 0;
function animatePipeline(cls) {
  const stages = document.querySelectorAll('.stage');
  stages.forEach(s => s.classList.remove('active'));
  animIdx = (animIdx + 1) % stages.length;
  stages[animIdx].classList.add('active');
}

// Metrics polling
async function refreshMetrics() {
  try {
    const r = await fetch('/metrics');
    const m = await r.json();
    const fmt = v => (typeof v === 'number') ? (v * 100).toFixed(1) + '%' : v;
    document.getElementById('m-f1').textContent = fmt(m.f1);
    document.getElementById('m-auc').textContent = fmt(m.auc);
    document.getElementById('m-mttr').textContent = (typeof m.mttr_reduction === 'number')
        ? m.mttr_reduction.toFixed(1) + '%' : m.mttr_reduction;
    document.getElementById('m-heals').textContent = m.total_heals;
    document.getElementById('m-rate').textContent = m.success_rate;
    const acts = document.getElementById('m-actions');
    acts.innerHTML = Object.entries(m.top_actions || {})
      .map(([k,v]) => `<span>${k}</span>: ${v}`)
      .join('<br>') || '—';
  } catch {}
}
refreshMetrics();
setInterval(refreshMetrics, 5000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8503, log_level="info")
