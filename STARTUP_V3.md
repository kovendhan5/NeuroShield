# NeuroShield v3 - Startup Guide

**Get the system running in 30 seconds.**

## Quick Start (Docker)

```bash
docker-compose up -d
open http://localhost:8000
```

Wait ~5 seconds for API to start, then dashboard appears.

##Quick Start (Local)

```bash
pip install -r requirements.txt
python main.py              # Terminal 1
```

In another terminal (after 2 sec):
```bash
python demo.py              # See it work
```

Dashboard available at: http://localhost:8000

## What You'll See

**Dashboard Features:**
- Real-time metrics (CPU, Memory, Health)
- Live event stream (color-coded)
- Healing action history
- Demo scenario buttons

**Demo runs 5 scenarios showing auto-healing:**

1. Pod crashes →System restarts it
2. Memory leaks → System clears cache
3. CPU spike → System scales up
4. Bad deploy → System rolls back
5. Multiple failures → Multi-action recovery

Each scenario takes ~2 seconds. Total: **~5 minutes**.

## System URLs

| URL | Purpose |
|-----|---------|
| http://localhost:8000 | Dashboard + UI |
| http://localhost:8000/docs | API documentation |
| http://localhost:8000/health | Health check |

## Try These APIs

```bash
# Get current status
curl http://localhost:8000/api/status

# See healing history
curl "http://localhost:8000/api/history?limit=10"

# Trigger orchestration manually
curl -X POST http://localhost:8000/api/cycle/trigger

# Inject pod crash
curl -X POST "http://localhost:8000/api/demo/inject?scenario=pod_crash"
```

## Key Files

- `app/orchestrator.py` - Core state machine
- `demo.py` - Demo scenarios
- `config.yaml` - Thresholds + configuration
- `dashboard.html` - Web UI
- `main.py` - Entry point

## Stop System

```bash
# Docker
docker-compose down

# Local
CTRL+C in terminals
```

---

**That's it!** Questions? Read README.md for full details.
