# NeuroShield v3 - Quick Start

**Start here for LOCAL setup.**

## In 2 Commands

```bash
# 1. Start
bash scripts/start-local.sh

# 2. Demo
python demo.py
```

## Access

- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Verify Setup**: `python verify-setup.py`

## What's Included

✅ **Orchestrator**: State machine that detects & heals failures
✅ **API**: REST + WebSocket real-time events
✅ **Dashboard**: Beautiful real-time UI
✅ **Demo**: 5 scenarios showing auto-healing
✅ **Database**: SQLite with full audit trail
✅ **Tests**: Full test suite

## Architecture

```
Jenkins + Prometheus + Kubernetes
            ↓
    TelemetryCollector
            ↓
   Orchestrator (Detect→Decide→Execute)
            ↓
    Healing Actions (6 types)
            ↓
        Dashboard
```

## File Structure

```
neuroshield/
├── app/                 Core system
├── api/                 FastAPI server
├── scripts/             Quick start scripts
├── tests/               Test suite
├── dashboard.html       Web UI
├── config.yaml          Configuration
├── docker-compose.yml   Local setup
└── demo.py              Demo scenarios
```

## Next Steps

### For Judges/Demo

```bash
# 1. See the system
bash scripts/start-local.sh

# 2. Watch auto-healing
python demo.py

# 3. Check dashboard
open http://localhost:8000
```

### For Development

```bash
# Run tests
pytest tests/ -v

# Check status anytime
bash scripts/status.sh

# View logs
tail -f logs/neuroshield.log

# Stop system
bash scripts/stop-local.sh
```

### For Understanding

1. Read: `docs/LOCAL_SETUP.md` (detailed guide)
2. Review: `app/orchestrator.py` (main engine)
3. Explore: `demo.py` (see it working)
4. Learn: `README.md` (full overview)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `lsof -i :8000` and kill it |
| Docker not running | Start Docker Desktop |
| API not responding | `bash scripts/status.sh` and check logs |
| Database corrupted | `rm -f data/neuroshield.db` |

## When Ready for Azure

→ See `AZURE_DISCUSSION.md` (we'll create this)

---

**Status**: ✅ Local system **READY TO USE**

Start with: `bash scripts/start-local.sh`
