# NeuroShield v3 - Local Setup Guide

**Get NeuroShield running on your device in 2 minutes.**

## Prerequisites

- **Docker Desktop** (includes Docker & Docker Compose)
  - [Download for Mac](https://www.docker.com/products/docker-desktop)
  - [Download for Windows](https://www.docker.com/products/docker-desktop)
  - [Download for Linux](https://docs.docker.com/desktop/install/linux-install/)

- **Python 3.13** (for demo scripts)
  - [Download](https://www.python.org/downloads/)

## Quick Start (2 Commands)

```bash
# 1. Start the system
bash scripts/start-local.sh

# 2. In another terminal, run demo
python demo.py
```

That's it! ✅

---

## What's Running

After `start-local.sh` completes, you have:

| Component | Port | Status |
|-----------|------|--------|
| **Dashboard** | 8000 | http://localhost:8000 |
| **API Server** | 8000 | http://localhost:8000/docs |
| **WebSocket Events** | 8000 | ws://localhost:8000/ws/events |
| **Database** | file system | data/neuroshield.db |
| **Logs** | file system | logs/neuroshield.log |

---

## Usage

### Access the Dashboard

```bash
open http://localhost:8000    # Mac
start http://localhost:8000   # Windows
xdg-open http://localhost:8000 # Linux
```

**Dashboard Features:**
- Real-time metrics (CPU, Memory, Health)
- Live event stream
- Healing history
- Demo injection buttons

### Run the Demo

```bash
python demo.py
```

Shows 5 scenarios back-to-back (~5 minutes):
1. **Pod Crash** → Auto-restart
2. **Memory Leak** → Auto-cache-clear
3. **CPU Spike** → Auto-scale
4. **Bad Deploy** → Auto-rollback
5. **Cascading Failure** → Multi-action recovery

### View API Endpoints

```bash
# Interactive docs
open http://localhost:8000/docs

# Or curl examples
curl http://localhost:8000/api/status
curl http://localhost:8000/api/history?limit=10
curl http://localhost:8000/api/events?limit=5
```

### Check System Health

```bash
bash scripts/status.sh
```

Shows:
- Running containers
- API health
- Database stats
- Recent logs
- Current metrics

### View Logs

```bash
# Watch live logs
tail -f logs/neuroshield.log

# Or from Docker
docker-compose logs -f orchestrator
```

### Stop the System

```bash
bash scripts/stop-local.sh
```

---

## Troubleshooting

### Port 8000 Already in Use

```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port (edit docker-compose.yml)
ports:
  - "9000:8000"
```

### Docker Not Running

```bash
# Start Docker Desktop and wait for it to fully load
# Then run:
docker ps    # Should show containers
```

### API Not Responding

```bash
# Check logs
docker-compose logs

# Rebuild images
docker-compose build --no-cache

# Restart
docker-compose restart
```

### Database Corrupted

```bash
# Reset database (data will be lost)
rm -f data/neuroshield.db

# Restart
docker-compose restart
```

### "No space left on device"

```bash
# Clean up Docker
docker system prune -a

# Or remove old images
docker image prune -a
```

---

## Commands Reference

| Task | Command |
|------|---------|
| Start | `bash scripts/start-local.sh` |
| Stop | `bash scripts/stop-local.sh` |
| Status | `bash scripts/status.sh` |
| Logs | `tail -f logs/neuroshield.log` |
| Demo | `python demo.py` |
| Tests | `pytest tests/ -v` |
| Reset DB | `rm -f data/neuroshield.db` |
| Full Cleanup | `docker-compose down -v && rm -rf data logs` |

---

## Development

### Directory Structure

```
neuroshield-v3/
├── app/
│   ├── orchestrator.py     Main engine (state machine)
│   ├── models.py           Database schema
│   ├── connectors.py       Demo connectors
│   └── __init__.py
│
├── api/
│   ├── main.py             FastAPI server
│   └── __init__.py
│
├── scripts/
│   ├── start-local.sh      Start system
│   ├── stop-local.sh       Stop system
│   ├── status.sh           Check health
│   └── README.md           This file
│
├── tests/
│   └── test_orchestrator_v3.py    Tests
│
├── data/                    Database & metrics (gitignored)
├── logs/                    Application logs (gitignored)
├── dashboard.html          Web UI
├── demo.py                 Demo scenarios
├── main.py                 Entry point
├── config.yaml             Central configuration
├── Dockerfile              Container image
├── docker-compose.yml      Service orchestration
├── requirements.txt        Python dependencies
└── README.md               Project overview
```

### Editing Configuration

Edit `config.yaml` to change:

```yaml
orchestrator:
  check_interval: 10       # Cycle frequency (seconds)
  action_timeout: 300      # Max action time (seconds)

detection:
  cpu_threshold: 80        # CPU alarm threshold (%)
  memory_threshold: 85     # Memory alarm threshold (%)
  pod_restart_threshold: 3 # Pod restart alarm count
  error_rate_threshold: 0.3 # Error rate alarm (30%)
```

Changes take effect on next cycle (no restart needed).

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Modifying Code

**Edit any file** → Restart with:
```bash
docker-compose restart orchestrator
```

Or for API changes:
```bash
docker-compose down && docker-compose up -d
```

---

## Performance Tips

### If System is Slow

1. **Check Docker resources:**
   ```bash
   docker stats
   ```

2. **Reduce cycle frequency** (config.yaml):
   ```yaml
   orchestrator:
     check_interval: 20  # Increase from 10
   ```

3. **Restart containers:**
   ```bash
   docker-compose restart
   ```

### If Database is Large

```bash
# Check size
du -h data/neuroshield.db

# Archive old data
sqlite3 data/neuroshield.db "DELETE FROM events WHERE timestamp < datetime('now', '-7 days');"

# Or reset completely
rm data/neuroshield.db
```

---

## Best Practices

1. **Daily**: Check status
   ```bash
   bash scripts/status.sh
   ```

2. **Weekly**: Clean up
   ```bash
   docker system prune
   ```

3. **Before demos**: Full restart
   ```bash
   docker-compose restart
   bash scripts/status.sh
   ```

4. **Keep logs**: They're in `logs/neuroshield.log` by default

---

## Next Steps

- ✅ Local system running?
- ✅ Dashboard accessible?
- ✅ Demo works?
- ➜ **Ready to discuss Azure deployment** (when you are)

---

## Support

If something doesn't work:

1. **Check logs:**
   ```bash
   bash scripts/status.sh
   ```

2. **Verify prerequisites:**
   - Docker Desktop running?
   - Port 8000 free?
   - Python 3.13 installed?

3. **Restart from scratch:**
   ```bash
   docker-compose down -v
   bash scripts/start-local.sh
   ```

4. **Check internet:**
   - First pull might need internet for base images

---

**You're all set!** Start with `bash scripts/start-local.sh` 🚀
