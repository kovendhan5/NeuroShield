# NeuroShield v2.1.0 - Windows Quick Start Guide

## 🚀 Running on Windows

### Option 1: Using the Batch Wrapper (Easiest) ✅

```cmd
neuroshield.cmd version
neuroshield.cmd health
neuroshield.cmd help
```

### Option 2: Using Python Directly

```cmd
python neuroshield version
python neuroshield health
python neuroshield start
python neuroshield demo pod_crash
```

### Option 3: Using Start Script

```cmd
start-neuroshield.bat         # Full system
start-neuroshield.bat --quick # UI only
```

## 📋 Available Commands

```cmd
REM Start the system
neuroshield.cmd start                  # Full system
neuroshield.cmd start --quick          # UI only (5 seconds)

REM Stop services
neuroshield.cmd stop                   # Graceful shutdown
neuroshield.cmd stop --force           # Force shutdown

REM Status and monitoring
neuroshield.cmd status                 # Show health
neuroshield.cmd status --watch         # Live monitoring (updates every 2s)
neuroshield.cmd health                 # Full health check
neuroshield.cmd health --detailed      # Detailed report

REM Demo scenarios
neuroshield.cmd demo pod_crash         # Pod restart scenario
neuroshield.cmd demo cpu_spike         # CPU scaling scenario
neuroshield.cmd demo memory_pressure   # Memory clearing scenario
neuroshield.cmd demo build_fail        # Build retry scenario
neuroshield.cmd demo rollback          # Rollback scenario

REM Configuration
neuroshield.cmd config --show          # Show all settings
neuroshield.cmd config --validate      # Validate config
neuroshield.cmd config --edit          # Edit in VS Code

REM Logs and metrics
neuroshield.cmd logs                   # Show recent logs
neuroshield.cmd logs --tail=100        # Last 100 entries
neuroshield.cmd logs --filter=error    # Filter by term
neuroshield.cmd metrics                # Show metrics
neuroshield.cmd metrics --prometheus   # Open Prometheus
neuroshield.cmd metrics --grafana      # Open Grafana

REM Testing
neuroshield.cmd test                   # Run all tests
neuroshield.cmd test --coverage        # With coverage report

REM Maintenance
neuroshield.cmd backup                 # Create backup
neuroshield.cmd restore --input=FILE   # Restore from backup
neuroshield.cmd cleanup --force        # Clean old data

REM Version
neuroshield.cmd version                # Show version
```

## 🎯 Quick Start (3 Steps)

### Step 1: Start the System
```cmd
neuroshield.cmd start --quick
```

This opens:
- Dashboard: http://localhost:9999
- Takes about 5 seconds

### Step 2: View the Dashboard
```
Open browser → http://localhost:9999
```

You'll see:
- Real-time metrics
- System health
- Recent healing actions

### Step 3: Run a Demo

```cmd
neuroshield.cmd demo pod_crash
```

Watch the dashboard - it will:
1. Inject a failure (pod crash)
2. Detect it (red banner)
3. Make a decision (AI analysis)
4. Execute recovery
5. Return to healthy (green)

**All in ~12 seconds!**

## 🔗 Web Interfaces

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:9999 | Main UI |
| **Grafana** | http://localhost:3000 | Metrics/history |
| **Prometheus** | http://localhost:9090 | Raw metrics |
| **Jenkins** | http://localhost:8080 | CI/CD |
| **Streamlit** | http://localhost:8501 | Analytics |

## ⚙️ Configuration

Edit `config/neuroshield.yaml` to customize:

```yaml
application:
  environment: production
  log_level: INFO

orchestrator:
  poll_interval_seconds: 15

kubernetes:
  namespace: default

jenkins:
  url: http://localhost:8080
  username: admin
  password: admin123

prometheus:
  url: http://localhost:9090
```

Then restart:
```cmd
neuroshield.cmd stop
neuroshield.cmd start
```

## 🐛 Troubleshooting

### Command not recognized
```cmd
REM Use full path or Python:
python neuroshield start

REM Or use batch wrapper:
neuroshield.cmd start
```

### Docker not running
```cmd
REM Make sure Docker Desktop is running
docker ps
```

### Ports already in use
```cmd
REM Check what's using port 9999:
netstat -ano | findstr :9999

REM Kill the process (replace PID):
taskkill /PID 1234 /F
```

### WebSocket errors in browser console
- This is normal if backend is slow
- Refresh the page
- Check `neuroshield.cmd health`

## 📊 Verify Installation

```cmd
REM Check all systems are healthy
neuroshield.cmd health --detailed

REM Run integration tests
neuroshield.cmd test

REM Check logs
neuroshield.cmd logs --tail=10
```

All should show `[OK] Healthy`

## 🎓 For Demonstrations

**Show to judges:**

```cmd
REM 1. Show the unified CLI
neuroshield.cmd --help

REM 2. Show the config (centralized)
neuroshield.cmd config --show

REM 3. Show the dashboard (live)
neuroshield.cmd start --quick
REM → Opens http://localhost:9999

REM 4. Run a demo scenario
neuroshield.cmd demo pod_crash

REM 5. Show test results
neuroshield.cmd test
```

## 📝 Log Files

- **Main logs:** `data/logs/neuroshield.jsonl` (searchable JSON)
- **CLI logs:** `data/logs/neuroshield.log` (timestamped)
- **State DB:** `data/neuroshield.db` (SQLite)
- **Demo data:** `demo_data/*.json` (scenario data)

View with:
```cmd
neuroshield.cmd logs --format=json
```

## 🚀 Next Steps

1. **Start:** `neuroshield.cmd start --quick`
2. **Monitor:** http://localhost:9999
3. **Demo:** `neuroshield.cmd demo pod_crash`
4. **Full System:** `neuroshield.cmd start` (Docker Compose)
5. **Custom:** Edit `config/neuroshield.yaml`

---

**NeuroShield v2.1.0 | Windows Ready | Production Deployed**
