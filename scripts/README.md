# Scripts Directory Guide

Quick navigation for all utility scripts in NeuroShield.

---

## 📁 Directory Structure

```
scripts/
├── health_check.py           ─ Verify system readiness (MAIN UTILITY)
├── start_api.py              ─ Start FastAPI server (PORT: 8502)
├── setup_local.sh            ─ Legacy setup script (use docs/GUIDES/SETUP.md)
│
├── launcher/                 ─ Application launchers
│   ├── launch_orchestrator.bat
│   ├── launch_orchestrator.sh
│   ├── launch_dashboard.bat
│   └── launch_dashboard.sh
│
├── demo/                     ─ Demo scenario tools
│   ├── real_demo.py          - Run all 6 demo scenarios live
│   ├── demo_scenario_dep.py  - Dependency conflict scenario
│   ├── demo_simulation.py    - Full simulation mode demo
│   └── generate_model_report.py - Generate HTML model report
│
├── infra/                    ─ Infrastructure setup & failure injection
│   ├── inject_failure.py     - Inject failures (CPU spike, OOM, etc.)
│   ├── inject_dep_conflict.py - Create dependency conflicts
│   ├── create_real_jenkins_job.py - Create Jenkins build job
│   ├── upgrade_jenkins_job.py - Update existing Jenkins job
│   └── setup_neuroshield_cicd.py - Setup self-CI monitoring
│
├── test/                     ─ Testing & diagnostics
│   ├── live_brain_feed.py    - Real-time AI decision UI (PORT: 8503)
│   ├── test_email.py         - Test email notifications
│   ├── test_notifications.py - Test desktop alerts
│
└── debug/                    ─ Experimental / archived scripts
    ├── _diag.py
    ├── _debug_healing.py
    ├── _final_verify.py
    ├── _fix_*.py
    ├── _verify_fixes.py
    └── ... (10 experimental debug scripts)
```

---

## 🚀 Quick Start Commands

### Launch Application (2 minutes)

```bash
# Windows
double-click: launcher/launch_orchestrator.bat
double-click: launcher/launch_dashboard.bat

# Mac/Linux
bash launcher/launch_orchestrator.sh &
bash launcher/launch_dashboard.sh &
```

### Run Demo (8-10 minutes)

```bash
# See full demo script at: docs/GUIDES/DEMO.md

# Inject a failure to trigger healing
python infra/inject_failure.py --scenario cpu_spike

# View live AI decisions
python test/live_brain_feed.py
```

### Verify System

```bash
# Health check (all components ready?)
python health_check.py

# Run tests
pytest tests/ -v
```

---

## 📋 Script Reference

### Launcher Scripts (`launcher/`)

| Script | Purpose | Platform |
|--------|---------|----------|
| `launch_orchestrator.{bat,sh}` | Start ML orchestrator pipeline | Windows/Unix |
| `launch_dashboard.{bat,sh}` | Start Streamlit dashboard UI | Windows/Unix |

**Usage:** Double-click or `bash launcher/launch_orchestrator.sh`

---

### Demo Scripts (`demo/`)

| Script | Purpose | Duration |
|--------|---------|----------|
| `real_demo.py` | Run all 6 scenarios in sequence | 5-10 min |
| `demo_scenario_dep.py` | Dependency conflict scenario only | 2 min |
| `demo_simulation.py` | Full simulation (no Docker needed) | 3 min |
| `generate_model_report.py` | Generate HTML model summary | 1 min |

**Usage:**
```bash
# Run full demo with all scenarios
python demo/real_demo.py

# Generate report
python demo/generate_model_report.py
```

**Documentation:** See `docs/GUIDES/DEMO.md` for full demo script with presenter notes.

---

### Infrastructure Scripts (`infra/`)

| Script | Purpose | Input |
|--------|---------|-------|
| `inject_failure.py` | Inject failures for testing | `--scenario [cpu_spike, memory_pressure, ...]` |
| `inject_dep_conflict.py` | Create dependency conflicts | `--fix` (cleanup), `--status` (check) |
| `create_real_jenkins_job.py` | Create Jenkins build job | None (reads .env) |
| `upgrade_jenkins_job.py` | Update Jenkins job stages | None (reads .env) |
| `setup_neuroshield_cicd.py` | Setup self-monitoring CI job | None (reads .env) |

**Usage:**
```bash
# Inject CPU spike failure
python infra/inject_failure.py --scenario cpu_spike

# Create/update Jenkins job
python infra/create_real_jenkins_job.py

# Setup self-CI monitoring
python infra/setup_neuroshield_cicd.py
```

---

### Test & Diagnostic Scripts (`test/`)

| Script | Purpose | Port |
|--------|---------|------|
| `live_brain_feed.py` | Real-time AI decision visualizer | 8503 |
| `test_email.py` | Test email alert notifications | N/A |
| `test_notifications.py` | Test desktop notifications | N/A |

**Usage:**
```bash
# Run Brain Feed visualization
python test/live_brain_feed.py
# Open: http://localhost:8503

# Test notifications
python test/test_notifications.py
python test/test_email.py
```

---

### Main Utilities (Root)

| Script | Purpose | Usage |
|--------|---------|-------|
| `health_check.py` | System readiness verification | `python health_check.py` |
| `start_api.py` | Start REST API server | `python start_api.py` (PORT: 8502) |
| `setup_local.sh` | Legacy setup (deprecated) | Not recommended — use `docs/GUIDES/SETUP.md` |

---

## 🔍 Debug Scripts (`debug/`)

All scripts prefixed with `_` are **experimental/archived debug tools**. They were used during development for debugging specific issues but are no longer needed for normal operation.

Examples:
- `_fix_all.py` — Old bug fix script (merged into main code)
- `_verify_fixes.py` — Old verification test (tests now in pytest)
- `_diag.py` — Diagnostic helper (replaced by `health_check.py`)

**Status:** Kept for historical reference. Can be safely ignored.

---

## Environment Variables

Most scripts read from `.env` file. Key variables:

```bash
# Jenkins
JENKINS_URL=http://localhost:8080
JENKINS_USERNAME=admin
JENKINS_TOKEN=...
JENKINS_JOB=neuroshield-test-job

# Kubernetes
K8S_NAMESPACE=neuroshield
AFFECTED_SERVICE=dummy-app
SCALE_REPLICAS=3

# Infrastructure
PROMETHEUS_URL=http://localhost:9090
DUMMY_APP_URL=http://localhost:5000

# Configuration
POLL_INTERVAL=15
MODEL_PATH=models/
TELEMETRY_OUTPUT_PATH=data/telemetry.csv
```

Copy from `.env.example`:
```bash
cp .env.example .env
# Edit .env with your values
```

---

## Common Workflows

### Scenario 1: Quick Demo (No Docker)
```bash
# Terminal 1: Start orchestrator
bash launcher/launch_orchestrator.sh

# Terminal 2: Start dashboard (in new terminal)
bash launcher/launch_dashboard.sh

# Browser: Open http://localhost:8501
# Watch real-time healing in action
```

### Scenario 2: Live Demo with Failures
```bash
# Terminal 1: Start all services
docker compose up -d
bash launcher/launch_orchestrator.sh

# Terminal 2: Dashboard
bash launcher/launch_dashboard.sh

# Terminal 3: Inject failures
python infra/inject_failure.py --scenario cpu_spike

# Terminal 4 (optional): Watch AI decisions
python test/live_brain_feed.py

# Browser: Open http://localhost:8501 and http://localhost:8503
```

### Scenario 3: Full End-to-End Demo
```bash
# See docs/GUIDES/DEMO.md for complete step-by-step guide with timing
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **Port already in use** | Find process: `lsof -i :8501` (Mac/Linux) or `netstat -ano \| find :8501` (Windows) |
| **Models missing** | Run: `python src/prediction/train.py && python -m src.rl_agent.train` |
| **.env not loaded** | Verify `.env` file exists and has correct path |
| **Script not found** | Make sure you're in `k:/Devops/NeuroShield` directory |
| **Permission denied** | Run with Python: `python infra/inject_failure.py` (not just `inject_failure.py`) |

---

## 📚 Full Documentation

- **Setup Guide:** `docs/GUIDES/SETUP.md`
- **Demo Guide:** `docs/GUIDES/DEMO.md`
- **Architecture:** `docs/ARCHITECTURE.md`
- **Project Status:** `PROJECT_STATUS.md`

---

**Last Updated:** 2026-03-20
**Organization:** Phase 3 Reorganization Complete ✅
