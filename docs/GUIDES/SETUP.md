# NeuroShield - Complete Setup Guide

Welcome to NeuroShield! This guide consolidates setup and launch instructions for all different use cases.

---

## ⚡ Quick Start (2 minutes)

### Windows Users
Double-click these files in order (in separate windows):
1. `scripts/launcher/launch_orchestrator.bat` — Starts the ML pipeline
2. `scripts/launcher/launch_dashboard.bat` — Opens the web dashboard
3. Open http://localhost:8501 in your browser

### Mac/Linux Users
Run these in separate terminals:
```bash
bash scripts/launcher/launch_orchestrator.sh
bash scripts/launcher/launch_dashboard.sh
```
Then open http://localhost:8501 in your browser.

---

## 🚀 What Happens Automatically

The orchestrator runs this cycle every 15 seconds:

```
Collect Telemetry (52D state vector)
    ↓
Parse Build Logs with DistilBERT (768D → 16D via PCA)
    ↓
Run Failure Predictor (PyTorch neural net)
    ↓
Output: Failure probability (0-100%)
    ↓
PPO RL Agent selects best action from 6 options
    ↓
Rule-based overrides (e.g., CPU>80% → scale_up)
    ↓
Execute healing action
    ↓
Log metrics to data/healing_log.json
    ↓
Dashboard updates in real-time
```

---

## 📊 You'll See This on Dashboard

| Component | What It Shows |
|-----------|---------------|
| **Top Cards** | MTTR: 44% reduction \| F1: 1.000 \| Actions: N \| Health: 🟢 |
| **Main Chart** | Failure probability over time (line chart) |
| **Side Charts** | Resource usage (CPU, memory), action distribution |
| **Log Viewer** | Real-time decisions being made |
| **Buttons** | "Run Cycle" (manual trigger), "Mark Resolved" (dismiss alerts) |

---

## 🔄 Two Run Modes

### Mode 1: SIMULATION (Recommended for Demo) ⚡
- **Use Case:** Demo, testing, development
- **Speed:** Fast (no external services)
- **Requirements:** Python only (no Docker needed)
- **Setup:** Just run the launchers

```bash
# Terminal 1: Start Orchestrator
python src/orchestrator/main.py --mode simulate

# Terminal 2: Start Dashboard
python -m streamlit run src/dashboard/app.py
```

### Mode 2: LIVE (Production)
- **Use Case:** Real monitoring with live infrastructure
- **Speed:** Slower (depends on service latency)
- **Requirements:** Docker, Jenkins, Prometheus running

```bash
# 1. Start Docker Desktop and wait 30 seconds

# 2. Start infrastructure
docker compose up -d

# 3. Wait for services to be ready (check each is green), then:
python src/orchestrator/main.py --mode live
python -m streamlit run src/dashboard/app.py
```

**Services Available at:**
- Jenkins: http://localhost:8080 (admin/admin123)
- Prometheus: http://localhost:9090
- Dashboard: http://localhost:8501
- Dummy App: http://localhost:5000

---

## 🎓 What the System Does

**Problem:** CI/CD builds fail randomly. Manual fixes take ~12.4 minutes on average.

**Solution:** NeuroShield learns patterns and heals automatically in ~7.7 minutes.

**How:**
- Watches 52 system metrics
- Uses DistilBERT to understand logs
- Trains neural network to predict failures
- Uses PPO reinforcement learning to pick best healing action
- Executes actions that reduce MTTR by 44%

**Healing Actions:**
- `restart_pod` — Kill and restart the failed pod
- `scale_up` — Add more replicas (handles load issues)
- `retry_build` — Re-run the Jenkins job
- `rollback_deploy` — Revert to last stable version
- `clear_cache` — Flush frozen/corrupt caches
- `escalate_to_human` — Alert on-call engineer with incident report

---

## ✅ Readiness Checklist

- ✅ Code: 95 tests (pytest tests/)
- ✅ Models: Trained and loaded (9.3 KB + 222.6 KB + 53.3 KB)
- ✅ Config: .env configured with Jenkins token (if using LIVE mode)
- ✅ Data: Clean `data/` folder ready for logs

**For simulation mode: No external services needed!**

---

## 🆘 Troubleshooting

### Nothing happens when launching?
- **Orchestrator:** Should show "Cycle 0", "Cycle 1"... within 5 seconds
- **Dashboard:** Should automatically open browser to http://localhost:8501
- **Fix:** Check Windows firewall isn't blocking ports 8501 or 8080

### Dashboard shows old data?
- Refresh the browser (Ctrl+R)
- Check orchestrator is still running (should see new "Cycle N" every 15s)

### Docker not starting?
- Check Docker Desktop is installed
- Manually start Docker Desktop from Start menu
- Alternative: Use simulation mode (no Docker needed)

### Port conflicts?
- Jenkins (8080), Prometheus (9090), Streamlit (8501)
- Check and close conflicting processes

### Models not found?
- Models are already trained in `models/` directory
- If missing, run: `python src/prediction/train.py && python -m src.rl_agent.train`

---

## 📖 Detailed Components

### Orchestrator (`src/orchestrator/main.py`)
**Role:** Main control loop - predicts failures and executes healing actions

**Workflow (every 15 seconds):**
1. **Collect Telemetry** → Build 52D state vector
2. **Predict Failure** → FailurePredictor (PyTorch NN)
3. **RL Agent Decision** → PPO Policy (Stable Baselines3)
4. **Rule-Based Override** → Apply domain logic (CPU>80% → scale_up, etc)
5. **Execute Action** → Log to healing_log.json, send notifications
6. **Dashboard Update** → Real-time metrics refresh

### Dashboard (`src/dashboard/app.py`)
**Role:** Real-time monitoring UI

**Key Features:**
- **Metric Cards** (top row): MTTR 44%, F1 1.000, Actions count, Health status
- **Charts:** Failure probability, resource usage, action distribution, MTTR trend
- **Interactive:** "Run Healing Cycle", "Mark as Resolved" buttons
- **Auto-Refresh:** Every 10 seconds

### Supporting Modules

**FailurePredictor** (`src/prediction/predictor.py`)
- PyTorch neural network
- Input: 52D state vector
- Output: failure probability (0.0-1.0)
- Uses DistilBERT for log embeddings

**PPOAgent** (`src/rl_agent/env.py`)
- Stable Baselines3 PPO policy
- Action space: 6 discrete healing actions
- Trained to maximize MTTR reduction

**Healing Action Mapping:**
| Action ID | Action | Description |
|-----------|--------|-------------|
| 0 | restart_pod | Kill and recreate failed pod |
| 1 | scale_up | Increase replicas for capacity |
| 2 | retry_build | Re-trigger Jenkins build |
| 3 | rollback_deploy | Revert to last stable version |
| 4 | clear_cache | Flush dependency/build caches |
| 5 | escalate_to_human | Alert on-call engineer |

---

## 🎬 Getting Started

1. **First Time?** Read "Quick Start" section above (2 minutes)
2. **Run orchestrator** in window 1
3. **Run dashboard** in window 2
4. **Watch the dashboard** at http://localhost:8501
5. **Click buttons** to see real-time healing in action

---

## 📚 Want More Info?

- **Architecture Deep-Dive:** See `docs/ARCHITECTURE.md`
- **Product Requirements:** See `docs/PRD.md`
- **Demo Scenarios:** See `docs/GUIDES/DEMO.md`
- **API Reference:** See `src/api/main.py`
- **Source Code:** See `src/orchestrator/main.py` (main loop)

---

**Status:** ✅ Project Ready to Run
**Last Updated:** 2026-03-20
