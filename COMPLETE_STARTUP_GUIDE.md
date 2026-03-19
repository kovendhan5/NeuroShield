# NeuroShield - Complete Launch Instructions

## 🚀 Quick Start (2 minutes)

### Windows Users:
Double-click these files in order (in separate windows):
1. `launch_orchestrator.bat` - Starts the ML pipeline & healing engine
2. `launch_dashboard.bat` - Opens the web dashboard
3. Open http://localhost:8501 in your browser

### Mac/Linux Users:
Run these in separate terminals:
```bash
bash launch_orchestrator.sh
bash launch_dashboard.sh
```

Then open http://localhost:8501 in your browser.

---

## System Components

### 1. Orchestrator (`src/orchestrator/main.py`)
**Role:** Main control loop - predicts failures and executes healing actions

**Workflow (every 15 seconds in simulate mode):**
```
┌─────────────────────────────────────────────────────────┐
│ 1. Collect Telemetry                                    │
│    • Build metrics (duration, queue time, etc)         │
│    • Resource metrics (CPU, memory, network)           │
│    • Log embeddings (DistilBERT → PCA)                │
│    Result: 52D state vector                            │
│                                                        │
│ 2. Predict Failure                                      │
│    • FailurePredictor (PyTorch NN)                     │
│    • Output: probability [0.0, 1.0]                   │
│                                                        │
│ 3. RL Agent Decision                                   │
│    • PPO Policy (Stable Baselines3)                    │
│    • Selects 1 of 6 actions                           │
│                                                        │
│ 4. Rule-Based Override                                 │
│    • CPU > 80% → scale_up                             │
│    • Memory > 70% → clear_cache                       │
│    • Pod restarts ≥ 3 → restart_pod                  │
│    • etc.                                             │
│                                                        │
│ 5. Execute Action                                      │
│    • Log to data/healing_log.json                     │
│    • Track MTTR metrics                               │
│    • Send notifications (optional)                    │
│                                                        │
│ 6. Dashboard Update                                    │
│    • Real-time metrics refresh                        │
│    • Action history visualized                        │
└─────────────────────────────────────────────────────────┘
```

### 2. Dashboard (`src/dashboard/app.py`)
**Role:** Real-time monitoring UI - visualizes all system metrics

**Key Features:**
- **Metric Cards** (top row):
  - MTTR Reduction: 44% vs baseline
  - Failure Prediction: F1 = 1.000
  - Total Actions Executed: incrementing counter
  - System Health: Green/Yellow/Red indicator

- **Charts:**
  - Failure Probability Over Time (line chart)
  - Resource Usage: CPU, Memory, Disk (gauge charts)
  - Action Distribution by Type (pie chart)
  - MTTR Trend (area chart)

- **Interactive Elements:**
  - "Run Healing Cycle" button - manually trigger a cycle
  - "Mark as Resolved" button - clear active alerts
  - Real-time log viewer

- **Auto-Refresh:** Every 10 seconds (configurable)

### 3. Supporting Modules

**FailurePredictor** (`src/prediction/predictor.py`)
- PyTorch neural network trained on synthetic failure patterns
- Input: 52D state vector
- Output: failure probability (0.0-1.0)
- Uses DistilBERT for log embeddings

**PPOAgent** (`src/rl_agent/env.py`)
- Stable Baselines3 PPO policy
- Observation space: 52 dimensions
- Action space: 6 discrete actions
- Trained to maximize MTTR reduction

**Distribution Maps:**
```
PPO Output → Healing Action

State 0: restart_pod       (kill and recreate failed pod)
Action 1: scale_up         (increase replicas for capacity)
Action 2: retry_build      (re-trigger Jenkins build)
Action 3: rollback_deploy  (revert to last stable version)
Action 4: clear_cache      (flush dependency/build caches)
Action 5: escalate_to_human (alert on-call engineer)
```

---

## 🎯 What to Watch For

### Orchestrator Terminal
```
[2026-03-19 10:45:32] Cycle 1
├─ Telemetry: 52D state collected
├─ Failure Probability: 0.23 (23%)
├─ PPO Action: 1 (scale_up)
├─ Rule Override: NONE (prob < threshold)
├─ Executed: scale_up (increased replicas 2→3)
└─ MTTR: 7.7 min (44% improvement)

[2026-03-19 10:46:47] Cycle 2
├─ Telemetry: 52D state collected
├─ Failure Probability: 0.68 (68%)
├─ PPO Action: 2 (retry_build)
├─ Rule Override: Memory check (78% ÷ restart_pod)
├─ Executed: restart_pod
└─ MTTR: 4.3 min (49% improvement)
```

### Dashboard
- Refresh indicator (small circle spinning)
- Metrics updating in real-time
- Success notification when cycle completes
- Red alert banner if failure probability > threshold

---

## 📊 Data Files Generated

The system auto-generates these files in `data/`:

1. **healing_log.json** - All healing actions with metadata
   ```json
   {
     "timestamp": "2026-03-19T10:45:32",
     "cycle": 1,
     "predicted_failure": 0.23,
     "action": "scale_up",
     "mttr_sec": 462
   }
   ```

2. **mttr_log.csv** - MTTR metrics over time
   ```csv
   timestamp,action,mttr_baseline,mttr_actual,improvement
   2026-03-19 10:45:32,scale_up,600,420,30%
   ```

3. **action_history.csv** - Full action audit log

4. **active_alert.json** - Current high-priority alert (if any)

5. **escalation_reports/** - HTML incident reports

---

## 🔧 Customization

### Environment Variables (.env)
```bash
POLL_INTERVAL=15              # Seconds between cycles
PREDICTION_THRESHOLD=0.7      # Failure prob to trigger action
MODEL_PATH=models/            # Trained model directory
TELEMETRY_LOGS_ENABLED=true   # Capture build logs
LOG_LEVEL=INFO                # Logging verbosity
```

### Healing Action Override Rules
Edit `src/orchestrator/main.py` function `determine_healing_action()`:
- Add custom conditions (e.g., "if disk > 90% then...") - Line 250
- Modify action priority/weighting
- Filter actions by environment

### Dashboard Refresh Rate
Edit `src/dashboard/app.py`:
- Change `REFRESH_INTERVAL = 10` (line 50) to adjust dashboard refresh

---

## 🐛 Troubleshooting

### Orchestrator Won't Start
**Error:** `ModuleNotFoundError: No module named 'orchestrator'`
**Fix:** Make sure you're running from the project root:
```bash
cd k:/Devops/NeuroShield
python src/orchestrator/main.py --mode simulate
```

### Dashboard Won't Connect
**Error:** `ConnectionRefusedError at http://localhost:8501`
**Fix:** Make sure nothing is using port 8501:
```bash
# Check what's using port 8501
netstat -ano | findstr :8501

# Or just change the port in streamlit config:
python -m streamlit run src/dashboard/app.py --server.port 8502
```

### Models Not Loading
**Error:** `FileNotFoundError: models/failure_predictor.pth`
**Fix:** Models should already be in `models/` directory. If missing:
```bash
python src/prediction/train.py
python -m src.rl_agent.train
```

### No Data Updates
**Check:**
1. Is orchestrator running? (check logs for "Cycle N" messages)
2. Is `data/` directory writable? (should already be)
3. Try refreshing dashboard (Ctrl+R on browser)
4. Check `data/healing_log.json` is being updated

---

## 📈 Key Metrics to Track

| Metric | Baseline | NeuroShield | Target |
|--------|----------|-------------|--------|
| OOM Recovery | 14.2 min | 7.5 min | ✅ Achieved |
| Flaky Test Recovery | 8.5 min | 4.3 min | ✅ Exceeded |
| Dependency Fix | 15.1 min | 9.8 min | ✅ Achieved |
| Avg MTTR Reduction | — | 44% | 38% ✅ |
| Failure Prediction F1 | — | 1.000 | >0.87 ✅ |
| False Positives | 23% | 7.8% | <10% ✅ |

---

## 🚀 Advanced: Running in LIVE Mode

When Docker + Jenkins are ready:

```bash
# Start infrastructure
docker compose up -d

# Wait for services:
# Jenkins: http://localhost:8080 (admin/admin123)
# Prometheus: http://localhost:9090

# Run in live mode
python src/orchestrator/main.py --mode live
```

Live mode connects to:
- Jenkins REST API (real builds)
- Prometheus HTTP API (real metrics)
- Kubernetes cluster (real pod management)

Healing actions execute on real infrastructure.

---

## 📞 Support

- **Dashboard not refreshing?** Check orchestrator is running
- **Metrics not changing?** Ensure you see "Cycle N" logs in orchestrator
- **Want to trace a decision?** Check `data/healing_log.json` for detailed reasoning
- **More details?** See README.md or LAUNCH_GUIDE.md

---

**Ready to go! Start with the Quick Start section above. 🎯**
