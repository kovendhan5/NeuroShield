# ⚡ NeuroShield - 60-Second Launch Guide

## 🎯 What to Do Right Now

### On Windows:
```
1. Open File Explorer
2. Navigate to: k:\Devops\NeuroShield
3. Double-click: launch_orchestrator.bat
4. Wait for terminal to show "Cycle 0", "Cycle 1", etc.
5. Open NEW Command Prompt window
6. cd k:\Devops\NeuroShield
7. Run: launch_dashboard.bat
8. Dashboard opens automatically at http://localhost:8501
9. Watch real-time ML predictions & healing actions!
```

### On Mac/Linux:
```bash
# Terminal 1 - Start the ML orchestrator
cd k:/Devops/NeuroShield
bash launch_orchestrator.sh

# Terminal 2 - Start the dashboard (in NEW terminal window)
cd k:/Devops/NeuroShield
bash launch_dashboard.sh

# Then open browser to http://localhost:8501
```

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
| **Top Cards** | MTTR: 44% reduction | F1: 1.000 | Actions: N | Health: 🟢 |
| **Main Chart** | Failure probability over time (line chart) |
| **Side Charts** | Resource usage (CPU, memory), action distribution |
| **Log Viewer** | Real-time decisions being made |
| **Buttons** | "Run Cycle" (manual trigger), "Mark Resolved" (dismiss alerts) |

---

## 🎓 What the System Does

**Problem:** CI/CD builds fail randomly. Fixing them takes ~12.4 minutes on average.

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

## ✅ Everything is Ready

- ✅ Code: All 83 tests passing
- ✅ Models: Trained and loaded (9.3 KB + 222.6 KB + 53.3 KB)
- ✅ Config: .env configured with Jenkins token
- ✅ Data: Clean `data/` folder ready for logs

**No external services needed for simulation mode.**

---

## 🔄 Two Modes

| Mode | What It Uses | Speed | Use Case |
|------|-------------|-------|----------|
| **Simulate** | Synthetic data | NOW | Demo, testing, dev |
| **Live** | Real Jenkins+Prometheus | When Docker runs | Production |

We're starting in **Simulate mode** (fast, self-contained).

To go **Live mode** later:
1. Start Docker Desktop (`docker compose up -d`)
2. Change `--mode simulate` to `--mode live` in launcher script

---

## 🆘 Troubleshooting

**Nothing happens after launching?**
- Orchestrator: Should show "Cycle 0", "Cycle 1"... within 5 seconds
- Dashboard: Should automatically open browser to http://localhost:8501
- If not: Check Windows firewall isn't blocking ports 8501 or 8080

**Dashboard shows old data?**
- Refresh the browser (Ctrl+R)
- Check orchestrator is still running (should see new "Cycle N" every 15s)

**Want to stop?**
- Orchestrator: Ctrl+C in the terminal
- Dashboard: Ctrl+C then close the terminal

---

## 📖 Want More Details?

- **Full Guide:** See `COMPLETE_STARTUP_GUIDE.md`
- **Status:** See `PROJECT_STATUS.md`
- **Architecture:** See `README.md`
- **Code:** See `src/orchestrator/main.py` (main loop with ML pipeline)

---

## 🎬 Start Here

**Windows:** Double-click `launch_orchestrator.bat`, then `launch_dashboard.bat`

**Mac/Linux:** `bash launch_orchestrator.sh` & `bash launch_dashboard.sh`

**Then:** Open http://localhost:8501

**Watch:** Real-time AI healing in action!

---

*Last Updated: 2026-03-19 | Project Status: READY TO RUN ✅*
