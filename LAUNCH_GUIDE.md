# NeuroShield Project Launch Guide

## Current System State
- ✗ Docker: Not running (Jenkins, Prometheus unavailable)
- ✓ Models: Trained and ready (failure_predictor.pth, log_pca.joblib, ppo_policy.zip)
- ✓ Configuration: .env configured
- ✓ Code: All 83 tests passing

## Two Ways to Run

### Option 1: SIMULATION MODE (No Docker needed) ⚡ RECOMMENDED
Runs with synthetic data - perfect for demonstration

```bash
# Terminal 1: Start Orchestrator
python src/orchestrator/main.py --mode simulate

# Terminal 2: Start Dashboard (in another terminal)
python -m streamlit run src/dashboard/app.py
```

Open http://localhost:8501 to see:
- Real-time failure predictions
- PPO RL agent healing actions (6 discrete actions)
- MTTR reduction metrics
- Active alerts and healing history
- Self-CI monitoring

### Option 2: LIVE MODE (Requires Docker + Jenkins + Prometheus)
Real production monitoring with live infrastructure

```bash
# 1. Start Docker Desktop and wait 30 seconds for it to initialize

# 2. Start infrastructure
docker compose up -d

# 3. Wait for services to be ready, then run
python src/orchestrator/main.py --mode live
python -m streamlit run src/dashboard/app.py
```

Services will be available at:
- Jenkins: http://localhost:8080 (admin/admin123)
- Prometheus: http://localhost:9090
- Dashboard: http://localhost:8501
- Dummy App: http://localhost:5000

## Quick Demo (2 minutes)

1. Start orchestrator in simulate mode
2. Start dashboard
3. Click "Run Healing Cycle" button in dashboard
4. Watch the system:
   - Poll Jenkins/Prometheus (simulated data)
   - Build 52D state vector
   - Run DistilBERT log analysis
   - PPO agent predicts action
   - Execute healing + log results
   - Update real-time charts

## What You'll See

**Metric Cards:**
- MTTR Reduction: 44% vs baseline
- Failure Prediction: F1 = 1.000
- Total Actions: Updated per cycle
- System Health: Green/Yellow/Red

**Charts:**
- Failure Probability Over Time
- Resource Monitoring (CPU, Memory)
- Action Distribution by Type
- MTTR Trend

**Healing Actions:**
- Restart Pod
- Scale Up Resources
- Retry Build
- Rollback Deploy
- Clear Cache
- Escalate to Human

## Troubleshooting

**Docker not starting:**
- Check Docker Desktop is installed
- Manually start Docker Desktop from Start menu
- Alternative: Use simulation mode

**Port conflicts:**
- Jenkins (8080), Prometheus (9090), Streamlit (8501)
- Check and close conflicting processes

**Models not found:**
- Models are already trained in `models/` directory
- If missing, run: `python src/prediction/train.py && python -m src.rl_agent.train`

## Architecture Summary

```
Jenkins/Prometheus → TelemetryCollector
    ↓
Build 52D State (10 build + 12 resource + 16 log + 14 dep metrics)
    ↓
DistilBERT Log Encoding + PCA (768D → 16D)
    ↓
Failure Predictor (PyTorch NN)
    ↓
PPO Agent (Stable Baselines3) → Selects 1 of 6 Actions
    ↓
Action Executor (execute_healing_action)
    ↓
Dashboard + Logs (data/healing_log.json, data/mttr_log.csv)
```

---

**Ready to go! Choose simulation or live mode above.**
