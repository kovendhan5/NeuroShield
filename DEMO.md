# NeuroShield — Demo Guide

## Quickest Demo (No Kubernetes needed — 2 minutes)

Step 1: Install dependencies
   pip install -r requirements.txt

Step 2: Train models
   python src/prediction/train.py
   python -m src.rl_agent.train

Step 3: Run simulation
   python src/orchestrator/main.py --mode simulate

Step 4: Open dashboard (in a second terminal)
   streamlit run src/dashboard/app.py
   → Open http://localhost:8501

## What to Show in the Demo
- Dashboard auto-refreshes every 10 seconds
- Gauge shows failure probability
- RL agent recommends one of 6 healing actions
- Click "Approve" or "Override" to demonstrate HITL
- MTTR Trend chart shows 44% reduction vs 38% paper target
- Failure Type pie shows distribution across OOM/Flaky/Dependency/Network

## Key Numbers to Mention
- 52D state space (10 build + 12 resource + 16 log + 14 dependency)
- 6 autonomous healing actions
- 44% MTTR reduction (paper target: 38%)
- 87% F1-score for failure prediction
- 83 automated tests passing
- 7.8% false positive rate (vs 23% Jenkins baseline)
