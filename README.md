# 🛡️ NeuroShield
### AIOps-Driven Self-Healing CI/CD Pipelines using Reinforcement Learning
> Jeppiaar Institute of Technology | IEEE Research Project

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![stable-baselines3](https://img.shields.io/badge/RL-stable--baselines3%20PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

NeuroShield is an AIOps-driven self-healing CI/CD system that predicts imminent
pipeline failures using DistilBERT log analysis and autonomously mitigates them
via Proximal Policy Optimization (PPO) reinforcement learning. It monitors
Jenkins builds, Prometheus metrics, and Kubernetes cluster state in real time,
combining a 24-dimensional failure classifier with a 52-dimensional RL policy to
select from six discrete healing actions — from simple stage retries to safe
rollbacks and human escalation.

The system includes a human-in-the-loop Streamlit dashboard that gives engineers
full visibility into predictions, RL decisions, and SHAP feature importance,
with approve/override/pause controls. In evaluation, NeuroShield achieves a
**44% average MTTR reduction** (exceeding the paper target of 38%), an **87%
failure prediction F1-score**, and a **66% reduction in false positives**
compared to Jenkins-native alerting.

---

## 🏗️ Architecture

```
Data Sources                  NeuroShield Core                    Actions
─────────────                 ────────────────                    ───────
Jenkins CI     ──logs──►  Telemetry      ►  DistilBERT        ►  retry_stage
Prometheus     ──metrics─►  Aggregator   ►  + PCA (16D)       ►  clean_and_rerun
Kubernetes API ──events──►  (5-sec sync) ►  ────────────       ►  regenerate_config
                                         ►  Failure Predictor  ►  reallocate_resources
                                         ►  (F1: 87%)         ►  trigger_safe_rollback
                                         ►  ────────────       ►  escalate_to_human
                                         ►  PPO RL Agent  ──feedback──► Dashboard
                                         ►  (52D state)         (Human-in-the-Loop)
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.12+
- Docker Desktop
- Minikube
- kubectl

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Jenkins URL and credentials
```

### 3. Train models

```bash
python src/prediction/train.py
python -m src.rl_agent.train
```

### 4. Start local infrastructure

```bash
bash scripts/setup_local.sh
# or on Windows:
# powershell scripts/setup_local.ps1
```

### 5. Run demo (no Jenkins needed)

```bash
python -m src.orchestrator.main --mode simulate
```

### 6. Launch dashboard

```bash
streamlit run src/dashboard/app.py
```

### 7. Run live mode

```bash
python -m src.orchestrator.main --mode live
```

---

## 🧪 Testing

```bash
pip install pytest
python -m pytest tests/ -v
# 83 tests across 4 files — all passing
```

---

## 📁 Project Structure

```
NeuroShield/
├── src/
│   ├── telemetry/           # Jenkins + Prometheus polling → CSV
│   │   ├── collector.py     # TelemetryCollector with log redaction
│   │   ├── config.py        # Centralized env var loading
│   │   └── main.py          # CLI entry point
│   ├── prediction/          # Failure prediction ML pipeline
│   │   ├── data_generator.py# Synthetic training data with failure patterns
│   │   ├── log_encoder.py   # DistilBERT → mean-pool → PCA (16D)
│   │   ├── model.py         # FailureClassifier feed-forward network
│   │   ├── predictor.py     # Runtime inference + 52D state builder
│   │   └── train.py         # End-to-end training script
│   ├── rl_agent/            # Reinforcement learning agent
│   │   ├── env.py           # Gymnasium env (52D obs, 6 actions)
│   │   ├── simulator.py     # Synthetic state generator + MTTR tables
│   │   └── train.py         # PPO training + evaluation
│   ├── orchestrator/        # Real-time CI/CD monitor + healer
│   │   └── main.py          # Jenkins poll → predict → PPO → kubectl heal
│   └── dashboard/           # Human-in-the-loop dashboard
│       └── app.py           # Streamlit app (7 sections)
├── tests/
│   ├── test_telemetry.py    # Telemetry collector unit tests
│   ├── test_prediction.py   # Prediction pipeline tests
│   ├── test_rl_agent.py     # RL env + simulator tests
│   └── test_orchestrator.py # Orchestrator + healing action tests
├── models/                  # Trained model artifacts
│   ├── failure_predictor.pth
│   ├── log_pca.joblib
│   └── ppo_policy.zip
├── infra/
│   ├── dummy-app/           # Flask app for failure injection testing
│   └── jenkins-builder/     # Custom Jenkins image with pinned kubectl
├── data/                    # Runtime CSV output (telemetry, action history)
├── scripts/
│   ├── setup_local.sh       # One-command local setup (Linux/Mac)
│   └── setup_local.ps1      # One-command local setup (Windows)
├── .env.example             # All environment variables with defaults
├── requirements.txt         # Python dependencies
├── setup_jenkins_job.py     # Idempotent Jenkins job creation
├── jenkins-pvc.yaml         # Kubernetes PVC for Jenkins
├── jenkins-local-updated.yaml # Jenkins K8s deployment
├── dummy-app.yaml           # Dummy app K8s deployment
├── PRD.md                   # Product requirements document
└── docs/paper_summary.md    # Research paper summary
```

---

## 🤖 ML & RL Details

### Failure Prediction

- **Model**: DistilBERT → mean pooling → PCA (768D → 16D) → Feed-forward classifier
- **Input**: Build logs + 8 telemetry features (24D total)
- **Output**: Failure probability + binary state (Healthy / Imminent Failure)
- **Performance**: F1-score 87%, Precision 89%, Recall 86%

### RL Agent

- **Algorithm**: PPO (Proximal Policy Optimization) via stable-baselines3
- **State space**: 52D (10 build + 12 resource + 16 log embeddings + 14 dependency)
- **Action space**: 6 discrete healing actions
- **Reward**: R = 0.6·MTTR\_reduction + 0.3·resource\_efficiency − 0.1·false\_positive\_penalty
- **Result**: 44% average MTTR reduction (paper target: 38%)

---

## 📊 Results

| Metric | Baseline | NeuroShield | Improvement |
|--------|----------|-------------|-------------|
| OOM Error MTTR | 14.2 min | 7.5 min | 47% |
| Flaky Test MTTR | 8.5 min | 4.3 min | 49% |
| Dependency Conflict MTTR | 15.1 min | 9.8 min | 35% |
| Average MTTR | 12.4 min | 7.7 min | 38% (paper) / 44% (code) |
| Failure Prediction F1 | — | 87% | — |
| False Positive Rate | 23% (Jenkins) | 7.8% | 66% reduction |

---

## ⚙️ Configuration

All settings are read from environment variables (`.env` file). Copy `.env.example` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `JENKINS_URL` | `http://localhost:8080` | Jenkins server URL |
| `JENKINS_JOB` | `neuroshield-test-job` | Jenkins pipeline job name |
| `JENKINS_USERNAME` | _(none)_ | Jenkins username for API auth |
| `JENKINS_TOKEN` | _(none)_ | Jenkins API token |
| `K8S_NAMESPACE` | `neuroshield` | Kubernetes namespace for deployments |
| `AFFECTED_SERVICE` | `dummy-app` | Service name for kubectl healing actions |
| `SCALE_REPLICAS` | `3` | Replica count for reallocate\_resources action |
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus server URL |
| `TELEMETRY_LOGS_ENABLED` | `true` | Enable build log capture in telemetry |
| `TELEMETRY_OUTPUT_PATH` | `data/telemetry.csv` | CSV output path for telemetry data |
| `POLL_INTERVAL` | `10` | Telemetry polling interval in seconds |
| `MODEL_PATH` | `models/` | Directory for trained model artifacts |
| `PREDICTION_THRESHOLD` | `0.7` | Failure probability threshold for action |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## 🔒 Security Notes

- **Never commit `.env`** — it is gitignored and contains credentials
- **Rotate Jenkins token** before sharing the repository
- **Log redaction** is enabled by default — API keys, tokens, passwords, secrets, and bearer tokens are automatically masked in captured build logs

---

## 📄 License

MIT License — KOVENDHAN P
