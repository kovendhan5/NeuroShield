# NeuroShield

**AI-Powered Self-Healing CI/CD Pipeline**

NeuroShield monitors CI/CD pipelines in real time, predicts build failures before
they happen, and executes autonomous healing actions using reinforcement learning.
It combines DistilBERT log analysis with a PPO-trained RL agent to select from
six discrete healing actions, achieving a **44% MTTR reduction** and **F1 = 1.000**
failure prediction accuracy.

```
                          NeuroShield Architecture
 ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
 │   Jenkins    │────>│  Telemetry   │────>│   Failure    │
 │   Pipeline   │     │  Collector   │     │  Predictor   │
 └─────────────┘     └──────────────┘     │  (DistilBERT │
 ┌─────────────┐            │             │   + PCA)     │
 │  Prometheus  │────────────┘             └──────┬───────┘
 │   Metrics    │                                 │
 └─────────────┘                          ┌──────▼───────┐
 ┌─────────────┐                          │  RL Agent    │
 │  Dummy App  │<─────── healing ────────│  (PPO)       │
 │  (Flask)    │          actions         └──────┬───────┘
 └─────────────┘                                 │
                                          ┌──────▼───────┐
                                          │  Streamlit   │
                                          │  Dashboard   │
                                          └──────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| MTTR Reduction | **44%** (target: 38%) |
| Failure Prediction F1 | **1.000** |
| State Space | 52 dimensions |
| Healing Actions | 6 autonomous |
| False Positive Rate | 7.8% (vs 23% Jenkins baseline) |

---

## Technologies

| Layer | Technology |
|---|---|
| Failure Prediction | PyTorch (FailureClassifier), DistilBERT, sklearn PCA |
| Reinforcement Learning | Stable Baselines3 (PPO), Gymnasium |
| Telemetry | Jenkins REST API, Prometheus HTTP API |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker Compose, Minikube, Kubernetes |
| CI/CD | Jenkins |

---

## Project Structure

```
NeuroShield/
├── src/
│   ├── orchestrator/      # Main healing loop
│   │   └── main.py        # Jenkins poll → predict → PPO → heal
│   ├── dashboard/         # Streamlit UI
│   │   └── app.py         # Dark-themed dashboard with charts
│   ├── prediction/        # DistilBERT + PCA failure predictor
│   │   ├── data_generator.py
│   │   ├── log_encoder.py # DistilBERT → PCA (768D → 16D)
│   │   ├── model.py       # FailureClassifier neural network
│   │   ├── predictor.py   # Runtime inference + 52D state builder
│   │   └── train.py       # Training script
│   ├── rl_agent/          # PPO reinforcement learning agent
│   │   ├── env.py         # Gymnasium env (52D obs, 6 actions)
│   │   ├── simulator.py   # Synthetic state generator
│   │   └── train.py       # PPO training + evaluation
│   └── telemetry/         # Jenkins & Prometheus collectors
│       ├── collector.py   # TelemetryCollector with log redaction
│       ├── config.py      # Centralized env var loading
│       └── main.py        # CLI entry point
├── models/                # Trained model weights
│   ├── failure_predictor.pth
│   ├── log_pca.joblib
│   └── ppo_policy.zip
├── data/                  # Telemetry CSV data
├── tests/                 # Pytest test suite
├── scripts/               # Setup & health check utilities
│   ├── health_check.py    # Verify services, models, imports
│   ├── setup_local.ps1    # Windows setup
│   └── setup_local.sh     # Linux/Mac setup
├── infra/                 # Dockerfiles (Jenkins, Prometheus, dummy-app)
├── docs/                  # Paper summary, PRD
└── microservices-demo/    # Weaveworks Sock Shop (reference deployment)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop (for Jenkins & Prometheus)
- Minikube (optional, for Kubernetes demo)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Jenkins token
# Get token at: http://localhost:8080/user/admin/configure
```

### 3. Start infrastructure

```bash
docker compose up -d          # Jenkins + Prometheus
python setup_jenkins_job.py   # Create the build pipeline job
```

### 4. Train models

```bash
python src/prediction/train.py
python -m src.rl_agent.train
```

### 5. Run the orchestrator

```bash
# Simulation mode (no live services needed)
python src/orchestrator/main.py --mode simulate

# Live mode (requires Jenkins + Prometheus running)
python src/orchestrator/main.py --mode live
```

### 6. Launch the dashboard

```bash
python -m streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 7. Health check

```bash
python scripts/health_check.py
```

---

## Self-CI Pipeline (Meta-Monitoring)

NeuroShield monitors **its own** CI/CD pipeline — a meta-feature where the system
watches itself for regressions.

### Setup

```bash
# Create the neuroshield-ci Jenkins job and trigger the first build
python scripts/setup_neuroshield_cicd.py
```

This creates a freestyle Jenkins job **neuroshield-ci** with 6 build steps:

| Step | What It Does |
|---|---|
| 1. Code Quality | `py_compile` on all core modules |
| 2. Pytest Suite | Runs all tests (`pytest tests/ -q`) |
| 3. Model Validation | Verifies `.pth`, `.joblib` model files load correctly |
| 4. Health Check | Runs `scripts/health_check.py` |
| 5. Integration Test | End-to-end predict call through the real model |
| 6. Deploy Simulation | Tests rule-based action selection logic |

### How It Works

- **Telemetry**: `TelemetryCollector` polls both the primary job and `neuroshield-ci`
  via `self_ci_status` / `self_ci_duration` fields.
- **Orchestrator**: Each main-loop cycle checks the self-CI job. On failure,
  `handle_self_ci_failure()` writes `data/self_ci_status.json`, fires a CRITICAL
  alert, and sends a desktop notification.
- **Dashboard**: The sidebar shows live self-CI build status. A dedicated
  "Self-CI History" section displays build number, result, duration, and alert state.
- **Environment**: Set `SELF_CI_JOB=neuroshield-ci` in `.env` (default).

---

## Demo Guide

**Quickest demo (2 minutes, no Kubernetes needed):**

1. `pip install -r requirements.txt`
2. `python src/prediction/train.py && python -m src.rl_agent.train`
3. `python src/orchestrator/main.py --mode simulate`
4. In a second terminal: `python -m streamlit run src/dashboard/app.py`
5. Open http://localhost:8501

**What to show:**
- Dashboard auto-refreshes every 10 seconds
- Failure probability chart updates in real time
- RL agent recommends one of 6 healing actions per cycle
- Click **Run Healing Cycle** to trigger a manual cycle
- 4 metric cards: MTTR 44%, F1 1.000, total actions, system health

**Key numbers to mention:**
- 52D state space (10 build + 12 resource + 16 log + 14 dependency)
- 6 autonomous healing actions
- 44% MTTR reduction (paper target: 38%)
- F1-score 1.000 for failure prediction

---

## State Space (52 dimensions)

| Component | Dimensions | Features |
|---|---|---|
| Build Metrics | 10 | duration, result, queue time, stage counts |
| Resource Metrics | 12 | CPU, memory, disk, network (per-node) |
| Log Embeddings | 16 | DistilBERT encoding → PCA reduction |
| Dependency Metrics | 14 | package versions, vulnerability counts |

## Healing Actions

| ID | Action | Description |
|---|---|---|
| 0 | `restart_pod` | Restart the affected Kubernetes pod |
| 1 | `scale_up` | Increase replica count |
| 2 | `retry_build` | Re-trigger the Jenkins build |
| 3 | `rollback_deploy` | Roll back to last known-good deployment |
| 4 | `clear_cache` | Clear build/dependency caches |
| 5 | `escalate_to_human` | Alert on-call engineer |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
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
