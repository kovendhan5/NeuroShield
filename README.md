# NeuroShield 🛡️

**PPO-Driven AIOps Self-Healing Framework for CI/CD Pipelines**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ray](https://img.shields.io/badge/Ray-RLlib-orange.svg)](https://docs.ray.io/en/latest/rllib/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Turning pipeline chaos into resilient intelligence** – An autonomous CI/CD self-healing system using transformer-based failure prediction and PPO reinforcement learning.

## 📋 Overview

NeuroShield is a proof-of-concept implementation of an AIOps-driven self-healing CI/CD system that:

- **Predicts** imminent pipeline failures using transformer-based log analysis (DistilBERT)
- **Autonomously mitigates** issues via Proximal Policy Optimization (PPO) reinforcement learning
- **Reduces MTTR** by ~40%+ through intelligent, automated recovery actions
- **Integrates** with Jenkins, Prometheus, and Kubernetes (Sock Shop microservices benchmark)

Inspired by research in "AIOps-Driven Self-Healing Pipelines" (2025), NeuroShield demonstrates how ML/RL can shift DevOps from reactive troubleshooting to predictive, autonomous recovery.

## 🎯 Key Features

- **Real-time Telemetry Collection** from Jenkins API, Prometheus metrics, and Kubernetes
- **Transformer-based Failure Prediction** (≥80% accuracy) using fine-tuned DistilBERT
- **PPO RL Agent** with 4 core actions: Retry, Scale pods, Rollback, No-op
- **Simulation Environment** with synthetic failure injection (OOM, flaky tests, dependencies)
- **Human-in-the-loop Dashboard** using Streamlit for monitoring and manual overrides
- **Local Development Setup** – runs entirely on Windows 11 with Minikube + Jenkins

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuroShield System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │   Telemetry      │─────▶│  Log Encoder +   │            │
│  │   Aggregator     │      │  Failure         │            │
│  │  (Jenkins/Prom/  │      │  Predictor       │            │
│  │   Kubernetes)    │      │  (DistilBERT)    │            │
│  └──────────────────┘      └─────────┬────────┘            │
│                                       │                      │
│                                       ▼                      │
│                            ┌──────────────────┐             │
│                            │   RL Agent       │             │
│                            │   (PPO)          │             │
│                            │  Action Select   │             │
│                            └─────────┬────────┘             │
│                                      │                      │
│                                      ▼                      │
│                            ┌──────────────────┐             │
│                            │  Orchestrator    │             │
│                            │  (Execute via    │             │
│                            │  Jenkins API +   │             │
│                            │  kubectl)        │             │
│                            └──────────────────┘             │
│                                                               │
│                    ┌────────────────────┐                   │
│                    │  Streamlit UI      │                   │
│                    │  (Monitor/Override)│                   │
│                    └────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker Desktop** (for Minikube)
- **Minikube** (Kubernetes local cluster)
- **Jenkins** (local or containerized)
- **Prometheus** (for metrics)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuroShield.git
cd NeuroShield

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Jenkins/Prometheus URLs
```

### Configuration

Create a `.env` file with the following:

```env
JENKINS_URL=http://localhost:8080
JENKINS_JOB=build-pipeline
JENKINS_USERNAME=admin
JENKINS_TOKEN=your_api_token

PROMETHEUS_URL=http://localhost:9090
POLL_INTERVAL=10

TELEMETRY_OUTPUT=telemetry.csv
LOG_LEVEL=INFO
```

### Running the Telemetry Collector

```bash
# Start collecting telemetry data
python -m src.telemetry.main

# With custom parameters
python -m src.telemetry.main --jenkins-url http://jenkins:8080 --interval 5
```

### Running Tests

```bash
pytest tests/test_telemetry.py -v
```

## 📊 Components

### 1. Telemetry Collection (`src/telemetry/`)

Real-time data aggregation from multiple sources:

- **Jenkins API**: Build status, duration, queue length
- **Prometheus**: CPU/memory usage, pod count, error rates
- **Output**: Time-synced CSV for analysis and ML training

```python
from src.telemetry import TelemetryCollector

collector = TelemetryCollector(
    jenkins_url="http://localhost:8080",
    prometheus_url="http://localhost:9090",
    poll_interval=10
)
collector.start()
```

See [src/telemetry/README.md](src/telemetry/README.md) for detailed documentation.

### 2. Failure Prediction (`src/prediction/`)

*Coming soon* – Transformer-based model for predicting CI/CD failures:

- Fine-tuned DistilBERT on log sequences
- Binary classification (failure/success)
- State vector generation (~20 dimensions)

### 3. RL Agent (`src/rl_agent/`)

*Coming soon* – PPO agent with custom Gym environment:

- **State**: Pipeline health, resource usage, log embeddings
- **Actions**: Retry, Scale pods (+20%), Rollback, No-op
- **Reward**: `R = 0.5(1 - MTTR_norm) + 0.3(Success) - 0.2(Cost)`

### 4. Orchestration (`src/orchestration/`)

*Coming soon* – Action execution layer:

- Jenkins REST API integration
- kubectl command wrappers
- Human-in-the-loop feedback via Streamlit

## 📈 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Failure Prediction Accuracy | ≥80% | 🔄 In Progress |
| MTTR Reduction | ≥40% | 🔄 In Progress |
| False Positive Rate | ≤10% | 🔄 In Progress |
| Inference Time | <5s | 🔄 In Progress |

## 🗓️ Development Timeline

- **Week 1** ✅: Setup (Minikube, Jenkins, Sock Shop) + Telemetry scripts
- **Week 2** 🔄: Failure predictor + synthetic data generation
- **Week 3** 📅: RL agent + simulation environment + actions
- **Week 4** 📅: Integration, dashboard, evaluation, demo video

## 🛠️ Tech Stack

- **ML/RL**: PyTorch, Transformers (Hugging Face), Ray RLlib
- **Infrastructure**: Jenkins, Prometheus, Kubernetes (Minikube)
- **Orchestration**: Python, REST APIs, kubectl
- **Dashboard**: Streamlit
- **Benchmark**: Sock Shop microservices
- **Dev Tools**: pytest, black, flake8

## 📚 Project Structure

```
NeuroShield/
├── README.md                 # This file
├── PRD.md                    # Product requirements document
├── requirements.txt          # Python dependencies
├── docs/
│   └── paper_summary.md      # Research paper summary
├── src/
│   ├── telemetry/            # Telemetry collection (Week 1) ✅
│   │   ├── collector.py
│   │   ├── config.py
│   │   ├── main.py
│   │   └── README.md
│   ├── prediction/           # Failure prediction (Week 2) 🔄
│   ├── rl_agent/             # PPO agent (Week 3) 📅
│   └── orchestration/        # Action execution (Week 3-4) 📅
├── simulations/              # Failure injection scenarios
└── tests/
    └── test_telemetry.py     # Unit tests
```

## 🎓 Research Background

This project is inspired by the paper "AIOps-Driven Self-Healing Pipelines" (Kovendhan P et al., 2025), which demonstrated:

- **47% MTTR reduction** on Sock Shop benchmark
- **92% prediction accuracy** with transformer-based log encoding
- **6 autonomous actions** via PPO reinforcement learning

NeuroShield implements a simplified MVP version focusing on 4 core actions and ~40% MTTR reduction target.

## 🤝 Contributing

This is a personal portfolio project, but feedback and suggestions are welcome! Feel free to open issues or reach out.

## 🙏 Acknowledgments

- Sock Shop microservices demo
- Ray RLlib and Hugging Face Transformers communities
- Research inspiration from "AIOps-Driven Self-Healing Pipelines"

---

**NeuroShield** – Building the future of resilient, intelligent DevOps. 🛡️🔥

