# NeuroShield — Project State Tracker
> Last Updated: 2026-03-04 (rev 4)
> Overall Status: 🟡 IN PROGRESS

---

## ✅ COMPLETED

- [x] **Telemetry Collector** — Polls Jenkins API + Prometheus metrics, saves to CSV. Log redaction, append-safe CSV, configurable log capture toggle. (`src/telemetry/`)
- [x] **Failure Predictor (ML Pipeline)** — DistilBERT log encoder → PCA (16D) → feed-forward classifier. Synthetic data generator, training script, inference with `predict_with_state()`. (`src/prediction/`)
- [x] **RL Environment (52D / 6 actions)** — Gymnasium env matching paper spec: 10D build + 12D resource + 16D log + 14D dependency signals. 6 discrete actions. Reward: R = 0.6·mttr_reduction + 0.3·resource_efficiency − 0.1·false_positive_penalty. (`src/rl_agent/env.py`)
- [x] **RL Simulator** — Generates realistic 52D synthetic states conditioned on failure type. MTTR tables from paper Table 1. Per-action resource costs and false-positive tracking. (`src/rl_agent/simulator.py`)
- [x] **PPO Training** — stable-baselines3 PPO with eval callback, saves `models/ppo_policy.zip`. Auto-deletes stale model before retraining. Evaluation prints MTTR reduction + action distribution for 6 actions. (`src/rl_agent/train.py`)
- [x] **Real-Time Orchestrator** — Monitors Jenkins builds, runs failure prediction with 24D classifier, builds 52D state for PPO, selects healing action from 6-action space, executes kubectl/Jenkins actions. All actions logged to `data/action_history.csv`. Retry helper with exponential back-off. `--mode live|simulate` argparse. (`src/orchestrator/main.py`)
- [x] **Jenkins Job Setup** — Creates pipeline job via REST API with crumb auth. Idempotent: treats "already exists" as success. (`setup_jenkins_job.py`)
- [x] **Security Hardening** — Log redaction (apikey/token/password/secret/bearer patterns), kubectl pinned with SHA256 checksum in Dockerfile, Jenkins resource limits in K8s manifest, credential exposure mitigated.
- [x] **Config Unification** — `.env.example` has all variables. Orchestrator uses `K8S_NAMESPACE`, `AFFECTED_SERVICE`, `SCALE_REPLICAS` env vars. No hardcoded `sock-shop` or `carts`.
- [x] **Prebuilt Dummy App Image** — Flask-based dummy app with Dockerfile, referenced in K8s manifests. (`infra/dummy-app/`)
- [x] **Telemetry Unit Tests** — Tests for JenkinsPoll, PrometheusPoll, TelemetryCollector, TelemetryData. (`tests/test_telemetry.py`)
- [x] **Streamlit Dashboard** — Human-in-the-loop AIOps dashboard with: failure-probability gauge, RL agent decision panel, SHAP feature importance, MTTR trend chart, failure-type breakdown pie, feedback approve/override/pause buttons, escalation review, sidebar controls. Auto-refreshes every 10s. (`src/dashboard/app.py`)

## 🔄 IN PROGRESS

_(All three previous high-severity items resolved — see rev 2 notes below.)_

## ❌ NOT STARTED

- [ ] **Streamlit Dashboard** — ~~Paper describes a real-time dashboard~~ **MOVED TO COMPLETED (rev 4)**.
- [ ] **End-to-End Integration Tests** — No tests for prediction pipeline, RL training, or orchestrator loop. Needs: `tests/test_prediction.py`, `tests/test_rl_agent.py`, `tests/test_orchestrator.py`.
- [ ] **Automated Local Setup Script** — No single command to spin up Minikube + Jenkins + Prometheus + dummy app + start telemetry. Needs: `scripts/setup_local.sh` or `Makefile` target.
- [ ] **CI/CD Pipeline for NeuroShield Itself** — No GitHub Actions or Jenkinsfile for linting, testing, building container images.
- [ ] **Grafana/Prometheus Monitoring Stack** — Docker-compose files exist in `microservices-demo/` but are not wired to NeuroShield's own metrics.
- [ ] **Model Retraining Pipeline** — No automated retraining from production telemetry data. Paper mentions continuous learning loop.

## ⚠️ KNOWN ISSUES

- ~~ISSUE: Predictor returns 24D state but PPO expects 52D~~ → **RESOLVED**: `build_52d_state()` added to `predictor.py`; orchestrator uses 52D for PPO and 24D for classifier separately.
- ~~ISSUE: `src/orchestration/main.py` stale 4-action map~~ → **RESOLVED**: Deprecated with warning; `run_once()` consolidated into `src/orchestrator/main.py --mode simulate`.
- ~~ISSUE: `execute_healing_action()` only handles 0-3~~ → **RESOLVED**: Full 6-action handler with CSV logging to `data/action_history.csv`.
- ~~ISSUE: Stale PPO model trained on 24D/4-action env~~ → **RESOLVED**: `train.py` auto-deletes stale model. Retrained 52D/6-action PPO — 44% avg MTTR reduction, 56% success rate.
- ~~ISSUE: pytest not installed~~ → **RESOLVED**: pytest 9.0.2 installed; 8/8 telemetry tests pass.
- ISSUE: README.md claims Ray RLlib but code uses stable-baselines3 PPO | FILE: README.md | SEVERITY: **Low**
- ISSUE: `.env` file contains real Jenkins token — must rotate before sharing repo | FILE: .env | SEVERITY: **Medium**
- ISSUE: `_init_csv()` output path defaults differ between collector ("data/telemetry.csv") and .env.example (`TELEMETRY_OUTPUT_PATH`) — key name is `TELEMETRY_OUTPUT` in config.py | FILE: src/telemetry/config.py | SEVERITY: **Low**

## 📐 PAPER vs CODE ALIGNMENT

| Component | Paper Says | Code Has | Status |
|-----------|-----------|----------|--------|
| Observation Space | 52D (10 build + 12 resource + 16 log + 14 dep) | 52D matching spec | ✅ Done |
| Action Space | 6 discrete actions | 6 in RL env + 6 in orchestrator | ✅ Done |
| Reward Function | R = 0.6·MTTR + 0.3·efficiency − 0.1·FP | Implemented in env.py | ✅ Done |
| RL Algorithm | PPO (paper is ambiguous — mentions both PPO and DQN) | stable-baselines3 PPO | ✅ Done |
| Log Encoder | DistilBERT → PCA (16D) | DistilBERT → PCA (16D) | ✅ Done |
| Failure Classifier | Feed-forward neural net | 24→ReLU→Dropout→2 | ✅ Done |
| Real-Time Orchestrator | Jenkins polling → predict → act → measure MTTR | Implemented with retry | ✅ Done |
| Dashboard | Streamlit with MTTR charts, action log, failure heatmap | Full dashboard with 7 sections + human-in-the-loop | ✅ Done |
| MTTR Reduction Target | 38% average | 44% avg in 50-episode eval (52D/6-action PPO) | ✅ Exceeds target |
| Failure Types | OOM, FlakyTest, DependencyConflict, NetworkLatency | All four + Healthy | ✅ Done |
| Kubernetes Integration | kubectl healing actions | All 6 actions implemented | ✅ Done |
| Continuous Retraining | Online learning from production data | Not implemented | ❌ Missing |
| Telemetry Collection | Jenkins + Prometheus polling → CSV | Implemented | ✅ Done |

## 🗓️ NEXT ACTIONS (ordered by priority)

1. ~~Build 52D state vector in orchestrator~~ — **DONE** (rev 2)
2. ~~Expand `execute_healing_action()` to 6 actions~~ — **DONE** (rev 2)
3. ~~Retrain PPO on new 52D/6-action env~~ — **DONE** (rev 3 — 44% MTTR reduction)
4. **Add pytest to requirements.txt and write core tests** — `tests/test_prediction.py`, `tests/test_rl_agent.py`, `tests/test_orchestrator.py` — No tests exist outside telemetry.
5. ~~Consolidate or delete `src/orchestration/main.py`~~ — **DONE** (rev 2 — deprecated; `run_once()` in orchestrator)
6. ~~Build Streamlit dashboard~~ — **DONE** (rev 4 — `src/dashboard/app.py` with 7 sections)
7. **Fix README.md** — Replace "Ray RLlib" with "stable-baselines3 PPO", update architecture diagram, fill in "Coming soon" sections.
8. **Create local setup automation** — Script or Makefile to: start Minikube, deploy dummy-app, start Jenkins container, run telemetry, launch orchestrator.
9. **Align telemetry config var names** — Standardize `TELEMETRY_OUTPUT` vs `TELEMETRY_OUTPUT_PATH` across config.py, .env.example, and collector.py.
10. **Rotate Jenkins token in `.env`** — Current token is live; generate a new one before any repo sharing.

## 📁 KEY FILES REFERENCE

- `setup_jenkins_job.py` — Creates Jenkins pipeline job (idempotent, crumb auth)
- `scripts/create_jenkins_job.py` — **DEPRECATED** older Jenkins setup script
- `.env.example` — All environment variables with defaults
- `src/telemetry/collector.py` — Jenkins + Prometheus polling, CSV output, log redaction
- `src/telemetry/config.py` — Centralized env var loading for telemetry
- `src/telemetry/main.py` — CLI entry point for telemetry collector
- `src/prediction/train.py` — End-to-end ML training (data gen → encode → PCA → classifier)
- `src/prediction/predictor.py` — Runtime inference: log text + telemetry → failure probability + state vector
- `src/prediction/log_encoder.py` — DistilBERT mean-pooled embeddings + PCA reduction
- `src/prediction/model.py` — FailureClassifier feed-forward network
- `src/prediction/data_generator.py` — Synthetic training data with injected failure patterns
- `src/rl_agent/env.py` — Gymnasium env (52D obs, 6 actions, paper reward function)
- `src/rl_agent/simulator.py` — Synthetic 52D state generator with failure-type-conditioned distributions
- `src/rl_agent/train.py` — PPO training + evaluation script
- `src/orchestrator/main.py` — Live runtime: Jenkins monitor → predict → PPO action → kubectl heal
- `src/orchestration/main.py` — Simulation-only demo orchestrator (DUPLICATE — to be removed)
- `infra/dummy-app/app.py` — Flask dummy app for failure injection testing
- `infra/dummy-app/Dockerfile` — Prebuilt dummy app container image
- `infra/jenkins-builder/Dockerfile.jenkins` — Custom Jenkins image with pinned kubectl
- `tests/test_telemetry.py` — Unit tests for telemetry collector
- `PRD.md` — Product requirements document
- `docs/paper_summary.md` — Research paper summary
