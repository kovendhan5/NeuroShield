# NeuroShield — Project State Tracker
> Last Updated: 2026-03-05 (rev 7)
> Overall Status: 🟢 READY FOR DEMO

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
- [x] **End-to-End Integration Tests** — 83 tests across 4 files, all passing. Coverage: prediction pipeline (data generator, log encoder, classifier, 52D state builder), RL agent (simulator constants, sample_state, simulate_action, NeuroShieldEnv reset/step/termination/seeding), orchestrator (retry_call, detect_failure_pattern, kubectl parsing, CSV logging, all 6 healing actions with mocked Jenkins/K8s, BuildInfo). (`tests/test_prediction.py`, `tests/test_rl_agent.py`, `tests/test_orchestrator.py`, `tests/test_telemetry.py`)
- [x] **README.md Rewrite** — Complete rewrite: accurate architecture diagram, 52D/6-action spec, stable-baselines3 PPO (not Ray RLlib), results table, configuration reference, security notes. All "Coming soon" sections replaced with real content.
- [x] **Telemetry Config Var Alignment** — Standardized `TELEMETRY_OUTPUT_PATH` across `config.py`, `main.py`, `.env.example`, and `dashboard/app.py`. Default: `data/telemetry.csv`.
- [x] **Automated Local Setup Scripts** — Windows PowerShell scripts: `start_services.ps1`, `start_minikube.ps1`, `start_neuroshield.ps1`, `stop_neuroshield.ps1`. Plus Linux/Mac `setup_local.sh`.
- [x] **Docker Compose Infrastructure** — Jenkins + Prometheus run as Docker containers (not in Minikube). Minikube runs dummy-app only. Reduces memory usage significantly. (`docker-compose.yml`, `infra/prometheus/prometheus.yml`)

## 🔄 IN PROGRESS

_(No items currently in progress.)_

## ❌ NOT STARTED
- [ ] **CI/CD Pipeline for NeuroShield Itself** — No GitHub Actions or Jenkinsfile for linting, testing, building container images.
- [ ] **Model Retraining Pipeline** — No automated retraining from production telemetry data. Paper mentions continuous learning loop.

## ⚠️ KNOWN ISSUES

- ~~ISSUE: Predictor returns 24D state but PPO expects 52D~~ → **RESOLVED**: `build_52d_state()` added to `predictor.py`; orchestrator uses 52D for PPO and 24D for classifier separately.
- ~~ISSUE: `src/orchestration/main.py` stale 4-action map~~ → **RESOLVED**: Deprecated with warning; `run_once()` consolidated into `src/orchestrator/main.py --mode simulate`.
- ~~ISSUE: `execute_healing_action()` only handles 0-3~~ → **RESOLVED**: Full 6-action handler with CSV logging to `data/action_history.csv`.
- ~~ISSUE: Stale PPO model trained on 24D/4-action env~~ → **RESOLVED**: `train.py` auto-deletes stale model. Retrained 52D/6-action PPO — 44% avg MTTR reduction, 56% success rate.
- ~~ISSUE: pytest not installed~~ → **RESOLVED**: pytest 9.0.2 installed; 8/8 telemetry tests pass.
- ~~ISSUE: README.md claims Ray RLlib but code uses stable-baselines3 PPO~~ → **RESOLVED** (rev 6): Full README rewrite with correct stable-baselines3 references.
- ISSUE: `.env` file contains real Jenkins token — must rotate before sharing repo | FILE: .env | SEVERITY: **Medium**
- ~~ISSUE: `_init_csv()` output path defaults differ between collector and .env.example~~ → **RESOLVED** (rev 6): Standardized to `TELEMETRY_OUTPUT_PATH` everywhere.

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
4. ~~Add pytest and write core tests~~ — **DONE** (rev 5 — 83 tests across 4 files)
5. ~~Consolidate or delete `src/orchestration/main.py`~~ — **DONE** (rev 2 — deprecated; `run_once()` in orchestrator)
6. ~~Build Streamlit dashboard~~ — **DONE** (rev 4 — `src/dashboard/app.py` with 7 sections)
7. ~~Fix README.md~~ — **DONE** (rev 6 — complete rewrite with accurate architecture, results, config table)
8. ~~Create local setup automation~~ — **DONE** (rev 6 — `scripts/setup_local.sh` + `scripts/setup_local.ps1`)
9. ~~Align telemetry config var names~~ — **DONE** (rev 6 — standardized to `TELEMETRY_OUTPUT_PATH`)
10. ~~Docker Compose infrastructure~~ — **DONE** (rev 7 — Jenkins + Prometheus as Docker containers, Minikube for dummy-app only)
11. **Rotate Jenkins token in `.env`** — Current token is live; generate a new one before any repo sharing.

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
- `tests/test_prediction.py` — Prediction pipeline tests (data gen, encoder, classifier, 52D state)
- `tests/test_rl_agent.py` — RL env + simulator tests (state sampling, actions, env API)
- `tests/test_orchestrator.py` — Orchestrator tests (retry, healing actions, CSV logging)
- `scripts/setup_local.sh` — One-command local setup (Linux/Mac)
- `scripts/setup_local.ps1` — One-command local setup (Windows)
- `scripts/start_services.ps1` — First-time Jenkins + Prometheus setup (Docker Compose)
- `scripts/start_minikube.ps1` — Start Minikube + build/deploy dummy-app
- `scripts/start_neuroshield.ps1` — Master quick-start (daily use after first-time setup)
- `scripts/stop_neuroshield.ps1` — Clean shutdown of all services
- `docker-compose.yml` — Jenkins + Prometheus container definitions
- `infra/prometheus/prometheus.yml` — Prometheus scrape config (Jenkins, dummy-app)
- `DEMO.md` — Quick demo guide with key numbers
- `PRD.md` — Product requirements document
- `docs/paper_summary.md` — Research paper summary
