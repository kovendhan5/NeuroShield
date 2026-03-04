# NeuroShield — Project State Tracker
> Last Updated: 2026-03-04
> Overall Status: 🟡 IN PROGRESS

---

## ✅ COMPLETED

- [x] **Telemetry Collector** — Polls Jenkins API + Prometheus metrics, saves to CSV. Log redaction, append-safe CSV, configurable log capture toggle. (`src/telemetry/`)
- [x] **Failure Predictor (ML Pipeline)** — DistilBERT log encoder → PCA (16D) → feed-forward classifier. Synthetic data generator, training script, inference with `predict_with_state()`. (`src/prediction/`)
- [x] **RL Environment (52D / 6 actions)** — Gymnasium env matching paper spec: 10D build + 12D resource + 16D log + 14D dependency signals. 6 discrete actions. Reward: R = 0.6·mttr_reduction + 0.3·resource_efficiency − 0.1·false_positive_penalty. (`src/rl_agent/env.py`)
- [x] **RL Simulator** — Generates realistic 52D synthetic states conditioned on failure type. MTTR tables from paper Table 1. Per-action resource costs and false-positive tracking. (`src/rl_agent/simulator.py`)
- [x] **PPO Training** — stable-baselines3 PPO with eval callback, saves `models/ppo_policy.zip`. Evaluation prints MTTR reduction + action distribution for 6 actions. (`src/rl_agent/train.py`)
- [x] **Real-Time Orchestrator** — Monitors Jenkins builds, runs failure prediction, selects healing action via PPO, executes kubectl/Jenkins actions. Retry helper with exponential back-off on all API/kubectl calls. All config externalized via env vars. (`src/orchestrator/main.py`)
- [x] **Jenkins Job Setup** — Creates pipeline job via REST API with crumb auth. Idempotent: treats "already exists" as success. (`setup_jenkins_job.py`)
- [x] **Security Hardening** — Log redaction (apikey/token/password/secret/bearer patterns), kubectl pinned with SHA256 checksum in Dockerfile, Jenkins resource limits in K8s manifest, credential exposure mitigated.
- [x] **Config Unification** — `.env.example` has all variables. Orchestrator uses `K8S_NAMESPACE`, `AFFECTED_SERVICE`, `SCALE_REPLICAS` env vars. No hardcoded `sock-shop` or `carts`.
- [x] **Prebuilt Dummy App Image** — Flask-based dummy app with Dockerfile, referenced in K8s manifests. (`infra/dummy-app/`)
- [x] **Telemetry Unit Tests** — Tests for JenkinsPoll, PrometheusPoll, TelemetryCollector, TelemetryData. (`tests/test_telemetry.py`)

## 🔄 IN PROGRESS

- [ ] **Prediction Model ↔ Orchestrator State Vector Mismatch** — Predictor outputs 24D (16D log + 8 telemetry), but RL env now expects 52D. The orchestrator feeds `predictor.predict_with_state()` output directly to PPO, which will crash at inference time. Needs a 52D state builder matching the RL env layout.
- [ ] **Duplicate Orchestration Paths** — `src/orchestration/main.py` (simulation demo) and `src/orchestrator/main.py` (live runtime) both exist with different action maps and logic. Need to consolidate or clearly separate.
- [ ] **Orchestrator Action Map vs RL Action Map** — Orchestrator uses 4 actions {Retry, Scale, Rollback, No-op} but RL env now has 6 {retry_stage, clean_and_rerun, regenerate_config, reallocate_resources, trigger_safe_rollback, escalate_to_human}. `execute_healing_action()` needs to handle all 6.

## ❌ NOT STARTED

- [ ] **Streamlit Dashboard** — Paper describes a real-time dashboard with MTTR charts, action history, failure heatmaps. No code exists. Needs: `src/dashboard/app.py` or similar.
- [ ] **End-to-End Integration Tests** — No tests for prediction pipeline, RL training, or orchestrator loop. Needs: `tests/test_prediction.py`, `tests/test_rl_agent.py`, `tests/test_orchestrator.py`.
- [ ] **Automated Local Setup Script** — No single command to spin up Minikube + Jenkins + Prometheus + dummy app + start telemetry. Needs: `scripts/setup_local.sh` or `Makefile` target.
- [ ] **CI/CD Pipeline for NeuroShield Itself** — No GitHub Actions or Jenkinsfile for linting, testing, building container images.
- [ ] **Grafana/Prometheus Monitoring Stack** — Docker-compose files exist in `microservices-demo/` but are not wired to NeuroShield's own metrics.
- [ ] **Model Retraining Pipeline** — No automated retraining from production telemetry data. Paper mentions continuous learning loop.

## ⚠️ KNOWN ISSUES

- ISSUE: Predictor returns 24D state but PPO expects 52D after env update — inference will fail | FILE: src/orchestrator/main.py + src/prediction/predictor.py | SEVERITY: **High**
- ISSUE: `src/orchestration/main.py` uses stale 4-action map {0:Retry, 1:Scale, 2:Rollback, 3:No-op} — conflicts with new 6-action RL env | FILE: src/orchestration/main.py | SEVERITY: **Medium**
- ISSUE: `execute_healing_action()` only handles actions 0-3; actions 4 (safe_rollback) and 5 (escalate_to_human) fall through to "Unknown action" | FILE: src/orchestrator/main.py | SEVERITY: **High**
- ISSUE: PPO model saved as `ppo_policy.zip` was trained on old 24D/4-action env — must be retrained before use with new 52D/6-action env | FILE: models/ppo_policy.zip | SEVERITY: **High**
- ISSUE: pytest not installed in active Python 3.13.1 interpreter — `pip install pytest` needed | FILE: requirements.txt (pytest missing) | SEVERITY: **Low**
- ISSUE: README.md claims Ray RLlib but code uses stable-baselines3 PPO | FILE: README.md | SEVERITY: **Low**
- ISSUE: `.env` file contains real Jenkins token — must rotate before sharing repo | FILE: .env | SEVERITY: **Medium**
- ISSUE: `_init_csv()` output path defaults differ between collector ("data/telemetry.csv") and .env.example (`TELEMETRY_OUTPUT_PATH`) — key name is `TELEMETRY_OUTPUT` in config.py | FILE: src/telemetry/config.py | SEVERITY: **Low**

## 📐 PAPER vs CODE ALIGNMENT

| Component | Paper Says | Code Has | Status |
|-----------|-----------|----------|--------|
| Observation Space | 52D (10 build + 12 resource + 16 log + 14 dep) | 52D matching spec | ✅ Done |
| Action Space | 6 discrete actions | 6 in RL env, 4 in orchestrator | ⚠️ Partial |
| Reward Function | R = 0.6·MTTR + 0.3·efficiency − 0.1·FP | Implemented in env.py | ✅ Done |
| RL Algorithm | PPO (paper is ambiguous — mentions both PPO and DQN) | stable-baselines3 PPO | ✅ Done |
| Log Encoder | DistilBERT → PCA (16D) | DistilBERT → PCA (16D) | ✅ Done |
| Failure Classifier | Feed-forward neural net | 24→ReLU→Dropout→2 | ✅ Done |
| Real-Time Orchestrator | Jenkins polling → predict → act → measure MTTR | Implemented with retry | ✅ Done |
| Dashboard | Streamlit with MTTR charts, action log, failure heatmap | Not started | ❌ Missing |
| MTTR Reduction Target | 38% average | Simulated only, not validated on live data | ⚠️ Unverified |
| Failure Types | OOM, FlakyTest, DependencyConflict, NetworkLatency | All four + Healthy | ✅ Done |
| Kubernetes Integration | kubectl healing actions | 4 of 6 actions implemented | ⚠️ Partial |
| Continuous Retraining | Online learning from production data | Not implemented | ❌ Missing |
| Telemetry Collection | Jenkins + Prometheus polling → CSV | Implemented | ✅ Done |

## 🗓️ NEXT ACTIONS (ordered by priority)

1. **Build 52D state vector in orchestrator** — `src/orchestrator/main.py` + `src/prediction/predictor.py` — PPO inference will crash without this; must map all 52 features from Jenkins/Prometheus/kubectl data.
2. **Expand `execute_healing_action()` to 6 actions** — `src/orchestrator/main.py` — Add handlers for `clean_and_rerun` (action 1), `regenerate_config` (action 2 remap), `trigger_safe_rollback` (action 4), `escalate_to_human` (action 5).
3. **Retrain PPO on new 52D/6-action env** — Run `python -m src.rl_agent.train --timesteps 100000` to produce a new `ppo_policy.zip` compatible with the updated env.
4. **Add pytest to requirements.txt and write core tests** — `tests/test_prediction.py`, `tests/test_rl_agent.py`, `tests/test_orchestrator.py` — No tests exist outside telemetry.
5. **Consolidate or delete `src/orchestration/main.py`** — Duplicate of `src/orchestrator/main.py` with stale action map. Either merge useful bits or remove.
6. **Build Streamlit dashboard** — `src/dashboard/app.py` — Paper's key deliverable: MTTR trend chart, action log table, failure type breakdown, live status.
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
