## NeuroShield Project Handoff Summary (for continuation)

### 1. Project Overview

**What this project does**
- NeuroShield is a local AIOps CI/CD self-healing prototype.
- It monitors Jenkins builds + telemetry, predicts failure risk using a DistilBERT-based model, and selects mitigation actions via PPO RL.
- It includes local Kubernetes/Jenkins manifests and scripts for spinning up a lightweight testbed (dummy app + Jenkins on Minikube).

**Tech stack**
- Language: Python 3.12+/3.13 (current env uses pyenv 3.13.1).
- ML/RL: `torch`, `transformers`, `scikit-learn`, `stable-baselines3`, `gymnasium`.
- Data/utility: `pandas`, `numpy`, `python-dotenv`, `requests`.
- Dev/testing: `pytest`, `black`, `flake8`.
- Infra/runtime: Docker Desktop, Minikube, Kubernetes manifests, Jenkins.
- Monitoring source: Prometheus HTTP API (polled via `requests`).
- Note: README says Ray RLlib, but actual RL code uses Stable-Baselines3 PPO.

**Folder/file structure (high-level)**
```text
NeuroShield/
  README.md
  PRD.md
  requirements.txt
  .env.example
  .env                         # local secrets/config (gitignored)
  setup_jenkins_job.py
  scripts/
    create_jenkins_job.py
  src/
    telemetry/
      collector.py
      config.py
      main.py
      README.md
    prediction/
      data_generator.py
      log_encoder.py
      model.py
      predictor.py
      train.py
    rl_agent/
      env.py
      simulator.py
      train.py
    orchestrator/
      main.py                  # runtime Jenkins/K8s orchestration loop
    orchestration/
      main.py                  # simulation/demo orchestrator path
  infra/
    jenkins-builder/
      Dockerfile.jenkins
    dummy-app/
      Dockerfile
      app.py
  jenkins-pvc.yaml
  jenkins-local-updated.yaml
  k8s-minimal.yaml
  dummy-app.yaml
  tests/
    test_telemetry.py
  microservices-demo/          # external benchmark repo content (large)
  models/                      # trained artifacts (pth/joblib/zip)
```

---

### 2. Current State

**Built and appears functional (code-level + prior runs)**
- Telemetry collector exists with Jenkins + Prometheus pollers and CSV output.
- Failure prediction pipeline exists end-to-end:
  - synthetic data generation
  - DistilBERT embedding + PCA
  - binary classifier training + inference
- RL components exist:
  - Gymnasium env
  - simulator
  - PPO training script and model save path
- Runtime orchestrator loop exists (`src/orchestrator/main.py`) with Jenkins polling, prediction, policy decision, and healing action execution.
- Local infra artifacts exist:
  - Jenkins PVC + deployment/service manifests
  - custom Jenkins image Dockerfile with pinned+verified `kubectl`
  - dummy-app prebuilt image flow (removed runtime `pip install` in pod).
- Security hardening already added:
  - telemetry log redaction patterns
  - optional telemetry log capture toggle (`TELEMETRY_LOGS_ENABLED`)
  - Jenkins resource limits in `jenkins-local-updated.yaml`
  - namespace param for healing actions (`K8S_NAMESPACE`) in orchestrator.

**Partially built / incomplete**
- Human-in-the-loop dashboard is mentioned in docs, but no Streamlit app is present.
- Prometheus integration is polling-based but simplistic and not robustly validated against real metrics schemas.
- Orchestration logic is split into two modules (`orchestrator` vs `orchestration`) with overlapping purpose and no unified entrypoint.
- End-to-end “production-like” flow is not fully codified into one reproducible script/task.
- Tests only cover telemetry module; prediction/RL/orchestrator integration tests are missing.

**Most important files**
- Runtime orchestration: `src/orchestrator/main.py`
- Telemetry ingestion: `src/telemetry/collector.py`, `src/telemetry/main.py`, `src/telemetry/config.py`
- Prediction model flow: `src/prediction/train.py`, `src/prediction/predictor.py`, `src/prediction/log_encoder.py`
- RL flow: `src/rl_agent/env.py`, `src/rl_agent/train.py`, `src/rl_agent/simulator.py`
- Jenkins job bootstrap: `setup_jenkins_job.py` (preferred over older script)
- Infra deploy: `jenkins-local-updated.yaml`, `jenkins-pvc.yaml`, `k8s-minimal.yaml`, `dummy-app.yaml`
- Custom images: `infra/jenkins-builder/Dockerfile.jenkins`, `infra/dummy-app/Dockerfile`

---

### 3. Known Issues

**Detected bugs/errors/inconsistencies**
- `pytest` is missing in the currently active interpreter (running tests fails with `No module named pytest`).
- Duplicate Jenkins job creation scripts with inconsistent behavior:
  - `setup_jenkins_job.py` includes required `mode=...WorkflowJob`
  - `scripts/create_jenkins_job.py` still omits mode and can fail with Jenkins 400 “No mode given”.
- Config mismatch risk:
  - `.env` may monitor `JENKINS_JOB=build-pipeline`, while setup scripts create `neuroshield-test-job`.
- Namespace/app mismatch risk:
  - orchestrator healing actions default to `K8S_NAMESPACE=sock-shop` and `AFFECTED_SERVICE=carts`; lightweight setup often uses dummy app/other namespace.
- Two orchestration paths create confusion:
  - `src/orchestrator/main.py` (real Jenkins loop)
  - `src/orchestration/main.py` (simulation-style run_once).
- Documentation drift:
  - README says multiple components are “Coming soon” though code exists.
  - README claims Ray RLlib, code uses Stable-Baselines3.
- Telemetry collector behavior:
  - `_init_csv()` overwrites CSV on each startup (history loss).
- Potential credential exposure:
  - `.env` currently contains real Jenkins token locally. It is gitignored, but should be rotated before sharing snapshots/logs.
- Prometheus metric extraction takes first result only; may not represent cluster-wide values accurately.
- `setup_jenkins_job.py` does not treat “already exists” as success (older script does).

**Missing imports/files/dependencies**
- Dependency missing in current env: `pytest`.
- No Streamlit dashboard implementation file, despite requirement/docs mention.
- No consolidated deployment automation script (build images + load Minikube + apply manifests + verify health) in one place.

---

### 4. Code Summary (major files)

- `src/telemetry/collector.py`  
  Implements Jenkins and Prometheus polling classes plus a `TelemetryCollector` loop writing CSV records. Includes optional Jenkins log capture and basic secret redaction regexes. Uses straightforward polling and coarse error handling.

- `src/telemetry/config.py`  
  Centralized env var loading for telemetry settings (`JENKINS_*`, `PROMETHEUS_URL`, output, interval, log level, `TELEMETRY_LOGS_ENABLED`). Default values are local-dev friendly but can hide misconfiguration.

- `src/telemetry/main.py`  
  CLI wrapper around `TelemetryCollector`. Parses args, logs startup config, then runs continuous collection.

- `src/prediction/data_generator.py`  
  Synthetic CI log + telemetry dataset generator for failure/healthy classes. Defines failure types and returns DataFrame suitable for model training.

- `src/prediction/log_encoder.py`  
  DistilBERT embedding pipeline with mean pooling and PCA reduction. Saves/loads PCA model via joblib.

- `src/prediction/model.py`  
  Simple feed-forward binary classifier (`FailureClassifier`) on combined log+telemetry features.

- `src/prediction/train.py`  
  End-to-end training script: generate synthetic data, encode logs, fit PCA, train classifier, compute F1, save artifacts (`log_pca.joblib`, `failure_predictor.pth`).

- `src/prediction/predictor.py`  
  Runtime predictor utility; converts telemetry to feature vector, concatenates with log embeddings, runs classifier inference, returns failure probability/state.

- `src/rl_agent/simulator.py`  
  Deterministic synthetic MTTR/action simulation tables and state sampling used by RL environment/training.

- `src/rl_agent/env.py`  
  Gymnasium custom env (`NeuroShieldEnv`) with 24D observations and 4 actions; reward blends MTTR improvement, success, and action cost.

- `src/rl_agent/train.py`  
  PPO trainer using Stable-Baselines3, with eval callback and summary output, saving `models/ppo_policy.zip`.

- `src/orchestrator/main.py`  
  Main live loop: polls Jenkins build/logs, computes failure probability, selects action from PPO/pattern heuristics, executes Jenkins retry or kubectl scale/rollback, logs MTTR metrics. Contains key operational logic and most production risk.

- `src/orchestration/main.py`  
  Separate simulation-oriented orchestrator path (`run_once`) combining predictor + policy + simulator for local demonstration. Not the same as live runtime loop.

- `setup_jenkins_job.py`  
  Reads `.env`, generates pipeline XML, retrieves crumb, and creates Jenkins pipeline job via REST API. Uses correct job mode query parameter.

- `scripts/create_jenkins_job.py`  
  Older alternative Jenkins job creation script with similar logic but outdated endpoint behavior (missing `mode`).

- `jenkins-local-updated.yaml`  
  Jenkins namespace/deployment/service manifest with PVC mount and now resource limits.

- `k8s-minimal.yaml`  
  Lightweight all-in-one manifest for namespace, PVC, dummy app, Jenkins, and services for low-resource local testing.

- `dummy-app.yaml`  
  Standalone dummy-app deployment in default namespace using prebuilt `neuroshield-dummy-app:latest`.

- `infra/jenkins-builder/Dockerfile.jenkins`  
  Custom Jenkins image extending `jenkins:lts-jdk17`, installs curl and pinned `kubectl` with checksum verification.

- `infra/dummy-app/app.py` and `infra/dummy-app/Dockerfile`  
  Minimal Flask app with `/` and `/fail` endpoints; containerized to avoid runtime package installation in cluster.

**Hardcoded values/placeholders/TODO-like concerns**
- Hardcoded defaults: `K8S_NAMESPACE=sock-shop`, `AFFECTED_SERVICE=carts`, `JENKINS_JOB=build-pipeline`.
- Hardcoded action parameters: scale replicas fixed to `3`.
- Old docs still indicate “Coming soon” for components that now exist.
- Security-sensitive `.env` token present locally (must not be shared).
- Synthetic constants in simulator/training are fixed and may not represent real workloads.

---

### 5. What’s Needed to Finish

**Critical remaining tasks**
- Unify orchestration architecture:
  - Decide one canonical runtime module (`src/orchestrator` vs `src/orchestration`) and remove/deprecate the other.
- Make end-to-end config coherent:
  - Ensure `.env` job name, namespace, and affected service match actual deployed Jenkins job and K8s objects.
- Stabilize Jenkins job bootstrap:
  - Keep only one script (prefer `setup_jenkins_job.py`) and delete/merge outdated duplicate.
- Add missing dashboard:
  - Implement Streamlit monitoring/override UI promised in PRD/README.
- Improve telemetry persistence:
  - switch CSV init from overwrite to append-or-create mode.
- Add robust test coverage:
  - prediction unit tests
  - RL env tests
  - orchestrator integration tests with mocked Jenkins/K8s.
- Resolve packaging/dev-env reproducibility:
  - install and pin test/developer tooling in active environment
  - provide bootstrap script for local setup (`make`/`powershell`).
- Harden operational behavior:
  - add retries/backoff around Jenkins and kubectl operations
  - add explicit handling for CSRF crumbs and auth edge cases
  - sanitize/guard all external command outputs.
- Validate low-resource profile flow:
  - scripted sequence to build/load images, apply manifests, port-forward Jenkins, and run orchestrator.
- Documentation cleanup:
  - update README to match current implementation and remove stale “Coming soon”.

**Recommended finish order**
1. Config unification + duplicate script cleanup.  
2. End-to-end automation script for local setup/run.  
3. Test suite expansion + CI check.  
4. Dashboard implementation.  
5. Docs polish + final demo workflow.

---

## Quick “Guide Context” Notes

- Current code compiles (`compileall` succeeded).
- Current Python env dependency check passes (`pip check`), but `pytest` is missing in that interpreter.
- There is substantial progress implemented beyond README status text; the main blockers are integration consistency, cleanup, and missing UX/testing layers.
- Before sharing logs/config with others, rotate Jenkins token and redact `.env` content.
