# NeuroShield — Comprehensive Project Analysis Report

**Generated:** 2026-03-06  
**Analyst:** Automated deep-code review  
**Scope:** Full codebase, infrastructure, ML models, testing, demo fidelity

---

## 1. Architecture Overview

NeuroShield is an **AIOps self-healing CI/CD platform** with five core modules:

```
┌──────────────────────────────────────────────────────────────┐
│  Telemetry Collector  →  Failure Predictor  →  RL Agent      │
│  (Jenkins + Prom +      (DistilBERT +         (PPO, 52D      │
│   K8s polling)           PyTorch, 24D)         → 6 actions)  │
│           │                    │                    │         │
│           ▼                    ▼                    ▼         │
│     Orchestrator ──────────────────────────────────────────── │
│     (main.py: live monitor → predict → decide → heal)        │
│           │                                                   │
│           ▼                                                   │
│     Dashboard (Streamlit + Plotly)   ←  data/*.csv + .json   │
└──────────────────────────────────────────────────────────────┘
     ↕              ↕                ↕
  Jenkins API    kubectl          Prometheus API
  localhost:8080 (Minikube)       localhost:9090
```

**Key design principle:** The orchestrator runs a continuous poll loop (default 15s) that collects telemetry from Jenkins/Prometheus/K8s, feeds it through the ML pipeline, and executes real infrastructure healing actions.

---

## 2. Component Quality Assessment

### 2.1 Telemetry Collector (`src/telemetry/`)
| Aspect | Rating | Notes |
|--------|--------|-------|
| Design | ★★★☆☆ | Polling-based; no event-driven or webhook support |
| Robustness | ★★☆☆☆ | No retry logic in the standalone collector; orchestrator has its own |
| Config | ★★★★☆ | Clean env-var config via `config.py` |

### 2.2 Failure Predictor (`src/prediction/`)
| Aspect | Rating | Notes |
|--------|--------|-------|
| Architecture | ★★★★☆ | DistilBERT → PCA(16D) + Telemetry(8D) → PyTorch classifier — clean pipeline |
| Model Quality | ★★☆☆☆ | Trained on synthetic data only; classifier is a single hidden-layer (64 units) |
| 52D State Builder | ★★★★☆ | Well-documented, clean mapping to RL observation space |
| Inference Speed | ★★★☆☆ | CPU-only DistilBERT tokenization adds ~200ms per call |

### 2.3 RL Agent (`src/rl_agent/`)
| Aspect | Rating | Notes |
|--------|--------|-------|
| Env Design | ★★★★☆ | Gymnasium-compliant, clean reward function (0.6·MTTR + 0.3·efficiency − 0.1·FP) |
| Simulator | ★★★☆☆ | Heuristic-based MTTR simulation; not learned from real incident data |
| Training | ★★★☆☆ | PPO with default hyperparameters; no hyperparameter tuning or ablation |
| Action Space | ★★★★☆ | 6 discrete actions covering key healing scenarios |

### 2.4 Orchestrator (`src/orchestrator/main.py`)
| Aspect | Rating | Notes |
|--------|--------|-------|
| Architecture | ★★★★☆ | Clean live-mode loop: collect → predict → decide → heal → log |
| Healing Actions | ★★★★★ | All 6 actions use REAL kubectl/Jenkins API calls |
| Error Handling | ★★★☆☆ | Basic try/except; no circuit breaker or rate limiting |
| Logging | ★★★★☆ | Dual logging to file + stdout; JSON healing log; CSV telemetry |

### 2.5 Dashboard (`src/dashboard/app.py`)
| Aspect | Rating | Notes |
|--------|--------|-------|
| Visual Quality | ★★★★☆ | Dark theme, Plotly charts, professional CSS |
| Real-Time | ★★★☆☆ | 10s auto-refresh but relies on polling, not WebSocket |
| Live Controls | ★★★★☆ | 4 real buttons (Trigger Build, Crash Pod, Stress Memory, Bad Deploy) |
| Data Binding | ★★★☆☆ | Reads CSV files on every refresh; no caching layer |

---

## 3. What Actually Works (Verified)

| # | Capability | Status | Evidence |
|---|-----------|--------|----------|
| 1 | pip install dependencies | ✅ PASS | psutil, colorama, torch, transformers, stable_baselines3, streamlit all present |
| 2 | Jenkins job creation via API | ✅ PASS | `neuroshield-app-build` freestyle job created, verified via REST API |
| 3 | Docker image build & load | ✅ PASS | Built with `--network host`, loaded into Minikube via `minikube image load` |
| 4 | K8s deployment (kubectl apply) | ✅ PASS | Deployment + Service applied, pod Running |
| 5 | All 6 dummy-app endpoints | ✅ PASS | /health(200), /version(200), /metrics(200), /stress(200), /crash(200→exit), /fail(500) |
| 6 | Scenario 1: Flaky Build → Retry | ✅ PASS | Build triggered, failure detected, retry triggered via Jenkins API |
| 7 | Scenario 2: Pod Crash → Restart | ✅ PASS | /crash killed pod, kubectl detected it, rollout restart succeeded |
| 8 | Scenario 3: Bad Deploy → Rollback | ✅ PASS | APP_VERSION=v2-broken deployed, rollback executed, healthy pod restored |
| 9 | Orchestrator live mode | ✅ PASS | 4 cycles: Jenkins/Prometheus/App all ONLINE, PPO model loaded, actions executed |
| 10 | Health check (20 checks) | ✅ PASS | 20/20 — .env, 3 models, telemetry.csv, 5 sources, 7 imports, 3 services |

---

## 4. What Doesn't Work / Is Fragile

### CRITICAL Issues

1. **Test Suite Broken (7/83 failures)**  
   The orchestrator tests reference old action names (`retry_stage`, `clean_and_rerun`, `regenerate_config`, `reallocate_resources`) that were replaced with real actions (`restart_pod`, `scale_up`, `retry_build`, `rollback_deploy`). The test mocks don't match the new `subprocess.run` + `requests.Session` implementations.

2. **Jenkins CSRF Session Bug (was present, NOW FIXED)**  
   The original `real_demo.py` and `orchestrator/main.py` made separate `requests.get()` (crumb) and `requests.post()` (build trigger) calls without sharing cookies. Modern Jenkins requires the session cookie from the crumb response to accompany the build trigger. Fixed by using `requests.Session()`.

3. **Port-Forward Drops After Pod Restart**  
   `kubectl port-forward` binds to a specific pod, not the Service. When any scenario restarts/crashes a pod, the port-forward silently dies. All subsequent health checks from the orchestrator or demo script fail with "connection refused." There is no automatic reconnection logic anywhere.

### HIGH Issues

4. **Minikube DNS Completely Broken**  
   Docker builds inside Minikube fail with "Temporary failure in name resolution." Even host Docker requires `--network host` to build. This means the standard workflow documented in README (minikube image build) does not work out of the box.

5. **Orchestrator Spams Escalation on Every Cycle**  
   When the last Jenkins build is FAILURE (which persists until a new build succeeds), the orchestrator triggers a healing action on **every single 15s cycle**. There's no "already handled this build number" deduplication. After 10 minutes, you get 40 escalation reports in `data/escalation_reports/`.

6. **RL Agent and Orchestrator Have Mismatched Action Names**  
   - RL env (`env.py`): `retry_stage, clean_and_rerun, regenerate_config, reallocate_resources, trigger_safe_rollback, escalate_to_human`
   - Orchestrator (`main.py`): `restart_pod, scale_up, retry_build, rollback_deploy, clear_cache, escalate_to_human`
   - The PPO policy was trained with env.py's semantics but the orchestrator maps the same action IDs to different real-world commands. For example, action 0 in training was "retry_stage" but in production executes "restart_pod" (a kubectl rollout restart, not a test retry).

7. **Model Trained Entirely on Synthetic Data**  
   `generate_sample()` produces random telemetry + log text from templates. The DistilBERT encoder + PCA + classifier pipeline has never seen a real Jenkins log. The classifier has a single 64-unit hidden layer. Predictions cluster around 0.06 (LOW) for all real inputs because the synthetic distribution has no overlap with actual Jenkins output.

### MODERATE Issues

8. **`datetime.utcnow()` Deprecation Warnings**  
   Python 3.12+ warns on every call. The orchestrator generates 12+ warnings per cycle across 7 callsites. Noisy but not broken.

9. **No Graceful Shutdown / Signal Handling**  
   The orchestrator catches `KeyboardInterrupt` but doesn't flush pending writes or close connections cleanly.

10. **Dashboard's "Run Healing Cycle" Uses Hardcoded Job Name**  
    `run_single_cycle()` had `JENKINS_JOB` default as `build-pipeline` (now fixed to `neuroshield-app-build`).

---

## 5. Code Quality Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Python LOC (src/) | ~2,400 | Appropriate for scope |
| Test count | 83 | Good coverage breadth |
| Test pass rate | 76/83 (91.6%) | 7 failures from orchestrator action rename |
| Type annotations | Partial | Present in predictor/env, sparse in scripts |
| Error handling | Basic | try/except exists but no retry budgets or circuit breakers |
| Documentation | PRD + paper_summary + READMEs | Good high-level docs, sparse inline comments |
| Security | Acceptable | No hardcoded secrets in source (credentials in .env, loaded at runtime) |
| Dependency count | ~15 direct | Heavy ML stack (torch, transformers) for binary classification |

---

## 6. ML Pipeline Honest Assessment

### What's Real
- DistilBERT tokenizer processes actual log text into 768D embeddings
- PCA reduces to 16D (fitted and saved as `log_pca.joblib`)
- PyTorch classifier takes 24D input (16 log + 8 telemetry) → binary probability
- PPO policy takes 52D state → selects from 6 discrete actions
- All models load and produce valid outputs during live operation

### What's Theater
- **Training data is 100% synthetic** — `data_generator.py` creates templated logs like "ERROR: OutOfMemoryError Java heap space" + random telemetry values. Real Jenkins logs look nothing like these templates.
- **The classifier always predicts ~0.06** for real input because the distribution mismatch is total. The `detect_failure_pattern` regex fallback (checking for "flaky", "oom", "timeout" substrings) drives all real decisions.
- **PPO policy and env have different action semantics** than the orchestrator. The policy was trained to minimize simulated MTTR; the orchestrator maps those action IDs to kubectl commands that may not correspond to the trained behavior.
- **No evaluation on real incidents** — no precision/recall/F1 reported, no A/B testing framework, no feedback loop from healing outcomes.

### Honest Prediction Accuracy
For the demo, the ML pipeline is window dressing. The **actual decision logic** is:
1. If Jenkins build status == FAILURE → trigger healing
2. `detect_failure_pattern()` regex picks action based on log keywords
3. PPO action is overridden when pattern match confidence is high (failure_prob > 0.7)

---

## 7. Infrastructure Assessment

### Jenkins
- Freestyle jobs only (no Pipeline plugin available)
- CSRF protection enabled with crumb issuer
- Build has 60% random failure rate (useful for demo, not for real testing)
- Authentication: Basic auth with admin/admin123

### Kubernetes (Minikube)
- Docker driver on Windows
- DNS broken (cannot pull from internet inside Minikube containers)
- Single-node cluster, no resource quotas or network policies
- Image management: local build + `minikube image load` workflow
- No liveness/readiness probes configured on the dummy-app deployment

### Prometheus
- Running at localhost:9090
- Collecting metrics but no custom recording rules or alerts configured
- `kube_pod_info` and `container_cpu_usage_seconds_total` queries work
- No Grafana dashboards connected

---

## 8. Security & Production Readiness

| Area | Status | Risk |
|------|--------|------|
| Credentials | `.env` file, not in git | LOW — acceptable for dev |
| Jenkins auth | Basic auth over HTTP | HIGH for production, OK for local demo |
| K8s RBAC | None (default ServiceAccount) | N/A for Minikube demo |
| Input validation | Minimal | MEDIUM — dummy-app trusts all input |
| Secrets in Docker | None hardcoded | OK |
| HTTP-only (no TLS) | All services | Expected for local demo |
| Escalation reports | Written to local disk | Appropriate for demo |

**Verdict:** Acceptable for a local demo. Not remotely production-ready (no TLS, no RBAC, no secret management, no network policies).

---

## 9. Comparison to Paper/PRD Claims

| PRD Claim | Reality |
|-----------|---------|
| "DistilBERT-based log analysis" | ✅ Real — tokenizer + PCA + classifier loaded and running |
| "PPO RL agent for action selection" | ⚠️ Partially — policy loads and runs, but trained on synthetic data with different action semantics |
| "Real-time telemetry from Jenkins/Prometheus/K8s" | ✅ Real — polling works on live infrastructure |
| "Self-healing actions" | ✅ Real — kubectl and Jenkins API calls execute actual remediation |
| "40-60% MTTR reduction" | ❌ Unverifiable — no baseline measurement infrastructure, no real incident data |
| "Professional dashboard" | ✅ Real — Streamlit dark theme with live charts and control buttons |
| "Closed-loop autonomous healing" | ⚠️ Partial — loop works but prediction model is non-functional on real data |

---

## 10. Priority Improvements

### Priority 1 — Must Fix Before Demo

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| **P1.1** | Fix 7 broken tests (action name mismatch) | Tests prove nothing if they don't pass | 30 min |
| **P1.2** | Add build-number deduplication in orchestrator | Prevents spam of 40 escalation reports per 10 min | 15 min |
| **P1.3** | Fix `datetime.utcnow()` deprecation warnings | 12+ noisy warnings per cycle pollute console output | 10 min |

### Priority 2 — Should Fix for Credibility

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| P2.1 | Add liveness/readiness probes to dummy-app K8s deployment | K8s can't auto-detect crashes without probes | 20 min |
| P2.2 | Auto-reconnect port-forward or use NodePort/`minikube service` | Demo breaks silently after any pod restart | 30 min |
| P2.3 | Align RL env action names with orchestrator action names | Professor will notice the mismatch | 20 min |

### Priority 3 — Nice to Have

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| P3.1 | Add a few real Jenkins log samples to training data | Reduces "always predicts 0.06" problem | 2-4 hrs |
| P3.2 | Add integration test that runs scenario 1-3 end-to-end | Prevents future regressions | 2 hrs |
| P3.3 | Connect Prometheus alerting → orchestrator webhook | Moves from polling to event-driven | 4+ hrs |

---

*End of Report*
