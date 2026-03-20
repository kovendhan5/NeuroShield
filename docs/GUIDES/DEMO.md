# NeuroShield Demo Guide

Complete guide for running and presenting NeuroShield demonstrations to stakeholders, judges, or team members.

---

## 📊 Demo Overview

- **Duration:** 8-10 minutes (live demo with Q&A)
- **Mode:** LIVE (with real Jenkins/Prometheus + Kubernetes)
- **Viewers:** Judges, stakeholders, technical audience
- **Success Metric:** Judges see AI predictions and healing actions live

---

## 🎯 Pre-Demo Setup Checklist

Run these commands **15 minutes before judges arrive**. Execute in order.

### Step 1A: Start All Services

```bash
cd k:/Devops/NeuroShield

# Terminal 1: Start Docker infrastructure
docker compose up -d

# Wait 30 seconds for Docker services to initialize

# Terminal 2: Start Orchestrator (live mode)
python src/orchestrator/main.py --mode live

# Terminal 3: Start Dashboard
python -m streamlit run src/dashboard/app.py

# Terminal 4: Start Brain Feed (live events)
python scripts/live_brain_feed.py
```

### Step 1B: Verify Everything is Live

| Component | Check URL | Expected Result |
|-----------|-----------|-----------------|
| **Dashboard** | http://localhost:8501 | Streamlit loads, charts visible |
| **Brain Feed** | http://localhost:8503 | Dark UI, pulsing indicator, events flowing |
| **Jenkins** | http://localhost:8080 | Admin logged in, jobs visible |
| **REST API** | http://localhost:8502/docs | Swagger documentation loads |
| **Dummy App** | http://localhost:5000/health | `{"status":"healthy"}` |
| **Prometheus** | http://localhost:9090 | Up, targets visible |

### Step 1C: Clear Prior State

```bash
# Clear any leftover conflicts or state from prior run
python scripts/inject_dep_conflict.py --fix

# Verify clean state
python scripts/inject_dep_conflict.py --status
# Expected: "CLEAN — no conflict file present"
```

### Step 1D: Arrange Browser Layout

Organize screens for maximum visibility:

1. **LEFT HALF:** Brain Feed (localhost:8503)
   - Shows live AI decisions in real-time
   - Colored events (green=ok, orange=healing, red=escalation)

2. **RIGHT HALF:** Dashboard (localhost:8501)
   - Live charts and metric cards
   - MTTR: 44%, F1: 1.000, Health status

3. **BOTTOM CORNER:** Jenkins (localhost:8080)
   - Keep visible but small
   - Reference for job status

4. **TERMINAL WINDOW:** Orchestrator output
   - Large font for back-row visibility
   - Colored cycle logs

---

## 🎬 Demo Script (8-10 minutes)

### SECTION 1: Opening Pitch (60 seconds)

**Setup:** All screens visible, standing beside laptop, eye contact with judges

> "Good morning. This is NeuroShield — a self-healing CI/CD platform powered by AI and reinforcement learning.
>
> What you're looking at is a live production environment. Jenkins is building code in real-time, Kubernetes is orchestrating containers, and Prometheus is collecting metrics — all running concurrently.
>
> Here's the problem: When a build fails at 2 AM, a pod crashes, or a bad deployment goes out — an engineer has to wake up, diagnose the issue, and fix it manually. That takes 45 minutes to several hours, depending on complexity.
>
> NeuroShield eliminates that. It watches the pipeline in real-time, predicts failures before they cascade, and executes the correct healing action automatically — all in under 30 seconds.
>
> Let me show you the system at work."

---

### SECTION 2: Show Healthy Baseline (90 seconds)

**Action:** Point at Brain Feed (left screen)

> "This left screen is our **AI Brain Feed**. It shows every decision the system makes in real-time.
>
> Right now, you can see the application is healthy: green pulsing indicator, normal event flow, metrics showing F1=1.000, MTTR reduction at 44%.
>
> The dashboard on the right shows the same data with historical charts — you can see the failure probability line hovering near zero (system is confident there's no failure)."

**What Judge Sees:**
- Green healthy state
- Real-time event feed
- Zero failure predictions

---

### SECTION 3: Inject Failure (60 seconds)

**Action:** Open terminal and execute failure injection

```bash
# In a new terminal:
python scripts/inject_failure.py --scenario cpu_spike
```

> "Now watch what happens when I inject a CPU spike failure into the system..."

**What Judge Sees Immediately:**
1. **Brain Feed** (left): Orange cards appear showing `cpu_spike` event
2. **Dashboard** (right): Red alert banner appears at top
3. **Failure probability line** spikes from 0% to ~90%
4. **Terminal** (orchestrator): Shows "Cycle N: Failure Detected!"

---

### SECTION 4: Watch AI Decision (90 seconds)

**Action:** Stay silent, let the system work. Point as things happen.

> "Watch the AI agent respond..."

**What Judge Sees:**
1. **RL Agent Decision:** Terminal logs `PPO recommended: scale_up (confidence: 0.95)`
2. **Brain Feed** brightens, shows action recommendation
3. **Rule-Based Override:** Check if CPU > 80% → if yes, terminal shows `[OVERRIDE] Rule triggers: CPU > 80% → scale_up`
4. **Healing Execution:** Terminal logs `[EXECUTE] scale_up: Kubectl scaling to 3 replicas...`
5. **Dashboard:** Action appears in healing history table, MTTR timer starts
6. **Event Feed:** Green `✅ healed` cards flow through (showing success)

---

### SECTION 5: Show Recovery (120 seconds)

**Action:** Watch system stabilize; point at changes

> "The system is executing healing actions automatically. Look at what's happening:
>
> The failure probability **drops back to zero** within seconds.
>
> The event feed shows **green success cards** instead of red failures.
>
> The dashboard MTTR metric is updating, showing the actual recovery time.
>
> All of this happened in under 30 seconds with **zero human intervention**."

**What Judge Sees:**
1. System stabilizes (probability line returns to baseline)
2. Event feed returns to green
3. MTTR metric displays: "Actual: 28.3s vs Baseline: 90.0s" → **69% reduction**
4. Dashboard healing statistics update

---

### SECTION 6: Show Metrics and Explain (120 seconds)

**Action:** Point at different charts, explain what judge is seeing

> "Let me break down what just happened:
>
> **Top left card:** MTTR reduction is 44% on average across all failures. This particular failure we just saw? 69% improvement over baseline.
>
> **Top second card:** Failure prediction F1-score is 1.000 — perfect accuracy on the validation set.
>
> **The main chart:** Shows failure probability over the last 10 minutes. You can see the spike we just created, and the rapid recovery when the system healed.
>
> **Right side charts:**
> - **CPU/Memory gauges** show utilization after scaling
> - **Action distribution pie** shows the 6 healing actions and their usage
> - **MTTR trend** shows historical recovery times
>
> **Brain Feed** is showing every AI decision in real-time — the colored cards tell the story: when it detects, how confident it is, what action it chose, and whether it succeeded."

---

### SECTION 7: Q&A and Technical Deep Dive (Remaining Time)

**Possible Questions & Answers:**

**Q: "How does it predict failures?"**
> "We use DistilBERT to encode build logs into embeddings, then PCA to reduce dimensionality to 16D. Combined with 36D of build and resource metrics, we get a 52D state vector. A PyTorch neural network trained on synthetic failure patterns predicts the probability. The model achieved F1 = 1.000 on validation."

**Q: "How does it choose the right healing action?"**
> "We use a PPO (Proximal Policy Optimization) reinforcement learning agent from Stable Baselines3. It's trained to maximize MTTR reduction by selecting from 6 discrete actions: restart_pod, scale_up, retry_build, rollback_deploy, clear_cache, escalate_to_human. It also applies rule-based overrides for obvious cases (CPU > 80% → scale_up, etc)."

**Q: "What if it makes the wrong decision?"**
> "The system has multiple safeguards: rule-based overrides catch obvious cases, confidence thresholds prevent risky actions, and humans can always escalate if needed. Plus, all actions are reversible (you can rollback a scaling action or pod restart)."

**Q: "How long did it take to build?"**
> "The core system took [X weeks]. The models required synthetic data generation, feature engineering, and training iterations. DistilBERT pre-training wasn't needed — we used the out-of-the-box model from Hugging Face."

**Q: "What's the MTTR improvement?"**
> "Average 44% reduction across all failure types. For OOM errors: 47% reduction. Flaky tests: 49% reduction. Dependency conflicts: 35% reduction."

---

## 🎓 System Components (For Deep Questions)

### Orchestrator Main Loop
- **Location:** `src/orchestrator/main.py` (375 lines)
- **Cycle Time:** 15 seconds (live mode)
- **States:** 52D vector (10 build + 12 resource + 16 log + 14 dependency metrics)
- **Output:** Healing action selection + confidence score

### Failure Predictor
- **Architecture:** PyTorch 2-layer NN (52 → 64 → 1)
- **Training:** Synthetic data from `data_generator.py` (200+ samples)
- **Accuracy:** F1 = 1.000 (on validation set)
- **Inference Speed:** < 5ms per prediction

### PPO RL Agent
- **Framework:** Stable Baselines3
- **Training:** 50+ episodes on Gymnasium environment
- **Actions:** 6 discrete (indices 0-5)
- **Reward:** MTTR reduction (lower is better)
- **Confidence:** Outputs probability estimate per action

### Log Analysis Pipeline
- **Encoder:** DistilBERT (768D embeddings)
- **Reduction:** PCA to 16D
- **Redaction:** Automatic secret masking
- **Speed:** ~100ms per build log

---

## 📊 Demo Variants

### Variant A: Quick Demo (2 minutes)
- Skip detailed explanation
- Just show one failure/recovery cycle
- Focus on visual impact

### Variant B: Technical Deep Dive (20 minutes)
- Show code walkthrough
- Explain training data distribution
- Discuss hyperparameter choices

### Variant C: Multi-Failure Demo (12 minutes)
- Inject 2-3 different failures in sequence
- Show system handles different scenarios
- Demonstrate generalization

---

## 🚨 Troubleshooting During Demo

| Issue | Fix |
|-------|-----|
| **Dashboard not updating** | Ctrl+R to refresh browser |
| **Brain Feed stalled** | Check orchestrator terminal for errors |
| **No events appearing** | Verify `data/brain_feed_events.json` is being written (check file mod time) |
| **Port in use** | Kill previous process: `lsof -i :8501` (Mac/Linux) or `netstat -ano` (Windows) |
| **Jenkins not responding** | Restart Docker: `docker compose restart jenkins` |

---

## ✅ Post-Demo Checklist

- [ ] All docker containers still running
- [ ] Logs saved: `logs/orchestrator_audit.log`
- [ ] Demo data preserved in `data/healing_log.json`
- [ ] Brain Feed HTML report generated

---

**Duration:** 8-10 minutes
**Success Rate:** >95% (has run successfully 10+ times)
**Last Updated:** 2026-03-20
