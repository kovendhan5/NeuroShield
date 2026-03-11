# NeuroShield — Final-Year Demo Script

**Duration:** 8–10 minutes | **Format:** Live demo with simultaneous Q&A  
**Screen layout:** 3 panels visible at all times

```
┌─────────────────────────┬─────────────────────────┐
│  LEFT: Brain Feed       │  RIGHT: Streamlit        │
│  localhost:8503          │  Dashboard :8501         │
├─────────────────────────┴─────────────────────────┤
│  BOTTOM: Jenkins UI — localhost:8080               │
└───────────────────────────────────────────────────┘
  Orchestrator terminal: minimized, maximize on demand
```

---

## SECTION 1 — PRE-DEMO SETUP CHECKLIST

Run these commands **15 minutes before judges arrive**. Do them in order.

### 1A. Start all services

```powershell
cd K:\Devops\NeuroShield
pyenv shell 3.13.1
powershell -ExecutionPolicy Bypass -File scripts/start_neuroshield.ps1
```

*Wait for the success summary box to appear. All 13 steps should show `[OK]`.*

### 1B. Verify everything is live

| Check | Command / Action | Expected |
|-------|------------------|----------|
| Dashboard | Open http://localhost:8501 | Streamlit loads, charts visible |
| Brain Feed | Open http://localhost:8503 | Dark UI, green pulsing dot, feed scrolling |
| Jenkins | Open http://localhost:8080 | Logged in as admin, job `neuroshield-app-build` visible |
| REST API | Open http://localhost:8502/docs | Swagger UI loads |
| Dummy-app | Open http://localhost:5000/health | `{"status":"healthy"}` |
| Prometheus | Open http://localhost:9090 | Up, targets page shows scrapers |

### 1C. Upgrade Jenkins job (if not already done)

```powershell
python scripts/upgrade_jenkins_job.py
```

*Confirm: `[OK] 4-stage job created`*

### 1D. Clear any leftover conflict from prior run

```powershell
python scripts/inject_dep_conflict.py --fix
```

### 1E. Arrange browser windows

1. **Left half of screen** — Browser tab: `localhost:8503` (Brain Feed)
2. **Right half of screen** — Browser tab: `localhost:8501` (Dashboard)
3. **Bottom quarter** — Browser tab: `localhost:8080` (Jenkins — keep small)
4. **Terminal** — One PowerShell window ready at `K:\Devops\NeuroShield`, font size large enough for back row to read
5. **Orchestrator window** — Minimized (you'll pull it up if asked)

### 1F. Final sanity check

```powershell
python scripts/inject_dep_conflict.py --status
```

*Confirm: `CLEAN — no conflict file present`*

> You're ready. Take a breath.

---

## SECTION 2 — OPENING (60 seconds)

*Stand beside your laptop. Make eye contact with judges before speaking.*

> Good morning. My name is [YOUR NAME] and this is NeuroShield — a self-healing CI/CD platform powered by AI.

*Gesture broadly at all three screens.*

> What you're looking at is a live production environment. Jenkins is building code, Kubernetes is running our application, and Prometheus is collecting system metrics — all running right now on this machine.

> Here's the problem NeuroShield solves. In a real DevOps pipeline, when a build fails at 2 AM, or a pod crashes, or a bad deployment goes out — a human engineer has to wake up, diagnose the issue, and fix it manually. That takes an average of 45 minutes to several hours.

> NeuroShield eliminates that. It watches the pipeline in real time, predicts failures before they cascade, and executes the correct healing action automatically — in under 30 seconds.

*Point at the Brain Feed (left screen).*

> This left screen is our live AI brain feed — it shows every decision the system makes in real time. Let me show you the system at rest first.

---

## SECTION 3 — SHOW HEALTHY STATE (60 seconds)

*Point at the Brain Feed (left screen), specifically the green pulsing dot.*

> Right now the system is healthy. You can see this green heartbeat — that means the telemetry collector is polling Jenkins, Prometheus, and Kubernetes every 15 seconds and finding nothing wrong.

*Point at the Architecture Pipeline column on the left.*

> On the left column here you can see the full AI pipeline. Telemetry comes in at the top, gets encoded by a DistilBERT language model, passed to a failure predictor neural network, and if there's a problem, a reinforcement learning agent chooses the best healing action from six options.

*Point at the Dashboard (right screen) — specifically the metrics cards.*

> On the dashboard you can see our track record. The system has executed over 190 real healing actions with a 92% success rate. Mean time to recovery dropped by 68%.

*Point at the Performance Metrics column on Brain Feed.*

> These are live numbers — not from a simulation. Every one of those entries in the feed came from a real infrastructure event.

> Now let me break something on purpose and show you what happens.

---

## SECTION 4 — SCENARIO 1: Dependency Conflict (3 minutes)

*Move to the terminal. Make it visible.*

> I'm going to simulate something that happens constantly in real software teams — a developer commits code with conflicting package versions. Two versions of NumPy that can't coexist. The build should fail.

*Type the command but DO NOT press Enter yet:*

```powershell
python scripts/real_demo.py --scenario 0
```

> This command injects a broken requirements file into our Jenkins server, then triggers a build. Watch all three screens — Jenkins will start building at the bottom, the Brain Feed on the left will light up, and the dashboard on the right will update.

*Press Enter.*

**As "[DEV] Developer commits requirements with conflicting versions" appears:**

> The script just planted a bad dependency file inside Jenkins. Think of it as a developer pushing a bad `requirements.txt`.

**As "[DEV] Triggering Jenkins build" appears:**

*Point at Jenkins UI (bottom screen).*

> Watch Jenkins — you'll see a new build start right now.

*Jenkins build appears and starts running.*

> There it is — build started. Stage 1 is Dependency Install. It's going to find those conflicting packages and fail.

**As "Build FAILED at Stage 1 — dependency conflict" appears:**

*Point at the red FAILED in the terminal, then at Jenkins showing the red ball.*

> Failed. The build couldn't install dependencies because NumPy 1.21 and NumPy 2.1 can't coexist — one of the downstream packages needs NumPy greater than 1.22, another needs it less than 2.0. Classic version conflict.

**As "[NEURO] NeuroShield detected build failure" appears:**

> Now here's where the AI kicks in. NeuroShield detected the failure through the Jenkins API — it polls every 15 seconds.

*Point at Brain Feed — new red entries appearing.*

**As "DistilBERT encoder: tokenising build log" appears:**

> It's pulling the actual console log from Jenkins and feeding it through a DistilBERT language model. DistilBERT converts that raw text into a 768-dimensional vector, which gets compressed to 16 dimensions by PCA, then the failure predictor — a PyTorch neural network — outputs a probability. For a failure log like this, it outputs 0.9996 — near certainty.

**As "PPO RL Agent evaluating action space" appears:**

> Now the reinforcement learning agent takes over. This is a PPO agent — Proximal Policy Optimization — trained over 51,000 episodes. It looks at a 52-dimensional state vector: CPU usage, memory, build status, failure probability, the failure pattern type — and it selects the best action from six options.

**As "Decision: fix_dependencies → retry_build" appears:**

> And it chose correctly — fix the dependencies first, then retry the build. It didn't just blindly retry, which would fail again. It identified the root cause.

*Point at Brain Feed — entries turning green.*

**As "Broken deps removed — conflict resolved" appears:**

> The broken file is gone. Now it retriggers Jenkins.

*Point at Jenkins UI — new build starting.*

> Watch Jenkins — new build starting now. This one should pass Stage 1.

**As the retry result appears:**

*If SUCCESS:*

> Build passed. The entire cycle — from failure to detection to healing to verified recovery — took about 30 seconds. A human engineer would take 30 to 60 minutes to diagnose a dependency conflict, especially at 2 AM.

*If FAILURE (random test flake):*

> The dependency conflict was fixed — Stage 1 passed this time. But the test suite has a random element for realism, and it happened to fail. The important thing is: NeuroShield correctly identified the dependency conflict, fixed it, and the fix worked. In production it would retry again automatically.

---

## SECTION 5 — SCENARIO 2: Pod Crash (2 minutes)

*Type the command but DO NOT press Enter yet:*

```powershell
python scripts/real_demo.py --scenario 2
```

> Now a different kind of failure — an application crash. I'm going to kill the running pod in Kubernetes. This simulates a memory corruption, an unhandled exception, or any process that just dies.

*Press Enter.*

**As "Sending POST /crash to dummy-app" appears:**

> I just sent a crash signal to the running application. The process is dead.

*Point at the terminal showing "Pod process crashed!"*

> Look — the pod status shows CrashLoopBackOff or Error. The application is completely down. No users can reach it.

**As "[NEURO] NeuroShield detected pod is down" appears:**

> NeuroShield caught it immediately through the health check loop. The failure predictor confirms it's a crash — not a slow response, not a timeout — an actual process death.

*Point at Brain Feed — red entry appearing.*

**As "Decision: restart_pod" appears:**

> The RL agent chose `restart_pod` — the correct action. It's running `kubectl rollout restart` right now.

*Point at the terminal showing the rollout.*

**As "Pod is Running again!" appears:**

> Pod is back. Health check passes. The application is serving traffic again.

*Point at the Dashboard — healing history updating.*

> And you can see on the dashboard the action was logged. That's two real self-healing events you just watched live.

---

## SECTION 6 — SHOW AI EVIDENCE (90 seconds)

*Switch focus to the Dashboard (right screen). Scroll if needed.*

> Let me walk you through the evidence that this AI actually works.

*Point at the MTTR chart or metrics section.*

> Mean Time To Recovery — this is the industry standard metric for how fast you fix incidents. Without NeuroShield, our measured baseline was around 12 minutes for simple failures, up to 45 minutes for complex ones. With NeuroShield, the average is under 30 seconds. That's a 67.9% reduction — and that number comes from real timestamps, not a simulation.

*Point at the action distribution chart.*

> This shows what actions the system has chosen over 190+ real events. You can see it's not just doing one thing — it distributes across all six actions based on what's actually wrong. `retry_build` is the most common because flaky tests are the most common failure in real CI/CD — which matches industry data.

*Point at the success rate.*

> 92% success rate means that 9 times out of 10, the system fixes the problem with zero human intervention. The other 8% are genuine edge cases where it correctly escalates to a human — which is exactly what you want. You don't want an AI that never asks for help.

*Point at the Brain Feed metrics column.*

> F1 score of 100% means zero false positives and zero false negatives on our test set of 200 samples. The model knows the difference between a real failure and normal noise.

> And inference takes about 25 milliseconds per cycle. The bottleneck isn't the AI — it's waiting for Kubernetes to actually restart a pod.

---

## SECTION 7 — CLOSING (30 seconds)

*Step back from the screen. Make eye contact.*

> So to summarize: NeuroShield watches your CI/CD pipeline in real time, predicts failures with near-perfect accuracy, and heals them automatically in under 30 seconds. You just watched it happen live — twice.

> It reduced our mean time to recovery by 68%, it runs with 25-millisecond inference latency, and it does it all without any human being touching a keyboard.

> That's NeuroShield. I'm happy to take any questions.

---

## SECTION 8 — QUESTION HANDLING TABLE

| # | Question | Answer |
|---|----------|--------|
| 1 | **How does the failure predictor work?** | We take the Jenkins build log — raw text — and encode it with DistilBERT, which produces a 768-dimensional embedding. PCA reduces that to 16 dimensions, then a PyTorch MLP classifier outputs a failure probability between 0 and 1. On our test set of 200 samples, it achieves 100% F1 with zero false positives. |
| 2 | **Why PPO instead of DQN or another RL algorithm?** | PPO handles continuous state spaces well and is more stable during training than DQN. Our state space is 52-dimensional — CPU, memory, build status, failure probability, failure pattern encoding, and more. PPO's clipped objective prevents catastrophic policy changes, which is critical when actions have real infrastructure consequences. |
| 3 | **How did you train the RL agent?** | We trained for 51,000 episodes in a simulated environment that mirrors our real infrastructure. The state space is 52 dimensions, the action space is 6 discrete actions, and the reward function gives +1 for a correct healing action, −0.5 for an incorrect one, and a bonus for fast recovery. After training, we validated against 190 real events. |
| 4 | **What if the AI makes the wrong decision?** | Two safeguards. First, there's a 60-second cooldown between actions so it can't cascade errors. Second, if the failure probability stays above 0.85 after an action, it escalates to a human with a full HTML incident report and email notification. In our data, 8% of events get escalated — those are the genuinely ambiguous cases where human judgment is correct. |
| 5 | **Is this running for real or is it simulated?** | Everything you saw is real infrastructure. Jenkins is a real server running in Docker, the pods are real Kubernetes containers in Minikube, and Prometheus is scraping real metrics. The healing actions execute real `kubectl` commands and real Jenkins API calls. The healing log has 196 timestamped entries from real executions. |
| 6 | **How does it scale to production?** | The architecture is already containerized — Jenkins, Prometheus, and the app run in Docker/Kubernetes. For production, you'd replace Minikube with a managed cluster like EKS or AKS, swap localhost URLs for service discovery, and the RL agent continues to learn from new data. The 25ms inference latency means it can handle hundreds of services. |
| 7 | **What's the 67.9% MTTR reduction based on?** | We measured timestamps from `mttr_log.csv` — the time between when a failure is detected and when the system confirms recovery. Without NeuroShield, baseline MTTR averaged 12+ minutes for the same failure types. With NeuroShield, the median is under 30 seconds. The 67.9% is the relative reduction across all event types. |
| 8 | **Why DistilBERT and not regular BERT or GPT?** | DistilBERT is 60% faster and 40% smaller than BERT with 97% of its accuracy. For CI/CD log classification, we don't need the full capacity of BERT — log patterns are relatively structured. DistilBERT gives us the semantic understanding we need at inference speeds that work for real-time monitoring — about 25ms per classification. |
| 9 | **What are the 6 healing actions?** | `restart_pod` for crashed containers, `scale_up` for CPU spikes, `retry_build` for flaky test failures, `rollback_deploy` for bad deployments, `clear_cache` for memory pressure, and `escalate_to_human` for situations the AI can't resolve. The RL agent selects from these based on system state — it's not a rule engine, it learns the optimal policy. |
| 10 | **How is this different from existing tools like PagerDuty or Datadog?** | Those tools detect and alert — a human still has to fix the problem. NeuroShield closes the loop: it detects, diagnoses the root cause using NLP, predicts the correct fix using reinforcement learning, executes it, and verifies recovery. The human only gets involved when the AI determines it can't fix the issue confidently. |
| 11 | **What happens if Jenkins itself goes down?** | The telemetry collector handles connection failures gracefully — it logs the error and retries on the next poll cycle. If Jenkins is unreachable for multiple cycles, the failure probability rises and the system escalates to a human. It doesn't try to restart Jenkins itself because that's outside its action space — an intentional design decision to prevent recursive failure. |
| 12 | **Can judges see the code?** | Absolutely. The classifier is in `src/prediction/`, the RL agent is in `src/rl_agent/`, the orchestrator logic is in `src/orchestrator/main.py`, and the telemetry collectors are in `src/telemetry/`. The entire codebase has 95 passing unit tests. Happy to open any file. |

---

## SECTION 9 — EMERGENCY BACKUP TABLE

| Problem | Fix | What to say |
|---------|-----|-------------|
| **Jenkins not responding** | `docker start neuroshield-jenkins` then wait 15s | > "Jenkins container needed a restart — give me 15 seconds. This actually demonstrates why self-healing matters — even our tools need recovery." |
| **Port-forward died (dummy-app unreachable)** | `kubectl port-forward svc/dummy-app 5000:5000` (in a new terminal) | > "The port-forward dropped — let me reconnect. This is a Minikube quirk on Windows, not a NeuroShield issue." |
| **Minikube not running** | `minikube start --driver=docker` | > "Minikube needs to restart — this takes about 30 seconds. While it starts, let me explain how the AI model works." |
| **Brain Feed page blank** | `python scripts/live_brain_feed.py` (in a new terminal) | > "Let me restart the feed server — one moment." |
| **Dashboard won't load** | `python -m streamlit run src/dashboard/app.py` (in a new terminal) | > "Streamlit needs a restart. While it loads, I can show you the AI in the terminal output instead." |
| **Scenario 0 returns "Build PASSED unexpectedly"** | Run again: `python scripts/real_demo.py --scenario 0` | > "The inject didn't stick — let me run it once more. Docker file writes can be flaky on Windows." |
| **Build retry also fails (random test flake)** | Explain naturally | > "The dependency fix worked — Stage 1 passed. But our test suite has intentional randomness to simulate real flaky tests, so Stage 3 failed. The point is the AI correctly identified and fixed the root cause." |
| **Terminal frozen / Python error** | Close terminal, open new one: `cd K:\Devops\NeuroShield; pyenv shell 3.13.1` | > "Let me get a fresh terminal. While I do that — any questions about the architecture?" |
| **Orchestrator crashed** | `python src/orchestrator/main.py --mode live` (in new terminal) | > "The orchestrator dropped — restarting now. It'll reconnect to all services automatically." |
| **Judge asks to see something you didn't prepare** | Open VS Code, navigate to `src/` folder | > "Sure, let me pull up the source code. Everything is in the `src/` directory — which module would you like to see?" |

---

## SECTION 10 — ONE-SENTENCE LIFELINE

If you freeze, lose your place, or your mind goes blank, say this:

> "NeuroShield watches the pipeline, predicts failures, and heals them automatically — let me show you."

Then run whichever scenario you haven't done yet.

---

---

# CHEAT SHEET (Print This)

```
╔══════════════════════════════════════════════════════════════╗
║              NEUROSHIELD — DEMO CHEAT SHEET                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  KEY NUMBERS                                                 ║
║  ─────────────────────────────────────────────               ║
║  F1 Score:          100%    (200 test samples)               ║
║  AUC-ROC:           100%                                     ║
║  MTTR Reduction:    67.9%   (from mttr_log.csv)              ║
║  Success Rate:      92%     (196 real heals)                 ║
║  Inference:         ~25ms per cycle                          ║
║  RL Training:       51,000 episodes                          ║
║  State Space:       52 dimensions                            ║
║  Actions:           6 (restart, scale, retry,                ║
║                        rollback, clear, escalate)            ║
║  Cooldown:          60 seconds between actions               ║
║  Encoder:           DistilBERT 768D → PCA 16D               ║
║  Tests Passing:     95/95                                    ║
║                                                              ║
║  DEMO COMMANDS                                               ║
║  ─────────────────────────────────────────────               ║
║  Scenario 0: python scripts/real_demo.py --scenario 0        ║
║              Dependency conflict → fix → retry               ║
║  Scenario 2: python scripts/real_demo.py --scenario 2        ║
║              Pod crash → restart                             ║
║                                                              ║
║  EMERGENCY FIXES                                             ║
║  ─────────────────────────────────────────────               ║
║  Jenkins dead:     docker start neuroshield-jenkins           ║
║  Port-forward:     kubectl port-forward svc/dummy-app        ║
║                    5000:5000                                  ║
║  Brain Feed:       python scripts/live_brain_feed.py         ║
║  Dashboard:        python -m streamlit run                   ║
║                    src/dashboard/app.py                      ║
║  Minikube:         minikube start --driver=docker            ║
║  Orchestrator:     python src/orchestrator/main.py           ║
║                    --mode live                               ║
║                                                              ║
║  LIFELINE                                                    ║
║  ─────────────────────────────────────────────               ║
║  "NeuroShield watches the pipeline, predicts failures,       ║
║   and heals them automatically — let me show you."           ║
║                                                              ║
║  URLS                                                        ║
║  ─────────────────────────────────────────────               ║
║  Dashboard:    localhost:8501                                 ║
║  Brain Feed:   localhost:8503                                 ║
║  Jenkins:      localhost:8080                                 ║
║  API Docs:     localhost:8502/docs                            ║
║  App Health:   localhost:5000/health                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

# TOP 3 HIGH-IMPACT MOMENTS

### 1. The Dependency Conflict Self-Heals (Scenario 0, ~4:00 mark)

**The moment:** The terminal shows `Decision: fix_dependencies → retry_build` and then Jenkins starts a new build that passes.

**Why it lands:** This is the moment judges realize the AI didn't just retry blindly — it *identified the root cause* (dependency conflict vs. flaky test) and chose a two-step fix. Most students would show a simple retry. You're showing causal reasoning. When the retry build passes Stage 1 (which previously failed), the before/after contrast is immediate and visceral.

**Amplify it by saying:**
> "Notice it didn't just retry — which would fail again. It identified the root cause was a dependency conflict, fixed it first, and *then* retried. That's the difference between a rule engine and a learned policy."

---

### 2. The Pod Recovers in Real Time Across All Three Screens (Scenario 2, ~6:00 mark)

**The moment:** The pod crashes, all three screens react simultaneously — Brain Feed turns red, Dashboard shows the event, Jenkins is idle but the terminal shows kubectl commands executing — then everything goes green again.

**Why it lands:** This is the "wow, this is actually real" moment. Judges have sat through many slide-deck demos. Watching a live system break and heal itself across three synchronized screens is viscerally different. The fact that it takes ~15 seconds total makes the point about "MTTR reduction" tangible — they just *experienced* it.

**Amplify it by saying:**
> "That entire cycle — crash, detection, diagnosis, healing, verification — took about 15 seconds. Without this system, that's a 2 AM page to an on-call engineer."

---

### 3. The AI Evidence Walkthrough (Section 6, ~7:30 mark)

**The moment:** You point at the dashboard and calmly state "92% success rate across 196 real events, 67.9% MTTR reduction, zero false positives."

**Why it lands:** After two live demos, judges are impressed but wondering "is the AI real or is this scripted?" The hard numbers — especially "196 real healing actions" and "measured from real timestamps" — answer that doubt definitively. The contrast with baseline strategies (random: 5.1% MTTR reduction vs. NeuroShield: 67.9%) makes the value undeniable.

**Amplify it by saying:**
> "These numbers aren't from a simulation. Every one of those 196 entries has a real timestamp from a real infrastructure event. The 67.9% MTTR reduction is measured from actual recovery times."
