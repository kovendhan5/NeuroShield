# NeuroShield - How to Run the Orchestrator

This guide explains how to start and run the **NeuroShield orchestrator** — the core AI system that detects failures and executes healing actions.

## What is the Orchestrator?

The orchestrator is the main system that:
1. **Monitors** your Jenkins CI/CD pipeline and Kubernetes cluster
2. **Predicts** failures before they cause problems
3. **Decides** which healing action to take (restart pod, scale up, etc.)
4. **Executes** the healing action automatically
5. **Tracks** recovery metrics (MTTR, success rate, etc.)

## Prerequisites

Before starting, you need:
- ✅ **kubectl** configured and accessing your Kubernetes cluster
- ✅ **Jenkins** running and accessible (usually localhost:8080)
- ✅ **Prometheus** running (usually localhost:9090)
- ✅ **Python 3.13** with dependencies installed
- ✅ **.env file** configured with service URLs

### Install Dependencies

```bash
# Install Python packages if not already done
pip install -r requirements.txt

# Verify kubectl can access cluster
kubectl cluster-info

# Verify Jenkins is accessible
curl http://localhost:8080/api/json

# Verify Prometheus is accessible
curl http://localhost:9090/-/healthy
```

## Option 1: Quick Start (Recommended)

Run the automated environment check and launch orchestrator:

```bash
bash scripts/launcher/quick_start.sh
```

This script will:
1. ✓ Check that Python dependencies are installed
2. ✓ Verify kubectl can access your cluster
3. ✓ Verify Jenkins is online and responding
4. ✓ Verify Prometheus is online
5. ✓ Verify the dummy-app is reachable
6. ✓ If all checks pass, start the orchestrator

### Output Example
```
==================================================
NeuroShield Quick Start - Environment Check
==================================================

Prerequisites:
Checking python ... ✓ Available
Checking kubectl ... ✓ Available
Checking docker ... ✓ Available

Services:
Checking Jenkins ... ✓ Online
Checking Prometheus ... ✓ Online
Checking Dummy App ... ✓ Online

Kubernetes:
Checking kubectl cluster access ... ✓ Connected
Checking namespace 'default' ... ✓ Exists

NeuroShield:
Checking Python dependencies ... ✓ All packages installed

==================================================
All checks passed! Starting orchestrator...

==========================================
NeuroShield Orchestrator
==========================================
Starting orchestrator with config:
  Jenkins:     http://localhost:8080
  Prometheus:  http://localhost:9090
  Dummy App:   http://localhost:5000
  Interval:    15s
  Namespace:   default

Logs: logs/orchestrator.log
Healing history: data/healing_log.json
MTTR metrics: data/mttr_log.csv

Press Ctrl+C to stop
==========================================

[CYCLE 1] ← Orchestrator is now running!
```

## Option 2: Manual Start

Start the orchestrator directly with custom settings:

```bash
# Simple start (uses defaults from .env)
bash scripts/launcher/run_orchestrator.sh

# Or run directly with Python
python -m src.orchestrator.main
```

## Option 3: Docker Container

Run the orchestrator in a Docker container:

```bash
# Build the orchestrator image
docker-compose -f docker-compose-orchestrator.yml build

# Start the orchestrator container
docker-compose -f docker-compose-orchestrator.yml up -d

# View logs in real-time
docker-compose -f docker-compose-orchestrator.yml logs -f neuroshield-orchestrator

# Stop when done
docker-compose -f docker-compose-orchestrator.yml down
```

## Monitoring the Orchestrator

While the orchestrator is running, you can monitor it in multiple ways:

### View Live Logs
```bash
# Real-time log stream
tail -f logs/orchestrator.log

# Or use docker logs if running in container
docker-compose -f docker-compose-orchestrator.yml logs -f
```

### Check Healing History
The orchestrator writes every action to `data/healing_log.json`:

```bash
# View all healing actions (JSON lines format)
cat data/healing_log.json

# View last 10 actions
tail -10 data/healing_log.json

# Parse and display prettily
python -c "
import json
with open('data/healing_log.json') as f:
    for i, line in enumerate(f):
        action = json.loads(line)
        print(f'{i+1}. {action[\"action_name\"]}: {\"SUCCESS\" if action[\"success\"] else \"FAILED\"} ({action[\"duration_ms\"]}ms)')
        if i >= 9: break
"
```

### Check MTTR Metrics
```bash
# View Mean Time To Recovery metrics
cat data/mttr_log.csv

# Analysis: Show average MTTR by action type
tail -20 data/mttr_log.csv | cut -d, -f3,4,6 | column -t -s,
```

### View Detailed Decision Log
The orchestrator records why it chose each action:

```bash
# View recent decisions with reasons
tail -5 logs/orchestrator_audit.log

# Filter for failures only
grep "FAILED" logs/orchestrator_audit.log
```

## Configuration

The orchest rator reads configuration from `.env` file and environment variables:

### Essential Configuration

```bash
# Jenkins integration
export JENKINS_URL=http://localhost:8080
export JENKINS_USERNAME=admin
export JENKINS_PASSWORD=your_password_or_token
export JENKINS_JOB=neuroshield-app-build

# Prometheus integration
export PROMETHEUS_URL=http://localhost:9090

# Kubernetes integration
export K8S_NAMESPACE=default
export AFFECTED_SERVICE=dummy-app

# App monitoring
export DUMMY_APP_URL=http://localhost:5000

# Monitoring interval (seconds)
export POLL_INTERVAL=15
```

### Create .env File

```bash
# Copy template and edit
cp .env.example .env

# Edit with your values
nano .env
```

## Validate Integration

Before running orchestrator long-term, validate that each component is working:

### Test Jenkins Integration
```bash
python scripts/test/jenkins_integration_test.py
```

Output should show:
- ✓ Jenkins Connection
- ✓ Get Latest Build
- ✓ Get Build Log
- ✓ Build Trigger Capable

### Test Prometheus Integration
```bash
python scripts/test/prometheus_integration_test.py
```

Output should show:
- ✓ Prometheus API Connection
- ✓ Check Active Targets
- ✓ Key Metrics Available
- ✓ Range Queries Working

### Test Kubernetes Integration
```bash
python scripts/test/k8s_integration_test.py
```

Output should show:
- ✓ Kubectl Connection
- ✓ Namespace Exists
- ✓ Get Deployments
- ✓ Pod Operations

### Run Real End-to-End Demo
```bash
# This shows orchestrator actually detecting and healing a real failure
python scripts/demo/e2e_real_system.py
```

This will:
1. Create a baseline of system health
2. Inject a real failure (pod crash)
3. Wait for orchestrator to detect it
4. Verify healing action was executed
5. Confirm recovery

## What the Orchestrator Does

### Main Loop (every 15 seconds by default)

1. **Collect Telemetry**
   - Latest Jenkins build status
   - CPU/memory from Prometheus
   - Pod health from Kubernetes
   - App health from /health endpoint

2. **Predict Failure**
   - DistilBERT encodes recent build logs
   - ML classifier predicts failure probability
   - Returns 0-100% risk level

3. **Choose Action**
   - PPO RL agent recommends action (1-5 options)
   - Rule engine may override (e.g., if app is DOWN, always restart_pod)
   - Action chosen with explanation

4. **Execute Action**
   - If probability > threshold, execute healing:
     - **restart_pod**: Kill and restart failed pod
     - **scale_up**: Add more replicas for load
     - **retry_build**: Re-run failed build job
     - **rollback_deploy**: Undo recent deployment
     - **escalate_to_human**: Alert team + create incident

5. **Log & Measure**
   - Record action to data/healing_log.json
   - Measure actual MTTR vs baseline
   - Write metrics to data/mttr_log.csv

## Understanding the Output

### Cycle Header
```
[CYCLE 123] Telemetry Cycle @ 2024-03-23 14:32:45
```
Shows the main loop iteration number and timestamp.

### Service Status
```
Jenkins:     ✓ Online (45ms)
Prometheus:  ✓ Online (32ms)
Dummy App:   ✓ Online (18ms)
```
Health of each service being monitored.

### Telemetry
```
Build #456: SUCCESS (65000ms)
CPU: 45.3%  |  Memory: 62.1%
Pods: 3  |  Error Rate: 0.0023
App Health: 98% | Response: 125ms
```
Current system metrics collected from all sources.

### Prediction & Decision
```
[PREDICTION] Failure probability: 12.5%
[THRESHOLD] 12.5% < 50% → No action needed (healthy)
```
ML prediction and whether action was taken.

### Healing Action (if triggered)
```
[ACTION] restart_pod -- dummy-app in default
[ACTION] Waiting for pod readiness (may take 30-60s)...
[ACTION] Pod restarted successfully (45s)
```
The healing action that was executed and its result.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for common issues:
- Jenkins not responding
- Prometheus metrics empty
- Kubernetes commands failing
- Python dependency errors
- ML models not loading

## Performance Expectations

- **Detection Time**: <30s (next monitoring cycle)
- **Decision Time**: <2s (ML pipeline)
- **Action Time**: 5-90s (depends on action type)
- **Total MTTR**: 15-120s vs 300-3600s manual

## Production Deployment

For production use on Azure:
- See: [docs/PRODUCTION_DEPLOYMENT.md](../PRODUCTION_DEPLOYMENT.md)
- Run in AKS with persistent volumes
- Use Terraform for IaC
- Enable alerting and dashboards

## Next Steps

1. **Start orchestrator**: Run `bash scripts/launcher/quick_start.sh`
2. **Monitor execution**: Watch `tail -f logs/orchestrator.log`
3. **Validate recovery**: Check `data/healing_log.json` for actions
4. **Run demo scenario**: Execute `python scripts/demo/e2e_real_system.py`
5. **Configure alerts**: See [docs/GUIDES/SETUP.md](./SETUP.md)

---

**Questions?** See [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md) or check the orchestrator source at `src/orchestrator/main.py`
