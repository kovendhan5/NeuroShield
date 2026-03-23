# NeuroShield - Troubleshooting Guide

Common issues and how to fix them.

## Quick Diagnosis

Run these commands to identify the problem:

```bash
# Check if orchestrator is running
ps aux | grep python | grep orchestrator

# Check logs for errors
tail -50 logs/orchestrator.log

# Validate all services
python scripts/test/jenkins_integration_test.py
python scripts/test/prometheus_integration_test.py
python scripts/test/k8s_integration_test.py

# Check environment
cat .env | grep JENKINS_URL
cat .env | grep PROMETHEUS_URL
cat .env | grep K8S_NAMESPACE
```

---

## Issue: "Connection refused" / Services Offline

### Jenkins Not Responding

**Symptom:**
```
✗ Jenkins Offline
Error: Connection refused @ http://localhost:8080
```

**Fix:**

1. **Check if Jenkins is running**
   ```bash
   # Docker container
   docker ps | grep jenkins

   # Or access the web UI
   curl http://localhost:8080/api/json
   ```

2. **If not running, start it**
   ```bash
   # Docker
   docker-compose up -d jenkins

   # Or standalone (if installed)
   java -jar jenkins.war
   ```

3. **If running but unreachable**
   - Check firewall: `netstat -an | grep 8080`
   - Verify port: Jenkins usually runs on 8080, check with: `docker port <container_name>`
   - Check for network issues: `ping localhost`

4. **If using different URL**
   - Update .env: `JENKINS_URL=http://your-jenkins-host:port`
   - Restart orchestrator

### Prometheus Not Responding

**Symptom:**
```
✗ Prometheus Offline
Error: Connection refused @ http://localhost:9090
```

**Fix:**

1. **Check if Prometheus is running**
   ```bash
   docker ps | grep prometheus
   curl http://localhost:9090/-/healthy
   ```

2. **Start Prometheus**
   ```bash
   docker-compose up -d prometheus
   ```

3. **Check Prometheus targets**
   ```bash
   # Access: http://localhost:9090/targets
   # All targets should be "UP" (not "DOWN")
   # If targets are DOWN, check their scrape configs
   ```

4. **Update URL in .env if needed**
   ```bash
   PROMETHEUS_URL=http://your-prometheus-host:9090
   ```

### Dummy App Not Responding

**Symptom:**
```
✗ Dummy App Offline
Error: Connection refused @ http://localhost:5000
```

**Fix:**

1. **Check if dummy-app is deployed in Kubernetes**
   ```bash
   kubectl get pods -n default | grep dummy-app
   kubectl get svc -n default | grep dummy-app
   ```

2. **If pod is not running, deploy it**
   ```bash
   # Create deployment
   kubectl create deployment dummy-app \
     --image=your-registry/dummy-app:latest \
     -n default

   # Expose service
   kubectl expose deployment dummy-app \
     --type=NodePort \
     --port=5000 \
     -n default
   ```

3. **If pod is running but not accessible**
   ```bash
   # Check pod logs
   kubectl logs -f deployment/dummy-app -n default

   # Port-forward to access
   kubectl port-forward svc/dummy-app 5000:5000 -n default
   ```

4. **Update URL if needed**
   ```bash
   DUMMY_APP_URL=http://your-app-host:5000
   ```

---

## Issue: Python Import Errors

### ModuleNotFoundError: No module named 'src'

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
Traceback in orchestrator/main.py line 27
```

**Fix:**

1. **Ensure pytest.ini exists**
   ```bash
   # Check if it exists
   ls -la pytest.ini

   # Create if missing
   echo "[pytest]
   pythonpath = ." > pytest.ini
   ```

2. **Verify Python path**
   ```bash
   # Check where orchestrator is running from
   cd k:\Devops\NeuroShield  # Correct location
   python -m src.orchestrator.main
   ```

3. **Install package in development mode**
   ```bash
   pip install -e .
   ```

### Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'torch'
or:
ModuleNotFoundError: No module named 'transformers'
```

**Fix:**

1. **Install all requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **If installation fails, try with retries**
   ```bash
   pip install --retries 10 --default-timeout=2000 -r requirements.txt
   ```

3. **Check what's missing**
   ```bash
   python -c "import torch; import transformers; print('OK')"
   ```

4. **Install specific packages**
   ```bash
   pip install torch transformers scikit-learn stable-baselines3
   ```

---

## Issue: Kubernetes Commands Failing

### "kubectl: command not found"

**Symptom:**
```
bash: kubectl: command not found
```

**Fix:**

1. **Install kubectl**
   ```bash
   # MacOS
   brew install kubectl

   # Windows
   # Download from: https://kubernetes.io/docs/tasks/tools/
   # Or using Chocolatey:
   choco install kubernetes-cli

   # Linux
   sudo apt-get install kubectl
   ```

2. **Verify installation**
   ```bash
   kubectl version --client
   ```

### "error: no context has been set"

**Symptom:**
```
error: no context has been set. Set the default context with 'kubectl config set-context'.
```

**Fix:**

1. **Check kubeconfig**
   ```bash
   kubectl config view
   ```

2. **Set default context**
   ```bash
   # List available contexts
   kubectl config get-contexts

   # Set default
   kubectl config use-context <context-name>
   ```

3. **Or set kubeconfig explicitly**
   ```bash
   export KUBECONFIG=/path/to/.kube/config
   ```

### "namespace not found"

**Symptom:**
```
Error from server (NotFound): namespaces "default" not found
```

**Fix:**

1. **List available namespaces**
   ```bash
   kubectl get namespaces
   ```

2. **Use an available namespace**
   ```bash
   # Update .env
   echo "K8S_NAMESPACE=kube-system" >> .env

   # Or create the namespace
   kubectl create namespace default
   ```

### "pods not found"

**Symptom:**
```
Error from server (NotFound): pods not found
```

**Fix:**

1. **Check pods exist**
   ```bash
   kubectl get pods -n default
   ```

2. **Deploy the dummy-app if not there**
   ```bash
   kubectl create deployment dummy-app \
     --image=nginx:latest
   ```

3. **Wait for pod to be ready**
   ```bash
   kubectl wait --for=condition=Ready \
     pod -l app=dummy-app \
     -n default \
     --timeout=300s
   ```

---

## Issue: Metrics Not Available

### "Prometheus returning zeros"

**Symptom:**
```
CPU: 0.0%  |  Memory: 0.0%
Pods: 0  |  Error Rate: 0.0
```

**Causes:**

1. **Minikube node-exporter not running**
   ```bash
   # Check if metrics are available
   kubectl get nodes
   kubectl describe node minikube | grep -i "memory\|cpu"

   # Enable metrics for Minikube
   minikube addons enable metrics-server
   minikube addons enable heapster  # (deprecated but works)
   ```

2. **Prometheus targets down**
   - Access: http://localhost:9090/targets
   - Look for RED "DOWN" status
   - Check scrape config: http://localhost:9090/graph
   - Try manual query: `node_cpu_seconds_total`

3. **Metrics not yet collected**
   - Wait 1-2 minutes for Prometheus to scrape
   - Check scrape interval in prometheus.yml (usually 15s)

**Fix:**

```bash
# Add fallback to Minikube queries
kubectl top nodes
kubectl top pods -n default

# Or use psutil fallback in orchestrator
# (Already implemented, will auto-use if Prometheus returns 0)
```

### "No matching metrics"

**Symptom:**
```
Error executing query: no data matching the query
```

**Fix:**

1. **Check what metrics exist**
   ```bash
   # Query available metrics
   curl 'http://localhost:9090/api/v1/label/__name__/values' | grep node
   ```

2. **Try different metric names**
   ```bash
   # For CPU:
   node_cpu_seconds_total
   process_cpu_seconds_total
   container_cpu_usage_seconds_total

   # For Memory:
   node_memory_MemAvailable_bytes
   node_memory_MemTotal_bytes
   container_memory_usage_bytes
   ```

3. **Check node-exporter is running**
   ```bash
   # In Minikube, node-exporter is typically required
   kubectl get pods -A | grep exporter
   # or
   kubectl get pods -A | grep prometheus
   ```

---

## Issue: Orchestrator Crashes/Stops

### "Orchestrator suddenly stopped"

**Check logs for errors:**
   ```bash
   tail -100 logs/orchestrator.log | grep -i "error\|exception\|traceback"
   ```

### "FATAL: Could not load ML models"

**Symptom:**
```
ERROR: Could not load DistilBERT predictor
...
FATAL: ML models missing
Exiting orchestrator
```

**Fix:**

1. **Check if models directory exists**
   ```bash
   ls -la models/
   ```

2. **Download models**
   ```bash
   python scripts/download_models.py
   # or
   python -c "from src.prediction.predictor import FailurePredictor; p = FailurePredictor()"
   ```

3. **Verify model files**
   ```bash
   ls -la models/
   # Should contain: pytorch_model.bin, config.json, etc.
   ```

### "Orchestrator throwing assertion errors"

**Common cause: State vector dimension mismatch**

```python
AssertionError: Expected 52D state, got 48D
```

**Fix:**

Check `build_52d_state()` function in `src/prediction/predictor.py` — ensure it's building exactly 52 dimensions.

### "Out of memory"

**Symptom:**
```
MemoryError: Unable to allocate 2.50 GiB for an array
```

**Fix:**

1. **Reduce model batch size in orchestrator/main.py**
   ```python
   # Change from default 1024 to 256
   predictor = FailurePredictor(batch_size=256)
   ```

2. **Run on machine with more RAM**
   - NeuroShield requires ~4GB RAM for ML models
   - Check available RAM: `free -h` (Linux) or `Get-WmiObject -Class Win32_ComputerSystem` (Windows)

3. **Disable PyTorch cache**
   ```bash
   export PYTORCH_CACHE_DIR=/tmp/torch_cache
   ```

---

## Issue: Healing Actions Not Executing

### "No action taken even though failure detected"

**Check prediction probability:**

```bash
# In orchestrator logs, look for:
[PREDICTION] Failure probability: X%

# Actions only trigger if X% >= threshold (default 50%)
# And cooldown period has passed (60 seconds between actions)
```

**If probability too low:**

1. **Check if ML models are trained**
   - Untrained models default to ~50% random
   - Need to train on real build logs first

2. **Check build logs are being parsed**
   ```bash
   grep "build.*log" logs/orchestrator.log
   ```

**If probability high but no action:**

1. **Check cooldown**
   ```bash
   grep "COOLDOWN\|cooldown" logs/orchestrator.log
   # Actions need 60 seconds gap
   ```

2. **Check health is passing**
   ```bash
   grep "app health\|APP.*HEALTH" logs/orchestrator.log -i
   # If app health is >90%, system may not trigger healing for minor issues
   ```

### "Action started but didn't complete"

**Check orchestrator logs for:***

```bash
grep "ACTION.*FAILED\|timeout" logs/orchestrator.log -i
```

**Fix based on action type:**

**restart_pod timed out:**
```bash
# Pod may be stuck. Try manual restart:
kubectl delete pod -l app=dummy-app -n default --wait=false
kubectl wait --for=condition=Ready pod -l app=dummy-app -n default --timeout=60s
```

**scale_up timed out:**
```bash
# Scaling may be slow. Increase timeout in orchestrator/main.py:
# Change wait deadline from 60 to 120 seconds
```

**retry_build failed:**
```bash
# Check Jenkins job exists and is responsive
curl http://localhost:8080/job/neuroshield-app-build/api/json
```

**rollback_deploy failed:**
```bash
# Check rollout history exists
kubectl rollout history deployment/dummy-app -n default
```

---

## Issue: Data Not Being Logged

### "healing_log.json is empty"

**Check if orchestrator is running:**
```bash
ps aux | grep orchestrator
tail logs/orchestrator.log | head -20
```

### "MTTR metrics not updating"

**Check if actions are creating entries:**
```bash
tail data/healing_log.json
# Should show JSON lines with "action_name", "success", "duration_ms"
```

**If empty:**
1. No healing actions have been triggered
2. Run the e2e demo to trigger one:
   ```bash
   python scripts/demo/e2e_real_system.py
   ```

### "Telemetry CSV not updating"

**Check data collection:**
```bash
tail data/telemetry.csv
wc -l data/telemetry.csv
```

**Should show one row per monitoring cycle (~every 15s for ~100 rows/25min)**

If not updating:
1. Check Prometheus is returning data: `python scripts/test/prometheus_integration_test.py`
2. Check Jenkins is responding: `python scripts/test/jenkins_integration_test.py`

---

## Issue: Performance Problems

### "Orchestrator cycles taking too long"

**Symptom:**
```
[CYCLE 1] ... completed 89234ms
[CYCLE 2] ... completed 85123ms
# Should be <30 seconds
```

**Causes:**

1. **ML prediction slow**
   - DistilBERT tokenization is slow on large logs
   - Reduce log size: truncate build logs to <5000 chars

2. **Prometheus queries slow**
   - Range queries (history) can be slow
   - Use shorter time ranges (5m instead of 24h)

3. **Kubernetes operations slow**
   - kubectl calls have 30s timeout
   - Pod restarts may take 20-40s

**Fix:**

```python
# In orchestrator/main.py:

# Reduce log truncation
build_log_truncate = 2000  # was 5000

# Use shorter Prometheus ranges
prom_query_range = "5m"  # was "1h"

# Increase timeouts for slow environments
KUBECTL_TIMEOUT = 60  # was 30
```

---

## Advanced Debugging

### Enable Debug Logging

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Or in .env
echo "LOG_LEVEL=DEBUG" >> .env

# Restart orchestrator
bash scripts/launcher/run_orchestrator.sh
```

### Trace a Single Cycle

```bash
# Run orchestrator with verbose output
python -m src.orchestrator.main --verbose

# Or run Python debugger
python -m pdb -m src.orchestrator.main
# Then use: n (next), s (step), p variable (print)
```

### Check System Resources

```bash
# CPU and memory usage
top -b -n 1 | head -20

# Process memory
ps aux --sort=-%mem | grep python | head -5

# Disk space
df -h

# GPU if available
nvidia-smi
```

---

## Getting Help

1. **Check recent logs**
   ```bash
   tail -500 logs/orchestrator.log > debug.log
   cat debug.log | grep -i error
   ```

2. **Run integration tests**
   ```bash
   python scripts/test/jenkins_integration_test.py
   python scripts/test/prometheus_integration_test.py
   python scripts/test/k8s_integration_test.py
   ```

3. **Run end-to-end demo**
   ```bash
   python scripts/demo/e2e_real_system.py
   ```

4. **Check configuration**
   ```bash
   echo "=== .env Configuration ===" && cat .env | grep -v "^#"
   echo "=== Python Version ===" && python --version
   echo "=== kubectl ===" && kubectl version --client
   echo "=== Available Models ===" && ls -la models/ 2>/dev/null || echo "No models/"
   ```

5. **Create issue with full context**
   ```bash
   # Collect debugging info
   mkdir support
   cp logs/orchestrator.log support/
   cp logs/orchestrator_audit.log support/
   python scripts/test/*.py > support/integration_tests.txt 2>&1
   tar czf neuroshield_debug.tar.gz support/

   # Attach neuroshield_debug.tar.gz to GitHub issue
   ```

---

## Performance Baseline

For reference, a healthy system should show:

```
- Cycle duration:           10-30 seconds
- Prediction latency:       <2 seconds
- Prometheus query time:    <5 seconds
- Kubernetes operations:    5-60 seconds (depends on action)
- Memory usage:             2-4 GB
- CPU usage:                20-60% (during healing actions)
- Detection to action:      15-45 seconds (one full cycle)
- Action to recovery:       2-90 seconds (depends on action type)
```

If your numbers are significantly higher, see the corresponding troubleshooting section above.
