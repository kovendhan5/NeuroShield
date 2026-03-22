# NeuroShield Production-Grade Kubernetes Setup

**Status: PHASE 3/7 COMPLETE - MONITORING STACK OPERATIONAL** ✅

---

## 📊 WHAT HAS BEEN DEPLOYED

### PHASE 1: Kubernetes Infrastructure ✅
- Minikube cluster (5GB RAM, 4 CPU cores, Docker driver)
- Namespaces: neuroshield-prod (isolated environment)
- Storage: fast-ssd StorageClass with persistent volumes
- ServiceAccounts & RBAC configured

### PHASE 2: Microservices (3 services, 7 replicas total) ✅
**API Service** (3 replicas)
- Python Flask-based REST API
- Prometheus metrics exposed on port 9090
- Health checks & liveness probes configured
- Connected to PostgreSQL + Redis
- Auto-restart on failure

**Web Service** (2 replicas)
- Frontend dashboard (Python Flask)
- Calls API service internally
- Port 80 publicly exposed
- Load balanced across 2 replicas

**Worker Service** (2 replicas)
- Background job processor
- Simulated production workload
- Failure injection (10% random failures)
- Prometheus metrics on port 9091

### PHASE 3: Database & Cache ✅
**PostgreSQL**
- Stateful deployment with persistent storage (5GB)
- Replication-ready architecture
- Health checks: ready/liveness probes

**Redis**
- In-memory cache layer (2GB)
- LRU eviction policy
- Used by all microservices

### PHASE 4: Monitoring Stack ✅
**Prometheus**
- Scrapes metrics from all services every 15 seconds
- Alert rules configured (PodDown, HighMemory, HighErrorRate)
- 30-day retention, 10GB storage

**Grafana**
- Pre-configured Prometheus datasource
- Admin: admin/admin
- Ready for custom dashboards

**Alertmanager**
- Alert deduplication & routing
- Webhook integration ready
- Slack notifications configured

---

## 🚀 HOW TO ACCESS SERVICES

Run the port-forwarding script:
```bash
bash scripts/port-forward-k8s.sh
```

Then access:
- Web Dashboard: http://localhost:8080
- API Service: http://localhost:5000/health
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Alertmanager: http://localhost:9093

---

## 📈 MONITORING COMMANDS

```bash
# Check pod status
kubectl get pods -n neuroshield-prod -w

# View logs
kubectl logs deployment/api-service -n neuroshield-prod -f

# Check metrics
kubectl top pods -n neuroshield-prod

# Scale services
kubectl scale deployment api-service --replicas=5 -n neuroshield-prod
```

---

## 🔥 TESTING FAILURE SCENARIOS

Kill a pod and watch it auto-restart:
```bash
kubectl delete pod -l app=api -n neuroshield-prod
kubectl get pods -n neuroshield-prod -w
```

---

## ⚠️ REMAINING PHASES

**PHASE 4:** Jenkins Real CI/CD Pipeline
**PHASE 5:** Chaos Engineering (inject failures)
**PHASE 6:** NeuroShield Orchestrator (auto-healing)
**PHASE 7:** Testing & Optimization

---

Generated: 2026-03-22 | Production-Grade AIOps System
