# NeuroShield Full Mode - Quick Start Guide

## System Status ✅
**All 6 services are running and healthy**

```
docker-compose ps
```

## Access Services

### 1. Jenkins CI/CD
- **URL**: http://localhost:8080
- **Username**: admin
- **Password**: admin123
- **Purpose**: Build pipeline configuration and execution

### 2. Prometheus
- **URL**: http://localhost:9090
- **Purpose**: Metrics collection and time-series database
- **Sample Query**: `up` or `container_memory_usage_bytes`

### 3. Grafana
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin
- **Purpose**: Dashboards and visualization
- **Setup**: Add Prometheus datasource at http://prometheus:9090

### 4. Streamlit Dashboard
- **URL**: http://localhost:8501
- **Purpose**: Real-time system monitoring
- **Features**: Healing actions, metrics, alerts

### 5. ML Orchestrator API
- **URL**: http://localhost:8502
- **Health Check**: http://localhost:8502/health
- **Purpose**: DistilBERT + PPO RL agent for intelligent healing

### 6. Dummy App (Test Target)
- **URL**: http://localhost:5000/health
- **Purpose**: Application target for NeuroShield to manage
- **Create Failure**: http://localhost:5000/create-failure

---

## Useful Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f orchestrator
docker-compose logs -f dashboard
docker-compose logs -f jenkins
```

### Manage Services
```bash
# Stop all
docker-compose down

# Start all
docker-compose up -d

# Restart one
docker-compose restart orchestrator

# Rebuild and restart
docker-compose build orchestrator --no-cache && docker-compose up -d
```

### System Info
```bash
# Disk usage
docker system df

# Clean up (safe)
docker system prune

# Clean everything (CAREFUL!)
docker system prune -a --volumes
```

---

## Troubleshooting

### Service won't start
```bash
docker-compose logs [service] | tail -50
```

### Out of memory
```bash
# Reduce WSL allocation in ~/.wslconfig
memory=6GB  # from 8GB
```

### Port already in use
```bash
# Check what's using port 8080
lsof -i :8080

# Kill the process
kill -9 [PID]
```

### Docker won't start
```bash
# Restart Docker daemon
systemctl restart docker

# Or on Windows, restart Docker Desktop
taskkill /F /IM "Docker Desktop.exe"
# Wait 10 seconds
"C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

---

## Expected Behavior

1. **Initial Startup**: Services take 30-60 seconds to become healthy
2. **Memory Usage**: ~3-4 GB after stabilization
3. **CPU Usage**: 5-15% at idle
4. **First Build**: ~10-15 minutes (downloads base images)
5. **Subsequent Starts**: ~45 seconds

---

## Next Steps

1. Configure Jenkins job at http://localhost:8080
2. Add Jenkins webhooks to your Git repo
3. Monitor via Streamlit dashboard
4. Create Grafana dashboards for visualization
5. Trigger test failures to see orchestrator in action

---

Generated: 2026-03-22
