# NeuroShield Production Service Architecture

## Overview

NeuroShield is a standalone DevOps platform that runs as a set of microservices, similar to Jenkins. It provides AI-powered CI/CD monitoring, failure prediction, and automatic healing.

## Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    External Access Layer                     │
│  (All services bound to localhost, reverse proxy needed)    │
└──────────┬──────────────┬───────────────┬───────────────────┘
           │              │               │
    ┌──────▼──────┐ ┌────▼─────┐  ┌─────▼──────┐
    │  Dashboard  │ │   API    │  │  Grafana   │
    │  (Port 8501)│ │(Port 8000)│  │ (Port 3000)│
    └──────┬──────┘ └────┬─────┘  └─────┬──────┘
           │              │               │
           └──────────────┴───────────────┘
                          │
           ┌──────────────▼──────────────┐
           │    Orchestrator Worker      │
           │    (Background Daemon)      │
           │  - Continuous monitoring    │
           │  - Failure prediction       │
           │  - Auto-healing actions     │
           └──────────┬──────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
  ┌─────▼─────┐            ┌───────▼────────┐
  │PostgreSQL │            │     Redis      │
  │(Port 5432)│            │  (Port 6379)   │
  │- Events   │            │  - Cache       │
  │- Actions  │            │  - Pub/Sub     │
  │- Metrics  │            │  - Sessions    │
  └─────┬─────┘            └───────┬────────┘
        │                           │
        └───────────┬───────────────┘
                    │
        ┌───────────▼──────────────────────┐
        │    External Systems Layer        │
        ├──────────────┬───────────────────┤
        │   Jenkins    │   Prometheus      │
        │ (Port 8080)  │   (Port 9090)     │
        │              │                   │
        │ CI/CD Builds │ Metrics Collection│
        └──────────────┴───────────────────┘
```

## Service Components

### 1. API Service (FastAPI)
**Port:** 8000
**Purpose:** RESTful API for external control and monitoring

**Endpoints:**
```
GET  /health                  - Service health check
GET  /api/status              - System status and metrics
GET  /api/events              - Recent detection events
GET  /api/actions             - Recent healing actions
GET  /api/metrics             - Historical metrics
POST /api/trigger             - Manually trigger orchestration
POST /api/demo/inject         - Inject demo failures
GET  /docs                    - OpenAPI documentation
WS   /ws/events               - Real-time event stream
```

**Responsibilities:**
- Expose NeuroShield functionality via REST API
- Serve dashboard static files
- WebSocket real-time updates
- Authentication & authorization
- Request validation

### 2. Worker Service (Orchestrator Daemon)
**Purpose:** Continuous background monitoring and healing

**Components:**
- Telemetry Collector (Jenkins, Prometheus, K8s)
- Failure Predictor (DistilBERT + ML)
- RL Agent (PPO for decision making)
- Action Executor (restart, scale, rollback, etc.)

**Loop:**
```python
while True:
    1. Collect telemetry (Jenkins logs, Prometheus metrics, K8s status)
    2. Build 52D state vector
    3. Predict failure probability (DistilBERT)
    4. Decide action (PPO RL agent + rule overrides)
    5. Execute healing action if needed
    6. Log to PostgreSQL
    7. Publish to Redis pub/sub
    8. Sleep (configurable interval, default 10s)
```

**Runs as:** Background daemon (no HTTP server)

### 3. Dashboard Service (Streamlit)
**Port:** 8501
**Purpose:** Real-time monitoring UI

**Features:**
- Live system metrics (CPU, Memory, Health)
- Event stream with color-coding
- Healing history with decision reasoning
- Demo controls (inject failures)
- Real-time charts (Plotly)

**Data Source:** Queries API service and PostgreSQL

### 4. Database Service (PostgreSQL)
**Port:** 5432 (localhost only)
**Database:** neuroshield_db

**Tables:**
```sql
events          - Detection events (anomalies, alerts)
actions         - Healing actions taken
metrics         - Time-series metrics (snapshot every cycle)
system_state    - Orchestrator state snapshots
cicd_fix_log    - CI/CD auto-fix audit trail
```

**Security:**
- Row-Level Security (RLS) enabled
- Audit logging
- Separate app user (neuroshield_app) with limited permissions

### 5. Cache Service (Redis)
**Port:** 6379 (localhost only)

**Usage:**
- Session storage
- Pub/sub for real-time events
- Cache frequently accessed metrics
- Rate limiting counters
- Temporary state storage

### 6. Monitoring Stack
#### Jenkins (Port 8080)
- CI/CD pipeline execution
- Build job monitoring
- Log collection for failure prediction

#### Prometheus (Port 9090)
- Metrics collection from all services
- Time-series database
- Alert rule evaluation

#### Grafana (Port 3000)
- Metrics visualization
- Custom dashboards
- Alert management

#### AlertManager (Port 9093)
- Alert routing
- Notification management
- Alert grouping/deduplication

## Service Communication

### Inter-Service Communication
```
Dashboard  →  API (HTTP REST)
           →  PostgreSQL (Direct SQL)

API        →  PostgreSQL (SQLAlchemy ORM)
           →  Redis (redis-py)
           →  WebSocket clients

Worker     →  PostgreSQL (Write events/actions)
           →  Redis (Publish events)
           →  Jenkins API (HTTP)
           →  Prometheus API (HTTP)
           →  Kubernetes API (kubectl/k8s client)

All Services → Logs to stdout (Docker captures)
```

### Network
- **Name:** neuroshield-prod
- **Type:** Bridge network
- **Subnet:** 172.22.0.0/16
- All services on same network can resolve by service name

## Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 20GB disk space

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/kovendhan5/NeuroShield.git
cd NeuroShield

# 2. Configure environment
cp .env.example .env
# Edit .env with your values

# 3. Start all services
docker-compose up -d

# 4. Check status
docker-compose ps

# 5. Access services
open http://localhost:8501  # Dashboard
open http://localhost:8000  # API
open http://localhost:3000  # Grafana
```

### Service Dependencies
```
PostgreSQL  ← (required by) → Orchestrator, API
Redis       ← (required by) → Orchestrator, API
Prometheus  ← (optional)   → Orchestrator (telemetry)
Jenkins     ← (optional)   → Orchestrator (CI/CD monitoring)
```

## Configuration

### Environment Variables (.env)
```bash
# Database
DB_ADMIN_PASSWORD=strong-admin-password
DB_USER_PASSWORD=strong-user-password

# Redis
REDIS_PASSWORD=strong-redis-password

# API
API_SECRET_KEY=generate-random-key

# Monitoring
GRAFANA_PASSWORD=strong-grafana-password
GRAFANA_SECRET_KEY=generate-random-key

# Jenkins (optional)
JENKINS_URL=http://jenkins:8080
JENKINS_USERNAME=admin
JENKINS_PASSWORD=your-password

# Prometheus (optional)
PROMETHEUS_URL=http://prometheus:9090

# Kubernetes (optional)
K8S_NAMESPACE=default
```

### Service Configuration (config.yaml)
```yaml
orchestrator:
  check_interval: 10        # seconds between cycles
  action_timeout: 300       # max action duration

detection:
  cpu_threshold: 80         # %
  memory_threshold: 85      # %
  pod_restart_threshold: 3  # count
  error_rate_threshold: 0.3 # 30%

api:
  host: "0.0.0.0"
  port: 8000

database:
  path: "postgresql://neuroshield_app:${DB_USER_PASSWORD}@postgres:5432/neuroshield_db"
```

## Data Persistence

### Volumes
```
neuroshield-postgres-data    - Database files
neuroshield-redis-data       - Redis snapshots
neuroshield-jenkins-data     - Jenkins jobs/configs
neuroshield-prometheus-data  - Metrics time-series
neuroshield-grafana-data     - Dashboards/datasources
```

### Host Mounts
```
./data  → /app/data   - Local data (logs, exports)
./logs  → /app/logs   - Application logs
```

## Observability

### Logging
- **Format:** Structured JSON
- **Destination:** stdout (Docker captures to json-file driver)
- **Retention:** 50MB per service, 5 files rotated
- **Fields:** timestamp, level, component, event, correlation_id

### Metrics
- **Exposed:** `/prometheus_metrics` endpoint on API
- **Collected by:** Prometheus every 15s
- **Visualized in:** Grafana dashboards

### Health Checks
```
API         - GET /health (10s interval)
Orchestrator - GET /health (10s interval)
PostgreSQL  - pg_isready command
Redis       - redis-cli ping
Prometheus  - GET /-/healthy
```

## Security

### Network Security
- All services bound to localhost (127.0.0.1)
- Reverse proxy required for external access
- Internal network isolated (172.22.0.0/16)

### Application Security
- Non-root execution (UID 1000)
- Input validation (Marshmallow schemas)
- JWT authentication on API endpoints
- Rate limiting (100 req/min list, 20 req/min write)
- SQL injection prevention (ORM)

### Database Security
- Separate admin and app users
- Row-Level Security (RLS) enabled
- Audit logging for all mutations
- Connection pooling (2-20 connections)
- Password authentication required

### Secrets Management
- All secrets in .env (not committed)
- Environment variable injection at runtime
- No hardcoded credentials

## Scaling Considerations

### Vertical Scaling (Resource Limits)
```yaml
orchestrator:
  cpu: 1.0 core
  memory: 1GB

api:
  cpu: 0.5 core
  memory: 512MB

postgres:
  cpu: 1.0 core
  memory: 1GB
```

### Horizontal Scaling
- **API:** Can run multiple replicas behind load balancer
- **Worker:** Should run as single instance (stateful)
- **Dashboard:** Can run multiple replicas
- **Database:** Use PostgreSQL replication for HA

## Maintenance

### Backup
```bash
# Database backup
docker exec neuroshield-postgres pg_dump -U postgres neuroshield_db > backup.sql

# Volume backup
docker run --rm -v neuroshield-postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

### Updates
```bash
# Pull latest images
docker-compose pull

# Recreate containers
docker-compose up -d --force-recreate

# Check logs
docker-compose logs -f orchestrator
```

### Monitoring
```bash
# Service status
docker-compose ps

# Logs
docker-compose logs -f [service_name]

# Resource usage
docker stats

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/api/status
```

## Troubleshooting

### Worker not healing failures
1. Check worker logs: `docker-compose logs orchestrator`
2. Verify Jenkins/Prometheus connectivity
3. Check database connection
4. Verify action permissions (K8s, Docker)

### API not responding
1. Check API health: `curl http://localhost:8000/health`
2. Check logs: `docker-compose logs api`
3. Verify database connection
4. Check port conflicts (8000)

### Dashboard not showing data
1. Check API connectivity from dashboard
2. Verify PostgreSQL access
3. Check WebSocket connection
4. Review browser console for errors

## Performance

### Expected Performance
- **Detection Cycle:** ~100ms (collect → detect → analyze)
- **Decision Making:** ~50ms (rule evaluation)
- **Action Execution:** 10-300ms (depends on action)
- **API Response:** <50ms (most endpoints)
- **Dashboard Update:** Real-time via WebSocket
- **Memory Footprint:** ~150MB per service
- **CPU (idle):** <1% per service

### Optimization Tips
1. Increase `check_interval` if monitoring overhead is high
2. Use Redis caching for frequently accessed metrics
3. Tune PostgreSQL connection pool
4. Enable query caching in Prometheus
5. Use Grafana dashboard refresh > 5s

## References

- [User Guide](USER_GUIDE.md) - How to use the system
- [API Documentation](http://localhost:8000/docs) - OpenAPI spec
- [Security Guide](SECURITY.md) - Security best practices
- [Deployment Guide](DEPLOYMENT_STATUS.md) - Production deployment
