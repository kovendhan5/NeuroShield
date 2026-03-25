# NeuroShield Quick Start Guide
## Transform into Production Service - One Command Deployment

### 🎯 What You Get

NeuroShield runs as a complete DevOps platform with:
- **API Service** - REST API for control and monitoring
- **Worker Service** - Background daemon for continuous CI/CD monitoring
- **Dashboard** - Real-time UI showing system status
- **PostgreSQL** - Persistent storage for events and metrics
- **Redis** - Caching and pub/sub
- **Jenkins** - CI/CD integration
- **Prometheus + Grafana** - Monitoring stack

### 🚀 Quick Start (5 Minutes)

#### Prerequisites
- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- 8GB RAM minimum
- 20GB disk space

#### 1. Clone Repository
```bash
git clone https://github.com/kovendhan5/NeuroShield.git
cd NeuroShield
```

#### 2. Configure Environment
```bash
# Create .env from template
cp .env.production .env

# Generate secure passwords
openssl rand -base64 32  # Run this multiple times

# Edit .env and update these values:
# - DB_ADMIN_PASSWORD
# - DB_USER_PASSWORD
# - REDIS_PASSWORD
# - API_SECRET_KEY
# - GRAFANA_PASSWORD
# - GRAFANA_SECRET_KEY
nano .env  # or use your favorite editor
```

#### 3. Start Everything
```bash
# Automated startup (recommended)
./start-production.sh

# OR manual startup
docker-compose -f docker-compose.production.yml up -d
```

#### 4. Access Services
- 📊 **Dashboard**: http://localhost:8501
- 🔌 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- 📈 **Grafana**: http://localhost:3000 (admin / see .env)
- 🔧 **Jenkins**: http://localhost:8080
- 📊 **Prometheus**: http://localhost:9090

### 📋 Service Details

#### Core Services

**API Service (Port 8000)**
- REST API for external control
- WebSocket for real-time updates
- Health checks and status endpoints
- OpenAPI documentation

**Worker Service (Background)**
- Continuous CI/CD monitoring (default: every 10s)
- Failure prediction using ML (DistilBERT)
- Automatic healing actions
- Logs all activities to PostgreSQL

**Dashboard (Port 8501)**
- Real-time system metrics
- Event stream with color coding
- Healing history with reasoning
- Demo controls for testing

#### Infrastructure Services

**PostgreSQL (Port 5432)**
- Stores events, actions, metrics
- Row-level security enabled
- Automatic backups via volumes

**Redis (Port 6379)**
- Caching layer
- Pub/sub for real-time events
- Session storage

**Jenkins (Port 8080)**
- CI/CD pipeline execution
- Build monitoring
- Auto-fix integration

**Prometheus (Port 9090)**
- Metrics collection
- 15 days retention
- Scrapes all services

**Grafana (Port 3000)**
- Metrics visualization
- Pre-configured dashboards
- Alert management

### 🔧 Configuration

#### Environment Variables (.env)

**Required (must change):**
```bash
DB_ADMIN_PASSWORD=your-secure-password-here
DB_USER_PASSWORD=your-secure-password-here
REDIS_PASSWORD=your-secure-password-here
API_SECRET_KEY=your-secret-key-here
GRAFANA_PASSWORD=your-secure-password-here
GRAFANA_SECRET_KEY=your-secret-key-here
```

**Optional (defaults work):**
```bash
ORCHESTRATOR_CHECK_INTERVAL=10  # seconds
PREDICTION_THRESHOLD=0.7        # 0.0-1.0
API_WORKERS=1                   # increase for high load
LOG_LEVEL=info                  # debug, info, warning, error
```

**Jenkins Integration (optional):**
```bash
JENKINS_URL=http://jenkins:8080
JENKINS_USERNAME=admin
JENKINS_PASSWORD=your-jenkins-password
JENKINS_JOB=build-pipeline
```

#### Service Configuration (config.yaml)

The default `config.yaml` works out of the box, but you can customize:

```yaml
orchestrator:
  check_interval: 10        # seconds between monitoring cycles
  action_timeout: 300       # max time for healing action

detection:
  cpu_threshold: 80         # CPU alert threshold (%)
  memory_threshold: 85      # Memory alert threshold (%)
  error_rate_threshold: 0.3 # 30% error rate triggers action
```

### 📊 How It Works

```
┌─────────────────────────────────────────────────────────┐
│  1. Worker collects telemetry every 10s                │
│     - Jenkins build status & logs                       │
│     - Prometheus metrics                                │
│     - Kubernetes pod health                             │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  2. ML predicts failure probability                     │
│     - DistilBERT analyzes logs                          │
│     - Pattern matching for known issues                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  3. RL agent decides best action                        │
│     - restart_pod, scale_up, retry_build, rollback      │
│     - Rule overrides for critical situations            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  4. Execute healing action                              │
│     - Apply fix to infrastructure                       │
│     - Log to PostgreSQL with full audit trail           │
│     - Publish to Redis for real-time updates            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  5. Dashboard shows results                             │
│     - Real-time metrics update                          │
│     - Event appears in stream                           │
│     - Charts update via WebSocket                       │
└─────────────────────────────────────────────────────────┘
```

### 🛠️ Common Operations

#### View Logs
```bash
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service
docker-compose -f docker-compose.production.yml logs -f worker
docker-compose -f docker-compose.production.yml logs -f api

# Last 100 lines
docker-compose -f docker-compose.production.yml logs --tail=100 worker
```

#### Check Service Status
```bash
docker-compose -f docker-compose.production.yml ps
```

#### Restart a Service
```bash
docker-compose -f docker-compose.production.yml restart worker
docker-compose -f docker-compose.production.yml restart api
```

#### Stop All Services
```bash
docker-compose -f docker-compose.production.yml down
```

#### Stop and Remove All Data
```bash
# WARNING: This deletes all data!
docker-compose -f docker-compose.production.yml down -v
```

#### Update to Latest Version
```bash
git pull
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d
```

### 🧪 Testing the System

#### 1. Check Health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/status
```

#### 2. View Recent Events
```bash
curl http://localhost:8000/api/events?limit=10 | jq
```

#### 3. Manually Trigger Orchestration
```bash
curl -X POST http://localhost:8000/api/trigger
```

#### 4. Inject Demo Failure (Testing)
```bash
curl -X POST "http://localhost:8000/api/demo/inject?scenario=pod_crash"
```

#### 5. Check Worker Logs
```bash
docker-compose -f docker-compose.production.yml logs -f worker
```

### 🔍 Troubleshooting

#### Worker Not Running
```bash
# Check logs
docker-compose -f docker-compose.production.yml logs worker

# Check environment
docker-compose -f docker-compose.production.yml exec worker env | grep JENKINS

# Restart worker
docker-compose -f docker-compose.production.yml restart worker
```

#### API Not Responding
```bash
# Check if port is available
netstat -tuln | grep 8000

# Check API health
curl http://localhost:8000/health

# Check API logs
docker-compose -f docker-compose.production.yml logs api
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose -f docker-compose.production.yml ps postgres

# Test connection
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U postgres -d neuroshield_db -c "SELECT 1"

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres
```

#### Dashboard Not Loading
```bash
# Check dashboard logs
docker-compose -f docker-compose.production.yml logs dashboard

# Verify API is accessible from dashboard
docker-compose -f docker-compose.production.yml exec dashboard \
  curl http://api:8000/health
```

### 📈 Performance Tuning

#### Increase API Workers
```bash
# Edit .env
API_WORKERS=4

# Restart API
docker-compose -f docker-compose.production.yml restart api
```

#### Adjust Worker Interval
```bash
# Edit .env (reduce load by checking less frequently)
ORCHESTRATOR_CHECK_INTERVAL=30

# Restart worker
docker-compose -f docker-compose.production.yml restart worker
```

#### Increase Resource Limits
Edit `docker-compose.production.yml`:
```yaml
worker:
  deploy:
    resources:
      limits:
        cpus: '2.0'    # increase from 1.0
        memory: 2G     # increase from 1G
```

### 🔒 Security Checklist

- [ ] Changed all default passwords in .env
- [ ] Used strong random passwords (32+ characters)
- [ ] .env file is not committed to git (.gitignore includes it)
- [ ] All services bound to localhost (127.0.0.1)
- [ ] Reverse proxy configured for external access (if needed)
- [ ] Jenkins configured with authentication
- [ ] Grafana default password changed
- [ ] Database backups configured
- [ ] Firewall rules configured (if exposing externally)

### 📦 Data Backup

#### Manual Backup
```bash
# Database backup
docker exec neuroshield-postgres \
  pg_dump -U postgres neuroshield_db > backup_$(date +%Y%m%d).sql

# Volume backup
docker run --rm \
  -v neuroshield-postgres-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/postgres-backup.tar.gz /data
```

#### Restore from Backup
```bash
# Restore database
cat backup_20260325.sql | docker exec -i neuroshield-postgres \
  psql -U postgres neuroshield_db
```

### 🌐 External Access (Production)

For production deployments accessible from the internet, use a reverse proxy:

#### Nginx Example
```nginx
# /etc/nginx/sites-available/neuroshield
server {
    listen 80;
    server_name neuroshield.yourcompany.com;

    location / {
        proxy_pass http://127.0.0.1:8501;  # Dashboard
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;  # API
    }
}
```

### 📚 Additional Resources

- **Architecture Details**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Documentation**: http://localhost:8000/docs (when running)
- **User Guide**: See [USER_GUIDE.md](USER_GUIDE.md)
- **Security Guide**: See [SECURITY.md](SECURITY.md)

### ✨ Features

✅ **Standalone Service** - Runs like Jenkins, one command to start
✅ **Background Worker** - Continuous monitoring daemon
✅ **REST API** - Full control via HTTP endpoints
✅ **Real-Time Dashboard** - WebSocket-powered live updates
✅ **Persistent Storage** - PostgreSQL + Redis
✅ **Auto-Healing** - ML-powered failure prediction and recovery
✅ **Full Observability** - Prometheus metrics + Grafana dashboards
✅ **Production Ready** - Health checks, logging, monitoring
✅ **Secure** - Non-root execution, localhost-only binding
✅ **Scalable** - Docker Compose for single host, K8s-ready architecture

### 🎓 Next Steps

1. **Configure Jenkins** - Set up your CI/CD jobs at http://localhost:8080
2. **Customize Dashboards** - Add your own Grafana dashboards
3. **Integrate with K8s** - Connect to your Kubernetes cluster
4. **Set Up Alerts** - Configure email/Slack notifications
5. **Train ML Models** - Fine-tune prediction models with your data

### 🆘 Support

- **Issues**: https://github.com/kovendhan5/NeuroShield/issues
- **Documentation**: See `/docs` folder
- **Logs**: `docker-compose logs -f`

---

**Built with** Python | FastAPI | PostgreSQL | Redis | ML/RL | Docker

**License**: MIT
