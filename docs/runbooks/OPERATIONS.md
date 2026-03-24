# NeuroShield Operations Runbook

**Target Audience:** DevOps Engineers, SREs
**Frequency:** Consult during incidents
**Last Updated:** 2026-03-24

---

## Table of Contents

1. [Daily Health Checks](#daily-health-checks)
2. [Common Incidents & Resolution](#common-incidents--resolution)
3. [Emergency Procedures](#emergency-procedures)
4. [Backup & Recovery](#backup--recovery)
5. [Scaling & Performance](#scaling--performance)
6. [Security Incidents](#security-incidents)

---

## Daily Health Checks

### 08:00 - Morning Standup Checklist

```bash
# 1. Verify all services running
docker-compose ps | grep "Up"
# Expected: 9/9 services showing "Up"

# 2. Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length'
# Expected: 7 active targets

# 3. Recent healing actions
tail -20 data/healing_log.json | grep -c "action"
# Expected: >3 actions per hour during business hours

# 4. Database size (warn if >5GB)
docker-compose exec postgres du -sh /var/lib/postgresql/data
# Expected: <5GB

# 5. Disk usage
docker system df
# Expected: <80% of available space

# 6. Check for errors in past hour
docker-compose logs --since=1h | grep -i "error" | wc -l
# Expected: <10 errors (not including recoverable transients)
```

### Alert Dashboard

Access Grafana: http://localhost:3000
- CPU usage trending >80%?
- Memory usage trending >85%?
- Disk usage trending >90%?
- Pod restart rate normal?

---

## Common Incidents & Resolution

### INCIDENT 1: Orchestrator Not Making Decisions

**Symptoms:**
- No healing_log entries for >10 minutes
- Orchestrator container running but CPU at 0%

**Root Causes & Fixes:**

```bash
# 1. Check orchestrator logs
docker-compose logs orchestrator | tail -50

# 2. Most common: TensorFlow/PyTorch model loading stalled
#    FIX: Restart orchestrator
docker-compose restart orchestrator
sleep 30

# 3. If still stuck, check disk space (model cache)
docker system df
# If >90% disk used:
docker system prune -a  # WARNING: deletes all dangling images

# 4. If still stuck, check database connection
docker-compose exec orchestrator python -c "
from src.database import Session
session = Session()
result = session.execute('SELECT 1')
print('✓ DB connected')
"

# 5. If DB connection fails: restart postgres
docker-compose restart postgres
docker-compose restart orchestrator
```

---

### INCIDENT 2: High Error Rate in Jenkins Integration

**Symptoms:**
- Prometheus shows intel_jenkins_api_errors_total increasing
- No builds getting detected

**Diagnosis:**

```bash
# 1. Is Jenkins running?
curl -v http://localhost:8080/

# 2. Check Jenkins credentials
docker-compose logs orchestrator | grep -i "jenkins\|auth\|401"

# 3. Test Jenkins API directly
JENKINS_URL="http://jenkins:8080"
curl -u admin:${JENKINS_PASSWORD} \
  ${JENKINS_URL}/api/json | head -20
```

**Fixes:**

```bash
# If Jenkins is down:
docker-compose restart jenkins
# Allow 60 seconds for start

# If auth fails:
# 1. Get new API token: http://localhost:8080/user/admin/configure
# 2. Update .env with new token
# 3. Restart orchestrator:
docker-compose restart orchestrator

# If Jenkins is very slow:
# Likely GC pause, restart with more memory
# Edit docker-compose-hardened.yml:
# JAVA_OPTS: "-Xmx1g -Xms512m"
# Then: docker-compose up -d jenkins
```

---

### INCIDENT 3: Database Errors

**Symptoms:**
- "connection refused" errors
- Queries timing out

**Diagnostic:**

```bash
# 1. Is postgres running?
docker-compose ps postgres

# 2. Check postgres logs
docker-compose logs postgres | tail -30

# 3. Test connection
docker-compose exec postgres psql -U postgres -d neuroshield_db -c "SELECT 1"

# 4. Check active connections (max is 100)
docker-compose exec postgres psql -U postgres -d neuroshield_db -c \
  "SELECT count(*) FROM pg_stat_activity"
```

**Fixes:**

```bash
# If too many connections:
# Option 1: Graceful restart
docker-compose stop microservice orchestrator
sleep 5
docker-compose start microservice orchestrator

# Option 2: Hard restart if unresponsive
docker-compose restart postgres
sleep 30
docker-compose restart microservice orchestrator

# If schema corrupted (after crash):
docker-compose exec postgres psql -U postgres -d neuroshield_db \
  -f /docker-entrypoint-initdb.d/init.sql
```

---

### INCIDENT 4: High Memory Usage

**Symptoms:**
- Docker warning: "Memory limit exceeded"
- OOMKilled containers

**Diagnosis:**

```bash
# 1. Which container is leaking?
docker stats --no-stream | sort -k4 -h

# 2. Memory usage over time
docker-compose logs orchestrator | grep -i "memory\|mb"

# 3. Check PyTorch cache
du -sh /tmp/torch*
```

**Fixes:**

```bash
# Immediate:
docker-compose restart <container>

# If TensorFlow caching issue:
docker-compose exec orchestrator rm -rf /tmp/torch*

# Increase limits in docker-compose-hardened.yml:
# deploy:
#   resources:
#     limits:
#       memory: 2G  # was 1G

# Then: docker-compose up -d
```

---

## Emergency Procedures

### FULL SYSTEM RESTART

Use when multiple services are failing:

```bash
# Step 1: Graceful shutdown (with draining)
docker-compose stop

# Step 2: Wait for connections to close
sleep 10

# Step 3: Full start
docker-compose up -d

# Step 4: Monitor startup
docker-compose logs -f

# Step 5: Verify after 5 minutes
curl http://localhost:8000/health/ready
```

### CRITICAL DATA LOSS RECOVERY

**If database gets corrupted:**

```bash
# 1. Create backup of current data
docker-compose exec postgres pg_dump -U neuroshield_app neuroshield_db \
  > backup_corrupted_$(date +%s).sql

# 2. Drop corrupted database
docker-compose exec postgres psql -U postgres -c "DROP DATABASE neuroshield_db"

# 3. Restore from backup
# Assuming you have daily backup in backups/
docker-compose exec postgres psql -U postgres \
  < backups/neuroshield_db_YYYYMMDD.sql

# 4. Restart services
docker-compose restart microservice orchestrator
```

---

## Backup & Recovery

### Daily Automated Backup

Enable by adding to crontab:

```bash
# In /etc/crontab (or crontab -e)
# Runs at 2 AM daily
0 2 * * * root /opt/neuroshield/scripts/backup.sh >> /var/log/neuroshield-backup.log 2>&1
```

### Manual Backup

```bash
# Backup database
docker-compose exec postgres pg_dump -U neuroshield_app neuroshield_db > \
  backups/db_$(date +%Y%m%d_%H%M%S).sql

# Backup data directory
tar -czf backups/data_$(date +%Y%m%d_%H%M%S).tar.gz data/

# Upload to S3
aws s3 cp backups/ s3://neuroshield-backups/ --recursive
```

### Restore from Backup

```bash
# Restore database
docker-compose stop microservice orchestrator
docker-compose exec postgres psql -U postgres \
  < backups/db_20260324_020000.sql
docker-compose start microservice orchestrator

# Restore data files
tar -xzf backups/data_20260324_020000.tar.gz
```

---

## Scaling & Performance

### CPU at 90%+ for >10 min?

```bash
# 1. Identify bottleneck
# Top consumers:
docker stats --no-stream

# 2. If orchestrator CPU-bound:
#    - Reduce ML model update frequency
#    - Edit orchestrator/main.py UPDATE_INTERVAL

# 3. If Jenkins CPU-bound:
#    - Reduce reachable pipeline complexity
#    - Archive old jobs

# 4. Scale by adding more replicas (K8s only)
```

### Response Time >500ms?

```bash
# 1. Check prometheus query latency
time curl http://prometheus:9090/api/v1/targets

# 2. Check database query time
docker-compose exec postgres psql -U postgres -d neuroshield_db -c \
  "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 5"

# 3. Add database indexes
docker-compose exec postgres psql -U postgres -d neuroshield_db -c \
  "CREATE INDEX idx_healing_timestamp ON healing_log(timestamp DESC)"
```

---

## Security Incidents

### INCIDENT: Suspicious Activity Detected

**If malicious containers detected:**

```bash
# 1. Preserve evidence
docker-compose logs > incident_logs_$(date +%s).txt

# 2. Stop potentially compromised service
docker-compose stop <service>

# 3. Inspect for modifications
docker diff <container_id>

# 4. Rebuild from known-good image
docker-compose rebuild <service>

# 5. Check secrets not exposed
grep -r "TOKEN\|PASSWORD\|SECRET" data/ logs/
# If any found: they're in logs only,logs are safe to delete
```

### INCIDENT: Secrets Exposed in Logs

```bash
# 1. Stop the application
docker-compose stop

# 2. Audit logs for secrets
grep -r "Bearer\|token\|APIKey\|password" logs/ data/

# 3. Rotate all exposed credentials
#    Generate new API_SECRET_KEY
#    Update .env and restart

# 4. Clean logs
rm -rf logs/*

# 5. Restart clean
docker-compose up -d

# 6. Alert team (changed credentials, new hashes)
```

---

## Escalation Path

**Tier 1 (First Response):**
- Check docker-compose ps
- Restart affected container
- Check logs from past hour

**Tier 2 (10 min unresolved):**
- Call DevOps on-call
- Follow diagnosis steps above
- Check if database backup needed

**Tier 3 (30 min unresolved):**
- Scale team: 2-3 engineers
- Consider failover to backup system
- Prepare incident report

**Tier 4 (>1 hour unresolved):**
- Declare SEV-1 incident
- Involve VP Engineering
- Activate disaster recovery plan

---

## Useful Commands

```bash
# View real-time logs for all services
docker-compose logs -f

# Logs for specific service
docker-compose logs -f orchestrator

# Exec into container for debugging
docker-compose exec orchestrator bash

# Check which port a service is using
docker-compose port microservice 5000

# Monitor resource usage live
docker stats

# Validate docker-compose.yml
docker-compose config

# Check network connectivity between services
docker-compose exec orchestrator ping prometheus
```

---

## Contact & Escalation

- **On-Call Pager:** [YOUR PAGER SERVICE]
- **Slack Channel:** #neuroshield-incidents
- **War Room:** [YOUR ZOOM LINK]
- **Postmortem Template:** docs/postmortems/INCIDENT_TEMPLATE.md
