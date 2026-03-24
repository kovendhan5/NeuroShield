# Docker Cleanup Plan - Careful Analysis

## Current State

### Unwanted Images (Orphaned, untagged, but RUNNING)
```
Image ID: a3407196e3e4 (13.3GB)
- Currently used by: neuroshield-orchestrator container
- Status: OLD BUILD - needs to be replaced

Image ID: 8a8ca761ca6c (259MB)  
- Currently used by: neuroshield-microservice container
- Status: OLD BUILD - needs to be replaced

Image: neuroshield-dashboard:latest (be44fe04951b, 1.01GB)
- Currently used by: NONE (no running container)
- Status: ORPHANED - can be safely deleted
```

### Newer Tagged Images (Not running)
```
neuroshield-orchestrator:latest (48651553d609, 13.3GB)
neuroshield-microservice:latest (8fed8dafd597, 252MB)
neuroshield-dashboard:latest (be44fe04951b, 1.01GB)
```

**Issue:** Containers are stuck on old untagged builds; new :latest tags exist but aren't being used.

---

## Unwanted Volumes (Duplicates - Old Format)

These are NOT in use (old underscored naming convention):
```
neuroshield_alertmanager_data   (duplicate of neuroshield-alertmanager-data)
neuroshield_grafana_data        (duplicate of neuroshield-grafana-data)
neuroshield_jenkins_home        (old variant of neuroshield-jenkins-data)
neuroshield_prometheus_data     (duplicate of neuroshield-prometheus-data)
neuroshield_redis_data          (duplicate of neuroshield-redis-data)
```

**Safe to delete?** YES - they're not mounted by any running containers

---

## Cleanup Strategy (Safe Approach)

### STEP 1: Delete unused volumes (5 min)
- Delete 5 old/duplicate volumes
- They're not in use by any container
- Data is on active volumes (hyphen-based)
- No risk to running system

### STEP 2: Purge dangling images (5 min)
- Delete orphaned/untagged images
- But FIRST update docker-compose to pull fresh
- Remove old untagged images
- Keep new :latest tagged images

### STEP 3: Restart containers (5 min)
- Restart with docker-compose
- Containers will use new :latest images
- Old images deleted

---

## Step-by-Step Commands

### Remove duplicate volumes (SAFE - not in use)
```bash
docker volume rm \
  neuroshield_alertmanager_data \
  neuroshield_grafana_data \
  neuroshield_jenkins_home \
  neuroshield_prometheus_data \
  neuroshield_redis_data
```

### Prune dangling/unused images
```bash
docker image prune --all --force
# This will remove:
# - All untagged images (a3407196e3e4, 8a8ca761ca6c)
# - Any images without tags not referenced by containers
```

### Restart containers to use clean images
```bash
docker-compose -f docker-compose-hardened.yml restart
```

---

## Verification

```bash
# Should show only active volumes (hyphen format)
docker volume ls | grep neuroshield

# Should show only tagged images
docker images | grep neuroshield

# Should show all containers healthy
docker-compose -f docker-compose-hardened.yml ps
```

---

## Estimated Cleanup

| Category | Count | Savings |
|----------|-------|---------|
| Duplicate volumes | 5 | ~300MB |
| Dangling images | 2 | ~13.5GB |
| Total | 7 | ~13.8GB |

