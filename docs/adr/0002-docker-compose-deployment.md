# ADR 0002: Use Docker Compose for Single-Node Deployment

**Status:** Accepted (2026-03-24)

## Context

NeuroShield deployment model:
- Development: Single machine
- Production (Phase 1): Single node (localhost-only)
- Production (Phase 2+): Multi-node Kubernetes

We need a deployment model that:
1. Matches single-machine constraints
2. Enables easy port mapping & isolation
3. Supports health checks & auto-restart
4. Scales to K8s later

## Decision

Use **Docker Compose** for Phase 1 deployment (single node).

### Alternatives

| Tool | Pros | Cons |
|------|------|------|
| **Docker Compose** ✅ | Simple, built-in, networking | Not HA, single node |
| Kubernetes (Minikube) | Production-ready, HA | Overkill for single node |
| Docker Swarm | Simpler than K8s | Deprecated, no community |
| Manual Docker CLI | Maximum control | No orchestration, no health checks |
| VM (no containers) | Traditional | Less reproducibility |

## Implementation

- Uses `docker-compose-hardened.yml` for security-focused config
- All services localhost-only (127.0.0.1 binding)
- Reverse proxy (nginx) in front for external access
- Health checks on every service
- Resource limits to prevent noisy neighbors

## Migration Path to Kubernetes

Files already prepared in `infra/k8s/`:
```
jenkins-production.yaml
postgres-production.yaml
prometheus-production.yaml
grafana-production.yaml
microservice-api.yaml
orchestrator-app.yaml
```

When ready for HA: `docker-compose convert | kubectl apply`

## Monitoring

Track in Prometheus:
- Container restart counts
- Service health check failures
- Resource utilization trends
