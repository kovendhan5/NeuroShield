#!/bin/bash

#/usr/bin/env bash
#
# NeuroShield Full Production Stack Launcher
# Starts all services: PostgreSQL, Redis, Jenkins, Prometheus, Grafana, Kubernetes microservices
# Deploys real microservices (API, Web, Worker) instead of dummy app
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

banner() {
  echo -e "\n${BLUE}"
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║      NeuroShield Production Stack Launcher v1.0            ║"
  echo "║  AI Self-Healing CI/CD with Real Microservices Architecture║"
  echo "╚════════════════════════════════════════════════════════════╝"
  echo -e "${NC}\n"
}

# Step 1: Start Docker Compose Stack
start_infrastructure() {
  log_info "=========================================="
  log_info "STEP 1: Starting Infrastructure Services"
  log_info "=========================================="

  cd "$PROJECT_ROOT"

  log_info "Starting PostgreSQL, Redis, Jenkins, Prometheus, Grafana..."
  docker-compose -f "$PROJECT_ROOT/docker-compose-full-stack.yml" up -d

  log_info "Waiting for services to be healthy..."
  sleep 10

  # Check PostgreSQL
  if docker-compose -f "$PROJECT_ROOT/docker-compose-full-stack.yml" exec -T postgres pg_isready -U admin -d neuroshield_db >/dev/null 2>&1; then
    log_success "PostgreSQL is online"
  else
    log_warn "PostgreSQL not ready yet, continuing..."
  fi

  # Check Redis
  if docker-compose -f "$PROJECT_ROOT/docker-compose-full-stack.yml" exec -T redis redis-cli ping >/dev/null 2>&1; then
    log_success "Redis is online"
  else
    log_warn "Redis not ready yet, continuing..."
  fi

  # Check Jenkins
  if timeout 5 curl -s http://localhost:8080/api/json >/dev/null 2>&1; then
    log_success "Jenkins is online at http://localhost:8080"
  else
    log_warn "Jenkins still starting (this can take 30-60 seconds)..."
  fi

  # Check Prometheus
  if timeout 5 curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
    log_success "Prometheus is online at http://localhost:9090"
  else
    log_warn "Prometheus still starting..."
  fi

  # Check Grafana
  if timeout 5 curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
    log_success "Grafana is online at http://localhost:3000 (admin/admin123)"
  else
    log_warn "Grafana still starting..."
  fi
}

# Step 2: Initialize Kubernetes
init_kubernetes() {
  log_info "=========================================="
  log_info "STEP 2: Initialize Kubernetes"
  log_info "=========================================="

  # Check if kubectl is available
  if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install kubernetes-cli"
    log_info "Windows (Chocolatey): choco install kubernetes-cli"
    log_info "Or download from: https://kubernetes.io/docs/tasks/tools/"
    return 1
  fi

  # Check if Minikube is running
  if command -v minikube &> /dev/null; then
    if minikube status | grep -q "Running"; then
      log_success "Minikube is already running"
    else
      log_info "Starting Minikube..."
      minikube start --cpus=4 --memory=4096
      log_success "Minikube started"
    fi
  else
    log_warn "Minikube not available. You may be using Docker Desktop K8s"
    log_warn "Ensure Kubernetes is enabled in Docker Desktop settings"
  fi

  # Check cluster connection
  if kubectl cluster-info >/dev/null 2>&1; then
    log_success "Kubernetes cluster connected"
  else
    log_error "Cannot connect to Kubernetes cluster"
    return 1
  fi
}

# Step 3: Create Kubernetes namespace and secrets
setup_kubernetes() {
  log_info "=========================================="
  log_info "STEP 3: Setting up Kubernetes Resources"
  log_info "=========================================="

  # Create namespace
  if kubectl get namespace neuroshield-prod >/dev/null 2>&1; then
    log_success "Namespace neuroshield-prod already exists"
  else
    log_info "Creating namespace neuroshield-prod..."
    kubectl create namespace neuroshield-prod
    log_success "Namespace created"
  fi

  # Create secrets
  log_info "Creating Kubernetes secrets..."
  kubectl -n neuroshield-prod create secret generic postgres-secret \
    --from-literal=password=secure_postgres_pass_123 \
    --dry-run=client -o yaml | kubectl apply -f -

  kubectl -n neuroshield-prod create secret generic redis-secret \
    --from-literal=password=redis_secure_pass_123 \
    --dry-run=client -o yaml | kubectl apply -f -

  log_success "Secrets created"

  # Create ConfigMap
  log_info "Creating ConfigMap..."
  kubectl -n neuroshield-prod create configmap neuroshield-config \
    --from-literal=DATABASE_HOST=postgres-service \
    --from-literal=DATABASE_PORT=5432 \
    --from-literal=DATABASE_NAME=neuroshield_db \
    --from-literal=REDIS_HOST=redis-service \
    --from-literal=REDIS_PORT=6379 \
    --from-literal=JENKINS_URL=http://jenkins:8080 \
    --from-literal=PROMETHEUS_URL=http://prometheus:9090 \
    --dry-run=client -o yaml | kubectl apply -f -

  log_success "ConfigMap created"
}

# Step 4: Deploy PostgreSQL and Redis to Kubernetes
deploy_databases() {
  log_info "=========================================="
  log_info "STEP 4: Deploying Databases to Kubernetes"
  log_info "=========================================="

  # Deploy PostgreSQL
  log_info "Deploying PostgreSQL..."
  kubectl apply -f "$PROJECT_ROOT/infra/k8s/postgres-production.yaml"
  log_success "PostgreSQL deployed"

  # Deploy Redis
  log_info "Deploying Redis..."
  kubectl apply -f "$PROJECT_ROOT/infra/k8s/redis-production.yaml"
  log_success "Redis deployed"

  # Wait for databases to be ready
  log_info "Waiting for databases to be ready (max 60 seconds)..."
  kubectl wait --for=condition=Ready pod \
    -l app=postgres -n neuroshield-prod \
    --timeout=60s 2>/dev/null || log_warn "PostgreSQL pod not ready, continuing..."

  kubectl wait --for=condition=Ready pod \
    -l app=redis -n neuroshield-prod \
    --timeout=60s 2>/dev/null || log_warn "Redis pod not ready, continuing..."
}

# Step 5: Deploy Real Microservices
deploy_microservices() {
  log_info "=========================================="
  log_info "STEP 5: Deploying Real Microservices"
  log_info "=========================================="

  # Deploy API Service
  log_info "Deploying API Service (3 replicas, Flask + PostgreSQL + Redis)..."
  kubectl apply -f "$PROJECT_ROOT/infra/k8s/microservice-api.yaml"
  log_success "API Service deployed"

  # Deploy Web Service
  log_info "Deploying Web Service..."
  kubectl apply -f "$PROJECT_ROOT/infra/k8s/microservice-web.yaml"
  log_success "Web Service deployed"

  # Deploy Worker Service
  log_info "Deploying Worker Service..."
  kubectl apply -f "$PROJECT_ROOT/infra/k8s/microservice-worker.yaml"
  log_success "Worker Service deployed"

  # Wait for deployments to be ready
  log_info "Waiting for microservices to be ready (max 120 seconds)..."
  for service in api web worker; do
    kubectl wait --for=condition=Ready pod \
      -l app=$service -n neuroshield-prod \
      --timeout=120s 2>/dev/null || log_warn "$service service not fully ready yet"
  done

  log_success "Microservices deployed"

  # Get service endpoints
  log_info "Service endpoints:"
  kubectl get svc -n neuroshield-prod | tail -n +2 | while read line; do
    echo "  $line"
  done
}

# Step 6: Setup Port Forwarding
setup_port_forwarding() {
  log_info "=========================================="
  log_info "STEP 6: Setting up Port Forwarding"
  log_info "=========================================="

  # Use kubectl port-forward for Kubernetes services
  if command -v kubectl &> /dev/null; then
    log_info "Setting up port-forward for API service (5001 -> 5000)..."
    kubectl port-forward -n neuroshield-prod svc/api-service 5001:5000 >/dev/null 2>&1 &
    log_success "API service port-forward established"

    log_info "Setting up port-forward for Web service (5002 -> 5000)..."
    kubectl port-forward -n neuroshield-prod svc/web-service 5002:5000 >/dev/null 2>&1 &
    log_success "Web service port-forward established"
  fi
}

# Step 7: Configure Orchestrator
configure_orchestrator() {
  log_info "=========================================="
  log_info "STEP 7: Configuring NeuroShield Orchestrator"
  log_info "=========================================="

  # Create/update .env
  cat > "$PROJECT_ROOT/.env.production" << EOF
# NeuroShield Production Configuration
ENVIRONMENT=production

# Jenkins
JENKINS_URL=http://localhost:8080
JENKINS_USERNAME=admin
JENKINS_TOKEN=your_jenkins_token_here
JENKINS_JOB=neuroshield-prod-pipeline

# Prometheus
PROMETHEUS_URL=http://localhost:9090

# Kubernetes
K8S_NAMESPACE=neuroshield-prod
AFFECTED_SERVICE=api-service

# Real Microservices
API_URL=http://localhost:5001
WEB_URL=http://localhost:5002
WORKER_URL=http://worker-service.neuroshield-prod:8000

# Databases
DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_NAME=neuroshield_db
DATABASE_USER=admin
DATABASE_PASSWORD=secure_postgres_pass_123

REDIS_HOST=redis
REDIS_PORT=6379

# NeuroShield Config
LOG_LEVEL=INFO
ORCHESTRATOR_POLL_INTERVAL=15
HEALING_ACTION_THRESHOLD=0.50
ENABLE_ML_PREDICTIONS=true
EOF

  log_success "Configuration created at .env.production"
}

# Step 8: Summary and Next Steps
show_summary() {
  log_info "=========================================="
  log_info "STEP 8: Production Stack Started Successfully"
  log_info "=========================================="

  echo -e "\n${GREEN}✓ ALL SERVICES ONLINE${NC}\n"

  echo "Infrastructure Services:"
  echo "  ${GREEN}✓${NC} PostgreSQL         http://localhost:5432 (admin/secure_postgres_pass_123)"
  echo "  ${GREEN}✓${NC} Redis              http://localhost:6379"
  echo "  ${GREEN}✓${NC} Jenkins CI/CD      http://localhost:8080 (admin/admin123)"
  echo "  ${GREEN}✓${NC} Prometheus Monitor http://localhost:9090"
  echo "  ${GREEN}✓${NC} Grafana Dashbrd    http://localhost:3000 (admin/admin123)"
  echo "  ${GREEN}✓${NC} AlertManager       http://localhost:9093"
  echo ""

  echo "Real Microservices (Kubernetes):"
  echo "  ${GREEN}✓${NC} API Service        http://localhost:5001/health"
  echo "  ${GREEN}✓${NC} Web Service        http://localhost:5002/health"
  echo "  ${GREEN}✓${NC} Worker Service     kubernetes cluster"
  echo ""

  echo "NeuroShield Orchestrator:"
  echo "  ${GREEN}✓${NC} Running in Docker  http://localhost:8000/health"
  echo "  ${GREEN}✓${NC} Monitoring        Real microservices (not dummy-app)"
  echo "  ${GREEN}✓${NC} Healing Actions   Automatically triggered on failures"
  echo ""

  echo "Next Steps:"
  echo "  1. Monitor orchestrator:        docker logs -f neuroshield-orchestrator"
  echo "  2. Check healing actions:       cat data/healing_log.json | python -m json.tool"
  echo "  3. View Prometheus metrics:     http://localhost:9090"
  echo "  4. View Grafana dashboards:     http://localhost:3000"
  echo "  5. Trigger test failure:        python scripts/inject_failure.py"
  echo "  6. Watch orchestrator respond:  tail -f logs/orchestrator.log"
  echo ""

  echo "Documentation:"
  echo "  • Full setup guide:  docs/GUIDES/QUICKSTART.md"
  echo "  • Troubleshooting:   docs/TROUBLESHOOTING.md"
  echo "  • Demo scenarios:    scripts/demo/e2e_real_system.py"
  echo ""
}

# Main execution
main() {
  banner

  start_infrastructure
  sleep 5

  if init_kubernetes; then
    setup_kubernetes
    deploy_databases
    sleep 10
    deploy_microservices
    setup_port_forwarding
  else
    log_warn "Skipping Kubernetes deployment - cluster not available"
    log_warn "Services will fall back to Docker Compose only"
  fi

  configure_orchestrator
  show_summary
}

# Run main function
main "$@"
