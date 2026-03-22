#!/bin/bash

#############################################################################
# Failover Manager - Switch between Azure and Local Deployment
# Usage: ./failover.sh [--to-azure|--to-local|--status]
#############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CONFIG_FILE=".failover-config"
FALLBACK_CONFIG=".env"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

#############################################################################
# STATUS CHECK
#############################################################################

check_azure_status() {
    log_info "Checking Azure deployment..."

    if ! command -v az &> /dev/null; then
        log_error "Azure CLI not installed"
        return 1
    fi

    if ! az account show &> /dev/null; then
        log_error "Not logged into Azure"
        return 1
    fi

    # Check if cluster exists
    CLUSTER=$(az aks list --query "[?name=='aks-neuroshield-prod'].name" -o tsv)
    if [ -z "$CLUSTER" ]; then
        log_error "AKS cluster not found"
        return 1
    fi

    # Check cluster health
    if kubectl cluster-info &> /dev/null; then
        NODES=$(kubectl get nodes --no-headers | wc -l)
        PODS=$(kubectl get pods -n neuroshield-prod --no-headers 2>/dev/null | wc -l || echo "0")
        log_success "Azure cluster healthy: $NODES nodes, $PODS pods running"
        return 0
    else
        log_error "Cannot connect to Azure cluster"
        return 1
    fi
}

check_local_status() {
    log_info "Checking local deployment..."

    if ! command -v minikube &> /dev/null; then
        log_error "Minikube not installed"
        return 1
    fi

    if ! docker ps &> /dev/null; then
        log_error "Docker not running"
        return 1
    fi

    # Check minikube status
    if minikube status &> /dev/null; then
        PODS=$(kubectl get pods -n neuroshield-prod --no-headers 2>/dev/null | wc -l || echo "0")
        log_success "Local Minikube healthy: $PODS pods running"
        return 0
    else
        log_error "Minikube is not running"
        return 1
    fi
}

status_all() {
    echo
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Deployment Status                                             ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo

    CURRENT_DEPLOYMENT=$(cat $CONFIG_FILE 2>/dev/null || echo "unknown")

    echo -e "Current Deployment: ${YELLOW}$CURRENT_DEPLOYMENT${NC}"
    echo

    echo "Azure Deployment:"
    check_azure_status >/dev/null 2>&1 && log_success "ONLINE" || log_error "OFFLINE"
    echo

    echo "Local Deployment:"
    check_local_status >/dev/null 2>&1 && log_success "ONLINE" || log_error "OFFLINE"
    echo
}

#############################################################################
# FAILOVER FUNCTIONS
#############################################################################

failover_to_azure() {
    log_info "Starting failover to Azure..."
    echo

    # Verify Azure is healthy
    if ! check_azure_status; then
        log_error "Azure cluster is not healthy. Aborting."
        exit 1
    fi

    log_info "Azure cluster is healthy and ready"

    # Update kubeconfig to point to Azure
    log_info "Updating kubeconfig to Azure cluster"
    az aks get-credentials \
        --resource-group rg-neuroshield-prod \
        --name aks-neuroshield-prod \
        --overwrite-existing

    # Verify connectivity
    if kubectl cluster-info &> /dev/null; then
        log_success "Connected to Azure cluster"
    else
        log_error "Failed to connect to Azure cluster"
        exit 1
    fi

    # Check deployment status
    log_info "Checking application status in Azure"
    kubectl get all -n neuroshield-prod

    # Save state
    echo "azure" > $CONFIG_FILE
    echo "DEPLOYMENT_TARGET=azure" >> $FALLBACK_CONFIG

    log_success "Failover to Azure complete"
    echo
    log_info "Next steps:"
    echo "  1. Monitor logs: kubectl logs -f -n neuroshield-prod <pod-name>"
    echo "  2. Check services: kubectl get svc -n neuroshield-prod"
    echo "  3. View dashboard: kubectl port-forward svc/dashboard 8080:80 -n neuroshield-prod"
}

failover_to_local() {
    log_info "Starting failover to Local..."
    echo

    # Verify local is healthy
    if ! check_local_status; then
        log_warning "Local Minikube cluster may not be healthy"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Failover cancelled"
            exit 1
        fi

        # Try to start Minikube
        log_info "Starting Minikube..."
        minikube start --cpus=4 --memory=8192 --disk-size=50g
    fi

    log_info "Local cluster is ready"

    # Update kubeconfig to point to local
    log_info "Updating kubeconfig to local Minikube"
    kubectl config use-context minikube

    # Verify connectivity
    if kubectl cluster-info &> /dev/null; then
        log_success "Connected to local Minikube"
    else
        log_error "Failed to connect to local Minikube"
        exit 1
    fi

    # Start Docker Compose services if needed
    log_info "Ensuring Docker Compose services are running"
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d
        log_success "Docker Compose services started"
    fi

    # Check deployment status
    log_info "Checking application status locally"
    kubectl get all -n neuroshield-prod

    # Save state
    echo "local" > $CONFIG_FILE
    echo "DEPLOYMENT_TARGET=local" >> $FALLBACK_CONFIG

    log_success "Failover to Local complete"
    echo
    log_info "Next steps:"
    echo "  1. Access dashboard: http://localhost:5000"
    echo "  2. Monitor logs: kubectl logs -f -n neuroshield-prod <pod-name>"
    echo "  3. Or use Docker Compose: docker-compose logs -f orchestrator"
}

sync_config_from_git() {
    log_info "Syncing configuration from Git..."

    if ! git fetch origin main &> /dev/null; then
        log_error "Git fetch failed"
        return 1
    fi

    log_info "Pulling latest configuration"
    git reset --hard origin/main

    log_info "Applying Kubernetes manifests"
    kubectl apply -f infra/k8s/ --namespace neuroshield-prod

    log_success "Configuration synced"
}

#############################################################################
# HEALTH CHECK & AUTO-FAILOVER
#############################################################################

health_check_loop() {
    INTERVAL=${1:-300}  # Default: 5 minutes
    CURRENT_TARGET=$(cat $CONFIG_FILE 2>/dev/null || echo "azure")

    echo
    log_info "Starting health check loop (interval: ${INTERVAL}s)"
    log_info "Current target: $CURRENT_TARGET"
    echo

    while true; do
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

        if [ "$CURRENT_TARGET" == "azure" ]; then
            if check_azure_status >/dev/null 2>&1; then
                echo "[$TIMESTAMP] Azure: OK"
            else
                log_warning "[$TIMESTAMP] Azure is DOWN! Attempting failover to local..."
                failover_to_local
                CURRENT_TARGET="local"
            fi
        else
            if check_local_status >/dev/null 2>&1; then
                echo "[$TIMESTAMP] Local: OK"
            else
                log_warning "[$TIMESTAMP] Local is DOWN! Cannot failover (Azure is primary)"
            fi
        fi

        # Sync config every 10 health checks
        if [ $(($(date +%s) % 3000)) -lt $INTERVAL ]; then
            sync_config_from_git || true
        fi

        sleep $INTERVAL
    done
}

#############################################################################
# MAIN
#############################################################################

main() {
    COMMAND=${1:-status}

    case $COMMAND in
        --to-azure)
            failover_to_azure
            ;;
        --to-local)
            failover_to_local
            ;;
        --status)
            status_all
            ;;
        --sync)
            sync_config_from_git
            ;;
        --health-check)
            INTERVAL=${2:-300}
            health_check_loop $INTERVAL
            ;;
        --help)
            echo "Usage: $0 [COMMAND]"
            echo
            echo "Commands:"
            echo "  --to-azure       Failover to Azure deployment"
            echo "  --to-local       Failover to local Minikube"
            echo "  --status         Check status of both deployments"
            echo "  --sync           Sync config from Git"
            echo "  --health-check   Run continuous health monitoring (TODO: 300s interval)"
            echo "  --help           Show this message"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
