#!/bin/bash

#############################################################################
# NeuroShield Azure Deployment Automation Script
# Purpose: Automated setup of production Azure infrastructure
# Requires: Azure CLI, kubectl, Terraform, Azure Student Pack activated
#############################################################################

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#############################################################################
# CONFIGURATION
#############################################################################

SUBSCRIPTION_NAME="NeuroShield-Production"
RESOURCE_GROUP="rg-neuroshield-prod"
LOCATION="East US"
CLUSTER_NAME="aks-neuroshield-prod"
ACR_NAME="acrneuroshieldprod"
POSTGRES_NAME="psql-neuroshield-prod"
REDIS_NAME="redis-neuroshield-prod"
KEYVAULT_NAME="kv-neuroshield-prod-001"

#############################################################################
# UTILITY FUNCTIONS
#############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is not installed. Please install it first."
        exit 1
    fi
}

#############################################################################
# PHASE 0: PRE-FLIGHT CHECKS
#############################################################################

phase_0_checks() {
    log_info "Running pre-flight checks..."

    check_command "az"
    check_command "kubectl"
    check_command "terraform"

    # Check Azure login
    if ! az account list &> /dev/null; then
        log_error "Not logged in to Azure. Run: az login"
        exit 1
    fi

    # Generate strong password
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    log_success "Generated PostgreSQL password (saved to: azure_secrets.txt)"
    echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD" > azure_secrets.txt
    chmod 600 azure_secrets.txt

    log_success "Pre-flight checks passed"
}

#############################################################################
# PHASE 1: AZURE ACCOUNT SETUP
#############################################################################

phase_1_azure_account() {
    log_info "PHASE 1: Azure Account Setup"

    # Get subscription ID
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    log_success "Using subscription: $SUBSCRIPTION_ID"

    # Create resource group
    log_info "Creating resource group: $RESOURCE_GROUP"
    az group create \
        --name $RESOURCE_GROUP \
        --location "$LOCATION"
    log_success "Resource group created"

    # Save subscription info
    cat > azure_info.txt << EOF
SUBSCRIPTION_ID=$SUBSCRIPTION_ID
RESOURCE_GROUP=$RESOURCE_GROUP
LOCATION=$LOCATION
CREATED_AT=$(date)
EOF

    log_success "PHASE 1 complete"
}

#############################################################################
# PHASE 2: CONTAINER REGISTRY
#############################################################################

phase_2_container_registry() {
    log_info "PHASE 2: Container Registry Setup"

    log_info "Creating Azure Container Registry: $ACR_NAME"
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --admin-enabled true

    log_success "ACR created"

    # Get credentials
    ACR_USER=$(az acr credential show --name $ACR_NAME --query username -o tsv)
    ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' -o tsv)
    ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)

    log_info "Logging in to ACR"
    az acr login --name $ACR_NAME

    # Save ACR credentials
    cat >> azure_secrets.txt << EOF
ACR_NAME=$ACR_NAME
ACR_USER=$ACR_USER
ACR_PASSWORD=$ACR_PASSWORD
ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER
EOF

    log_success "PHASE 2 complete"
}

#############################################################################
# PHASE 3: KUBERNETES CLUSTER
#############################################################################

phase_3_kubernetes() {
    log_info "PHASE 3: Kubernetes Cluster Setup"

    log_info "Creating AKS cluster: $CLUSTER_NAME"
    az aks create \
        --resource-group $RESOURCE_GROUP \
        --name $CLUSTER_NAME \
        --node-count 1 \
        --vm-set-type VirtualMachineScaleSets \
        --load-balancer-sku standard \
        --enable-managed-identity \
        --network-plugin azure \
        --network-policy azure \
        --docker-bridge-address 172.17.0.1/16 \
        --service-cidr 10.1.0.0/16 \
        --dns-service-ip 10.1.0.10 \
        --vm-sku-name Standard_B3s \
        --aks-custom-headers UseGPUDedicatedVHD=false

    log_success "AKS cluster created"

    # Get credentials
    log_info "Configuring kubectl credentials"
    az aks get-credentials \
        --resource-group $RESOURCE_GROUP \
        --name $CLUSTER_NAME \
        --overwrite-existing

    # Verify connection
    log_info "Verifying cluster connection"
    if kubectl cluster-info &> /dev/null; then
        log_success "Kubernetes cluster is accessible"
    else
        log_error "Failed to connect to Kubernetes cluster"
        exit 1
    fi

    # Enable auto-scaling
    log_info "Enabling cluster auto-scaling (1-3 nodes)"
    az aks nodepool update \
        --resource-group $RESOURCE_GROUP \
        --cluster-name $CLUSTER_NAME \
        --name nodepool1 \
        --enable-cluster-autoscaler \
        --min-count 1 \
        --max-count 3

    log_success "PHASE 3 complete"
}

#############################################################################
# PHASE 4: DATABASE SERVICES
#############################################################################

phase_4_databases() {
    log_info "PHASE 4: Database Services Setup"

    # PostgreSQL
    log_info "Creating PostgreSQL Flexible Server: $POSTGRES_NAME"
    az postgres flexible-server create \
        --resource-group $RESOURCE_GROUP \
        --name $POSTGRES_NAME \
        --location "$LOCATION" \
        --admin-user "dbadmin" \
        --admin-password "$POSTGRES_PASSWORD" \
        --sku-name Standard_B1ms \
        --tier Burstable \
        --storage-size 32 \
        --version 14 \
        --high-availability Disabled

    log_success "PostgreSQL server created"

    # Create database
    log_info "Creating database: neuroshield"
    az postgres flexible-server db create \
        --server-name $POSTGRES_NAME \
        --resource-group $RESOURCE_GROUP \
        --name neuroshield

    log_success "Database created"

    # Allow Azure services
    az postgres flexible-server firewall-rule create \
        --server-name $POSTGRES_NAME \
        --resource-group $RESOURCE_GROUP \
        --name allow-azure-services \
        --start-ip-address 0.0.0.0 \
        --end-ip-address 0.0.0.0

    # Redis
    log_info "Creating Redis Cache: $REDIS_NAME"
    az redis create \
        --resource-group $RESOURCE_GROUP \
        --name $REDIS_NAME \
        --location "$LOCATION" \
        --sku Basic \
        --vm-size c0 \
        --enable-non-ssl-port false

    log_success "Redis cache created"

    # Get connection details
    POSTGRES_HOST=$(az postgres flexible-server show --name $POSTGRES_NAME --resource-group $RESOURCE_GROUP --query fullyQualifiedDomainName -o tsv)
    REDIS_HOST=$(az redis show --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query hostName -o tsv)
    REDIS_KEY=$(az redis list-keys --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query primaryKey -o tsv)

    # Save database credentials
    cat >> azure_secrets.txt << EOF
POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_USER=dbadmin
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DATABASE=neuroshield
REDIS_HOST=$REDIS_HOST
REDIS_KEY=$REDIS_KEY
REDIS_PORT=6380
EOF

    log_success "PHASE 4 complete"
}

#############################################################################
# PHASE 5: KEY VAULT
#############################################################################

phase_5_keyvault() {
    log_info "PHASE 5: Key Vault Setup"

    log_info "Creating Azure Key Vault: $KEYVAULT_NAME"
    az keyvault create \
        --name $KEYVAULT_NAME \
        --resource-group $RESOURCE_GROUP \
        --location "$LOCATION" \
        --enabled-for-deployment true \
        --enabled-for-template-deployment true

    log_success "Key Vault created"

    # Add secrets
    log_info "Adding secrets to Key Vault"

    az keyvault secret set \
        --vault-name $KEYVAULT_NAME \
        --name "postgres-password" \
        --value "$POSTGRES_PASSWORD"

    az keyvault secret set \
        --vault-name $KEYVAULT_NAME \
        --name "redis-key" \
        --value "$REDIS_KEY"

    az keyvault secret set \
        --vault-name $KEYVAULT_NAME \
        --name "acr-password" \
        --value "$ACR_PASSWORD"

    log_success "PHASE 5 complete"
}

#############################################################################
# PHASE 6: KUBERNETES CONFIGURATION
#############################################################################

phase_6_kubernetes_config() {
    log_info "PHASE 6: Kubernetes Configuration"

    # Create namespace
    log_info "Creating Kubernetes namespace: neuroshield-prod"
    kubectl create namespace neuroshield-prod || true

    # Label namespace
    kubectl label namespace neuroshield-prod \
        environment=production \
        monitoring=enabled \
        --overwrite

    # Create secrets
    log_info "Creating Kubernetes secrets"
    kubectl create secret generic neuroshield-secrets \
        --from-literal=postgres-password="$POSTGRES_PASSWORD" \
        --from-literal=redis-key="$REDIS_KEY" \
        --from-literal=acr-password="$ACR_PASSWORD" \
        --namespace neuroshield-prod \
        --dry-run=client -o yaml | kubectl apply -f -

    # Create ACR secret for pod image pulls
    log_info "Creating ACR image pull secret"
    kubectl create secret docker-registry regcred \
        --docker-server=$ACR_LOGIN_SERVER \
        --docker-username=$ACR_USER \
        --docker-password=$ACR_PASSWORD \
        --docker-email=noreply@neuroshield.dev \
        --namespace neuroshield-prod \
        --dry-run=client -o yaml | kubectl apply -f -

    log_success "PHASE 6 complete"
}

#############################################################################
# PHASE 7: INGRESS CONTROLLER
#############################################################################

phase_7_ingress() {
    log_info "PHASE 7: Ingress Controller Setup"

    # Add Helm repo
    log_info "Adding NGINX Helm repository"
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update

    # Install ingress controller
    log_info "Installing NGINX Ingress Controller"
    helm install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.service.type=LoadBalancer \
        --set controller.resources.requests.cpu=100m \
        --set controller.resources.requests.memory=128Mi \
        --set controller.metrics.enabled=true \
        --set controller.podAnnotations."prometheus\.io/scrape"=true \
        --set controller.podAnnotations."prometheus\.io/port"=10254

    log_info "Waiting for LoadBalancer IP..."
    sleep 10

    INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "PENDING")

    if [ "$INGRESS_IP" != "PENDING" ]; then
        log_success "LoadBalancer IP: $INGRESS_IP"
        echo "INGRESS_IP=$INGRESS_IP" >> azure_info.txt
    else
        log_warning "LoadBalancer IP still pending. Run later: kubectl get svc -n ingress-nginx"
    fi

    log_success "PHASE 7 complete"
}

#############################################################################
# PHASE 8: BUILD AND PUSH IMAGES
#############################################################################

phase_8_build_images() {
    log_info "PHASE 8: Build and Push Docker Images"

    if [ ! -f "Dockerfile.orchestrator" ]; then
        log_warning "Dockerfiles not found in current directory. Skipping image build."
        log_info "To build images later, run:"
        echo "  az acr build --registry $ACR_NAME --image neuroshield-orchestrator:latest -f Dockerfile.orchestrator ."
        return
    fi

    log_info "Building orchestrator image"
    az acr build \
        --registry $ACR_NAME \
        --image neuroshield-orchestrator:latest \
        --file Dockerfile.orchestrator \
        .

    log_info "Building dashboard image"
    az acr build \
        --registry $ACR_NAME \
        --image neuroshield-dashboard:latest \
        --file Dockerfile.streamlit \
        .

    log_info "Building pipeline-watch image"
    az acr build \
        --registry $ACR_NAME \
        --image pipeline-watch:latest \
        --file pipeline-watch/Dockerfile \
        .

    log_info "Building dummy-app image"
    az acr build \
        --registry $ACR_NAME \
        --image dummy-app:latest \
        --file infra/dummy-app/Dockerfile \
        .

    log_success "PHASE 8 complete"
}

#############################################################################
# PHASE 9: DEPLOY APPLICATIONS
#############################################################################

phase_9_deploy() {
    log_info "PHASE 9: Deploy Applications"

    if [ ! -f "infra/k8s/namespace-production.yaml" ]; then
        log_warning "Kubernetes manifests not found. Skipping deployment."
        return
    fi

    log_info "Updating image references in manifests"
    sed -i "s|docker.io/neuroshield|${ACR_LOGIN_SERVER}/neuroshield|g" infra/k8s/*.yaml
    sed -i "s|postgres:14|${POSTGRES_HOST}|g" infra/k8s/*.yaml
    sed -i "s|redis:6379|${REDIS_HOST}:6380|g" infra/k8s/*.yaml

    log_info "Deploying infrastructure"
    kubectl apply -f infra/k8s/ --namespace neuroshield-prod

    log_info "Waiting for pods to be ready (max 300 seconds)"
    if kubectl wait --for=condition=ready pod -l app=neuroshield -n neuroshield-prod --timeout=300s 2>/dev/null; then
        log_success "All pods are ready"
    else
        log_warning "Some pods not ready after 5 minutes. Check with: kubectl get pods -n neuroshield-prod"
    fi

    log_success "PHASE 9 complete"
}

#############################################################################
# PHASE 10: VERIFICATION
#############################################################################

phase_10_verify() {
    log_info "PHASE 10: Verification"

    log_info "Checking cluster status"
    kubectl get nodes

    log_info "Checking namespaces"
    kubectl get namespaces

    log_info "Checking pods in neuroshield-prod"
    kubectl get pods -n neuroshield-prod

    log_info "Checking services"
    kubectl get svc -n ingress-nginx

    log_success "PHASE 10 complete"
}

#############################################################################
# MAIN EXECUTION
#############################################################################

main() {
    echo
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  NeuroShield Azure Deployment Automation                       ║${NC}"
    echo -e "${BLUE}║  GitHub Student Pack: $100 credits                              ║${NC}"
    echo -e "${BLUE}║  Estimated Cost: $70/month (within budget)                     ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo

    read -p "Continue with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_error "Deployment cancelled"
        exit 1
    fi

    start_time=$(date +%s)

    phase_0_checks
    phase_1_azure_account
    phase_2_container_registry
    phase_3_kubernetes
    phase_4_databases
    phase_5_keyvault
    phase_6_kubernetes_config
    phase_7_ingress
    phase_8_build_images
    phase_9_deploy
    phase_10_verify

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Deployment Complete!                                         ║${NC}"
    echo -e "${GREEN}║  Duration: $((duration / 60)) minutes $((duration % 60)) seconds                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo

    log_info "Credentials saved to: azure_secrets.txt (chmod 600, keep secure!)"
    log_info "Cluster info saved to: azure_info.txt"
    echo

    log_success "Next steps:"
    echo "  1. Get LoadBalancer IP: kubectl get svc -n ingress-nginx"
    echo "  2. Update DNS to point your domain to LoadBalancer IP"
    echo "  3. Deploy applications: kubectl apply -f infra/k8s/"
    echo "  4. Monitor logs: kubectl logs -f -n neuroshield-prod <pod-name>"
    echo "  5. Check costs: az cost management query --timeframe MonthToDate"
    echo
}

# Run main
main
