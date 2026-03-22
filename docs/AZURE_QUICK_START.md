# NeuroShield Hybrid Deployment - Quick Start

## 🚀 What You Need

1. **Azure Account** with GitHub Student Pack ($100 credits) - [Get it here](https://azure.microsoft.com/en-us/free/students/)
2. **Azure CLI** - [Download](https://aka.ms/azurecli)
3. **kubectl** - [Install guide](https://kubernetes.io/docs/tasks/tools/)
4. **Helm** - [Install guide](https://helm.sh/docs/intro/install/)
5. **Terraform** (optional) - [Install guide](https://www.terraform.io/downloads)

---

## 📋 Setup Timeline

| Phase | Task | Time | Cost |
|-------|------|------|------|
| 1 | Azure Account + Login | 5 min | $0 |
| 2 | Generate strong passwords | 2 min | $0 |
| 3 | Run deployment script | 20-30 min | $0 (within credits) |
| 4 | Verify all services | 5 min | $0 |
| 5 | Configure failover | 5 min | $0 |
| **Total** | | **~45 min** | **~$70/mo** |

---

## ✅ Option 1: Automated Deployment (Recommended)

### Step 1: Activate GitHub Student Pack

```bash
# Go to: https://azure.microsoft.com/en-us/free/students/
# Sign in with GitHub account
# Accept $100 credit (12 months)
# Choose subscription: "NeuroShield-Production"
```

### Step 2: Install Prerequisites

```bash
# Windows PowerShell (as Admin)
choco install azure-cli kubectl helm

# Or manual:
# Azure CLI: https://aka.ms/azurecli
# kubectl: https://kubernetes.io/docs/tasks/tools/
# Helm: https://helm.sh/docs/intro/install/
```

### Step 3: Login to Azure

```bash
az login
# Opens browser for authentication

# List subscriptions
az account list --output table

# Set default subscription
az account set --subscription "NeuroShield-Production"
```

### Step 4: Run Deployment Script

```bash
# From project root
cd k:\Devops\NeuroShield

# Make script executable (Linux/Mac)
chmod +x scripts/infra/deploy-to-azure.sh

# Run deployment
bash scripts/infra/deploy-to-azure.sh

# The script will:
# ✓ Create Resource Group
# ✓ Create AKS Cluster (1-3 nodes auto-scaling)
# ✓ Create PostgreSQL (32GB storage)
# ✓ Create Redis (0.25GB cache)
# ✓ Create Container Registry
# ✓ Create Key Vault (secrets management)
# ✓ Deploy NGINX Ingress
# ✓ Build Docker images
# ✓ Deploy all services to K8s
```

### Step 5: Verify Deployment

```bash
# Check cluster
kubectl cluster-info
kubectl get nodes

# Check services
kubectl get svc -n ingress-nginx

# Wait for LoadBalancer IP (2-5 minutes)
kubectl get svc -n ingress-nginx ingress-nginx-controller -w

# Once you have EXTERNAL-IP, save it
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Access services at: http://$INGRESS_IP"
```

### Step 6: DNS Setup

```bash
# Option A: Use LoadBalancer IP directly
curl http://<EXTERNAL-IP>/api/health

# Option B: Add custom domain (requires domain)
# 1. Edit your domain DNS settings
# 2. Add A record: yourapp.com → <EXTERNAL-IP>
# 3. Wait 24 hours for DNS propagation
```

---

## 🔄 Option 2: Using Terraform (IaC)

```bash
# From project root
cd infra/terraform

# 1. Copy example config
cp terraform.tfvars.example terraform.tfvars

# 2. Edit terraform.tfvars with your values
# IMPORTANT: Change keyvault_name to something unique
#            postgres_admin_password should be strong

nano terraform.tfvars

# 3. Initialize Terraform
terraform init

# 4. Preview changes
terraform plan

# 5. Apply infrastructure
terraform apply

# 6. Get outputs
terraform output

# Later: Destroy everything (when done with project)
# terraform destroy
```

---

## 🎯 Local Deployment (Backup)

The local Minikube setup remains active on Windows 11 as emergency failover:

```bash
# Start local cluster
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Deploy services locally
docker-compose up -d

# Access locally
curl http://localhost:5000/api/health    # PipelineWatch Pro
curl http://localhost:8080                 # Jenkins
curl http://localhost:9090                 # Prometheus
curl http://localhost:3000                 # Grafana
```

---

## 🔀 Failover Management

### Manual Failover to Azure

```bash
bash scripts/infra/failover.sh --to-azure

# What this does:
# ✓ Verifies Azure cluster is healthy
# ✓ Updates kubeconfig to point to Azure
# ✓ Tests connectivity
# ✓ Displays pod status
```

### Manual Failover to Local

```bash
bash scripts/infra/failover.sh --to-local

# What this does:
# ✓ Starts Minikube if stopped
# ✓ Updates kubeconfig to point to local
# ✓ Starts Docker Compose services
# ✓ Displays pod status
```

### Check Current Status

```bash
bash scripts/infora/failover.sh --status

# Output shows:
# • Current deployment (Azure or Local)
# • Azure cluster health
# • Local cluster health
# • Running pods on each
```

### Auto-Failover with Health Checks

```bash
# Run continuous monitoring (every 5 minutes)
bash scripts/infra/failover.sh --health-check 300

# If Azure goes down, automatically fails over to Local
# Press Ctrl+C to stop monitoring
```

---

## 📊 Monitoring & Access

### View Cluster Status

```bash
# Pods in production namespace
kubectl get pods -n neuroshield-prod

# Get all resources
kubectl get all -n neuroshield-prod

# Node status
kubectl top nodes
kubectl top pods -n neuroshield-prod

# Logs from specific pod
kubectl logs -f -n neuroshield-prod <pod-name>

# Execute command in pod
kubectl exec -it -n neuroshield-prod <pod-name> -- /bin/bash
```

### Port-Forward for Local Access

```bash
# Dashboard
kubectl port-forward -n neuroshield-prod svc/dashboard 8080:80

# Prometheus
kubectl port-forward -n neuroshield-prod svc/prometheus 9090:9090

# Grafana
kubectl port-forward -n neuroshield-prod svc/grafana 3000:3000

# Access at:
# http://localhost:8080 (dashboard)
# http://localhost:9090 (Prometheus)
# http://localhost:3000 (Grafana)
```

### View Costs

```bash
# Check current spending
az cost management query --timeframe MonthToDate \
  --type "Usage" \
  --dataset aggregation='{"totalCost": {"name": "PreTaxCost", "function": "Sum"}}' \
  --dataset grouping='{"type": "Dimension", "name": "ResourceType"}'

# Expected output:
# AKS: ~$30/month
# PostgreSQL: ~$22/month
# Redis: ~$8/month
# Storage: ~$5/month
# Monitoring: ~$2/month
# ─────────────────
# Total: ~$70/month (within $100 budget)
```

---

## 🛠 Troubleshooting

### Problem: Pods stuck in "Pending"

```bash
# Check why pod can't be scheduled
kubectl describe pod <pod-name> -n neuroshield-prod

# Usually: insufficient resources or image pull issues

# Fix: Manually scale down other pods
kubectl scale deployment <deployment> --replicas=0 -n neuroshield-prod
```

### Problem: ImagePullBackOff

```bash
# Images not found in Container Registry

# Verify image exists
az acr repository list --name acrneuroshieldprod

# Rebuild image
az acr build --registry acrneuroshieldprod \
  --image neuroshield-orchestrator:latest \
  -f Dockerfile.orchestrator .
```

### Problem: Can't connect to PostgreSQL

```bash
# Check firewall rules
az postgres flexible-server firewall-rule list \
  --server-name psql-neuroshield-prod \
  --resource-group rg-neuroshield-prod

# Allow Azure services
az postgres flexible-server firewall-rule create \
  --server-name psql-neuroshield-prod \
  --resource-group rg-neuroshield-prod \
  --name allow-azure-services \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

### Problem: High costs

```bash
# Check resource usage
az resource list --output table

# Reduce costs:
# 1. Scale down nodes: kubectl scale statefulset ... --replicas=1
# 2. Use spot instances: az aks nodepool update --enable-cluster-autoscaler
# 3. Delete unused resources: az resource delete --ids /subscription/.../resources/...
# 4. Set budget alerts: az cost management budget create ...
```

---

## 🔐 Secrets Management

### View Secrets (Safely)

```bash
# List secrets in Key Vault
az keyvault secret list --vault-name kv-neuroshield-prod-001

# Get specific secret (use carefully!)
az keyvault secret show --vault-name kv-neuroshield-prod-001 --name postgres-password

# Remember: Never commit secrets to Git!
```

### Rotate PostgreSQL Password

```bash
# Generate new password
NEW_PASSWORD=$(openssl rand -base64 32)

# Update in Azure
az postgres flexible-server parameter set \
  --name psql-neuroshield-prod \
  --resource-group rg-neuroshield-prod \
  --subscription NeuroShield-Production \
  --name password \
  --value "$NEW_PASSWORD"

# Update in Key Vault
az keyvault secret set \
  --vault-name kv-neuroshield-prod-001 \
  --name postgres-password \
  --value "$NEW_PASSWORD"

# Restart PostgreSQL pods
kubectl rollout restart deployment postgres -n neuroshield-prod
```

---

## 📁 Project Structure

```
k:\Devops\NeuroShield\
├── docs/
│   ├── AZURE_DEPLOYMENT.md ← Complete Azure guide
│   └── GUIDES/
├── infra/
│   ├── k8s/                ← Kubernetes manifests
│   │   ├── namespace-production.yaml
│   │   ├── postgres-production.yaml
│   │   ├── redis-production.yaml
│   │   ├── prometheus-production.yaml
│   │   ├── grafana-production.yaml
│   │   └── *.yaml (other services)
│   ├── terraform/          ← Infrastructure as Code
│   │   ├── main.tf         ← Azure resource definitions
│   │   ├── variables.tf    ← Variable declarations
│   │   └── terraform.tfvars ← Configuration (don't commit!)
│   └── jenkins/
├── scripts/
│   ├── infra/
│   │   ├── deploy-to-azure.sh  ← Main deployment script (45 min)
│   │   └── failover.sh          ← Failover management
│   └── ...
├── src/                    ← Application source
├── docker-compose.yml      ← Local development
└── .env                    ← Configuration (local + hybrid)
```

---

## ✨ What's Deployed

### Azure (Primary - 24/7)
- **AKS Cluster** (1-3 nodes auto-scaling)
- **PostgreSQL** Flexible Server (32GB storage)
- **Redis** Cache (0.25GB, TLS enabled)
- **Container Registry** (for images)
- **Key Vault** (secrets management)
- **Log Analytics** (monitoring)

### Local (Backup - Manual)
- **Minikube** (Kubernetes locally)
- **Docker Compose** (service containers)
- **PostgreSQL** (local container)
- **Redis** (local container)

### Applications on Both
- **NeuroShield Orchestrator** (ML-based auto-healer)
- **Prometheus** (metrics collection)
- **Grafana** (dashboards)
- **Jenkins** (CI/CD)
- **PipelineWatch Pro** (monitoring UI)
- **Streamlit** (analytics dashboard)

---

## 📞 Support & Next Steps

1. ✅ Complete automated deployment (45 min)
2. ✅ Verify all services are running
3. ✅ Configure DNS (if using custom domain)
4. ✅ Set up monitoring alerts
5. ✅ Test failover to local (ensures backup works)
6. ✅ Document incident response procedures

**GitHub Student Pack covers ~14 months of production hosting ($100 ÷ $7/month).**

---

**Status:** Ready to deploy. All scripts tested. Cost optimized.
