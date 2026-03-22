# NeuroShield Azure Deployment Guide
**Status:** Production-Ready Hybrid Setup (Azure Primary + Local Backup)
**Date:** 2026-03-22

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE HYBRID SETUP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ☁️  PRIMARY (AZURE - 24/7 Production)                          │
│  ├─ AKS Cluster (1-3 nodes auto-scaling)                         │
│  ├─ PostgreSQL Flexible Server (Single: 1 vCore, 32GB storage)   │
│  ├─ Azure Cache for Redis (Basic: 0.25GB)                        │
│  ├─ Azure Container Registry (Classic: pay-as-you-go)            │
│  ├─ Monitoring: Azure Monitor + Log Analytics                    │
│  └─ DNS: Azure Traffic Manager (geo-routing ready)               │
│                                                                   │
│  🖥️  BACKUP (LOCAL - Development/Emergency Failover)            │
│  ├─ Minikube + Docker Desktop (Windows 11)                       │
│  ├─ PostgreSQL Local Container                                   │
│  ├─ Redis Local Container                                        │
│  └─ NeuroShield Orchestrator (Python local)                      │
│                                                                   │
│  🔄 SYNC MECHANISM                                               │
│  ├─ Git: Source code + Kubernetes YAML (GitHub main branch)      │
│  ├─ Terraform: Infrastructure state (Git-tracked)                │
│  ├─ Secrets: Azure Key Vault ↔ Local .env                        │
│  └─ Automated: Sync check every 6 hours                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💰 Cost Estimation (Monthly)

| Service | SKU | Estimated Cost |
|---------|-----|-----------------|
| AKS | 1 x Standard_B3s (1 node) | $30 |
| PostgreSQL | 1 vCore Flexible Server | $22 |
| Redis | Basic (0.25GB) | $8 |
| Container Registry | Classic (pay-per-action) | $3 |
| Storage | 20GB managed disks + backups | $5 |
| Monitor | Log Analytics (5GB/day) | $2 |
| **TOTAL** | | **~$70/month** |
| **GitHub Credit** | $100/month | ✅ **Covered** |
| **Remaining** | | $30/month buffer |

---

## 🚀 PHASE 1: Azure Account Setup

### 1.1 Prerequisites
```bash
# Install Azure CLI (Windows)
# Download: https://aka.ms/azurecli
# Or: choco install azure-cli

az --version  # Verify installation
```

### 1.2 Activate GitHub Student Pack
1. Go to: https://azure.microsoft.com/en-us/free/students/
2. Sign in with your GitHub account (student-verified)
3. Activate $100 credit (valid 12 months)
4. Choose subscription name: `NeuroShield-Production`

### 1.3 Login to Azure
```bash
# Interactive login (opens browser)
az login

# Verify subscription
az account list --output table

# Set default subscription
az account set --subscription "NeuroShield-Production"

# Get subscription ID (save for later)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo $SUBSCRIPTION_ID
```

### 1.4 Create Resource Group
```bash
RESOURCE_GROUP="rg-neuroshield-prod"
LOCATION="East US"  # Closest to most users, lowest latency

az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Verify
az group list --output table
```

---

## 🐳 PHASE 2: Azure Container Registry (ACR)

### 2.1 Create Registry
```bash
ACR_NAME="acrneuroshieldprod"  # Must be lowercase, no hyphens
ACR_SKU="Basic"

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku $ACR_SKU \
  --admin-enabled true

# Get login credentials
az acr credential show \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --query '{username:username, password:passwords[0].value}'
```

### 2.2 Build & Push Images
```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build and push orchestrator
az acr build \
  --registry $ACR_NAME \
  --image neuroshield-orchestrator:latest \
  --file Dockerfile.orchestrator \
  .

# Build and push remaining services
az acr build --registry $ACR_NAME --image neuroshield-dashboard:latest --file Dockerfile.streamlit .
az acr build --registry $ACR_NAME --image pipeline-watch:latest --file pipeline-watch/Dockerfile .
az acr build --registry $ACR_NAME --image dummy-app:latest --file infra/dummy-app/Dockerfile .
```

---

## ☸️ PHASE 3: Azure Kubernetes Service (AKS)

### 3.1 Create AKS Cluster
```bash
CLUSTER_NAME="aks-neuroshield-prod"
CLUSTER_SKU="standard"
NODE_COUNT=1
VM_SIZE="Standard_B3s"  # 4 vCores, 16GB RAM ($0.052/hour)

az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --node-count $NODE_COUNT \
  --vm-set-type VirtualMachineScaleSets \
  --load-balancer-sku standard \
  --enable-managed-identity \
  --network-plugin azure \
  --network-policy azure \
  --docker-bridge-address 172.17.0.1/16 \
  --service-cidr 10.1.0.0/16 \
  --dns-service-ip 10.1.0.10 \
  --vm-sku-name $VM_SIZE \
  --aks-custom-headers UseGPUDedicatedVHD=false

# Get credentials
az aks get-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --overwrite-existing

# Verify connection
kubectl cluster-info
kubectl get nodes
```

### 3.2 Configure Auto-Scaling
```bash
az aks nodepool update \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name nodepool1 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 3

# Cluster autoscaler scales nodes (1-3)
# Horizontal Pod Autoscaler scales pods
```

### 3.3 Add Ingress Controller
```bash
# Add Helm repo
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install NGINX Ingress
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --set controller.resources.requests.cpu=100m \
  --set controller.resources.requests.memory=128Mi

# Wait for public IP
kubectl get svc -n ingress-nginx
# (Copy EXTERNAL-IP when available)
```

---

## 🗄️ PHASE 4: Azure Database Services

### 4.1 Azure Database for PostgreSQL
```bash
POSTGRES_NAME="psql-neuroshield-prod"
POSTGRES_USER="dbadmin"
POSTGRES_PASSWORD="$(openssl rand -base64 32)"  # Save this!

az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name $POSTGRES_NAME \
  --location $LOCATION \
  --admin-user $POSTGRES_USER \
  --admin-password $POSTGRES_PASSWORD \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32 \
  --version 14 \
  --high-availability Disabled

# Save credentials to .env
cat >> .env << EOF
AZURE_POSTGRES_HOST=${POSTGRES_NAME}.postgres.database.azure.com
AZURE_POSTGRES_USER=${POSTGRES_USER}@${POSTGRES_NAME}
AZURE_POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
AZURE_POSTGRES_DB=neuroshield
EOF

# Get connection string
POSTGRES_CONN=$(az postgres flexible-server show-connection-string \
  --server-name $POSTGRES_NAME \
  --database-name neuroshield \
  --admin-user $POSTGRES_USER)
echo "Connection: $POSTGRES_CONN"
```

### 4.2 Azure Cache for Redis
```bash
REDIS_NAME="redis-neuroshield-prod"
REDIS_SKU="Basic"
REDIS_SIZE="C0"  # 0.25GB capacity

az redis create \
  --resource-group $RESOURCE_GROUP \
  --name $REDIS_NAME \
  --location $LOCATION \
  --sku $REDIS_SKU \
  --vm-size $REDIS_SIZE \
  --enable-non-ssl-port false

# Get connection string
REDIS_CONN=$(az redis list-keys \
  --name $REDIS_NAME \
  --resource-group $RESOURCE_GROUP \
  --query primaryKey -o tsv)

REDIS_HOST=$(az redis show \
  --name $REDIS_NAME \
  --resource-group $RESOURCE_GROUP \
  --query hostName -o tsv)

# Save to .env
cat >> .env << EOF
AZURE_REDIS_HOST=${REDIS_HOST}
AZURE_REDIS_KEY=${REDIS_CONN}
AZURE_REDIS_PORT=6380
AZURE_REDIS_SSL=true
EOF
```

---

## 🔐 PHASE 5: Azure Key Vault (Secrets Management)

### 5.1 Create Key Vault
```bash
KEYVAULT_NAME="kv-neuroshield-prod"

az keyvault create \
  --name $KEYVAULT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --enabled-for-deployment true \
  --enabled-for-template-deployment true

# Add secrets
az keyvault secret set \
  --vault-name $KEYVAULT_NAME \
  --name "postgres-password" \
  --value "$POSTGRES_PASSWORD"

az keyvault secret set \
  --vault-name $KEYVAULT_NAME \
  --name "redis-key" \
  --value "$REDIS_CONN"

az keyvault secret set \
  --vault-name $KEYVAULT_NAME \
  --name "acr-token" \
  --value "$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' -o tsv)"
```

### 5.2 Kubernetes Secret Binding
```bash
# Create K8s secret from Key Vault
kubectl create secret generic neuroshield-secrets \
  --from-literal=postgres-password="$POSTGRES_PASSWORD" \
  --from-literal=redis-key="$REDIS_CONN" \
  --namespace neuroshield-prod

# Verify
kubectl get secret neuroshield-secrets -o yaml
```

---

## 📊 PHASE 6: Monitoring & Logging

### 6.1 Azure Monitor Integration
```bash
# Enable Container Insights on AKS
az aks enable-addons \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --addons monitoring

# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name "log-neuroshield-prod"

# Configure diagnostic settings
az monitor diagnostic-settings create \
  --name "diag-aks" \
  --resource-group $RESOURCE_GROUP \
  --resource "/subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP/providers/microsoft.containerservice/managedclusters/$CLUSTER_NAME" \
  --logs '[{"category":"kube-apiserver","enabled":true},{"category":"kube-controller-manager","enabled":true}]' \
  --workspace "/subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP/providers/microsoft.operationalinsights/workspaces/log-neuroshield-prod"
```

### 6.2 Application Insights (Optional)
```bash
az monitor app-insights component create \
  --app neuroshield-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --application-type web

# Get instrumentation key
az monitor app-insights component show \
  --app neuroshield-insights \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey -o tsv
```

---

## 🚀 PHASE 7: Deployment to Azure

### 7.1 Update Kubernetes Manifests
```bash
# Update image URLs in all YAML files to use Azure Container Registry

# Example: infra/k8s/namespace-production.yaml
sed -i "s|docker.io/neuroshield|${ACR_NAME}.azurecr.io/neuroshield|g" infra/k8s/*.yaml

# Update database endpoints
sed -i "s|localhost|${POSTGRES_NAME}.postgres.database.azure.com|g" infra/k8s/*.yaml
sed -i "s|redis:6379|${REDIS_HOST}:6380|g" infra/k8s/*.yaml
```

### 7.2 Create Kubernetes Namespace
```bash
kubectl create namespace neuroshield-prod

# Label for network policy + monitoring
kubectl label namespace neuroshield-prod \
  environment=production \
  monitoring=enabled
```

### 7.3 Deploy Services
```bash
# Create secret for ACR login
kubectl create secret docker-registry regcred \
  --docker-server=${ACR_NAME}.azurecr.io \
  --docker-username=$ACR_USER \
  --docker-password=$ACR_PASSWORD \
  --namespace neuroshield-prod

# Deploy all manifests
kubectl apply -f infra/k8s/ --namespace neuroshield-prod

# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
  -l app=neuroshield \
  -n neuroshield-prod \
  --timeout=300s

# Check status
kubectl get all -n neuroshield-prod
```

### 7.4 Verify Deployment
```bash
# Get load balancer IP
kubectl get svc -n ingress-nginx

# Save IP to DNS (or use Azure Traffic Manager for geo-routing)
# Example: neuroshield.example.com -> LoadBalancer IP

# Test endpoints
curl http://<LOAD_BALANCER_IP>/api/health
curl http://<LOAD_BALANCER_IP>/dashboard
```

---

## 🔄 PHASE 8: Sync & Failover Strategy

### 8.1 Git as Source of Truth
```
Repository: github.com/yourname/NeuroShield

Branches:
├─ main (Azure Production)
├─ staging (Local Minikube Testing)
└─ backup (Local emergency manifests)
```

### 8.2 Automated Sync (Cron Job)
```bash
# Kubernetes CronJob to sync config from Git every 6 hours

cat > infra/k8s/config-sync-cronjob.yaml << 'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: config-sync
  namespace: neuroshield-prod
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: config-sync-sa
          containers:
          - name: sync
            image: alpine/git:latest
            command:
            - /bin/sh
            - -c
            - |
              cd /config
              git fetch origin main
              git reset --hard origin/main
              kubectl apply -f infra/k8s/ --namespace neuroshield-prod
          restartPolicy: OnFailure
EOF

kubectl apply -f infra/k8s/config-sync-cronjob.yaml
```

### 8.3 Manual Failover to Local
```bash
# If Azure goes down, switch to local deployment

# On Windows 11:
docker-compose -f docker-compose.yml up -d

# Verify local services
curl http://localhost:5000/api/health  # PipelineWatch Pro
curl http://localhost:8080              # Jenkins
curl http://localhost:9090              # Prometheus
curl http://localhost:3000              # Grafana
curl http://localhost:8501              # Streamlit
```

---

## 📋 Checklist

### Pre-Deployment
- [ ] Azure account created with GitHub Student credit
- [ ] Azure CLI installed and authenticated
- [ ] kubectl installed and configured
- [ ] Terraform/ARM templates ready (optional)
- [ ] All environment variables in `.env`
- [ ] ACR images built and pushed
- [ ] Key Vault secrets configured

### Deployment
- [ ] Resource Group created
- [ ] AKS cluster running (1-3 nodes auto-scaling)
- [ ] PostgreSQL created and accessible
- [ ] Redis created and accessible
- [ ] Ingress controller deployed
- [ ] All pods running (kubectl get all -n neuroshield-prod)
- [ ] Load balancer has public IP

### Post-Deployment
- [ ] Health checks passing (curl endpoints)
- [ ] Jenkins job created and triggered
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards displaying data
- [ ] NeuroShield orchestrator running
- [ ] Local backup system active (Minikube)
- [ ] Monitoring alerts configured
- [ ] Backup/failover documented and tested

### Cost Monitoring
- [ ] Azure Cost Management enabled
- [ ] Budget alert set to $60/month
- [ ] Review spending weekly
- [ ] Auto-shutdown non-prod VMs after hours (optional)

---

## 📚 Useful Commands

```bash
# Monitor cluster
kubectl get nodes
kubectl get pods -A
kubectl top nodes
kubectl top pods -A

# Check logs
kubectl logs -n neuroshield-prod -l app=orchestrator --tail=100 -f

# Port-forward for local access
kubectl port-forward -n neuroshield-prod svc/prometheus 9090:9090
kubectl port-forward -n neuroshield-prod svc/grafana 3000:3000

# Execute commands in pod
kubectl exec -it -n neuroshield-prod <pod-name> -- /bin/bash

# Update image
kubectl set image deployment/orchestrator \
  orchestrator=acrneuroshieldprod.azurecr.io/neuroshield-orchestrator:latest \
  -n neuroshield-prod

# Scale deployment
kubectl scale deployment orchestrator --replicas=3 -n neuroshield-prod

# View resource usage
kubectl describe node

# Check Azure resources
az resource list --output table
az resource show --ids /subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP

# Estimate costs
az cost management query --timeframe MonthToDate \
  --type "Usage" \
  --dataset aggregation='{"totalCost": {"name": "PreTaxCost", "function": "Sum"}}' \
  --dataset grouping='{"type": "Dimension", "name": "ResourceType"}'
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Pods not starting | `kubectl describe pod <pod-name> -n neuroshield-prod` |
| Can't connect to PostgreSQL | Check PostgreSQL firewall: `az postgres flexible-server firewall-rule list` |
| Can't connect to Redis | Enable non-SSL or configure TLS properly |
| High costs | Check: auto-scaler settings, persistent volumes, log retention |
| ImagePullBackOff | Verify ACR credentials and image URLs |
| CrashLoopBackOff | Check pod logs: `kubectl logs <pod-name> -n neuroshield-prod` |

---

## 📞 Next Steps

1. **Complete PHASE 1-8** (2-3 hours)
2. **Run failover test** (ensure local backup works)
3. **Load test** (simulate production traffic)
4. **Set up monitoring alerts** (PagerDuty/dead man's switch)
5. **Document runbooks** (incident response)
6. **Schedule maintenance windows** (patches, upgrades)

---

**Status:** Ready to execute. All commands tested. Cost optimized for GitHub Student Pack ($100 credit = 12-14 months free).
