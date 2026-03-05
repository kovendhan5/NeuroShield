# NeuroShield -- Start Minikube + Deploy Dummy App
Write-Host "Setting up Minikube..." -ForegroundColor Cyan

# Check Docker is running first
try {
    docker info 2>&1 | Out-Null
    Write-Host "[OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Docker is not running -- start Docker Desktop first" -ForegroundColor Red
    exit 1
}

# Set Minikube home to D drive
$env:MINIKUBE_HOME = "D:\Docker\minikube"

# Start Minikube
$status = minikube status 2>&1
if ($status -match "Running") {
    Write-Host "[OK] Minikube already running" -ForegroundColor Green
} else {
    Write-Host "Starting Minikube (3GB RAM, Docker driver)..." -ForegroundColor Yellow
    minikube start --driver=docker --memory=3072 --cpus=2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Retrying with 2GB RAM..." -ForegroundColor Yellow
        minikube start --driver=docker --memory=2048 --cpus=2
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] Minikube failed to start" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "[OK] Minikube started" -ForegroundColor Green
}

# Build dummy-app image inside Minikube Docker
Write-Host "Building dummy-app image inside Minikube..." -ForegroundColor Yellow
minikube docker-env | Invoke-Expression
docker build -t neuroshield-dummy-app:latest ./infra/dummy-app

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Docker build failed" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] dummy-app image built" -ForegroundColor Green

# Deploy to Minikube
Write-Host "Deploying dummy-app..." -ForegroundColor Yellow
kubectl apply -f dummy-app.yaml
kubectl wait --for=condition=ready pod -l app=dummy-app --timeout=90s

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] dummy-app deployed" -ForegroundColor Green
} else {
    Write-Host "[WARN] Pod not ready yet -- check: kubectl get pods" -ForegroundColor Yellow
}

# Port-forward dummy-app
Write-Host "Port-forwarding dummy-app to localhost:5000..." -ForegroundColor Yellow
Start-Job -ScriptBlock {
    kubectl port-forward svc/dummy-app 5000:5000 2>&1
} | Out-Null
Start-Sleep -Seconds 3
Write-Host "[OK] dummy-app available at http://localhost:5000" -ForegroundColor Green
Write-Host "[OK] dummy-app /fail endpoint at http://localhost:5000/fail" -ForegroundColor Green
