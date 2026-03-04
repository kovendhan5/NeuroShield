#Requires -Version 5.1
<#
.SYNOPSIS
    NeuroShield Local Setup — Windows PowerShell
.DESCRIPTION
    One-command setup: installs dependencies, trains models, starts Minikube,
    deploys Jenkins + dummy-app, creates Jenkins job, and runs tests.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ok($msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }
function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "  NeuroShield Local Setup" -ForegroundColor Cyan
Write-Host "  =======================" -ForegroundColor Cyan
Write-Host ""

# STEP 1: Check prerequisites
Info "Checking prerequisites..."
foreach ($cmd in @('docker', 'minikube', 'kubectl', 'python', 'pip')) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        Ok "$cmd found"
    } else {
        Fail "$cmd not found - please install it"
    }
}

# STEP 2: Install Python dependencies
Info "Installing Python dependencies..."
pip install -r requirements.txt -q
Ok "Dependencies installed"

# STEP 3: Train models if missing
if (-not (Test-Path "models/ppo_policy.zip") -or -not (Test-Path "models/failure_predictor.pth")) {
    Info "Training models (first time only)..."
    python src/prediction/train.py
    Ok "Prediction model trained"
    python -m src.rl_agent.train
    Ok "PPO model trained"
} else {
    Ok "Models already exist - skipping training"
}

# STEP 4: Start Minikube
Info "Starting Minikube..."
$minikubeStatus = minikube status 2>&1
if ($LASTEXITCODE -eq 0) {
    Ok "Minikube already running"
} else {
    minikube start --memory=4096 --cpus=2
    if ($LASTEXITCODE -ne 0) { Fail "Minikube failed to start" }
    Ok "Minikube started"
}

# STEP 5: Build and load Docker images
Info "Building Docker images..."
minikube docker-env | Invoke-Expression
docker build -t neuroshield-dummy-app:latest ./infra/dummy-app -q
Ok "dummy-app image built"
docker build -t neuroshield-jenkins:latest -f ./infra/jenkins-builder/Dockerfile.jenkins ./infra/jenkins-builder -q
Ok "jenkins image built"

# STEP 6: Apply Kubernetes manifests
Info "Applying Kubernetes manifests..."
if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }
kubectl apply -f jenkins-pvc.yaml
Ok "PVC applied"
kubectl apply -f jenkins-local-updated.yaml
Ok "Jenkins deployed"
kubectl apply -f dummy-app.yaml
Ok "dummy-app deployed"

Info "Waiting for pods to be ready (up to 120s)..."
kubectl wait --for=condition=ready pod -l app=jenkins --timeout=120s
if ($LASTEXITCODE -ne 0) { Fail "Jenkins pod not ready" }
Ok "Jenkins pod ready"
kubectl wait --for=condition=ready pod -l app=dummy-app --timeout=60s
if ($LASTEXITCODE -ne 0) { Fail "dummy-app pod not ready" }
Ok "dummy-app pod ready"

# STEP 7: Port-forward Jenkins
Info "Port-forwarding Jenkins on localhost:8080..."
Get-Job -Name 'JenkinsPortForward' -ErrorAction SilentlyContinue | Stop-Job -PassThru | Remove-Job
Start-Job -Name 'JenkinsPortForward' -ScriptBlock {
    kubectl port-forward svc/jenkins 8080:8080
} | Out-Null
Start-Sleep -Seconds 3
Ok "Jenkins available at http://localhost:8080"

# STEP 8: Create Jenkins job
Info "Creating Jenkins pipeline job..."
python setup_jenkins_job.py
Ok "Jenkins job ready"

# STEP 9: Run tests
Info "Running test suite..."
python -m pytest tests/ -q
if ($LASTEXITCODE -ne 0) { Fail "Tests failed - fix before proceeding" }
Ok "All tests passing"

# STEP 10: Print launch instructions
Write-Host ""
Write-Host "  ==========================================" -ForegroundColor Green
Write-Host "  NeuroShield setup complete!" -ForegroundColor Green
Write-Host "  ==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To start the demo, open 3 terminals:" -ForegroundColor White
Write-Host ""
Write-Host "    Terminal 1 - Telemetry collector:" -ForegroundColor White
Write-Host "      python -m src.telemetry.main" -ForegroundColor Cyan
Write-Host ""
Write-Host "    Terminal 2 - Orchestrator (simulate mode):" -ForegroundColor White
Write-Host "      python -m src.orchestrator.main --mode simulate" -ForegroundColor Cyan
Write-Host ""
Write-Host "    Terminal 3 - Dashboard:" -ForegroundColor White
Write-Host "      streamlit run src/dashboard/app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "    Then open: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
