# ═══════════════════════════════════════════════════════════════════════════════
# NeuroShield -- Single-Command Startup Script
# Usage: powershell -ExecutionPolicy Bypass -File scripts/start_neuroshield.ps1
# ═══════════════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host ""
Write-Host "  NeuroShield AIOps Platform -- Starting..." -ForegroundColor Cyan
Write-Host ""

# ─── STEP 1: Check prerequisites ─────────────────────────────────────────────
Write-Host "[1/13] Checking prerequisites..." -ForegroundColor Yellow

# Docker
try {
    docker info *>$null
    Write-Host "  [OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Docker is not running -- start Docker Desktop first" -ForegroundColor Red
    exit 1
}

# Minikube
$mkStatus = minikube status 2>&1 | Out-String
if ($mkStatus -match "apiserver: Running") {
    Write-Host "  [OK] Minikube is running" -ForegroundColor Green
} else {
    Write-Host "  [..] Starting Minikube..." -ForegroundColor Yellow
    minikube start --driver=docker --memory=3072 --cpus=2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAIL] Minikube failed to start" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Minikube started" -ForegroundColor Green
}

# Jenkins container
$jenkins = docker ps --filter "name=jenkins" --format "{{.Names}}" 2>$null
if (-not $jenkins) {
    Write-Host "  [..] Starting Jenkins + Prometheus via docker-compose..." -ForegroundColor Yellow
    docker compose up -d
    Start-Sleep -Seconds 5
} else {
    Write-Host "  [OK] Jenkins container running" -ForegroundColor Green
}

# Prometheus container
$prom = docker ps --filter "name=prometheus" --format "{{.Names}}" 2>$null
if (-not $prom) {
    Write-Host "  [..] Starting Prometheus..." -ForegroundColor Yellow
    docker compose up -d
    Start-Sleep -Seconds 3
} else {
    Write-Host "  [OK] Prometheus container running" -ForegroundColor Green
}

# ─── STEP 2: Deploy dummy-app ────────────────────────────────────────────────
Write-Host "[2/13] Deploying dummy-app to Kubernetes..." -ForegroundColor Yellow
kubectl apply -f dummy-app.yaml 2>$null | Out-Null
kubectl rollout status deployment/dummy-app --timeout=60s 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] dummy-app deployed" -ForegroundColor Green
} else {
    Write-Host "  [WARN] dummy-app rollout may not be ready" -ForegroundColor Yellow
}

# ─── STEP 3: Start port-forward ──────────────────────────────────────────────
Write-Host "[3/13] Starting kubectl port-forward..." -ForegroundColor Yellow

# Kill any existing port-forward on 5000
Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -eq "kubectl" -and $_.CommandLine -match "port-forward"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Process powershell -ArgumentList @(
    "-NoProfile", "-NoExit", "-Command",
    "Set-Location '$ProjectRoot'; kubectl port-forward svc/dummy-app 5000:5000"
) -WindowStyle Minimized
Write-Host "  [OK] Port-forward started (minimized window)" -ForegroundColor Green

# ─── STEP 4: Verify dummy-app health ─────────────────────────────────────────
Write-Host "[4/13] Waiting for dummy-app health check..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
try {
    $health = Invoke-WebRequest -Uri http://localhost:5000/health -TimeoutSec 5 -UseBasicParsing
    if ($health.StatusCode -eq 200) {
        Write-Host "  [OK] dummy-app responding on :5000" -ForegroundColor Green
    }
} catch {
    Write-Host "  [WARN] dummy-app not responding yet -- may need a few more seconds" -ForegroundColor Yellow
}

# ─── STEP 5: Start telemetry collector ────────────────────────────────────────
Write-Host "[5/13] Starting Telemetry Collector..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoProfile", "-NoExit", "-Command",
    "Set-Location '$ProjectRoot'; `$Host.UI.RawUI.WindowTitle = 'NeuroShield Telemetry'; python src/telemetry/main.py"
) -WindowStyle Normal
Write-Host "  [OK] Telemetry collector started" -ForegroundColor Green

# ─── STEP 6: Wait for first telemetry data ───────────────────────────────────
Write-Host "[6/13] Waiting 10s for telemetry to collect first data..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# ─── STEP 7: Start orchestrator ──────────────────────────────────────────────
Write-Host "[7/13] Starting Orchestrator (live mode)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoProfile", "-NoExit", "-Command",
    "Set-Location '$ProjectRoot'; `$Host.UI.RawUI.WindowTitle = 'NeuroShield Orchestrator'; python src/orchestrator/main.py --mode live"
) -WindowStyle Normal
Write-Host "  [OK] Orchestrator started" -ForegroundColor Green

# ─── STEP 8: Wait ────────────────────────────────────────────────────────────
Write-Host "[8/13] Waiting 5s for orchestrator initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# ─── STEP 9: Start dashboard ─────────────────────────────────────────────────
Write-Host "[9/13] Starting Dashboard..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoProfile", "-NoExit", "-Command",
    "Set-Location '$ProjectRoot'; `$Host.UI.RawUI.WindowTitle = 'NeuroShield Dashboard'; python -m streamlit run src/dashboard/app.py"
) -WindowStyle Normal
Write-Host "  [OK] Dashboard started" -ForegroundColor Green

# ─── STEP 10: Start REST API ─────────────────────────────────────────────────
Write-Host "[10/13] Starting REST API..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoProfile", "-NoExit", "-Command",
    "Set-Location '$ProjectRoot'; `$Host.UI.RawUI.WindowTitle = 'NeuroShield API'; python scripts/start_api.py"
) -WindowStyle Normal
Write-Host "  [OK] REST API started on :8502" -ForegroundColor Green

# ─── STEP 11: Start Brain Feed ───────────────────────────────────────────────
Write-Host "[11/13] Starting Live Brain Feed..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoProfile", "-NoExit", "-Command",
    "Set-Location '$ProjectRoot'; `$Host.UI.RawUI.WindowTitle = 'NeuroShield Brain Feed'; python scripts/live_brain_feed.py"
) -WindowStyle Normal
Write-Host "  [OK] Brain Feed started on :8503" -ForegroundColor Green

# ─── STEP 12: Open browser ───────────────────────────────────────────────────
Write-Host "[12/13] Opening dashboard in browser..." -ForegroundColor Yellow
Start-Sleep -Seconds 8
Start-Process "http://localhost:8501"

# ─── STEP 13: Success summary ────────────────────────────────────────────────
Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "   NeuroShield is now running!                " -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard:   http://localhost:8501" -ForegroundColor Cyan
Write-Host "  REST API:    http://localhost:8502" -ForegroundColor Cyan
Write-Host "  API Docs:    http://localhost:8502/docs" -ForegroundColor Cyan
Write-Host "  Brain Feed:  http://localhost:8503" -ForegroundColor Cyan
Write-Host "  Jenkins:     http://localhost:8080" -ForegroundColor Cyan
Write-Host "  Prometheus:  http://localhost:9090" -ForegroundColor Cyan
Write-Host "  Dummy-App:   http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Run demo:" -ForegroundColor Yellow
Write-Host "    python scripts/real_demo.py --scenario 1" -ForegroundColor White
Write-Host ""
Write-Host "  Stop all:" -ForegroundColor Yellow
Write-Host "    powershell -File scripts/stop_neuroshield.ps1" -ForegroundColor White
Write-Host ""
