# NeuroShield — Quick Start (after first-time setup)
Write-Host "Starting NeuroShield..." -ForegroundColor Cyan

# 1. Start Docker services
Write-Host "[1/5] Starting Jenkins + Prometheus..." -ForegroundColor Yellow
docker compose up -d
Start-Sleep -Seconds 5
Write-Host "[OK] Services started" -ForegroundColor Green

# 2. Start Minikube
Write-Host "[2/5] Starting Minikube..." -ForegroundColor Yellow
$env:MINIKUBE_HOME = "D:\Docker\minikube"
$status = minikube status 2>&1
if ($status -notmatch "Running") {
    minikube start --driver=docker --memory=3072 --cpus=2
}
Write-Host "[OK] Minikube running" -ForegroundColor Green

# 3. Port-forward dummy-app
Write-Host "[3/5] Port-forwarding dummy-app..." -ForegroundColor Yellow
Start-Job -ScriptBlock {
    kubectl port-forward svc/dummy-app 5000:5000 2>&1
} | Out-Null
Start-Sleep -Seconds 2
Write-Host "[OK] dummy-app at http://localhost:5000" -ForegroundColor Green

# 4. Train models if missing
Write-Host "[4/5] Checking ML models..." -ForegroundColor Yellow
if (-not (Test-Path "models/ppo_policy.zip") -or -not (Test-Path "models/failure_predictor.pth")) {
    Write-Host "Training models (first time — takes 3-5 mins)..." -ForegroundColor Yellow
    python src/prediction/train.py
    python -m src.rl_agent.train
    Write-Host "[OK] Models trained" -ForegroundColor Green
} else {
    Write-Host "[OK] Models already trained" -ForegroundColor Green
}

# 5. Run tests
Write-Host "[5/5] Running test suite..." -ForegroundColor Yellow
python -m pytest tests/ -q --tb=short
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] All tests passing" -ForegroundColor Green
} else {
    Write-Host "[WARN] Some tests failed — check output above" -ForegroundColor Yellow
}

# Final instructions
Write-Host ""
Write-Host "══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "NeuroShield is ready!" -ForegroundColor Green
Write-Host "══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open 3 NEW PowerShell terminals:" -ForegroundColor White
Write-Host ""
Write-Host "  Terminal 1 — Telemetry Collector:" -ForegroundColor Yellow
Write-Host "    python src/telemetry/main.py" -ForegroundColor White
Write-Host ""
Write-Host "  Terminal 2 — Orchestrator (simulate):" -ForegroundColor Yellow
Write-Host "    python src/orchestrator/main.py --mode simulate" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Terminal 3 — Dashboard:" -ForegroundColor Yellow
Write-Host "    streamlit run src/dashboard/app.py" -ForegroundColor White
Write-Host ""
Write-Host "  Dashboard URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host "  Jenkins URL:   http://localhost:8080" -ForegroundColor Cyan
Write-Host "  Prometheus:    http://localhost:9090" -ForegroundColor Cyan
Write-Host "  Dummy App:     http://localhost:5000" -ForegroundColor Cyan
