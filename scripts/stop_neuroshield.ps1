# ═══════════════════════════════════════════════════════════════════════════════
# NeuroShield -- Clean Shutdown Script
# Usage: powershell -ExecutionPolicy Bypass -File scripts/stop_neuroshield.ps1
# ═══════════════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "SilentlyContinue"

Write-Host ""
Write-Host "  Stopping NeuroShield..." -ForegroundColor Cyan
Write-Host ""

# Stop Streamlit dashboard
$streamlit = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -eq "streamlit" -or
    ($_.ProcessName -eq "python" -and $_.CommandLine -match "streamlit")
}
if ($streamlit) {
    $streamlit | Stop-Process -Force
    Write-Host "  [OK] Dashboard (Streamlit) stopped" -ForegroundColor Green
} else {
    Write-Host "  [--] Dashboard not running" -ForegroundColor Gray
}

# Stop Orchestrator
$orch = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -eq "python" -and $_.CommandLine -match "orchestrator"
}
if ($orch) {
    $orch | Stop-Process -Force
    Write-Host "  [OK] Orchestrator stopped" -ForegroundColor Green
} else {
    Write-Host "  [--] Orchestrator not running" -ForegroundColor Gray
}

# Stop Telemetry collector
$telem = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -eq "python" -and $_.CommandLine -match "telemetry"
}
if ($telem) {
    $telem | Stop-Process -Force
    Write-Host "  [OK] Telemetry collector stopped" -ForegroundColor Green
} else {
    Write-Host "  [--] Telemetry not running" -ForegroundColor Gray
}

# Stop kubectl port-forward
$portfwd = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -eq "kubectl" -and $_.CommandLine -match "port-forward"
}
if ($portfwd) {
    $portfwd | Stop-Process -Force
    Write-Host "  [OK] Port-forward stopped" -ForegroundColor Green
} else {
    Write-Host "  [--] Port-forward not running" -ForegroundColor Gray
}

# Kill any NeuroShield-titled PowerShell windows
Get-Process powershell -ErrorAction SilentlyContinue | Where-Object {
    $_.MainWindowTitle -match "NeuroShield"
} | Stop-Process -Force -ErrorAction SilentlyContinue

# Clean up background jobs
Get-Job -ErrorAction SilentlyContinue | Stop-Job -ErrorAction SilentlyContinue
Get-Job -ErrorAction SilentlyContinue | Remove-Job -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "  NeuroShield stopped." -ForegroundColor Green
Write-Host "  (Docker containers and Minikube left running)" -ForegroundColor Gray
Write-Host ""
