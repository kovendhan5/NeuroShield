#!/usr/bin/env pwsh
<#
.DESCRIPTION
  NeuroShield Demo — Start all services and open dashboards
#>

$ErrorActionPreference = "SilentlyContinue"

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║         NeuroShield AIOps Demo — Starting                     ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

# Call start_neuroshield.ps1 if it exists
$startScript = Join-Path -Path (Split-Path -Parent $PSScriptRoot) -ChildPath "scripts/launcher/start_neuroshield.ps1"
if (Test-Path $startScript) {
    Write-Host "[*] Starting services via $startScript"
    & $startScript
} else {
    Write-Host "[*] Assuming services are already running..."
}

Write-Host ""
Write-Host "[*] Waiting for services to be ready..."

$ports = @(5000, 8501, 8503)
$maxWait = 30
$elapsed = 0

foreach ($port in $ports) {
    $ready = $false
    while ($elapsed -lt $maxWait -and -not $ready) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -TimeoutSec 2 -ErrorAction Stop
            $ready = $true
            Write-Host "[✓] Port $port is responding"
        } catch {
            Start-Sleep -Seconds 1
            $elapsed++
        }
    }
    if (-not $ready) {
        Write-Host "[!] Port $port did not respond after $maxWait seconds"
    }
}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║          NeuroShield Demo Ready                               ║"
Write-Host "╠════════════════════════════════════════════════════════════════╣"
Write-Host "║  App:        http://localhost:5000  (IncidentBoard)            ║"
Write-Host "║  Brain Feed: http://localhost:8503  (Live AI Feed)             ║"
Write-Host "║  Dashboard:  http://localhost:8501  (Streamlit Dashboard)     ║"
Write-Host "╠════════════════════════════════════════════════════════════════╣"
Write-Host "║  Trigger Crash:                                               ║"
Write-Host "║    curl.exe -X POST http://localhost:5000/crash               ║"
Write-Host "║                                                                ║"
Write-Host "║  Watch all 3 screens. NeuroShield heals in ~25 seconds         ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

# Open browsers
Write-Host "[*] Opening browser tabs..."
Start-Process "http://localhost:5000"
Start-Sleep -Milliseconds 500
Start-Process "http://localhost:8503"
Start-Sleep -Milliseconds 500
Start-Process "http://localhost:8501"

Write-Host "[✓] Demo ready. Check your browser tabs."
