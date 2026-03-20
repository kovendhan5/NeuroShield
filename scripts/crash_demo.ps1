#!/usr/bin/env pwsh
<#
.DESCRIPTION
  Trigger crash and monitor healing across all 3 dashboards
#>

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║          Triggering NeuroShield Demo Crash                    ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "[!] Crashing IncidentBoard..."
curl.exe -X POST http://localhost:5000/crash | ConvertFrom-Json | Select-Object message, recovery_in
Write-Host ""
Write-Host "[*] Watch for 30 seconds:"
Write-Host "  • localhost:5000: Red crash screen → Green heal screen"
Write-Host "  • localhost:8503: New event card appearing"
Write-Host "  • localhost:8501: Action appears in healing table"
Write-Host ""

$start = Get-Date
$duration = 30
while ((Get-Date) - $start) -lt ([TimeSpan]::FromSeconds($duration)) {
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    Write-Progress -Activity "Monitoring healing" -SecondsRemaining ($duration - $elapsed) -PercentComplete (($elapsed / $duration) * 100)
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "[✓] Demo complete"
Write-Host "    Check if IncidentBoard recovered and Brain Feed shows new events"
