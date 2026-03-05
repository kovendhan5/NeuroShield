# Clean shutdown of all NeuroShield services
Write-Host "Stopping NeuroShield..." -ForegroundColor Cyan

docker compose down
Write-Host "[OK] Jenkins + Prometheus stopped" -ForegroundColor Green

minikube stop
Write-Host "[OK] Minikube stopped" -ForegroundColor Green

Get-Job | Stop-Job | Remove-Job
Write-Host "[OK] Background jobs cleared" -ForegroundColor Green

Write-Host "All services stopped." -ForegroundColor Green
