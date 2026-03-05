# NeuroShield -- Start Services
Write-Host "Starting NeuroShield Services..." -ForegroundColor Cyan

# Create required folders
New-Item -ItemType Directory -Force -Path "D:\Docker\jenkins_home" | Out-Null
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null

# Start Jenkins + Prometheus
Write-Host "Starting Jenkins and Prometheus..." -ForegroundColor Yellow
docker compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] docker compose failed. Is Docker Desktop running?" -ForegroundColor Red
    Write-Host "Start Docker Desktop from Start Menu and wait for whale icon in taskbar" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Containers started" -ForegroundColor Green
Write-Host "[OK] Jenkins  -> http://localhost:8080" -ForegroundColor Green
Write-Host "[OK] Prometheus -> http://localhost:9090" -ForegroundColor Green

# Wait for Jenkins to be ready
Write-Host ""
Write-Host "Waiting for Jenkins to be ready (this takes 1-2 minutes)..." -ForegroundColor Yellow
$waited = 0
$ready = $false
do {
    Start-Sleep -Seconds 5
    $waited += 5
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8080/login" -TimeoutSec 3 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Write-Host "  Still waiting... ($waited s)" -ForegroundColor Gray
} while ($waited -lt 180)

if (-not $ready) {
    Write-Host "[FAIL] Jenkins did not respond in 3 minutes" -ForegroundColor Red
    Write-Host "Check logs: docker logs neuroshield-jenkins" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Jenkins is ready!" -ForegroundColor Green

# Get initial admin password
Write-Host ""
Write-Host "======================================" -ForegroundColor Magenta
Write-Host "JENKINS FIRST-TIME SETUP REQUIRED" -ForegroundColor Magenta
Write-Host "======================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Initial Admin Password:" -ForegroundColor Cyan
$password = docker exec neuroshield-jenkins cat /var/jenkins_home/secrets/initialAdminPassword 2>&1
Write-Host "  $password" -ForegroundColor Yellow
Write-Host ""
Write-Host "FOLLOW THESE STEPS IN YOUR BROWSER:" -ForegroundColor White
Write-Host "  1. Open http://localhost:8080" -ForegroundColor White
Write-Host "  2. Paste the password above" -ForegroundColor White
Write-Host "  3. Click 'Install suggested plugins' -- wait for it to finish" -ForegroundColor White
Write-Host "  4. Create admin user:" -ForegroundColor White
Write-Host "     Username : admin" -ForegroundColor Yellow
Write-Host "     Password : admin123" -ForegroundColor Yellow
Write-Host "     Full name: NeuroShield Admin" -ForegroundColor Yellow
Write-Host "  5. Jenkins URL: http://localhost:8080/ (keep as is)" -ForegroundColor White
Write-Host "  6. Click Save and Finish -> Start using Jenkins" -ForegroundColor White
Write-Host ""
Write-Host "Press ENTER when done..." -ForegroundColor Cyan
Read-Host

# Get API token
Write-Host ""
Write-Host "NOW CREATE AN API TOKEN:" -ForegroundColor Magenta
Write-Host "  1. Go to: http://localhost:8080/user/admin/configure" -ForegroundColor White
Write-Host "  2. Scroll to 'API Token' section" -ForegroundColor White
Write-Host "  3. Click 'Add new Token'" -ForegroundColor White
Write-Host "  4. Name it: neuroshield" -ForegroundColor White
Write-Host "  5. Click Generate" -ForegroundColor White
Write-Host "  6. COPY the token (you will not see it again!)" -ForegroundColor Yellow
Write-Host ""
$token = Read-Host "Paste your Jenkins API token here"

# Save token to .env
if (Test-Path ".env") {
    (Get-Content .env) -replace 'JENKINS_TOKEN=.*', "JENKINS_TOKEN=$token" | Set-Content .env
    Write-Host "[OK] Token saved to .env" -ForegroundColor Green
} else {
    Copy-Item ".env.example" ".env"
    (Get-Content .env) -replace 'JENKINS_TOKEN=.*', "JENKINS_TOKEN=$token" | Set-Content .env
    Write-Host "[OK] .env created with token" -ForegroundColor Green
}

# Create Jenkins job
Write-Host ""
Write-Host "Creating Jenkins pipeline job..." -ForegroundColor Yellow
python setup_jenkins_job.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Jenkins job created!" -ForegroundColor Green
} else {
    Write-Host "[WARN] Jenkins job creation failed -- check .env credentials" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Jenkins setup complete!" -ForegroundColor Green
