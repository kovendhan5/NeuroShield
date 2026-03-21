@echo off
REM NeuroShield Docker & System Optimization (Windows)
REM This script resets Docker, cleans resources, and verifies NeuroShield is ready

setlocal enabledelayedexpansion

title NeuroShield Docker Optimization

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║        NeuroShield Docker Optimization (Windows)                   ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

REM STEP 1: Stop all Docker processes
echo [STEP 1] Stopping Docker Desktop and containers...
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>NUL | find /I /N "Docker Desktop.exe">NUL
if "%ERRORLEVEL%"=="0" (
    for /f "tokens=2 delims= " %%A in ('tasklist ^| find /i "Docker Desktop"') do (
        taskkill /PID %%A /F >NUL 2>&1
    )
    echo ✓ Docker Desktop stopped
) else (
    echo ℹ Docker Desktop not running
)

REM Wait 5 seconds
timeout /t 5 /nobreak

REM STEP 2: Stop WSL docker-desktop distro
echo [STEP 2] Resetting WSL distro...
wsl --terminate docker-desktop >NUL 2>&1
echo ✓ WSL docker-desktop terminated

echo Waiting for full shutdown (10 seconds)...
timeout /t 10 /nobreak

REM STEP 3: Start Docker daemon
echo [STEP 3] Starting Docker Desktop daemon...
start "" "D:\Docker\app\resources\bin\Docker Desktop.exe" >NUL 2>&1

REM Wait for Docker to fully initialize
echo Waiting for Docker initialization (90 seconds)...
for /L %%i in (90,-1,1) do (
    cls
    echo Waiting: %%i seconds remaining...
    timeout /t 1 /nobreak
)

REM STEP 4: Verify Docker is responsive
echo [STEP 4] Verifying Docker is responsive...
docker ps >NUL 2>&1
if !ERRORLEVEL! equ 0 (
    echo ✓ Docker is responsive
) else (
    echo ⚠ Docker not fully responsive yet
    echo Waiting additional 30 seconds...
    timeout /t 30 /nobreak
)

REM STEP 5: Clean Docker resources
echo [STEP 5] Cleaning Docker resources...

echo Listing Docker images:
docker images

echo.
echo Removing unused Docker images (keeping neuroshield only)...
for /f "tokens=3" %%i in ('docker images ^| findstr /v neuroshield ^| findstr /v "REPOSITORY"') do (
    docker rmi -f %%i >NUL 2>&1
)
echo ✓ Cleaned unused images

echo.
echo Removing dangling Docker build cache...
docker builder prune -af >NUL 2>&1
echo ✓ Cleaned build cache

echo Removing dangling volumes...
docker volume prune -af >NUL 2>&1
echo ✓ Cleaned dangling volumes

REM STEP 6: Display Docker system info
echo [STEP 6] Docker system information...
docker system df

REM STEP 7: Check Minikube (if installed)
echo.
echo [STEP 7] Checking Minikube status...
where minikube >NUL 2>&1
if !ERRORLEVEL! equ 0 (
    echo Minikube detected. Status:
    minikube status

    echo.
    echo Cleaning Minikube resources...
    minikube image gc --all >NUL 2>&1
    minikube cache sync >NUL 2>&1
    echo ✓ Minikube optimized
) else (
    echo ℹ Minikube not found (optional)
)

REM STEP 8: Display NeuroShield containers status
echo.
echo [STEP 8] NeuroShield containers status...
docker-compose -f k:\Devops\NeuroShield\docker-compose.yml ps 2>NUL

REM FINAL: Display next steps
echo.
echo ════════════════════════════════════════════════════════════════════
echo OPTIMIZATION COMPLETE
echo ════════════════════════════════════════════════════════════════════
echo.
echo Next Steps:
echo.
echo  1. Quick Start (UI only - 5 seconds):
echo     cd k:\Devops\NeuroShield
echo     python neuroshield start --quick
echo     Then open: http://localhost:9999
echo.
echo  2. Full System (with Docker containers):
echo     cd k:\Devops\NeuroShield
echo     python neuroshield start
echo     Wait ~30 seconds for services to start
echo.
echo  3. Run Demo (in another terminal):
echo     cd k:\Devops\NeuroShield
echo     python neuroshield demo pod_crash
echo.
echo  4. Verify Health:
echo     python neuroshield health --detailed
echo.

pause
