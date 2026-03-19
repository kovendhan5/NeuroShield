@echo off
REM NeuroShield Complete Startup Script - LIVE Mode
REM Requires: Docker running, Jenkins + Prometheus started

setlocal enabledelayedexpansion

cd /d "%~dp0"
cls

echo.
echo ========================================
echo   NeuroShield LIVE System Startup
echo ========================================
echo.

REM Start Orchestrator in a new window
echo Launching Orchestrator (main control loop)...
start "NeuroShield Orchestrator" cmd /k "python src/orchestrator/main.py --mode live"

REM Wait a bit for orchestrator to start
timeout /t 3 /nobreak

REM Start Dashboard in a new window
echo Launching Dashboard (web UI on port 8501)...
start "NeuroShield Dashboard" cmd /k "python -m streamlit run src/dashboard/app.py"

timeout /t 3 /nobreak

REM Show status
echo.
echo ========================================
echo   System Started Successfully!
echo ========================================
echo.
echo Access Points:
echo   Dashboard:  http://localhost:8501
echo   Jenkins:    http://localhost:8080 (admin/admin123)
echo   Prometheus: http://localhost:9090
echo.
echo The orchestrator is polling Jenkins every 15 seconds.
echo The dashboard auto-refreshes every 10 seconds.
echo.
echo Watch the orchestrator window for healing action decisions.
pause
