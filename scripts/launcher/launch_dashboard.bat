@echo off
REM NeuroShield Dashboard Launcher
cd /d "%~dp0"
cls
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  NeuroShield Dashboard - Web UI                            ║
echo ║  Opening http://localhost:8501 in browser                 ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Dashboard features:
echo   * Real-time failure probability charts
echo   * Healing action distribution and history
echo   * Resource monitoring (CPU, memory, disk)
echo   * MTTR metrics (44%% reduction vs baseline)
echo   * Manual healing cycle trigger
echo   * Active alerts and escalation reports
echo   * Self-CI monitoring
echo.
echo Starting Streamlit dashboard on http://localhost:8501
echo.
python -m streamlit run src/dashboard/app.py
pause
