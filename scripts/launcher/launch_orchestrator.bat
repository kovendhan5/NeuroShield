@echo off
REM NeuroShield Orchestrator Launcher (Simulation Mode)
cd /d "%~dp0"
cls
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  NeuroShield Orchestrator - Simulation Mode                ║
echo ║  (Synthetic data demonstrating PPO RL + DistilBERT)        ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Starting orchestrator...
echo The system will:
echo   * Poll telemetry every 15 seconds (simulated)
echo   * Build 52D state vectors
echo   * Run DistilBERT log analysis
echo   * Predict failures with neural network
echo   * Execute PPO RL agent - selects 1 of 6 healing actions
echo   * Log all decisions to data/healing_log.json
echo.
python src/orchestrator/main.py --mode simulate
pause
