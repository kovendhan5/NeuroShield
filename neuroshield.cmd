@echo off
REM NeuroShield CLI - Windows batch wrapper
REM This wrapper allows "neuroshield" command to work on Windows

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Run the Python script with all arguments passed through
python "%SCRIPT_DIR%neuroshield" %*

REM Preserve exit code
exit /b %errorlevel%
