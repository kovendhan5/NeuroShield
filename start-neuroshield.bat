@echo off
REM NeuroShield - Windows starter script
REM Usage: neuroshield-start.cmd [--quick|full]

setlocal enabledelayedexpansion

echo.
echo [96m=====================================================================[0m
echo  [1mNeuroShield v2.1.0 - Starting System[0m
echo [96m=====================================================================[0m
echo.

if "%1"=="--quick" (
    echo [93m[*] Quick mode: Starting UI only (5 seconds)[0m
    python scripts/manage.py start --quick
    timeout /t 3 /nobreak
    start http://localhost:9999
) else (
    echo [93m[*] Full mode: Starting all services[0m
    docker-compose -f docker-compose.yml up -d
    timeout /t 5 /nobreak
    echo.
    echo [92m[+] System starting. Check http://localhost:9999[0m
    echo [92m[+] Or run: neuroshield status[0m
)

echo.
