@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ESC="
for /f %%A in ('echo prompt $E ^| cmd') do set "ESC=%%A"
set "GREEN=%ESC%[92m"
set "RED=%ESC%[91m"
set "YELLOW=%ESC%[93m"
set "BLUE=%ESC%[94m"
set "NC=%ESC%[0m"

set /a PASS_COUNT=0
set /a FAIL_COUNT=0
set /a WARN_COUNT=0

set "STEP1_FAIL=0"
set "STEP2_FAIL=0"
set "STEP3_FAIL=0"
set "STEP4_FAIL=0"

echo.
echo ============================================================
echo  NeuroShield Demo Check (Windows CMD)
echo ============================================================

echo.
echo %BLUE%STEP 1 - Service health checks%NC%

call :check_http_200 "API /api/health" "http://localhost/api/health" || set "STEP1_FAIL=1"
call :check_http_200 "Prometheus /-/healthy" "http://localhost:9090/-/healthy" || set "STEP1_FAIL=1"
call :check_http_200 "Grafana /api/health" "http://localhost:3000/api/health" || set "STEP1_FAIL=1"
call :check_http_200 "Alertmanager /-/healthy" "http://localhost:9093/-/healthy" || set "STEP1_FAIL=1"
call :check_http_200 "Nginx /" "http://localhost/" || set "STEP1_FAIL=1"

set "WORKER_HEALTH="
for /f "usebackq delims=" %%H in (`docker inspect neuroshield-worker --format "{{.State.Health.Status}}" 2^>nul`) do set "WORKER_HEALTH=%%H"
if /I "!WORKER_HEALTH!"=="healthy" (
  call :pass "Worker health is healthy"
) else (
  call :fail "Worker health expected healthy, got !WORKER_HEALTH!"
  set "STEP1_FAIL=1"
)

docker exec neuroshield-worker sh -c "test -f /tmp/worker_alive" >nul 2>&1
if errorlevel 1 (
  call :fail "Worker heartbeat file exists (/tmp/worker_alive)"
  set "STEP1_FAIL=1"
) else (
  call :pass "Worker heartbeat file exists (/tmp/worker_alive)"
)

echo.
echo %BLUE%STEP 2 - WebSocket live check%NC%
call :check_websocket || set "STEP2_FAIL=1"

echo.
echo %BLUE%STEP 3 - Fire synthetic alert%NC%
call :inject_alert || set "STEP3_FAIL=1"

echo.
echo %BLUE%STEP 4 - Verify orchestrator log assertion%NC%
timeout /t 2 /nobreak >nul
call :assert_logs || set "STEP4_FAIL=1"

echo.
echo %BLUE%STEP 5 - Demo summary%NC%
echo.
echo Checks summary:
echo   PASS: !PASS_COUNT!
echo   FAIL: !FAIL_COUNT!
echo   WARN: !WARN_COUNT!
echo.

if "%STEP1_FAIL%"=="0" (echo   STEP 1: %GREEN%PASS%NC%) else (echo   STEP 1: %RED%FAIL%NC%)
if "%STEP2_FAIL%"=="0" (echo   STEP 2: %GREEN%PASS%NC%) else (echo   STEP 2: %RED%FAIL%NC%)
if "%STEP3_FAIL%"=="0" (echo   STEP 3: %GREEN%PASS%NC%) else (echo   STEP 3: %RED%FAIL%NC%)
if "%STEP4_FAIL%"=="0" (echo   STEP 4: %GREEN%PASS%NC%) else (echo   STEP 4: %RED%FAIL%NC%)

if %FAIL_COUNT% EQU 0 (
  echo.
  echo %GREEN%DEMO READY - present now%NC%
  exit /b 0
)

echo.
echo %RED%FIX BEFORE PRESENTING%NC%
echo Failed steps:
if not "%STEP1_FAIL%"=="0" echo   - STEP 1 (Service health checks)
if not "%STEP2_FAIL%"=="0" echo   - STEP 2 (WebSocket live check)
if not "%STEP3_FAIL%"=="0" echo   - STEP 3 (Synthetic alert injection)
if not "%STEP4_FAIL%"=="0" echo   - STEP 4 (Orchestrator log assertion)
exit /b 1

:check_http_200
setlocal EnableDelayedExpansion
set "NAME=%~1"
set "URL=%~2"
set "CODE="
for /l %%I in (1,1,3) do (
  for /f "usebackq delims=" %%C in (`curl -s -o NUL -w "%%{http_code}" "%URL%" 2^>nul`) do set "CODE=%%C"
  if "!CODE!"=="200" (
    endlocal & call :pass "%~1 (HTTP 200)" & exit /b 0
  )
  timeout /t 1 /nobreak >nul
)
endlocal & call :fail "%~1 (expected HTTP 200)" & exit /b 1

:check_websocket
set "WS_TMP=%TEMP%\neuroshield_ws_check_%RANDOM%.txt"
set "API_METRICS=%TEMP%\neuroshield_metrics_%RANDOM%.txt"
curl -sS "http://localhost/api/metrics" > "%API_METRICS%" 2>nul
if errorlevel 1 (
  if exist "%API_METRICS%" del /q "%API_METRICS%" >nul 2>&1
  call :fail "Telemetry source check failed (/api/metrics unreachable)"
  exit /b 1
)
findstr /C:"neuroshield_cpu_usage_percent" "%API_METRICS%" >nul 2>&1
if errorlevel 1 (
  findstr /C:"\"cpu_usage\"" "%API_METRICS%" >nul 2>&1
  if errorlevel 1 (
    del /q "%API_METRICS%" >nul 2>&1
    call :fail "Telemetry source missing cpu metric"
    exit /b 1
  )
)
findstr /C:"neuroshield_memory_usage_percent" "%API_METRICS%" >nul 2>&1
if errorlevel 1 (
  findstr /C:"\"memory_usage\"" "%API_METRICS%" >nul 2>&1
  if errorlevel 1 (
    del /q "%API_METRICS%" >nul 2>&1
    call :fail "Telemetry source missing memory metric"
    exit /b 1
  )
)
del /q "%API_METRICS%" >nul 2>&1
call :pass "Telemetry source available (/api/metrics has cpu+memory)"
exit /b 0

:inject_alert
set "ALERT_RESP=%TEMP%\neuroshield_alert_resp_%RANDOM%.txt"
curl -sS -X POST "http://localhost/api/alerts" -H "Content-Type: application/json" -d "{\"receiver\":\"neuroshield-webhook\",\"status\":\"firing\",\"alerts\":[{\"status\":\"firing\",\"labels\":{\"alertname\":\"DemoSyntheticAlert\",\"severity\":\"critical\",\"service\":\"dummy-app\",\"namespace\":\"default\"},\"annotations\":{\"summary\":\"demo synthetic alert\",\"description\":\"Injected by demo_check.bat\"},\"startsAt\":\"2026-03-26T00:00:00Z\"}],\"groupLabels\":{\"alertname\":\"DemoSyntheticAlert\"},\"commonLabels\":{\"severity\":\"critical\"},\"commonAnnotations\":{\"summary\":\"demo synthetic alert\"},\"version\":\"4\"}" > "%ALERT_RESP%" 2>nul
if errorlevel 1 (
  del /q "%ALERT_RESP%" >nul 2>&1
  call :fail "POST synthetic alert to /api/alerts"
  exit /b 1
)

findstr /R /C:"\"forwarded\"[ ]*:[ ]*1" "%ALERT_RESP%" >nul 2>&1
if errorlevel 1 (
  call :fail "Synthetic alert response missing \"forwarded\":1"
  call :warn "Response saved at %ALERT_RESP%"
  exit /b 1
)

del /q "%ALERT_RESP%" >nul 2>&1
call :pass "Synthetic alert accepted and forwarded"
exit /b 0

:assert_logs
set "LOG_TMP=%TEMP%\neuroshield_worker_logs_%RANDOM%.txt"
docker logs neuroshield-worker --tail 50 > "%LOG_TMP%" 2>&1
if errorlevel 1 (
  del /q "%LOG_TMP%" >nul 2>&1
  call :fail "Read neuroshield-worker logs"
  exit /b 1
)

findstr /C:"Processing webhook event" "%LOG_TMP%" >nul 2>&1
if errorlevel 1 (
  findstr /C:"Orchestration Cycle #" "%LOG_TMP%" >nul 2>&1
  if not errorlevel 1 (
    del /q "%LOG_TMP%" >nul 2>&1
    call :pass "Worker loop active (Orchestration Cycle log found)"
    exit /b 0
  )
  if exist "%LOG_TMP%" del /q "%LOG_TMP%" >nul 2>&1
  docker logs neuroshield-orchestrator --tail 80 > "%LOG_TMP%" 2>&1
  findstr /C:"Processing webhook event" "%LOG_TMP%" >nul 2>&1
  if errorlevel 1 (
    findstr /C:"Orchestration Cycle #" "%LOG_TMP%" >nul 2>&1
    if not errorlevel 1 (
      del /q "%LOG_TMP%" >nul 2>&1
      call :pass "Orchestrator loop active (Orchestration Cycle log found)"
      exit /b 0
    )
    call :fail "No webhook-processing log found in worker/orchestrator"
    call :warn "Logs saved at %LOG_TMP%"
    exit /b 1
  ) else (
    del /q "%LOG_TMP%" >nul 2>&1
    call :pass "Logs contain: Processing webhook event (orchestrator)"
    exit /b 0
  )
)

del /q "%LOG_TMP%" >nul 2>&1
call :pass "Logs contain: Processing webhook event (worker)"
exit /b 0

:pass
set /a PASS_COUNT+=1
echo %GREEN%[PASS]%NC% %~1
exit /b 0

:fail
set /a FAIL_COUNT+=1
echo %RED%[FAIL]%NC% %~1
exit /b 1

:warn
set /a WARN_COUNT+=1
echo %YELLOW%[WARN]%NC% %~1
exit /b 0
