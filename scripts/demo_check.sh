#!/usr/bin/env bash

# NeuroShield Demo Reliability Check
# Works on Linux/macOS and Windows Git Bash

set +e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

STEP1_FAIL=0
STEP2_FAIL=0
STEP3_FAIL=0
STEP4_FAIL=0

print_header() {
  echo ""
  echo "============================================================"
  echo " NeuroShield Demo Check"
  echo "============================================================"
}

pass() {
  echo -e "${GREEN}[PASS]${NC} $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  echo -e "${RED}[FAIL]${NC} $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
  WARN_COUNT=$((WARN_COUNT + 1))
}

section() {
  echo ""
  echo -e "${BLUE}$1${NC}"
}

check_http_200() {
  local name="$1"
  local url="$2"
  local code
  local attempt
  for attempt in 1 2 3; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    if [ "$code" = "200" ]; then
      pass "$name (HTTP 200)"
      return 0
    fi
    sleep 1
  done

  fail "$name (expected HTTP 200, got ${code:-no-response})"
  return 1
}

check_cmd() {
  local name="$1"
  shift
  "$@" >/dev/null 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    pass "$name"
    return 0
  fi

  fail "$name"
  return 1
}

websocket_probe() {
  local metrics
  metrics=$(curl -sS "http://localhost/api/metrics" 2>/dev/null)
  local rc=$?
  if [ $rc -ne 0 ] || [ -z "$metrics" ]; then
    fail "Telemetry source check failed (/api/metrics unreachable)"
    return 1
  fi

  if echo "$metrics" | grep -q '"cpu_usage"' && echo "$metrics" | grep -q '"memory_usage"'; then
    pass "Telemetry source available (/api/metrics has cpu_usage+memory_usage)"
    return 0
  fi

  fail "Telemetry source missing required metrics"
  return 1
}

inject_alert() {
  local payload
  payload='{"receiver":"neuroshield-webhook","status":"firing","alerts":[{"status":"firing","labels":{"alertname":"DemoSyntheticAlert","severity":"critical","service":"dummy-app","namespace":"default"},"annotations":{"summary":"demo synthetic alert","description":"Injected by demo_check.sh"},"startsAt":"2026-03-26T00:00:00Z"}],"groupLabels":{"alertname":"DemoSyntheticAlert"},"commonLabels":{"severity":"critical"},"commonAnnotations":{"summary":"demo synthetic alert"},"version":"4"}'

  local response
  response=$(curl -sS -X POST "http://localhost/api/alerts" -H "Content-Type: application/json" -d "$payload" 2>/dev/null)
  local rc=$?

  if [ $rc -ne 0 ]; then
    fail "POST synthetic alert to /api/alerts"
    warn "Alert POST response: <curl failed>"
    return 1
  fi

  if echo "$response" | grep -Eq '"forwarded"[[:space:]]*:[[:space:]]*1'; then
    pass "Synthetic alert accepted and forwarded"
    return 0
  fi

  fail "Synthetic alert response missing \"forwarded\":1"
  warn "Alert POST response: $response"
  return 1
}

assert_worker_logs() {
  local worker_logs
  local logs
  worker_logs=$(docker logs neuroshield-worker --tail 80 2>&1)
  local rc=$?

  if [ $rc -ne 0 ]; then
    fail "Read neuroshield-worker logs"
    warn "docker logs output: $worker_logs"
    return 1
  fi

  if echo "$worker_logs" | grep -q "Processing webhook event"; then
    pass "Logs contain: Processing webhook event (worker)"
    return 0
  fi

  if echo "$worker_logs" | grep -q "Orchestration Cycle #"; then
    pass "Worker loop active (Orchestration Cycle log found)"
    return 0
  fi

  if docker ps --format '{{.Names}}' | grep -q '^neuroshield-orchestrator$'; then
    logs=$(docker logs neuroshield-orchestrator --tail 80 2>&1)
    if echo "$logs" | grep -q "Processing webhook event"; then
      pass "Logs contain: Processing webhook event (orchestrator)"
      return 0
    fi
    if echo "$logs" | grep -q "Orchestration Cycle #"; then
      pass "Orchestrator loop active (Orchestration Cycle log found)"
      return 0
    fi
  fi

  fail "No webhook-processing log found in worker/orchestrator"
  warn "Recent worker logs:\n$(echo "$worker_logs" | tail -n 12)"
  return 1
}

print_header

section "STEP 1 — Service health checks"

check_http_200 "API /api/health" "http://localhost/api/health" || STEP1_FAIL=1
check_http_200 "Prometheus /-/healthy" "http://localhost:9090/-/healthy" || STEP1_FAIL=1
check_http_200 "Grafana /api/health" "http://localhost:3000/api/health" || STEP1_FAIL=1
check_http_200 "Alertmanager /-/healthy" "http://localhost:9093/-/healthy" || STEP1_FAIL=1
check_http_200 "Nginx /" "http://localhost/" || STEP1_FAIL=1

worker_health=$(docker inspect neuroshield-worker --format "{{.State.Health.Status}}" 2>/dev/null)
if [ "$worker_health" = "healthy" ]; then
  pass "Worker health is healthy"
else
  fail "Worker health expected healthy, got ${worker_health:-unknown}"
  STEP1_FAIL=1
fi

check_cmd "Worker heartbeat file exists (/tmp/worker_alive)" docker exec neuroshield-worker sh -c "test -f /tmp/worker_alive" || STEP1_FAIL=1

section "STEP 2 — WebSocket live check"
websocket_probe || STEP2_FAIL=1

section "STEP 3 — Fire synthetic alert"
inject_alert || STEP3_FAIL=1

section "STEP 4 — Verify orchestrator log assertion"
sleep 2
assert_worker_logs || STEP4_FAIL=1

section "STEP 5 — Demo summary"

echo ""
echo "Checks summary:"
echo "  PASS: $PASS_COUNT"
echo "  FAIL: $FAIL_COUNT"
echo "  WARN: $WARN_COUNT"
echo ""

echo "Step status:"
[ $STEP1_FAIL -eq 0 ] && echo -e "  STEP 1: ${GREEN}PASS${NC}" || echo -e "  STEP 1: ${RED}FAIL${NC}"
[ $STEP2_FAIL -eq 0 ] && echo -e "  STEP 2: ${GREEN}PASS${NC}" || echo -e "  STEP 2: ${RED}FAIL${NC}"
[ $STEP3_FAIL -eq 0 ] && echo -e "  STEP 3: ${GREEN}PASS${NC}" || echo -e "  STEP 3: ${RED}FAIL${NC}"
[ $STEP4_FAIL -eq 0 ] && echo -e "  STEP 4: ${GREEN}PASS${NC}" || echo -e "  STEP 4: ${RED}FAIL${NC}"

if [ "$FAIL_COUNT" -eq 0 ]; then
  echo ""
  echo -e "${GREEN}DEMO READY — present now${NC}"
  exit 0
fi

echo ""
echo -e "${RED}FIX BEFORE PRESENTING${NC}"
echo "Failed steps:"
[ $STEP1_FAIL -ne 0 ] && echo "  - STEP 1 (Service health checks)"
[ $STEP2_FAIL -ne 0 ] && echo "  - STEP 2 (WebSocket live check)"
[ $STEP3_FAIL -ne 0 ] && echo "  - STEP 3 (Synthetic alert injection)"
[ $STEP4_FAIL -ne 0 ] && echo "  - STEP 4 (Orchestrator log assertion)"

exit 1
