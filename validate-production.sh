#!/usr/bin/env bash
# NeuroShield Production System Validation Script
# Comprehensive validation of all services before merge

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
ISSUES=()

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
    ISSUES+=("$1")
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

run_test() {
    ((TESTS_RUN++))
    echo ""
    log_info "Test $TESTS_RUN: $1"
}

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       NeuroShield Production Validation Suite            ║"
echo "║           Comprehensive System Testing                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
log_info "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    log_fail "Docker not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    log_fail "Docker daemon not running"
    exit 1
fi

if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    log_fail "Docker Compose not available"
    exit 1
fi

log_success "Prerequisites met (Docker + Compose)"

# Setup test environment
log_info "Setting up test environment..."
if [ ! -f .env ]; then
    if [ -f .env.production ]; then
        cp .env.production .env
    else
        log_fail ".env.production template not found"
        exit 1
    fi

    # Set minimal test values
    sed -i 's/CHANGE_ME_STRONG_ADMIN_PASSWORD/test_admin_password_123/g' .env
    sed -i 's/CHANGE_ME_STRONG_APP_PASSWORD/test_app_password_123/g' .env
    sed -i 's/CHANGE_ME_STRONG_REDIS_PASSWORD/test_redis_password_123/g' .env
    sed -i 's/CHANGE_ME_GENERATE_RANDOM_256_BIT_KEY/test_secret_key_12345678901234567890/g' .env
    sed -i 's/CHANGE_ME_STRONG_GRAFANA_PASSWORD/test_grafana_password_123/g' .env

    log_info "Created test .env file"
fi

# ============================================================================
# TEST 1: Container Health
# ============================================================================
run_test "Container Health - Start all services"

log_info "Starting services with docker-compose..."
$COMPOSE_CMD -f docker-compose.production.yml up -d 2>&1 | tee /tmp/docker-compose-start.log

if [ $? -eq 0 ]; then
    log_success "Docker Compose started successfully"
else
    log_fail "Docker Compose failed to start"
    cat /tmp/docker-compose-start.log
    exit 1
fi

# Wait for services to initialize
log_info "Waiting 60 seconds for services to initialize..."
sleep 60

# ============================================================================
# TEST 2: Check running containers
# ============================================================================
run_test "Container Health - All services running"

EXPECTED_SERVICES=(
    "neuroshield-api"
    "neuroshield-worker"
    "neuroshield-dashboard"
    "neuroshield-postgres"
    "neuroshield-redis"
)

RUNNING_SERVICES=$($COMPOSE_CMD -f docker-compose.production.yml ps --services --filter "status=running")

for service in "${EXPECTED_SERVICES[@]}"; do
    if echo "$RUNNING_SERVICES" | grep -q "$service"; then
        log_success "Service $service is running"
    else
        log_fail "Service $service is NOT running"
    fi
done

# ============================================================================
# TEST 3: Check for restart loops
# ============================================================================
run_test "Container Health - No restart loops"

for service in "${EXPECTED_SERVICES[@]}"; do
    RESTART_COUNT=$(docker inspect --format='{{.RestartCount}}' "$service" 2>/dev/null || echo "0")
    if [ "$RESTART_COUNT" -gt 3 ]; then
        log_fail "Service $service has restarted $RESTART_COUNT times (possible crash loop)"
    else
        log_success "Service $service is stable (restarts: $RESTART_COUNT)"
    fi
done

# ============================================================================
# TEST 4: Service Connectivity - API
# ============================================================================
run_test "Service Connectivity - API accessible (port 8000)"

sleep 5  # Give API time to be ready

API_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")

if [ "$API_HEALTH" = "200" ]; then
    log_success "API health endpoint responding (HTTP 200)"
else
    log_fail "API health endpoint failed (HTTP $API_HEALTH)"
fi

# Test API status endpoint
API_STATUS=$(curl -s http://localhost:8000/api/status 2>/dev/null)
if echo "$API_STATUS" | grep -q "status"; then
    log_success "API status endpoint responding with valid JSON"
else
    log_fail "API status endpoint not responding properly"
fi

# ============================================================================
# TEST 5: Service Connectivity - Dashboard
# ============================================================================
run_test "Service Connectivity - Dashboard accessible (port 8501)"

DASHBOARD_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/_stcore/health 2>/dev/null || echo "000")

if [ "$DASHBOARD_HEALTH" = "200" ]; then
    log_success "Dashboard health endpoint responding (HTTP 200)"
else
    log_fail "Dashboard health endpoint failed (HTTP $DASHBOARD_HEALTH)"
fi

# ============================================================================
# TEST 6: Database connectivity
# ============================================================================
run_test "Service Connectivity - PostgreSQL connection"

DB_TEST=$(docker exec neuroshield-postgres psql -U postgres -d neuroshield_db -c "SELECT 1" 2>&1)

if echo "$DB_TEST" | grep -q "1 row"; then
    log_success "PostgreSQL connection successful"
else
    log_fail "PostgreSQL connection failed"
fi

# ============================================================================
# TEST 7: Redis connectivity
# ============================================================================
run_test "Service Connectivity - Redis connection"

REDIS_TEST=$(docker exec neuroshield-redis redis-cli --pass test_redis_password_123 PING 2>&1)

if echo "$REDIS_TEST" | grep -q "PONG"; then
    log_success "Redis connection successful"
else
    log_fail "Redis connection failed"
fi

# ============================================================================
# TEST 8: Worker behavior - Check logs
# ============================================================================
run_test "Worker Behavior - Continuous monitoring"

log_info "Checking worker logs for activity..."
WORKER_LOGS=$(docker logs neuroshield-worker 2>&1 | tail -100)

if echo "$WORKER_LOGS" | grep -q "Orchestration Cycle"; then
    log_success "Worker is running monitoring cycles"
else
    log_fail "Worker does not appear to be running monitoring cycles"
fi

if echo "$WORKER_LOGS" | grep -qi "error\|exception\|traceback" | grep -v "No module"; then
    log_warning "Worker logs contain errors (may be expected during startup)"
fi

# ============================================================================
# TEST 9: Worker behavior - No memory leaks
# ============================================================================
run_test "Worker Behavior - Memory usage check"

WORKER_MEM=$(docker stats neuroshield-worker --no-stream --format "{{.MemUsage}}" | awk '{print $1}')
log_info "Worker memory usage: $WORKER_MEM"

# Extract numeric value (remove MiB/GiB suffix)
MEM_VALUE=$(echo "$WORKER_MEM" | sed 's/[^0-9.]//g')
if (( $(echo "$MEM_VALUE < 1000" | bc -l) )); then
    log_success "Worker memory usage is normal (< 1GB)"
else
    log_fail "Worker memory usage is high (> 1GB)"
fi

# ============================================================================
# TEST 10: API endpoints test
# ============================================================================
run_test "API Functionality - Test key endpoints"

# Test /api/events
EVENTS=$(curl -s http://localhost:8000/api/events?limit=5 2>/dev/null)
if [ -n "$EVENTS" ]; then
    log_success "API /api/events endpoint responding"
else
    log_fail "API /api/events endpoint not responding"
fi

# Test /docs (OpenAPI)
DOCS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs 2>/dev/null)
if [ "$DOCS" = "200" ]; then
    log_success "API documentation (/docs) accessible"
else
    log_fail "API documentation not accessible"
fi

# ============================================================================
# TEST 11: Check service logs for errors
# ============================================================================
run_test "System Health - Check for critical errors in logs"

CRITICAL_ERRORS=0

for service in api worker dashboard; do
    log_info "Checking $service logs..."
    ERRORS=$(docker logs neuroshield-$service 2>&1 | grep -i "critical\|fatal" | wc -l)

    if [ "$ERRORS" -gt 0 ]; then
        log_warning "Found $ERRORS critical/fatal errors in $service logs"
        CRITICAL_ERRORS=$((CRITICAL_ERRORS + ERRORS))
    fi
done

if [ "$CRITICAL_ERRORS" -eq 0 ]; then
    log_success "No critical errors found in service logs"
else
    log_warning "Found $CRITICAL_ERRORS critical/fatal log entries"
fi

# ============================================================================
# TEST 12: Resource usage check
# ============================================================================
run_test "Resource Usage - CPU and Memory"

log_info "Collecting resource usage statistics..."
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | tee /tmp/resource-usage.txt

# Check if any service is using > 80% CPU
HIGH_CPU=$(cat /tmp/resource-usage.txt | awk '{print $2}' | sed 's/%//' | awk '$1 > 80 {print}' | wc -l)

if [ "$HIGH_CPU" -gt 0 ]; then
    log_warning "Some services using > 80% CPU"
else
    log_success "All services have normal CPU usage (< 80%)"
fi

# ============================================================================
# TEST 13: Network connectivity between services
# ============================================================================
run_test "Network Connectivity - Inter-service communication"

# Test API -> PostgreSQL
API_TO_DB=$(docker exec neuroshield-api sh -c "nc -zv postgres 5432 2>&1" | grep -c "open" || echo "0")
if [ "$API_TO_DB" -gt 0 ]; then
    log_success "API can reach PostgreSQL"
else
    log_fail "API cannot reach PostgreSQL"
fi

# Test Worker -> Redis
WORKER_TO_REDIS=$(docker exec neuroshield-worker sh -c "nc -zv redis 6379 2>&1" | grep -c "open" || echo "0")
if [ "$WORKER_TO_REDIS" -gt 0 ]; then
    log_success "Worker can reach Redis"
else
    log_fail "Worker cannot reach Redis"
fi

# ============================================================================
# TEST 14: Configuration validation
# ============================================================================
run_test "Configuration - Environment variables loaded"

# Check if worker has required env vars
ENV_CHECK=$(docker exec neuroshield-worker env | grep -E "JENKINS_URL|PROMETHEUS_URL|DATABASE_URL" | wc -l)

if [ "$ENV_CHECK" -ge 2 ]; then
    log_success "Worker has required environment variables"
else
    log_fail "Worker missing required environment variables"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    VALIDATION SUMMARY${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Tests Run:    $TESTS_RUN"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  ✅ READY TO MERGE                        ║${NC}"
    echo -e "${GREEN}║          All validation tests passed!                    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    VERDICT="READY TO MERGE"
    EXIT_CODE=0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                  ⚠️  NEEDS FIXES                          ║${NC}"
    echo -e "${RED}║          Some validation tests failed                    ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Issues Found:${NC}"
    for issue in "${ISSUES[@]}"; do
        echo "  • $issue"
    done
    VERDICT="NEEDS FIXES"
    EXIT_CODE=1
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Generate validation report
cat > /tmp/validation-report.txt <<EOF
NeuroShield Production Validation Report
Generated: $(date)

SUMMARY
=======
Tests Run: $TESTS_RUN
Tests Passed: $TESTS_PASSED
Tests Failed: $TESTS_FAILED

VERDICT: $VERDICT

ISSUES FOUND
============
EOF

for issue in "${ISSUES[@]}"; do
    echo "- $issue" >> /tmp/validation-report.txt
done

cat >> /tmp/validation-report.txt <<EOF

SERVICE STATUS
==============
EOF

$COMPOSE_CMD -f docker-compose.production.yml ps >> /tmp/validation-report.txt

cat >> /tmp/validation-report.txt <<EOF

RESOURCE USAGE
==============
EOF

cat /tmp/resource-usage.txt >> /tmp/validation-report.txt

log_info "Validation report saved to /tmp/validation-report.txt"

# Keep containers running for inspection
log_info "Containers are still running for inspection"
log_info "To stop: docker-compose -f docker-compose.production.yml down"

exit $EXIT_CODE
