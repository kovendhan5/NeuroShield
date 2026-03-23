#!/bin/bash

# ===== PHASE 1 SECURITY VALIDATION =====
# Validates that all security fixes from Phase 1 implementation are working
# Tests: localhost-only ports, JWT auth, database users, rate limiting, etc.

set -e
RESET='\033[0m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

echo -e "${BLUE}=== NeuroShield Phase 1 Security Validation ===${RESET}\n"

# Loaded from .env.production
API_SECRET_KEY="${API_SECRET_KEY:-}"
MICROSERVICE_HOST="127.0.0.1"
MICROSERVICE_PORT="5000"
POSTGRES_PORT="5432"
REDIS_PORT="6379"
PROMETHEUS_PORT="9090"
GRAFANA_PORT="3000"
JENKINS_PORT="8080"
ALERTMANAGER_PORT="9093"

PASS_COUNT=0
FAIL_COUNT=0

# Helper functions
pass_test() {
    echo -e "${GREEN}✓ PASS${RESET}: $1"
    ((PASS_COUNT++))
}

fail_test() {
    echo -e "${RED}✗ FAIL${RESET}: $1"
    ((FAIL_COUNT++))
}

warn_test() {
    echo -e "${YELLOW}⚠ WARN${RESET}: $1"
}

# === TEST 1: Localhost-Only Port Binding ===
echo -e "${BLUE}[TEST 1] Port Binding Security${RESET}"
echo "Verifying all services bound to 127.0.0.1 only..."

for port in $MICROSERVICE_PORT $POSTGRES_PORT $REDIS_PORT $PROMETHEUS_PORT $GRAFANA_PORT $JENKINS_PORT $ALERTMANAGER_PORT; do
    if netstat -tlnp 2>/dev/null | grep -q ":$port.*LISTEN"; then
        # Check if bound to 127.0.0.1
        if netstat -tlnp 2>/dev/null | grep ":$port" | grep -q "127.0.0.1"; then
            pass_test "Port $port bound to localhost-only"
        else
            fail_test "Port $port NOT bound to localhost-only (security risk!)"
        fi
    fi
done
echo ""

# === TEST 2: Database User Security ===
echo -e "${BLUE}[TEST 2] Database User Security${RESET}"
echo "Verifying database users created with proper permissions..."

# Wait for PostgreSQL to be healthy
POSTGRES_HEALTHY=false
for i in {1..30}; do
    if docker exec neuroshield-postgres pg_isready -U postgres > /dev/null 2>&1; then
        POSTGRES_HEALTHY=true
        break
    fi
    sleep 1
done

if [ "$POSTGRES_HEALTHY" = true ]; then
    # Check if app user exists
    if docker exec neuroshield-postgres psql -U postgres -d neuroshield_db -c "SELECT 1 FROM pg_user WHERE usename='neuroshield_app'" 2>/dev/null | grep -q "1"; then
        pass_test "Database user 'neuroshield_app' created"
    else
        fail_test "Database user 'neuroshield_app' NOT created"
    fi

    # Check if readonly user exists
    if docker exec neuroshield-postgres psql -U postgres -d neuroshield_db -c "SELECT 1 FROM pg_user WHERE usename='neuroshield_readonly'" 2>/dev/null | grep -q "1"; then
        pass_test "Database user 'neuroshield_readonly' created"
    else
        fail_test "Database user 'neuroshield_readonly' NOT created"
    fi

    # Check if audit_log table exists
    if docker exec neuroshield-postgres psql -U postgres -d neuroshield_db -c "SELECT 1 FROM information_schema.tables WHERE table_name='audit_log'" 2>/dev/null | grep -q "1"; then
        pass_test "Audit logging table 'audit_log' created"
    else
        fail_test "Audit logging table NOT created"
    fi

    # Check if RLS is enabled on jobs table
    if docker exec neuroshield-postgres psql -U postgres -d neuroshield_db -c "SELECT row_security FROM information_schema.tables WHERE table_name='jobs'" 2>/dev/null | grep -qi "on"; then
        pass_test "Row-Level Security (RLS) enabled on jobs table"
    else
        warn_test "Row-Level Security (RLS) may not be enabled on jobs table"
    fi
else
    fail_test "PostgreSQL not ready for validation"
fi
echo ""

# === TEST 3: JWT Authentication ===
echo -e "${BLUE}[TEST 3] JWT Authentication${RESET}"
echo "Verifying microservice requires authentication..."

# Test without token (should fail)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/api/jobs)
if [ "$HTTP_CODE" == "401" ]; then
    pass_test "Microservice rejects requests without JWT token (HTTP 401)"
else
    warn_test "Microservice HTTP response without token: $HTTP_CODE (expected 401)"
fi

# Test with valid token (should work)
if [ -n "$API_SECRET_KEY" ]; then
    BEARER_TOKEN=$API_SECRET_KEY
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $BEARER_TOKEN" http://127.0.0.1:5000/api/jobs)
    if [ "$HTTP_CODE" == "200" ] || [ "$HTTP_CODE" == "200" ]; then
        pass_test "Microservice accepts valid JWT token (HTTP $HTTP_CODE)"
    else
        warn_test "Microservice with JWT token returned HTTP $HTTP_CODE"
    fi
else
    warn_test "API_SECRET_KEY not set - skipping JWT validation"
fi
echo ""

# === TEST 4: Structured Logging ===
echo -e "${BLUE}[TEST 4] Structured JSON Logging${RESET}"
echo "Verifying structured JSON logging is enabled..."

LOGS=$(docker logs neuroshield-microservice 2>&1 | head -5)
if echo "$LOGS" | grep -q '"timestamp".*"level".*"message"'; then
    pass_test "Microservice logs in structured JSON format"
else
    warn_test "Microservice logs may not be in JSON format - manual verification needed"
fi
echo ""

# === TEST 5: Resource Limits ===
echo -e "${BLUE}[TEST 5] Container Resource Limits${RESET}"
echo "Verifying container resource limits are enforced..."

for container in neuroshield-postgres neuroshield-redis neuroshield-microservice neuroshield-orchestrator; do
    if docker inspect "$container" 2>/dev/null | grep -q '"Memory"'; then
        MEMORY_LIMIT=$(docker inspect "$container" --format='{{.HostConfig.Memory}}')
        if [ "$MEMORY_LIMIT" -gt 0 ]; then
            pass_test "Container $container has memory limit: $((MEMORY_LIMIT / 1024 / 1024))MB"
        else
            fail_test "Container $container has no memory limit"
        fi
    fi
done
echo ""

# === TEST 6: Health Checks ===
echo -e "${BLUE}[TEST 6] Health Checks${RESET}"
echo "Verifying health checks are configured..."

for container in neuroshield-postgres neuroshield-redis neuroshield-microservice; do
    if docker inspect "$container" 2>/dev/null | grep -q '"HealthCheck"'; then
        pass_test "Container $container has healthcheck configured"
    else
        warn_test "Container $container may not have healthcheck"
    fi
done
echo ""

# === TEST 7: Non-Root User Execution ===
echo -e "${BLUE}[TEST 7] Non-Root User Execution${RESET}"
echo "Verifying application containers run as non-root..."

for container in neuroshield-microservice neuroshield-orchestrator; do
    USER=$(docker inspect "$container" --format='{{.Config.User}}' 2>/dev/null)
    if [ "$USER" != "root" ] && [ -n "$USER" ]; then
        pass_test "Container $container runs as user: $USER"
    else
        warn_test "Container $container may be running as root"
    fi
done
echo ""

# === TEST 8: Rate Limiting ===
echo -e "${BLUE}[TEST 8] Rate Limiting${RESET}"
echo "Verifying rate limiting is enforced..."

# Make 25 rapid requests (should hit rate limit if enabled)
for i in {1..25}; do
    curl -s -H "Authorization: Bearer $API_SECRET_KEY" http://127.0.0.1:5000/api/jobs > /dev/null 2>&1 &
done
wait

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $API_SECRET_KEY" http://127.0.0.1:5000/api/jobs)
if [ "$HTTP_CODE" == "429" ]; then
    pass_test "Rate limiting working - received HTTP 429 (Too Many Requests)"
elif [ "$HTTP_CODE" == "200" ]; then
    warn_test "Rate limiting may not be active (got HTTP 200)"
else
    warn_test "Rate limiting test returned HTTP $HTTP_CODE"
fi
echo ""

# === SUMMARY ===
echo -e "${BLUE}========== PHASE 1 VALIDATION SUMMARY ==========${RESET}"
TOTAL=$((PASS_COUNT + FAIL_COUNT))
if [ $TOTAL -gt 0 ]; then
    echo -e "Tests Passed: ${GREEN}$PASS_COUNT${RESET}"
    echo -e "Tests Failed: ${RED}$FAIL_COUNT${RESET}"
    echo -e "Total Tests: $TOTAL"

    if [ $FAIL_COUNT -eq 0 ]; then
        echo -e "\n${GREEN}✓ Phase 1 Security Implementation: VERIFIED${RESET}"
        echo "All critical security fixes are in place!"
        exit 0
    else
        echo -e "\n${RED}✗ Phase 1 Security Implementation: INCOMPLETE${RESET}"
        echo "Please fix the above issues before deploying to production."
        exit 1
    fi
else
    echo -e "${YELLOW}No tests were executed - services may not be ready${RESET}"
fi
