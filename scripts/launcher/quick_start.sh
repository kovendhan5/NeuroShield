#!/bin/bash
# NeuroShield Quick Start - Environment Validation + Orchestrator Launch
# This script verifies all dependencies and starts the orchestrator

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

: ${JENKINS_URL:="http://localhost:8080"}
: ${PROMETHEUS_URL:="http://localhost:9090"}
: ${DUMMY_APP_URL:="http://localhost:5000"}
: ${K8S_NAMESPACE:="default"}

echo ""
echo "=================================================="
echo "NeuroShield Quick Start - Environment Check"
echo "=================================================="
echo ""

# Check function
check_service() {
    local name=$1
    local url=$2
    local timeout=$3

    echo -n "Checking $name ... "
    if timeout $timeout curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Online${NC}"
        return 0
    else
        echo -e "${RED}✗ Offline${NC}"
        return 1
    fi
}

# Check function for CLI tools
check_command() {
    local cmd=$1
    echo -n "Checking $cmd ... "
    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}✓ Available${NC}"
        return 0
    else
        echo -e "${RED}✗ Not found${NC}"
        return 1
    fi
}

# Track failures
FAILED=0

# Check prerequisites
echo "Prerequisites:"
check_command "python" || FAILED=$((FAILED+1))
check_command "kubectl" || FAILED=$((FAILED+1))
check_command "docker" || FAILED=$((FAILED+1))

echo ""
echo "Services:"
check_service "Jenkins" "$JENKINS_URL/api/json" 5 || FAILED=$((FAILED+1))
check_service "Prometheus" "$PROMETHEUS_URL/-/healthy" 5 || FAILED=$((FAILED+1))
check_service "Dummy App" "$DUMMY_APP_URL/health" 5 || FAILED=$((FAILED+1))

echo ""
echo "Kubernetes:"
echo -n "Checking kubectl cluster access ... "
if kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Connected${NC}"
else
    echo -e "${RED}✗ Not connected${NC}"
    FAILED=$((FAILED+1))
fi

echo -n "Checking namespace '$K8S_NAMESPACE' ... "
if kubectl get namespace "$K8S_NAMESPACE" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Exists${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    FAILED=$((FAILED+1))
fi

echo ""
echo "NeuroShield:"
echo -n "Checking Python dependencies ... "
if python -c "import numpy, requests, torch, transformers, stable_baselines3" 2>/dev/null; then
    echo -e "${GREEN}✓ All packages installed${NC}"
else
    echo -e "${RED}✗ Missing dependencies${NC}"
    FAILED=$((FAILED+1))
fi

echo ""
echo "=================================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Starting orchestrator...${NC}"
    echo ""
    exec bash "$PROJECT_ROOT/scripts/launcher/run_orchestrator.sh"
else
    echo -e "${RED}$FAILED check(s) failed. Cannot start orchestrator.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure Jenkins is running: $JENKINS_URL"
    echo "  2. Ensure Prometheus is running: $PROMETHEUS_URL"
    echo "  3. Ensure kubectl can access your cluster: kubectl cluster-info"
    echo "  4. Install missing Python dependencies: pip install -r requirements.txt"
    echo ""
    echo "See docs/TROUBLESHOOTING.md for more help."
    exit 1
fi
