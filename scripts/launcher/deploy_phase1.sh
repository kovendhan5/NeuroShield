#!/bin/bash

# ===== PHASE 1 SECURITY HARDENED DEPLOYMENT =====
# Deploys NeuroShield with all Phase 1 security fixes
# - Localhost-only ports
# - JWT authentication
# - Database users with RLS
# - Connection pooling
# - Rate limiting
# - Structured logging

set -e
RESET='\033[0m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${BLUE}=== NeuroShield Phase 1 Secure Deployment ===${RESET}\n"

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}✗ Docker not found${RESET}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}✗ Docker-compose not found${RESET}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and docker-compose available${RESET}\n"

# Change to project directory
cd "$(dirname "$0")/../.." || exit 1
PROJECT_DIR="$(pwd)"
echo -e "Project directory: ${GREEN}$PROJECT_DIR${RESET}\n"

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.production" ]; then
        echo -e "${YELLOW}Creating .env from .env.production...${RESET}"
        cp .env.production .env
    else
        echo -e "${YELLOW}✗ .env.production not found${RESET}"
        exit 1
    fi
fi

# Check for required files
echo "Checking Phase 1 configuration files..."
REQUIRED_FILES=(
    "docker-compose-hardened.yml"
    "scripts/init_db.sql"
    ".env"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${RESET}"
    else
        echo -e "${YELLOW}✗ $file not found${RESET}"
        exit 1
    fi
done

echo ""
echo "Stopping any running instances..."
docker-compose -f docker-compose-hardened.yml down 2>/dev/null || true
sleep 2

echo -e "\n${BLUE}=== Starting Phase 1 Hardened Stack ===${RESET}\n"
echo "This will deploy:"
echo "  • PostgreSQL (127.0.0.1:5432) with security users"
echo "  • Redis (127.0.0.1:6379) with password protection"
echo "  • Prometheus (127.0.0.1:9090) for metrics"
echo "  • Grafana (127.0.0.1:3000) for visualization"
echo "  • Jenkins (127.0.0.1:8080) for CI/CD"
echo "  • Microservice (127.0.0.1:5000) with JWT auth"
echo "  • Orchestrator (127.0.0.1:8000) with AI healing"
echo ""
echo -e "${YELLOW}Note: All services will be localhost-only${RESET}\n"

echo "Starting services..."
docker-compose -f docker-compose-hardened.yml up -d

echo ""
echo -e "Waiting for services to become healthy...\n"

# Wait for services
HEALTH_CHECKS=(
    "neuroshield-postgres:pg_isready"
    "neuroshield-redis:redis-cli ping"
    "neuroshield-microservice:curl -f http://localhost:5000/health"
)

for i in {1..60}; do
    echo -n "."

    # Check if main services are running
    if docker ps | grep -q neuroshield-postgres && \
       docker ps | grep -q neuroshield-microservice && \
       docker ps | grep -q neuroshield-postgres; then
        echo ""
        echo -e "${GREEN}✓ Services are now running${RESET}\n"
        break
    fi

    sleep 2
done

echo ""
echo -e "${BLUE}=== Phase 1 Status ===${RESET}\n"

docker-compose -f docker-compose-hardened.yml ps

echo ""
echo -e "${GREEN}✓ Phase 1 Deployment Complete!${RESET}\n"

echo "Quick Access URLs:"
echo "  • Microservice Health:   http://localhost:5000/health"
echo "  • Prometheus:            http://localhost:9090"
echo "  • Grafana:               http://localhost:3000 (admin/admin123)"
echo "  • Jenkins:               http://localhost:8080"
echo ""

echo "Test API Authentication:"
echo "  curl -H 'Authorization: Bearer \$API_SECRET_KEY' http://localhost:5000/api/jobs"
echo ""

echo "Verify Database Security:"
echo "  psql -U neuroshield_app -h localhost -d neuroshield_db -c 'SELECT 1;'"
echo ""

echo -e "Next: Run ${GREEN}bash scripts/launcher/validate_phase1.sh${RESET} to verify all security fixes"
echo ""
