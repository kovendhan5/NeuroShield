#!/usr/bin/env bash
# NeuroShield Production Startup Script
# Starts all services with proper initialization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           NeuroShield Production Deployment               ║"
echo "║       AI-Powered CI/CD Self-Healing Platform             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found${NC}"
    echo ""
    echo "Creating .env from template..."

    if [ -f .env.production ]; then
        cp .env.production .env
        echo -e "${GREEN}✅ Created .env file${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env and update the following:${NC}"
        echo "   - DB_ADMIN_PASSWORD"
        echo "   - DB_USER_PASSWORD"
        echo "   - REDIS_PASSWORD"
        echo "   - API_SECRET_KEY"
        echo "   - GRAFANA_PASSWORD"
        echo "   - GRAFANA_SECRET_KEY"
        echo "   - JENKINS_PASSWORD (if using Jenkins)"
        echo ""
        echo "Generate secure passwords with:"
        echo "  openssl rand -base64 32"
        echo ""
        read -p "Press Enter after updating .env file..."
    else
        echo -e "${RED}❌ .env.production template not found${NC}"
        exit 1
    fi
fi

# Validate required environment variables
echo ""
echo -e "${BLUE}Validating configuration...${NC}"

source .env

required_vars=(
    "DB_ADMIN_PASSWORD"
    "DB_USER_PASSWORD"
    "REDIS_PASSWORD"
    "API_SECRET_KEY"
    "GRAFANA_PASSWORD"
    "GRAFANA_SECRET_KEY"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ] || [ "${!var}" == "CHANGE_ME"* ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing or invalid configuration:${NC}"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please update .env with secure values"
    exit 1
fi

echo -e "${GREEN}✅ Configuration valid${NC}"

# Check Docker
echo ""
echo -e "${BLUE}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon not running. Please start Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker ready${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

# Determine compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

echo -e "${GREEN}✅ Docker Compose ready${NC}"

# Create required directories
echo ""
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p data logs models
mkdir -p data/archive
mkdir -p logs/archive
echo -e "${GREEN}✅ Directories created${NC}"

# Pull/build images
echo ""
echo -e "${BLUE}Building Docker images...${NC}"
$COMPOSE_CMD -f docker-compose.production.yml build --pull

# Start services
echo ""
echo -e "${BLUE}Starting services...${NC}"
$COMPOSE_CMD -f docker-compose.production.yml up -d

# Wait for services to be healthy
echo ""
echo -e "${BLUE}Waiting for services to be healthy...${NC}"
echo "(This may take 1-2 minutes)"

max_wait=120
elapsed=0
interval=5

while [ $elapsed -lt $max_wait ]; do
    healthy=$($COMPOSE_CMD -f docker-compose.production.yml ps | grep -c "(healthy)" || true)
    total=$($COMPOSE_CMD -f docker-compose.production.yml ps | grep -c "Up" || true)

    echo -ne "\r${YELLOW}⏳ $healthy/$total services healthy (${elapsed}s/${max_wait}s)${NC}"

    if [ "$healthy" -ge 5 ]; then
        echo ""
        echo -e "${GREEN}✅ Core services are healthy!${NC}"
        break
    fi

    sleep $interval
    elapsed=$((elapsed + interval))
done

echo ""

# Show service status
echo ""
echo -e "${BLUE}Service Status:${NC}"
$COMPOSE_CMD -f docker-compose.production.yml ps

# Show access URLs
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║             NeuroShield is Running!                       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Dashboard:${NC}      http://localhost:8501"
echo -e "${BLUE}🔌 API:${NC}            http://localhost:8000"
echo -e "${BLUE}📚 API Docs:${NC}       http://localhost:8000/docs"
echo -e "${BLUE}📈 Grafana:${NC}        http://localhost:3000 (admin / see .env)"
echo -e "${BLUE}🔧 Jenkins:${NC}        http://localhost:8080"
echo -e "${BLUE}📊 Prometheus:${NC}     http://localhost:9090"
echo ""
echo -e "${YELLOW}📝 View logs:${NC}"
echo "   $COMPOSE_CMD -f docker-compose.production.yml logs -f"
echo ""
echo -e "${YELLOW}🛑 Stop services:${NC}"
echo "   $COMPOSE_CMD -f docker-compose.production.yml down"
echo ""
echo -e "${GREEN}✨ Happy DevOps! ✨${NC}"
