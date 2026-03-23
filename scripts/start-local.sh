#!/bin/bash
# NeuroShield v3 - Local Quick Start
# One command to have everything running

set -e  # Exit on error

echo "========================================="
echo "NeuroShield v3 - Local Setup"
echo "========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Install from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "[1/5] Creating directories..."
mkdir -p data logs config

echo "[2/5] Building Docker image..."
docker-compose build

echo "[3/5] Starting services..."
docker-compose up -d

echo "[4/5] Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ API failed to start"
        docker-compose logs
        exit 1
    fi
    echo "  Attempt $i/30..."
    sleep 1
done

echo "[5/5] Verification..."
HEALTH=$(curl -s http://localhost:8000/health | grep -q "healthy" && echo "OK" || echo "FAIL")
if [ "$HEALTH" = "OK" ]; then
    echo "✓ System is healthy"
else
    echo "✗ Health check failed"
    exit 1
fi

echo ""
echo "========================================="
echo "SUCCESS! NeuroShield is running"
echo "========================================="
echo ""
echo "Access the system:"
echo "  Dashboard:  http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Health:     http://localhost:8000/health"
echo ""
echo "Next steps:"
echo "  1. Open dashboard: open http://localhost:8000"
echo "  2. Run demo: python demo.py"
echo "  3. Stop system: docker-compose down"
echo ""
echo "Logs:"
echo "  tail -f logs/neuroshield.log"
echo ""
