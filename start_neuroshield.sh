#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         NeuroShield - Self-Healing CI/CD System            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo "▶ $1"
    echo "─────────────────────────────────────────────────────────"
}

# Check if Docker daemon is running
print_section "1. Checking Docker"
max_attempts=30
attempt=0
while ! docker ps > /dev/null 2>&1 && [ $attempt -lt $max_attempts ]; do
    echo "  ⏳ Waiting for Docker daemon... ($((attempt+1))/$max_attempts)"
    sleep 2
    ((attempt++))
done

if ! docker ps > /dev/null 2>&1; then
    echo "  ❌ Docker daemon not responding. Please start Docker Desktop manually."
    exit 1
fi
echo "  ✓ Docker is running"

# Start infrastructure
print_section "2. Starting Infrastructure (Jenkins + Prometheus)"
docker compose up -d
echo "  ✓ Jenkins: http://localhost:8080 (admin/admin123)"
echo "  ✓ Prometheus: http://localhost:9090"

# Wait for services to be ready
print_section "3. Waiting for Services to Initialize"
attempt=0
while [ $attempt -lt 60 ]; do
    if curl -s http://localhost:8080/login > /dev/null 2>&1; then
        echo "  ✓ Jenkins is ready"
        break
    fi
    echo "  ⏳ Waiting for Jenkins... ($((attempt+1))/60)"
    sleep 2
    ((attempt++))
done

if [ $attempt -eq 60 ]; then
    echo "  ⚠️  Jenkins took longer than expected"
fi

# Setup Jenkins job if needed
print_section "4. Setting Up Jenkins Job"
if ! python scripts/setup_jenkins_job.py 2>/dev/null; then
    echo "  ⚠️  Jenkins job setup encountered an issue (this may be normal if already exists)"
else
    echo "  ✓ Jenkins job configured"
fi

# Show summary
print_section "5. Project Ready!"
echo ""
echo "  📊 Streamlit Dashboard:    http://localhost:8501"
echo "  🔧 Jenkins:                http://localhost:8080"
echo "  📈 Prometheus:             http://localhost:9090"
echo "  🤖 Dummy App:              http://localhost:5000"
echo ""
echo "To launch the full system, run in separate terminals:"
echo ""
echo "  Terminal 1 (Orchestrator):"
echo "  python src/orchestrator/main.py --mode live"
echo ""
echo "  Terminal 2 (Dashboard):"
echo "  python -m streamlit run src/dashboard/app.py"
echo ""
echo "  Terminal 3 (Real Demo - Optional):"
echo "  python scripts/real_demo.py"
echo ""
echo "╚════════════════════════════════════════════════════════════╝"
