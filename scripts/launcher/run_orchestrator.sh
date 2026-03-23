#!/bin/bash
# NeuroShield Orchestrator Launcher
# Simple script to start the orchestrator main loop

set -e

# Get project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure logs directory exists
mkdir -p logs data

# Load environment from .env if it exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set defaults if not in .env
: ${JENKINS_URL:="http://localhost:8080"}
: ${PROMETHEUS_URL:="http://localhost:9090"}
: ${DUMMY_APP_URL:="http://localhost:5000"}
: ${POLL_INTERVAL:="15"}
: ${K8S_NAMESPACE:="default"}
: ${AFFECTED_SERVICE:="dummy-app"}

# Print startup info
echo "=========================================="
echo "NeuroShield Orchestrator"
echo "=========================================="
echo "Starting orchestrator with config:"
echo "  Jenkins:     $JENKINS_URL"
echo "  Prometheus:  $PROMETHEUS_URL"
echo "  Dummy App:   $DUMMY_APP_URL"
echo "  Interval:    ${POLL_INTERVAL}s"
echo "  Namespace:   $K8S_NAMESPACE"
echo ""
echo "Logs: logs/orchestrator.log"
echo "Healing history: data/healing_log.json"
echo "MTTR metrics: data/mttr_log.csv"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Run orchestrator, logging to file and console
python -m src.orchestrator.main 2>&1 | tee -a logs/orchestrator.log
