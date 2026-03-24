#!/bin/bash
#
# Start NeuroShield Dashboard Simulation
#

echo ""
echo "Starting NeuroShield Dashboard with Live Simulation"
echo "===================================================="
echo ""

# Generate realistic scenarios
echo "[1/4] Generating realistic 24-hour scenario..."
python3 scripts/generate_realistic_scenario.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate scenarios"
    exit 1
fi
echo ""

# Verify backend
echo "[2/4] Verifying backend services..."
curl -s http://localhost:5000/health > /dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Microservice not responding on port 5000"
    exit 1
fi
echo "OK: Microservice running"
echo ""

# Show metrics
echo "[3/4] System metrics loaded:"
curl -s http://localhost:5000/api/dashboard/system-metrics | head -c 100
echo "..."
echo ""

echo "[4/4] Dashboard ready!"
echo ""
echo "Opening: http://localhost:5173"
echo ""
echo "Controls:"
echo "  - Click 'Start Simulation' button to enable"
echo "  - Click 'Stop Simulation' button to disable"
echo "  - View realistic CPU/memory spikes triggering incidents"
echo "  - Monitor healing actions in real-time"
echo ""
