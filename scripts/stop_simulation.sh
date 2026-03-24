#!/bin/bash
#
# Stop NeuroShield Dashboard Simulation
#

echo ""
echo "Stopping NeuroShield Dashboard Simulation"
echo "========================================"
echo ""

# Kill dashboard process if running on port 5173
echo "Stopping dashboard server (port 5173)..."
PIDS=$(lsof -ti:5173 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "Killing process: $PIDS"
    kill $PIDS 2>/dev/null
    sleep 1
fi

# Kill simulation generators
echo "Stopping any simulation generators..."
pkill -f "generate_realistic" 2>/dev/null

echo "Done!"
echo ""
echo "To restart: ./scripts/start_simulation.sh"
echo ""
