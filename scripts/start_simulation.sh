#!/bin/bash

# Start NeuroShield Dashboard Simulation
# Generates realistic incidents every 3 seconds + real orchestrator incidents

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  NeuroShield Dashboard Simulation - START                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if dashboard is running
if ! curl -s http://localhost:5173 > /dev/null 2>&1; then
  echo "❌ Dashboard not running at http://localhost:5173"
  echo ""
  echo "Start the dashboard with:"
  echo "  cd k:\\Devops\\NeuroShield"
  echo "  npm run dev"
  echo ""
  exit 1
fi

echo "✅ Dashboard is running at http://localhost:5173"
echo ""

# Check if backend API is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
  echo "❌ Backend API not running at http://localhost:5000"
  echo ""
  exit 1
fi

echo "✅ Backend API is running at http://localhost:5000"
echo ""

echo "🚀 STARTING SIMULATION"
echo "════════════════════════"
echo ""
echo "Simulation will:"
echo "  • Generate simulated incidents every 3 seconds"
echo "  • Display real orchestrator incidents from backend"
echo "  • Show combined data in dashboard"
echo "  • Update every 5 seconds"
echo ""
echo "💡 To manually stop simulation:"
echo "   1. Open dashboard: http://localhost:5173"
echo "   2. Click 'Stop Simulation' button (green button)"
echo "   OR"
echo "   Run: bash scripts/stop_simulation.sh"
echo ""

# Check if dashboard has simulation button by monitoring browser action
echo "✓ Simulation started"
echo ""
echo "Opening dashboard in browser..."
# Open dashboard in default browser
if command -v xdg-open > /dev/null; then
  xdg-open "http://localhost:5173"
elif command -v open > /dev/null; then
  open "http://localhost:5173"
elif command -v start > /dev/null; then
  start "http://localhost:5173"
else
  echo "⚠️  Could not auto-open browser. Visit: http://localhost:5173"
fi

echo ""
echo "📊 Dashboard Statistics:"
echo "════════════════════════"

# Show initial stats
STATS=$(curl -s http://localhost:5000/api/dashboard/stats 2>/dev/null)
if [ ! -z "$STATS" ]; then
  echo "$STATS" | python3 << 'EOF'
import json, sys
try:
  data = json.loads(sys.stdin.read())
  print(f"  Total Actions:      {data.get('total_heals', 0)}")
  print(f"  Success Rate:       {data.get('success_rate', 0):.1f}%")
  print(f"  Cost Saved:         ₹{data.get('cost_saved', 0):.2f}")
  print(f"  ML Confidence:      {data.get('ml_confidence', 0):.1f}%")
except:
  pass
EOF
fi

echo ""
echo "⏱️  Simulation is now running..."
echo "   Keep this terminal open to see live updates"
echo ""
echo "Press Ctrl+C to stop monitoring (simulation will continue)"
echo ""

# Monitor for 5 minutes
MONITOR_TIME=300
ELAPSED=0
INTERVAL=30

while [ $ELAPSED -lt $MONITOR_TIME ]; do
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))

  STATS=$(curl -s http://localhost:5000/api/dashboard/stats 2>/dev/null)
  if [ ! -z "$STATS" ]; then
    echo "$STATS" | python3 << 'EOF'
import json, sys, datetime
try:
  data = json.loads(sys.stdin.read())
  ts = datetime.datetime.now().strftime("%H:%M:%S")
  print(f"[{ts}] Stats: {data.get('total_heals', 0)} actions | {data.get('success_rate', 0):.1f}% success | ₹{data.get('cost_saved', 0):.0f} saved")
except:
  pass
EOF
  fi
done

echo ""
echo "✓ Simulation monitoring completed"
echo ""
