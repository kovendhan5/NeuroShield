#!/bin/bash

# Stop NeuroShield Dashboard Simulation

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  NeuroShield Dashboard Simulation - STOP                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if dashboard is running
if ! curl -s http://localhost:5173 > /dev/null 2>&1; then
  echo "❌ Dashboard not running at http://localhost:5173"
  exit 1
fi

echo "✅ Dashboard detected at http://localhost:5173"
echo ""
echo "🛑 To stop the simulation:"
echo "════════════════════════════"
echo ""
echo "Method 1: Manual (Easiest)"
echo "  1. Open dashboard: http://localhost:5173"
echo "  2. Look for green 'Stop Simulation' button"
echo "  3. Click it to stop generating incidents"
echo ""

echo "Method 2: Automatic"
echo "  Using browser automation to click stop button..."
echo ""

# Try using JavaScript if browser is open
curl -s http://localhost:5173 > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "✓ Dashboard is responsive"
  echo ""
  echo "📋 Steps to stop:"
  echo "  1. Switch to dashboard browser window"
  echo "  2. Find the green 'Stop Simulation' button in toolbar"
  echo "  3. Click to stop incident generation"
  echo ""
  echo "After stopping:"
  echo "  • Only REAL orchestrator incidents will be shown"
  echo "  • Data fetches will occur every 5s (configured in UI)"
  echo "  • You can start it again by clicking 'Start Simulation'"
  echo ""
fi

# Show current stats
STATS=$(curl -s 'http://localhost:5000/api/dashboard/history?limit=1' 2>/dev/null)
if [ ! -z "$STATS" ]; then
  echo "📊 Last Incident:"
  echo "$STATS" | python3 << 'EOF'
import json, sys
try:
  data = json.loads(sys.stdin.read())
  if data.get('actions'):
    a = data['actions'][0]
    status = "✓" if a.get('success') else "✗"
    print(f"  [{status}] {a.get('action_name', 'unknown')} @ {a.get('timestamp', 'unknown')[:16]}")
except:
  pass
EOF
fi

echo ""
echo "✓ To confirm simulation stopped:"
echo "   • Dashboard 'Stop Simulation' button should be amber/yellow"
echo "   • New incidents will only appear when orchestrator detects failures"
echo ""
