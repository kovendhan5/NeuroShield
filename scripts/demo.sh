#!/bin/bash

# NeuroShield Complete Demo - Real Incidents + Simulated Incidents
# Shows both real orchestrator healing and dashboard simulation

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                  NeuroShield Complete Demo                             ║"
echo "║         Real Orchestrator Incidents + Dashboard Simulation             ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Verify services
echo "STEP 1: Verifying Services"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

SERVICES_OK=true

# Check backend API
echo -n "Checking Backend API (localhost:5000)... "
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
  echo "✅ OK"
else
  echo "❌ FAILED"
  SERVICES_OK=false
fi

# Check dashboard dev server
echo -n "Checking Dashboard Dev Server (localhost:5173)... "
if curl -s http://localhost:5173 > /dev/null 2>&1; then
  echo "✅ OK"
else
  echo "⚠️  OFFLINE (will start it)"
fi

# Check Docker services
echo -n "Checking Docker Services (Orchestrator)... "
if docker ps 2>/dev/null | grep -q "neuroshield-orchestrator"; then
  echo "✅ OK"
else
  echo "⚠️  Not running (is optional for this demo)"
fi

echo ""

if [ "$SERVICES_OK" = false ]; then
  echo "❌ Required services are not running"
  echo ""
  echo "Start with:"
  echo "  cd k:\\Devops\\NeuroShield"
  echo "  docker-compose -f docker-compose-hardened.yml up -d"
  echo "  npm run dev  # in separate terminal"
  echo ""
  exit 1
fi

# Step 2: Show current data
echo "STEP 2: Current System Status"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

STATS=$(curl -s http://localhost:5000/api/dashboard/stats 2>/dev/null)
echo "📊 Real Orchestrator + Simulated Incidents:"
echo "$STATS" | python3 << 'EOF'
import json, sys
data = json.loads(sys.stdin.read())
print(f"  • Total Actions:  {data.get('total_heals', 0)}")
print(f"  • Success Rate:   {data.get('success_rate', 0):.1f}%")
print(f"  • Failed:         {data.get('failed_actions', 0)}")
print(f"  • Cost Saved:     ₹{data.get('cost_saved', 0):.2f}")
print(f"  • ML Confidence:  {data.get('ml_confidence', 0):.1f}%")

actions = data.get('action_distribution', {})
print(f"\n  Top Actions:")
for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True)[:5]:
  print(f"    - {action}: {count} times")
EOF

echo ""

# Step 3: Open dashboard and start simulation
echo "STEP 3: Launch Dashboard & Simulation"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

echo "Opening dashboard in browser..."
echo "URL: http://localhost:5173"
echo ""

# Open in browser
if command -v xdg-open > /dev/null 2>&1; then
  xdg-open "http://localhost:5173" 2>/dev/null &
elif command -v open > /dev/null 2>&1; then
  open "http://localhost:5173" 2>/dev/null &
elif command -v start > /dev/null 2>&1; then
  start "http://localhost:5173" 2>/dev/null
fi

echo "⏳ Waiting 3 seconds for dashboard to load..."
sleep 3

echo ""
echo "📝 Dashboard is now open. You should see:"
echo "   ✓ KPI Cards: Total Actions, Success Rate, Cost Saved, Avg Response Time"
echo "   ✓ Recent Actions Table: Shows incidents with status"
echo "   ✓ Control Panel: With 'Start Simulation' button"
echo ""

# Step 4: Show simulation controls
echo "STEP 4: Dashboard Configuration"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

echo "✓ Simulation is ENABLED by default"
echo ""
echo "Dashboard Controls Available:"
echo "  • Start/Stop Simulation: Toggle incident generation (default: ON)"
echo "  • Refresh: Manually fetch latest data"
echo "  • Update Frequency: 1s, 5s, 10s, 30s (default: 5s)"
echo "  • Theme: Toggle dark/light mode"
echo ""

# Step 5: Show trigger options
echo "STEP 5: Trigger Real Incidents (Optional)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

echo "To test with REAL incidents, trigger a Jenkins failure:"
echo ""
echo "  curl -X POST http://localhost:5000/api/trigger/jenkins-failure"
echo ""
echo "Or other incident types:"
echo "  • Pod Crash:   curl -X POST http://localhost:5000/api/trigger/pod-crash"
echo "  • CPU Spike:   curl -X POST http://localhost:5000/api/trigger/cpu-spike"
echo "  • Full Flow:   curl -X POST http://localhost:5000/api/test/full-demo-flow"
echo ""

# Step 6: Live monitoring
echo "STEP 6: Live Monitoring"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

echo "Monitoring dashboard metrics for 2 minutes..."
echo "Press Ctrl+C to stop monitoring"
echo ""

MONITOR_TIME=120
ELAPSED=0
INTERVAL=15
FIRST=true

while [ $ELAPSED -lt $MONITOR_TIME ]; do
  if [ "$FIRST" = false ]; then
    sleep $INTERVAL
  fi
  FIRST=false
  ELAPSED=$((ELAPSED + INTERVAL))

  STATS=$(curl -s http://localhost:5000/api/dashboard/stats 2>/dev/null)
  HISTORY=$(curl -s 'http://localhost:5000/api/dashboard/history?limit=1' 2>/dev/null)

  echo "$STATS" "$HISTORY" | python3 << 'EOF'
import json, sys, datetime
lines = sys.stdin.read().split('\n')
stats_line = lines[0]
history_line = lines[1] if len(lines) > 1 else "{}"

stats = json.loads(stats_line) if stats_line else {}
history = json.loads(history_line) if history_line else {}

ts = datetime.datetime.now().strftime("%H:%M:%S")
actions = stats.get('total_heals', 0)
success = stats.get('success_rate', 0)
saved = stats.get('cost_saved', 0)

last_action = "N/A"
if history.get('actions'):
  a = history['actions'][0]
  status = "✓" if a.get('success') else "✗"
  last_action = f"[{status}] {a.get('action_name', 'unknown')}"

print(f"[{ts}] Actions: {actions:3d} | Success: {success:5.1f}% | Saved: ₹{saved:7.0f} | Latest: {last_action}")
EOF
done

echo ""
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Demo Complete                                                     ║"
echo "║                                                                        ║"
echo "║  Dashboard is running with:                                           ║"
echo "║  • REAL incidents from orchestrator                                   ║"
echo "║  • SIMULATED incidents (3 sec interval)                               ║"
echo "║  • Auto-refreshing metrics (5 sec interval)                           ║"
echo "║                                                                        ║"
echo "║  To stop simulation:                                                  ║"
echo "║  • Click 'Stop Simulation' button in dashboard, OR                    ║"
echo "║  • Run: bash scripts/stop_simulation.sh                               ║"
echo "║                                                                        ║"
echo "║  Dashboard: http://localhost:5173                                     ║"
echo "║  API: http://localhost:5000                                           ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
