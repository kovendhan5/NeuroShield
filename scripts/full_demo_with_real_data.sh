#!/bin/bash

# NeuroShield Complete End-to-End Demo
# Shows REAL orchestrator incidents + manual triggers in dashboard

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  NeuroShield - Real Incidents + Manual Triggers Dashboard  ║"
echo "║  REAL ORCHESTRATOR INCIDENTS NOW IN DASHBOARD              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Show dashboard status
echo "STEP 1: Dashboard is Running"
echo "════════════════════════════"
echo "URL: http://localhost:5173"
echo ""
echo "⚡ ACTION: Open this URL in your browser:"
echo "   http://localhost:5173"
echo ""
read -p "Press ENTER when dashboard is open..."
echo ""

# Step 2: Show real API data
echo ""
echo "STEP 2: Real Data from API"
echo "═════════════════════════════"
echo "The dashboard fetches REAL data from the backend API every 5 seconds."
echo ""

echo "🔴 Current Stats (REAL orchestrator + manual triggers):"
API_STATS=$(curl -s http://localhost:5000/api/dashboard/stats)
echo "$API_STATS" | python3 << 'EOF'
import json, sys
data = json.loads(sys.stdin.read(-1))
print("")
print("  Total Healing Actions:  {}".format(data['total_heals']))
print("  Success Rate:           {:.1f}%".format(data['success_rate']))
print("  Cost Saved:             ${:.2f}".format(data['cost_saved']))
print("  Top Action:             {} ({} times)".format(
    max(data['action_distribution'], key=data['action_distribution'].get),
    data['action_distribution'][max(data['action_distribution'], key=data['action_distribution'].get)]
))
print("")
EOF

echo "✅ Recent Incidents (scroll down on dashboard to see):"
curl -s 'http://localhost:5000/api/dashboard/history?limit=5' | python3 << 'EOF'
import json, sys
data = json.loads(sys.stdin.read(-1))
for i, action in enumerate(data.get('actions', []), 1):
    status = "SUCCESS" if action['success'] else "FAILED"
    is_orchestrator = "Jenkins" in action.get('detail', '') or action['action_name'] == 'retry_build'
    source = "[REAL ORCHESTRATOR]" if is_orchestrator else "[Manual Trigger]"
    print("  {} {}  {}  - {}".format(i, source, action['action_name'].ljust(20), status))
EOF
echo ""

# Step 3: Trigger a real incident
echo ""
echo "STEP 3: Watch Dashboard Update with New Incidents"
echo "════════════════════════════════════════════════"
echo ""
echo "Manual Trigger Options:"
echo ""
echo "  A) Pod Crash:"
echo "     curl -X POST http://localhost:5000/api/trigger/pod-crash"
echo ""
echo "  B) Jenkins Failure:"
echo "     curl -X POST http://localhost:5000/api/trigger/jenkins-failure"
echo ""
echo "  C) CPU Spike:"
echo "     curl -X POST http://localhost:5000/api/trigger/cpu-spike"
echo ""
echo "  D) Full 3-Step Demo Flow:"
echo "     curl -X POST http://localhost:5000/api/test/full-demo-flow"
echo ""
read -p "Choose option (A/B/C/D): " choice

case $choice in
  A|a)
    echo "Triggering Pod Crash..."
    curl -X POST http://localhost:5000/api/trigger/pod-crash -s | grep -o '"status":"[^"]*"'
    ;;
  B|b)
    echo "Triggering Jenkins Build Failure..."
    curl -X POST http://localhost:5000/api/trigger/jenkins-failure -s | grep -o '"status":"[^"]*"'
    ;;
  C|c)
    echo "Triggering CPU Spike..."
    curl -X POST http://localhost:5000/api/trigger/cpu-spike -s | grep -o '"status":"[^"]*"'
    ;;
  D|d)
    echo "Triggering Full Demo Flow (Failure → Healing → Recovery)..."
    curl -X POST http://localhost:5000/api/test/full-demo-flow -s | python3 -c "import json,sys; d=json.load(sys.stdin); [print('  T+{}: {}'.format(s['time'], s['event'])) for s in d['flow']]"
    ;;
esac

echo ""
echo "STEP 4: Refresh Dashboard"
echo "═════════════════════════"
echo ""
echo "💡 New incident has been added to healing_log.json"
echo ""
echo "🔄 ACTION: Refresh dashboard in your browser (press F5 or click refresh button)"
echo ""
read -p "After refreshing, press ENTER..."
echo ""

# Show updated data
echo ""
echo "STEP 5: Verify New Incident in Dashboard"
echo "═════════════════════════════════════════"
echo ""
echo "The dashboard should now show:"
echo "  • New incident in 'Recent Actions' list"
echo "  • Updated statistics (total count increased)"
echo "  • New action state transitions"
echo ""

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Complete!                                             ║"
echo "║                                                            ║"
echo "║  Dashboard is now showing:                                ║"
echo "║  • REAL orchestrator incidents from production            ║"
echo "║  • MANUAL trigger simulations for testing                 ║"
echo "║  • All metrics from actual healing_log.json               ║"
echo "║                                                            ║"
echo "║  Refresh every 5 seconds to see live updates!             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
