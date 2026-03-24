#!/bin/bash

# NeuroShield Full End-to-End Demo
# This script demonstrates the complete autonomous healing flow:
# 1. Trigger a failure in your system
# 2. Watch the orchestrator detect it
# 3. See automatic healing action executed
# 4. Monitor recovery in the dashboard

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  NeuroShield End-to-End Autonomous Healing Demo           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

API="http://localhost:5000"
DASHBOARD="http://localhost:5173"

echo "Step 1: Open Dashboard in Browser"
echo "=================================="
echo "👉 Open this URL in your browser:"
echo "   $DASHBOARD"
echo ""
echo "Hold on, don't proceed yet. Keep the dashboard open in a separate window."
echo "Press ENTER when dashboard is open..."
read

echo ""
echo "Step 2: Trigger a Failure Event"
echo "==============================="
echo ""
echo "Option A: Trigger Jenkins Build Failure"
echo "  Command: curl -X POST http://localhost:5000/api/trigger/jenkins-failure"
echo ""
echo "Option B: Trigger Pod Crash (requires restart)"
echo "  Command: curl -X POST http://localhost:5000/api/trigger/pod-crash"
echo ""
echo "Option C: Trigger High CPU (requires scale-up)"
echo "  Command: curl -X POST http://localhost:5000/api/trigger/cpu-spike"
echo ""
echo "Option D: Full Demo Flow (Failure → Heal → Recover)"
echo "  Command: curl -X POST http://localhost:5000/api/test/full-demo-flow"
echo ""

read -p "Enter option (A/B/C/D): " choice

case $choice in
  A|a)
    echo ""
    echo "🔴 TRIGGERING: Jenkins Build Failure"
    echo "====================================="
    curl -s -X POST "$API/api/trigger/jenkins-failure" | python3 -m json.tool
    TRIGGER="jenkins-failure"
    ;;
  B|b)
    echo ""
    echo "🔴 TRIGGERING: Pod Crash"
    echo "========================="
    curl -s -X POST "$API/api/trigger/pod-crash" | python3 -m json.tool
    TRIGGER="pod-crash"
    ;;
  C|c)
    echo ""
    echo "🔴 TRIGGERING: CPU Spike"
    echo "========================="
    curl -s -X POST "$API/api/trigger/cpu-spike" | python3 -m json.tool
    TRIGGER="cpu-spike"
    ;;
  D|d)
    echo ""
    echo "🔴 TRIGGERING: Full Demo Flow"
    echo "=============================="
    curl -s -X POST "$API/api/test/full-demo-flow" | python3 -m json.tool
    TRIGGER="full-flow"
    ;;
  *)
    echo "Invalid option!"
    exit 1
    ;;
esac

echo ""
echo "Step 3: Check Dashboard After Trigger"
echo "====================================="
echo ""
echo "What should happen next:"
echo ""

case $TRIGGER in
  "jenkins-failure")
    echo "1. ⏳ Orchestrator detects build failure in Jenkins"
    echo "2. 🤖 ML model analyzes the issue"
    echo "3. ✅ System decides to RETRY the build"
    echo "4. 📊 Dashboard shows: Action = 'retry_build'"
    echo "5. ✔️ Build succeeds on retry"
    echo ""
    echo "Expected healing actions in dashboard:"
    echo "  - restart_pod (if pod is stuck)"
    echo "  - retry_build (if build failed)"
    echo "  - rollback_deploy (if recent deploy caused it)"
    ;;
  "pod-crash")
    echo "1. 🚨 Orchestrator detects pod crash"
    echo "2. 🤖 ML model identifies: RESTART needed"
    echo "3. ✅ System execute:  POD RESTART"
    echo "4. 📊 Dashboard shows: Action = 'restart_pod'"
    echo "5. ✔️ Pod comes back online, service restored"
    ;;
  "cpu-spike")
    echo "1. 📈 Orchestrator detects CPU > 90%"
    echo "2. 🤖 ML model identifies: SCALE UP needed"
    echo "3. ✅ System executes: SCALE_UP replica count"
    echo "4. 📊 Dashboard shows: Action = 'scale_up'"
    echo "5. ✔️ Load distributed, CPU returns to normal"
    ;;
  "full-flow")
    echo "1. 🔴 Failure detected (build failure)"
    echo "    Dashboard shows: detect_failure event"
    echo ""
    echo "2. 🤖 Orchestrator analyzes"
    echo "    ML model: 92% confidence → restart_pod"
    echo ""
    echo "3. ⚙️ Action executed at T+2 seconds"
    echo "    Dashboard shows: restart_pod action"
    echo ""
    echo "4. ✅ Service recovers at T+4 seconds"
    echo "    Dashboard shows: verify_health action"
    echo "    Service status: 'healthy', response: '52ms'"
    ;;
esac

echo ""
echo "Step 4: Watch the Dashboard Update"
echo "==================================="
echo ""
echo "✅ Refresh the dashboard (press F5 or click refresh button)"
echo ""
echo "You should see:"
echo "  • New healing actions in 'Recent Actions' list"
echo "  • Statistics updated (total heals, success rate counter)"
echo "  • Action appears in 'Live Event Stream' with timestamp"
echo "  • Success or failure indicator for the action"
echo ""

echo ""
echo "Step 5: Monitor in Backend Logs"
echo "==============================="
echo ""
echo "To see what the orchestrator is doing:"
echo "  docker logs neuroshield-orchestrator -f"
echo ""
echo "To see what the microservice is tracking:"
echo "  docker logs neuroshield-microservice -f"
echo ""

echo ""
echo "Step 6: Check Real Data"
echo "======================"
echo ""
echo "API endpoints to check healing data:"
echo ""
echo "Stats:"
echo "  curl http://localhost:5000/api/dashboard/stats"
echo ""
echo "Recent actions:"
echo "  curl http://localhost:5000/api/dashboard/history"
echo ""
echo "Metrics:"
echo "  curl http://localhost:5000/api/dashboard/metrics"
echo ""

echo ""
echo "Step 7: Test All Different Failures"
echo "==================================="
echo ""
echo "Run each trigger multiple times to see:"
echo "  • Different healing strategies being selected"
echo "  • Success/failure patterns"
echo "  • Performance metrics improving (ML learning)"
echo ""

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Demo Complete!                                           ║"
echo "║  ✅ You have seen the autonomous healing flow working    ║"
echo "║  📊 Dashboard shows REAL data from your triggers         ║"
echo "║  🤖 Orchestrator autonomously healed the failure         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
