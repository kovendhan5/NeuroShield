#!/bin/bash
# NeuroShield v3 - Health Check & Status

echo "========================================="
echo "NeuroShield v3 - System Status"
echo "========================================="
echo ""

# Check containers
echo "[1] Docker Containers:"
if docker ps | grep -q neuroshield; then
    echo "  ✓ Orchestrator running"
    docker ps --filter "name=neuroshield" --format "table {{.Names}}\t{{.Status}}"
else
    echo "  ✗ Orchestrator not running"
    docker-compose ps
fi
echo ""

# Check API health
echo "[2] API Health:"
HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo 'offline')
if echo "$HEALTH" | grep -q "healthy"; then
    echo "  ✓ API is responding"
    echo "  $HEALTH" | jq '.' 2>/dev/null || echo "  $HEALTH"
else
    echo "  ✗ API is not responding"
fi
echo ""

# Check database
echo "[3] Database:"
if [ -f "data/neuroshield.db" ]; then
    SIZE=$(du -h data/neuroshield.db | cut -f1)
    TABLES=$(sqlite3 data/neuroshield.db "SELECT count(*) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
    EVENTS=$(sqlite3 data/neuroshield.db "SELECT count(*) FROM events;" 2>/dev/null || echo "0")
    ACTIONS=$(sqlite3 data/neuroshield.db "SELECT count(*) FROM actions;" 2>/dev/null || echo "0")
    echo "  ✓ Database exists"
    echo "    Size: $SIZE"
    echo "    Tables: $TABLES"
    echo "    Events: $EVENTS"
    echo "    Actions: $ACTIONS"
else
    echo "  ✗ Database not found"
fi
echo ""

# Check logs
echo "[4] Recent Logs:"
if [ -f "logs/neuroshield.log" ]; then
    echo "  ✓ Log file exists"
    tail -3 logs/neuroshield.log | sed 's/^/    /'
else
    echo "  ✗ Log file not found"
fi
echo ""

# Port availability
echo "[5] Port Status:"
if lsof -i :8000 > /dev/null 2>&1; then
    echo "  ✓ Port 8000 is in use (as expected)"
else
    echo "  ✗ Port 8000 is not listening"
fi
echo ""

# Quick stats
echo "[6] System Stats:"
if [ -f "data/neuroshield.db" ]; then
    STATUS=$(curl -s http://localhost:8000/api/status 2>/dev/null || echo 'error')
    if echo "$STATUS" | grep -q "anomaly_score"; then
        STATE=$(echo "$STATUS" | jq -r '.state' 2>/dev/null)
        ANOMALY=$(echo "$STATUS" | jq -r '.anomaly_score' 2>/dev/null)
        CPU=$(echo "$STATUS" | jq -r '.metrics.cpu_percent' 2>/dev/null)
        MEM=$(echo "$STATUS" | jq -r '.metrics.memory_percent' 2>/dev/null)
        HEALTH=$(echo "$STATUS" | jq -r '.metrics.app_health_percent' 2>/dev/null)
        echo "  State: $STATE"
        echo "  Anomaly Score: $ANOMALY"
        echo "  CPU: ${CPU}%"
        echo "  Memory: ${MEM}%"
        echo "  App Health: ${HEALTH}%"
    fi
fi
echo ""

echo "========================================="
echo "Dashboard: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "========================================="
