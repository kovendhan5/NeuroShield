#!/bin/bash
# NeuroShield Docker & Project Verification

echo "=== NeuroShield Docker & Project Status ==="
echo ""

# Wait for Docker
echo "[1/5] Waiting for Docker to be ready..."
max_attempts=30
attempt=0
while ! docker ps &>/dev/null; do
  attempt=$((attempt + 1))
  if [ $attempt -gt $max_attempts ]; then
    echo "[FAIL] Docker did not respond after 30 seconds"
    exit 1
  fi
  echo "  Attempt $attempt/30..."
  sleep 1
done
echo "[OK] Docker is responsive"

# Check Docker root directory
echo ""
echo "[2/5] Checking Docker storage location..."
docker_root=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk -F': ' '{print $2}')
echo "  Docker Root Dir: $docker_root"
if [[ "$docker_root" == *"k:"* ]] || [[ "$docker_root" == *"K:"* ]]; then
  echo "[OK] Using K: drive (SUCCESS!)"
else
  echo "[WARN] Not on K: drive, but system is working"
fi

# List Docker images
echo ""
echo "[3/5] Docker images available:"
docker images --format "table {{.Repository}}\t{{.Size}}" 2>&1 | head -5

# Check project structure
echo ""
echo "[4/5] NeuroShield project structure:"
if [ -d "src/orchestrator" ]; then
  echo "[OK] src/orchestrator/"
fi
if [ -d "src/prediction" ]; then
  echo "[OK] src/prediction/"
fi
if [ -d "src/telemetry" ]; then
  echo "[OK] src/telemetry/"
fi
if [ -d "src/rl_agent" ]; then
  echo "[OK] src/rl_agent/"
fi
if [ -d "src/dashboard" ]; then
  echo "[OK] src/dashboard/"
fi

# Test Python imports
echo ""
echo "[5/5] Testing Python core imports..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    from src.orchestrator.main import determine_healing_action
    print("  [OK] Orchestrator")
except Exception as e:
    print(f"  [FAIL] Orchestrator: {e}")

try:
    from src.prediction.predictor import FailurePredictor
    print("  [OK] Predictor")
except Exception as e:
    print(f"  [FAIL] Predictor: {e}")

try:
    from src.telemetry.collector import TelemetryCollector
    print("  [OK] Telemetry")
except Exception as e:
    print(f"  [FAIL] Telemetry: {e}")
PYEOF

echo ""
echo "=== READY TO RUN NEUROSHIELD ==="
echo ""
echo "Next: python run_local_demo.py"
