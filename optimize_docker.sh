#!/bin/bash
# NeuroShield Docker & Minikube Optimization Script
# Run this to clean up Docker and optimize the system

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        NeuroShield Docker & Minikube Optimization                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

# STEP 1: Reset WSL and Docker (if Docker is unresponsive)
echo "STEP 1: Resetting Docker..."
wsl --shutdown
sleep 5
echo "✓ WSL shutdown complete"

# STEP 2: Wait for Docker to restart
echo
echo "STEP 2: Waiting for Docker to restart (60 seconds)..."
sleep 60
echo "✓ Docker restarted"

# STEP 3: List Docker images and volumes
echo
echo "STEP 3: Current Docker state..."
docker images --format "table {{.Repository}}\t{{.Size}}"
echo

# STEP 4: Remove unwanted Docker images (keep only neuroshield images)
echo "STEP 4: Removing unwanted images..."
docker images | grep -v neuroshield | grep -v "IMAGE ID" | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null
echo "✓ Cleaned unused images"

# STEP 5: Remove dangling layers, build cache, and volumes
echo
echo "STEP 5: Removing dangling Docker resources..."
docker builder prune -af 2>/dev/null || true
docker image prune -af 2>/dev/null || true
docker volume prune -af 2>/dev/null || true
echo "✓ Cleaned dangling resources"

# STEP 6: Check Minikube volumes
echo
echo "STEP 6: Checking Minikube status..."
if command -v minikube &> /dev/null; then
    minikube status
    echo
    echo "Minikube disk usage:"
    minikube ssh "df -h /" 2>/dev/null || echo "Could not access minikube"
    echo
    echo "Cleaning Minikube cache..."
    minikube cache sync 2>/dev/null || true
    minikube image gc --all 2>/dev/null || true
    echo "✓ Minikube optimized"
else
    echo "ℹ Minikube not installed or in PATH"
fi

# STEP 7: Display final Docker stats
echo
echo "STEP 7: Final Docker resource usage..."
docker system df 2>/dev/null || echo "Warning: docker system df unavailable during daemon start"

# STEP 8: Start NeuroShield
echo
echo "════════════════════════════════════════════════════════════════════"
echo "OPTIMIZATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════"
echo
echo "To start NeuroShield:"
echo "  python neuroshield start              # Full system"
echo "  python neuroshield start --quick      # UI only (faster)"
echo
echo "To verify it's working:"
echo "  python neuroshield health --detailed"
echo "  docker-compose ps"
echo

