#!/bin/bash
set -e

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

echo "🛡️  NeuroShield Local Setup"
echo "══════════════════════════"

# STEP 1: Check prerequisites
info "Checking prerequisites..."
for cmd in docker minikube kubectl python3 pip; do
    command -v $cmd &>/dev/null && ok "$cmd found" || fail "$cmd not found — please install it"
done

# STEP 2: Install Python dependencies
info "Installing Python dependencies..."
pip install -r requirements.txt -q && ok "Dependencies installed"

# STEP 3: Train models if missing
if [ ! -f "models/ppo_policy.zip" ] || [ ! -f "models/failure_predictor.pth" ]; then
    info "Training models (first time only)..."
    python3 src/prediction/train.py && ok "Prediction model trained"
    python3 -m src.rl_agent.train && ok "PPO model trained"
else
    ok "Models already exist — skipping training"
fi

# STEP 4: Start Minikube
info "Starting Minikube..."
minikube status &>/dev/null && ok "Minikube already running" || {
    minikube start --memory=4096 --cpus=2 && ok "Minikube started" || fail "Minikube failed to start"
}

# STEP 5: Build and load Docker images
info "Building Docker images..."
eval $(minikube docker-env)
docker build -t neuroshield-dummy-app:latest ./infra/dummy-app -q && ok "dummy-app image built"
docker build -t neuroshield-jenkins:latest -f ./infra/jenkins-builder/Dockerfile.jenkins ./infra/jenkins-builder -q && ok "jenkins image built"

# STEP 6: Apply Kubernetes manifests
info "Applying Kubernetes manifests..."
mkdir -p data
kubectl apply -f jenkins-pvc.yaml && ok "PVC applied"
kubectl apply -f jenkins-local-updated.yaml && ok "Jenkins deployed"
kubectl apply -f dummy-app.yaml && ok "dummy-app deployed"

info "Waiting for pods to be ready (up to 120s)..."
kubectl wait --for=condition=ready pod -l app=jenkins --timeout=120s && ok "Jenkins pod ready" || fail "Jenkins pod not ready"
kubectl wait --for=condition=ready pod -l app=dummy-app --timeout=60s && ok "dummy-app pod ready" || fail "dummy-app pod not ready"

# STEP 7: Port-forward Jenkins
info "Port-forwarding Jenkins on localhost:8080..."
pkill -f "kubectl port-forward.*8080" 2>/dev/null || true
kubectl port-forward svc/jenkins 8080:8080 &>/dev/null &
sleep 3 && ok "Jenkins available at http://localhost:8080"

# STEP 8: Create Jenkins job
info "Creating Jenkins pipeline job..."
python3 setup_jenkins_job.py && ok "Jenkins job ready"

# STEP 9: Run tests
info "Running test suite..."
python3 -m pytest tests/ -q && ok "All tests passing" || fail "Tests failed — fix before proceeding"

# STEP 10: Print launch instructions
echo ""
echo "══════════════════════════════════════════"
echo -e "${GREEN}✅ NeuroShield setup complete!${NC}"
echo "══════════════════════════════════════════"
echo ""
echo "To start the demo, open 3 terminals:"
echo ""
echo "  Terminal 1 — Telemetry collector:"
echo "    python3 -m src.telemetry.main"
echo ""
echo "  Terminal 2 — Orchestrator (simulate mode):"
echo "    python3 -m src.orchestrator.main --mode simulate"
echo ""
echo "  Terminal 3 — Dashboard:"
echo "    python -m streamlit run src/dashboard/app.py"
echo ""
echo "  Then open: http://localhost:8501"
echo ""
