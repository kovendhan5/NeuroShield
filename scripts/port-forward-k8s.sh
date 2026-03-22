#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     NeuroShield Kubernetes Port Forwarding Setup            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting port forwards..."
echo ""

# Function to start port forward in background
start_port_forward() {
  local service=$1
  local namespace=$2
  local local_port=$3
  local remote_port=$4
  
  kubectl port-forward -n $namespace svc/$service $local_port:$remote_port &
  echo "✅ $service: http://localhost:$local_port"
}

# Start all port forwards
start_port_forward "web-service" "neuroshield-prod" "8080" "80"
start_port_forward "api-service" "neuroshield-prod" "5000" "5000"
start_port_forward "prometheus-service" "neuroshield-prod" "9090" "9090"
start_port_forward "grafana-service" "neuroshield-prod" "3000" "3000"
start_port_forward "alertmanager-service" "neuroshield-prod" "9093" "9093"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     📊 KUBERNETES SERVICES ARE NOW ACCESSIBLE              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Services:"
echo "  • Web Dashboard:     http://localhost:8080"
echo "  • API Service:       http://localhost:5000/health"
echo "  • Prometheus:        http://localhost:9090"
echo "  • Grafana:           http://localhost:3000 (admin/admin)"
echo "  • Alertmanager:      http://localhost:9093"
echo ""
echo "Press Ctrl+C to stop port forwarding"
echo ""

# Keep process running
wait
