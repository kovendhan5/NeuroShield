#!/bin/bash
# NeuroShield v2.1.0 - Production Stack Configuration Generator

set -e

REPO_DIR="/app"
COMPOSE_DIR="$REPO_DIR"

echo "🚀 NeuroShield v2.1.0 - Production Stack Builder"
echo "=================================================="
echo ""

# Generate production docker-compose file
cat > "$COMPOSE_DIR/docker-compose-production.yml" << 'EOF'
version: '3.8'

services:
  # ===== DATABASE =====
  postgres:
    image: postgres:15-alpine
    container_name: neuroshield-postgres
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: neuroshield_db_pass_123
      POSTGRES_DB: neuroshield
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - neuroshield
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ===== CACHE =====
  redis:
    image: redis:7-alpine
    container_name: neuroshield-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - neuroshield
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ===== MONITORING =====
  prometheus:
    image: prom/prometheus:latest
    container_name: neuroshield-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./infra/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - neuroshield
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: neuroshield-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin123
      GF_SECURITY_ADMIN_USER: admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - neuroshield
    depends_on:
      - prometheus
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    container_name: neuroshield-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./infra/prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    networks:
      - neuroshield
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: neuroshield-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
    networks:
      - neuroshield
    restart: unless-stopped

  # ===== CI/CD =====
  jenkins:
    image: jenkins/jenkins:lts-alpine
    container_name: neuroshield-jenkins
    user: root
    ports:
      - "8080:8080"
      - "50000:50000"
    environment:
      JENKINS_HOME: /var/jenkins_home
    volumes:
      - jenkins_data:/var/jenkins_home
      - /var/run/docker.sock:/var/run/docker.sock
    networks:
      - neuroshield
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080/api/json || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  neuroshield:
    driver: bridge
    name: neuroshield

volumes:
  postgres_data:
  redis_data:
  jenkins_data:
  prometheus_data:
  grafana_data:
  alertmanager_data:
EOF

echo "✅ Generated docker-compose-production.yml"
echo ""
echo "📊 Service Summary:"
echo "  • PostgreSQL (port 5432) - Data persistence"
echo "  • Redis (port 6379) - Caching layer"
echo "  • Prometheus (port 9090) - Metrics collection"
echo "  • Grafana (port 3000) - Visualization dashboard"
echo "  • AlertManager (port 9093) - Alert management"
echo "  • Jenkins (port 8080) - CI/CD pipeline"
echo ""
echo "🚀 Starting services..."
cd "$COMPOSE_DIR"
docker-compose -f docker-compose-production.yml up -d
sleep 10

echo ""
echo "✅ Stack is starting up..."
echo ""
echo "📍 Access the services at:"
echo "  • Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "  • Prometheus: http://localhost:9090"
echo "  • Jenkins: http://localhost:8080 (initializing...)"
echo "  • AlertManager: http://localhost:9093"
echo ""
