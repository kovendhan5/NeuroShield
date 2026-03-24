#!/bin/bash
# Checkpoint Restore Script
# Restores from a saved checkpoint (Docker images, database, config)
# Usage: bash .checkpoints/restore.sh

set -e

CHECKPOINT_DIR=".checkpoints/latest"

echo "🔄 NeuroShield Checkpoint Restore Started..."
echo "============================================="

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "   Please run: bash .checkpoints/backup.sh first"
    exit 1
fi

# 1. Stop running services
echo "🛑 Stopping running services..."
docker-compose -f docker-compose-hardened.yml down 2>/dev/null || true
sleep 5

# 2. Restore Docker images
echo "📦 Restoring Docker images..."
if [ -d "$CHECKPOINT_DIR/docker" ]; then
    for image_tar in "$CHECKPOINT_DIR/docker"/*.tar.gz; do
        if [ -f "$image_tar" ]; then
            echo "  → Loading $(basename $image_tar)"
            docker load < "$image_tar"
        fi
    done
    echo "  ✓ All Docker images restored"
else
    echo "  ⚠️  No Docker images found in checkpoint"
fi

# 3. Restore configuration
echo "⚙️  Restoring configuration..."
if [ -f "$CHECKPOINT_DIR/config/.env" ]; then
    cp "$CHECKPOINT_DIR/config/.env" .env
    echo "  ✓ .env restored"
fi

if [ -f "$CHECKPOINT_DIR/config/docker-compose-hardened.yml" ]; then
    cp "$CHECKPOINT_DIR/config/docker-compose-hardened.yml" .
    echo "  ✓ docker-compose-hardened.yml restored"
fi

# 4. Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose-hardened.yml up -d
echo "  ✓ Services starting (wait 30 seconds for initialization)"

# 5. Wait for db to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec postgres_neuroshield pg_isready -U neuroshield -d neuroshield > /dev/null 2>&1; then
        echo "  ✓ PostgreSQL is ready"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 1
done

# 6. Restore database
echo "🗄️  Restoring PostgreSQL database..."
if [ -f "$CHECKPOINT_DIR/database/neuroshield_backup.sql.gz" ]; then
    # Drop and recreate database
    docker exec postgres_neuroshield psql -U neuroshield -d postgres \
        -c "DROP DATABASE IF EXISTS neuroshield;" 2>/dev/null || true
    docker exec postgres_neuroshield psql -U neuroshield -d postgres \
        -c "CREATE DATABASE neuroshield;" 2>/dev/null || true

    # Restore backup
    zcat "$CHECKPOINT_DIR/database/neuroshield_backup.sql.gz" | \
        docker exec -i postgres_neuroshield psql -U neuroshield -d neuroshield

    # Get record count
    RECORD_COUNT=$(docker exec postgres_neuroshield psql -U neuroshield -d neuroshield \
        -t -c "SELECT COUNT(*) FROM healing_log;" 2>/dev/null || echo "0")

    echo "  ✓ Database restored ($RECORD_COUNT healing entries)"
else
    echo "  ⚠️  No database backup found in checkpoint"
fi

# 7. Restore data files
echo "📂 Restoring data files..."
if [ -d "$CHECKPOINT_DIR/data" ]; then
    mkdir -p data
    cp "$CHECKPOINT_DIR/data"/* data/ 2>/dev/null || true
    echo "  ✓ Data files restored"
fi

# 8. Verify restoration
echo ""
echo "============================================="
echo "✅ Checkpoint Restore Complete!"
echo "============================================="
echo ""
echo "Verification:"
echo "  Services Status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | head -10
echo ""
echo "  Database Status:"
docker exec postgres_neuroshield psql -U neuroshield -d neuroshield \
    -t -c "SELECT COUNT(*) FROM healing_log;" | xargs echo "  Healing entries:"
echo ""
echo "Next steps:"
echo "  1. Wait 30 seconds for all services to initialize"
echo "  2. Start dashboard: cd dashboard && npm run dev"
echo "  3. Access: http://localhost:5173"
echo "  4. Run verification: bash scripts/verify_dashboard.sh"
echo ""
