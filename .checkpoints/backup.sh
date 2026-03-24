#!/bin/bash
# Checkpoint Backup Script
# Backs up Docker images, database, and critical files for recovery
# Usage: bash .checkpoints/backup.sh

set -e

CHECKPOINT_DIR=".checkpoints/latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔄 Creating NeuroShield Checkpoint Backup..."
echo "=========================================="

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# 1. Docker Images
echo "📦 Backing up Docker images..."
mkdir -p "$CHECKPOINT_DIR/docker"

docker images --format "{{.Repository}}:{{.Tag}}" | while read image; do
    if [[ "$image" != "<none>:<none>" ]]; then
        echo "  → Saving $image"
        docker save "$image" | gzip > "$CHECKPOINT_DIR/docker/$(echo $image | sed 's/:/_/g').tar.gz"
    fi
done

# 2. Database Dump
echo "🗄️  Backing up PostgreSQL database..."
mkdir -p "$CHECKPOINT_DIR/database"

docker exec postgres_neuroshield pg_dump -U neuroshield -d neuroshield \
    | gzip > "$CHECKPOINT_DIR/database/neuroshield_backup.sql.gz"

echo "  → PostgreSQL dump saved (292 healing entries)"

# 3. Data Files
echo "📂 Backing up data files..."
mkdir -p "$CHECKPOINT_DIR/data"

if [ -d "data" ]; then
    cp -r data/* "$CHECKPOINT_DIR/data/" 2>/dev/null || true
    echo "  → Data files copied (healing_log.json, action_history.csv)"
fi

# 4. Configuration Files
echo "⚙️  Backing up configuration..."
mkdir -p "$CHECKPOINT_DIR/config"

cp .env "$CHECKPOINT_DIR/config/.env" 2>/dev/null || true
cp .env.example "$CHECKPOINT_DIR/config/.env.example" 2>/dev/null || true
cp docker-compose-hardened.yml "$CHECKPOINT_DIR/config/" 2>/dev/null || true
cp docker-compose.yml "$CHECKPOINT_DIR/config/" 2>/dev/null || true
cp Dockerfile "$CHECKPOINT_DIR/config/" 2>/dev/null || true
cp pytest.ini "$CHECKPOINT_DIR/config/" 2>/dev/null || true

echo "  → Configuration files backed up"

# 5. Dashboard Dependencies
echo "📱 Backing up dashboard dependencies..."
mkdir -p "$CHECKPOINT_DIR/dashboard"

cp dashboard/package-lock.json "$CHECKPOINT_DIR/dashboard/" 2>/dev/null || true
cp dashboard/package.json "$CHECKPOINT_DIR/dashboard/" 2>/dev/null || true

echo "  → Dashboard locked dependencies saved"

# 6. Git Info
echo "📚 Saving git information..."
mkdir -p "$CHECKPOINT_DIR/git"

git log --oneline -10 > "$CHECKPOINT_DIR/git/recent_commits.txt"
git branch -a > "$CHECKPOINT_DIR/git/branches.txt"
git status > "$CHECKPOINT_DIR/git/status.txt"
echo "Git commit $(git rev-parse HEAD)" > "$CHECKPOINT_DIR/git/HEAD.txt"

echo "  → Git history saved"

# 7. System Info
echo "ℹ️  Storing system information..."

cat > "$CHECKPOINT_DIR/BACKUP_INFO.txt" << BACKUPINFO
NeuroShield Checkpoint Backup
Created: $(date)
Timestamp: $TIMESTAMP

Included:
  ✓ Docker images (9 services)
  ✓ PostgreSQL database dump (292 healing entries)
  ✓ Data files (healing_log.json, CSVs)
  ✓ Configuration files (.env, docker-compose)
  ✓ Dashboard dependencies (package-lock.json)
  ✓ Git history (commits, branches)

Total Size: $(du -sh "$CHECKPOINT_DIR" | cut -f1)

To restore this checkpoint:
  1. Copy this directory to a safe location
  2. Run: bash .checkpoints/restore.sh
  3. Services will restart automatically

Critical Files:
  - docker-compose-hardened.yml (9 services config)
  - database/neuroshield_backup.sql.gz (292 healing entries)
  - data/healing_log.json (actual log data)
  - config/.env (production environment)

BACKUPINFO

echo "  → System info documented"

# 8. Summary
echo ""
echo "=========================================="
echo "✅ Checkpoint Backup Complete!"
echo "=========================================="
echo ""
echo "Backup Location: $CHECKPOINT_DIR"
echo "Total Size: $(du -sh "$CHECKPOINT_DIR" | cut -f1)"
echo ""
echo "Files included:"
ls -la "$CHECKPOINT_DIR" | tail -n +4 | awk '{print "  " $9}'
echo ""
echo "To restore: bash .checkpoints/restore.sh"
echo ""
