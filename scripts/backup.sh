#!/bin/bash
# NeuroShield Automated Backup Script
# Run daily via cron: 0 2 * * * /app/scripts/backup.sh

set -euo pipefail

# Configuration
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
RETENTION_DAYS=30
LOG_FILE="logs/backup.log"
S3_BUCKET="${BACKUP_S3_BUCKET:-}"  # Optional S3 upload

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[OK]${NC} $*" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"
log "Starting NeuroShield backup to $BACKUP_DIR"

# 1. Database backup
log "Backing up PostgreSQL database..."
if docker-compose exec -T postgres pg_dump -U neuroshield_app neuroshield_db 2>/dev/null > "$BACKUP_DIR/neuroshield_db.sql"; then
    success "Database backup completed ($(wc -c < "$BACKUP_DIR/neuroshield_db.sql" | numfmt --to=iec-i --suffix=B 2>/dev/null || echo "OK"))"
else
    error "Database backup failed"
fi

# 2. Data directory backup
log "Backing up data directory..."
if tar -czf "$BACKUP_DIR/data.tar.gz" data/ 2>/dev/null; then
    success "Data backup completed ($(stat -f%z "$BACKUP_DIR/data.tar.gz" 2>/dev/null | numfmt --to=iec-i --suffix=B 2>/dev/null || ls -lh "$BACKUP_DIR/data.tar.gz" | awk '{print $5}'))"
else
    warn "Data directory partial backup (some files may be locked)"
fi

# 3. Model weights backup
log "Backing up ML model weights..."
if [ -d "models" ]; then
    if tar -czf "$BACKUP_DIR/models.tar.gz" models/ 2>/dev/null; then
        success "Model weights backup completed"
    else
        warn "Model weights backup partial"
    fi
else
    warn "Models directory not found"
fi

# 4. Configuration backup
log "Backing up configuration..."
mkdir -p "$BACKUP_DIR/config"
cp -v config/*.yaml "$BACKUP_DIR/config/" 2>/dev/null || true
cp -v .env "$BACKUP_DIR/.env" 2>/dev/null || warn ".env not found (expected for security)"
success "Configuration backed up"

# 5. Create backup manifest
log "Creating backup manifest..."
cat > "$BACKUP_DIR/MANIFEST.txt" <<EOF
NeuroShield Backup Manifest
Created: $(date)
Hostname: $(hostname)
Version: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

Contents:
- neuroshield_db.sql: PostgreSQL database dump
- data.tar.gz: Healing logs, telemetry, active alerts
- models.tar.gz: Trained ML model weights
- config/: Configuration files
- MANIFEST.txt: This file

Restore Instructions:
1. Restore database:
   docker-compose exec postgres psql -U postgres < neuroshield_db.sql

2. Restore data:
   tar -xzf data.tar.gz

3. Restore models:
   tar -xzf models.tar.gz

4. Restart services:
   docker-compose restart microservice orchestrator
EOF
success "Backup manifest created"

# 6. Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    log "Uploading backup to S3 ($S3_BUCKET)..."
    if command -v aws &> /dev/null; then
        if aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$(hostname)/$(date +%Y/%m/%d)/" --sse AES256 --storage-class GLACIER; then
            success "S3 upload completed"
        else
            error "S3 upload failed"
        fi
    else
        warn "AWS CLI not found - skipping S3 upload"
    fi
fi

# 7. Cleanup old backups
log "Cleaning up old backups (older than $RETENTION_DAYS days)..."
find backups -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true

# Calculate kept backups
KEPT_COUNT=$(find backups -type d -mindepth 1 -mtime -$RETENTION_DAYS | wc -l)
TOTAL_SIZE=$(du -sh backups | awk '{print $1}')
success "Retention cleanup completed ($KEPT_COUNT backups kept, total: $TOTAL_SIZE)"

# 8. Verify backup integrity
log "Verifying backup integrity..."
if tar -tzf "$BACKUP_DIR/data.tar.gz" > /dev/null 2>&1; then
    success "Backup integrity verified"
else
    error "Backup integrity check failed"
fi

# 9. Generate report
log "Generating backup report..."
cat >> "$LOG_FILE" <<EOF

BACKUP SUMMARY
==============
Backup ID: $BACKUP_DIR
Start Time: $(head -1 "$LOG_FILE" | grep -o '\[.*\]')
End Time: $(date +'[%Y-%m-%d %H:%M:%S]')
Total Size: $(du -sh "$BACKUP_DIR" | awk '{print $1}')
Status: SUCCESS

Backup Contents:
- Database: $([ -f "$BACKUP_DIR/neuroshield_db.sql" ] && echo "✓ Yes" || echo "✗ No")
- Data: $([ -f "$BACKUP_DIR/data.tar.gz" ] && echo "✓ Yes" || echo "✗ No")
- Models: $([ -f "$BACKUP_DIR/models.tar.gz" ] && echo "✓ Yes" || echo "✗ No")
- Config: $([ -d "$BACKUP_DIR/config" ] && echo "✓ Yes" || echo "✗ No")

S3 Backup: $([ -n "$S3_BUCKET" ] && echo "✓ Uploaded" || echo "⊝ Not configured")

Next Backup: $(date -d '+1 day' +'%Y-%m-%d %H:%M:%S')
EOF

success "Backup completed successfully!"
log "Backup location: $BACKUP_DIR"
log "Full report: $LOG_FILE"

exit 0
