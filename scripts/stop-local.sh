#!/bin/bash
# NeuroShield v3 - Stop/Cleanup

echo "Stopping NeuroShield..."
docker-compose down

echo "Cleanup options:"
echo "  Keep database: (default - data preserved)"
echo "  Reset database: rm -f data/neuroshield.db"
echo "  Full cleanup: docker-compose down -v && rm -rf data logs"
echo ""
echo "Stopped."
