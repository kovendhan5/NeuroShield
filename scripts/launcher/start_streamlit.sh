#!/bin/bash
# Start Streamlit Dashboard

cd "$(dirname "$0")/../.."

echo "🚀 Starting NeuroShield Streamlit Dashboard..."
echo "================================================"
echo ""
echo "Dashboard will be available at: http://localhost:8501"
echo ""

streamlit run src/dashboard/streamlit_dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --logger.level=info \
    --client.showErrorDetails=true
