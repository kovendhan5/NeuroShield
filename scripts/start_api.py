#!/usr/bin/env python3
"""Launch the NeuroShield REST API on port 8502.

Usage:
    python scripts/start_api.py
    Open http://localhost:8502/docs for Swagger UI
"""

import sys
import os
import uvicorn

# Add project root to path so uvicorn can find src.api.main
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8502,
        reload=False,
        log_level="info",
    )
