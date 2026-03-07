#!/usr/bin/env python3
"""Launch the NeuroShield REST API on port 8502.

Usage:
    python scripts/start_api.py
    Open http://localhost:8502/docs for Swagger UI
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8502,
        reload=False,
        log_level="info",
    )
