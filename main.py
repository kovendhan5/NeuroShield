#!/usr/bin/env python3
"""
NeuroShield v3 - Main entry point
Orchestrator runs autonomously, API serves dashboard
"""

import asyncio
import logging
import threading
import time
from pathlib import Path

import uvicorn
import yaml

from src.api.main import app as api_app
from src.orchestrator.main import main as orchestrator_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def run_api_server(config: dict):
    """Run FastAPI server in separate thread"""
    logger.info("Starting API server...")

    api_config = config["api"]
    uvicorn.run(
        api_app,
        host=api_config["host"],
        port=api_config["port"],
        log_level="info",
    )


def main():
    """Main entry point"""

    print("""

    ╔════════════════════════════════════════════════════════════╗
    ║                   NeuroShield v3                           ║
    ║         Intelligent CI/CD Self-Healing System              ║
    ║                                                            ║
    ║  AI-powered orchestrator for automatic failure recovery    ║
    ╚════════════════════════════════════════════════════════════╝

    """)

    config = load_config("config.yaml")

    # Start API server in background thread
    logger.info("Starting API server in background...")
    api_thread = threading.Thread(
        target=run_api_server,
        args=(config,),
        daemon=True,
    )
    api_thread.start()

    # Brief delay for API to start
    time.sleep(2)

    api_config = config["api"]
    logger.info(f"✅ API server running at http://{api_config['host']}:{api_config['port']}")
    logger.info(f"📊 Dashboard: http://{api_config['host']}:{api_config['port']}/")
    logger.info(f"📚 API docs: http://{api_config['host']}:{api_config['port']}/docs")

    # Start orchestrator daemon (blocking)
    asyncio.run(orchestrator_main())


if __name__ == "__main__":
    main()
