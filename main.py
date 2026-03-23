#!/usr/bin/env python3
"""
NeuroShield v3 - Main entry point
Orchestrator runs autonomously, API serves dashboard
"""

import asyncio
import logging
import time
import yaml
from pathlib import Path
from app import Database, Orchestrator, JenkinsConnector, KubernetesConnector, PrometheusConnector
from api.main import create_app
import uvicorn
import threading

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


def run_orchestrator(config: dict, db: Database, connectors: dict, orchestrator: Orchestrator):
    """Run orchestrator main loop"""
    logger.info("Starting orchestrator main loop...")

    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Orchestration Cycle #{cycle_count}")
            logger.info(f"{'='*60}")

            try:
                result = orchestrator.run_cycle()

                # Log cycle result
                if result["success"]:
                    logger.info(f"✅ Cycle successful ({result['duration_ms']}ms)")
                else:
                    logger.warning(f"⚠️ Cycle had issues ({result['duration_ms']}ms)")

                if result["actions_taken"]:
                    logger.info(f"🔧 Actions taken: {len(result['actions_taken'])}")
                    for action in result["actions_taken"]:
                        status = "✅" if action["success"] else "❌"
                        logger.info(f"  {status} {action['action'].upper()} ({action['duration_ms']}ms)")

            except Exception as e:
                logger.error(f"Cycle execution failed: {e}", exc_info=True)

            # Sleep before next cycle
            wait_time = config["orchestrator"]["check_interval"]
            logger.info(f"Next cycle in {wait_time}s...")
            time.sleep(wait_time)

    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")


def run_api_server(config: dict, orchestrator: Orchestrator, db: Database, connectors: dict):
    """Run FastAPI server in separate thread"""
    logger.info("Starting API server...")

    app = create_app(config, orchestrator, db, connectors)

    api_config = config["api"]
    uvicorn.run(
        app,
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

    # Load configuration
    config = load_config("config.yaml")

    # Initialize database
    logger.info("Initializing database...")
    db = Database(config["database"]["path"])

    # Initialize connectors
    logger.info("Initializing connectors...")
    connectors = {
        "jenkins": JenkinsConnector(config["connectors"]["jenkins"]),
        "kubernetes": KubernetesConnector(config["connectors"]["kubernetes"]),
        "prometheus": PrometheusConnector(config["connectors"]["prometheus"]),
    }

    # Initialize orchestrator
    logger.info("Initializing orchestrator...")
    orchestrator = Orchestrator(config, db, connectors)

    # Start API server in background thread
    logger.info("Starting API server in background...")
    api_thread = threading.Thread(
        target=run_api_server,
        args=(config, orchestrator, db, connectors),
        daemon=True,
    )
    api_thread.start()

    # Brief delay for API to start
    time.sleep(2)

    api_config = config["api"]
    logger.info(f"✅ API server running at http://{api_config['host']}:{api_config['port']}")
    logger.info(f"📊 Dashboard: http://{api_config['host']}:{api_config['port']}/")
    logger.info(f"📚 API docs: http://{api_config['host']}:{api_config['port']}/docs")

    # Start orchestrator main loop (blocking)
    run_orchestrator(config, db, connectors, orchestrator)


if __name__ == "__main__":
    main()
