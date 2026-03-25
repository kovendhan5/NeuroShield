#!/usr/bin/env python3
"""
NeuroShield Worker Service
Background daemon that continuously monitors CI/CD systems and performs auto-healing
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def main():
    """Start the worker service (orchestrator daemon)"""
    logger.info("="*60)
    logger.info("NeuroShield Worker Service Starting")
    logger.info("="*60)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Get configuration from environment
    check_interval = int(os.getenv("ORCHESTRATOR_CHECK_INTERVAL", "10"))

    logger.info(f"Worker Service Configuration:")
    logger.info(f"  Check Interval: {check_interval}s")
    logger.info(f"  Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"  Jenkins URL: {os.getenv('JENKINS_URL', 'not configured')}")
    logger.info(f"  Prometheus URL: {os.getenv('PROMETHEUS_URL', 'not configured')}")
    logger.info("="*60)

    # Import orchestrator components
    try:
        from src.orchestrator.main import (
            _load_env,
            _setup_logging,
            get_latest_build_info,
            get_build_log,
            build_52d_state,
        )
        from src.prediction.predictor import FailurePredictor

        logger.info("Orchestrator modules loaded successfully")
    except Exception as e:
        logger.error(f"Failed to import orchestrator modules: {e}", exc_info=True)
        sys.exit(1)

    # Initialize orchestrator components
    _load_env()
    _setup_logging()

    logger.info("Initializing Failure Predictor (ML Model)...")
    try:
        predictor = FailurePredictor()
        logger.info("✅ Failure Predictor initialized")
    except Exception as e:
        logger.warning(f"⚠️  Failed to initialize Failure Predictor: {e}")
        logger.warning("Continuing without ML predictions...")
        predictor = None

    # Main monitoring loop
    cycle_count = 0
    logger.info("="*60)
    logger.info("Starting continuous monitoring loop...")
    logger.info("Worker running as daemon - press Ctrl+C to stop")
    logger.info("="*60)

    while not shutdown_requested:
        cycle_count += 1
        cycle_start = time.time()

        try:
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"Orchestration Cycle #{cycle_count}")
            logger.info(f"{'='*60}")

            # Get Jenkins configuration
            jenkins_url = os.getenv("JENKINS_URL", "")
            jenkins_job = os.getenv("JENKINS_JOB", "build-pipeline")
            jenkins_username = os.getenv("JENKINS_USERNAME", "admin")
            jenkins_password = os.getenv("JENKINS_PASSWORD", "")

            if not jenkins_url or not jenkins_password:
                logger.warning("Jenkins not configured, skipping CI/CD monitoring")
            else:
                # Collect telemetry
                logger.info("Collecting telemetry from Jenkins...")
                build_info = get_latest_build_info(
                    jenkins_url, jenkins_job, jenkins_username, jenkins_password
                )

                if build_info:
                    logger.info(f"  Build #{build_info.number}: {build_info.result}")

                    # Check for failures
                    if build_info.result == "FAILURE":
                        logger.warning(f"⚠️  Detected build failure: #{build_info.number}")

                        # Get build logs for analysis
                        log_text = get_build_log(
                            jenkins_url, jenkins_job, build_info.number,
                            jenkins_username, jenkins_password
                        )

                        if log_text:
                            logger.info(f"  Retrieved {len(log_text)} chars of build log")

                            # Predict failure and suggest action
                            if predictor:
                                try:
                                    # Build state vector and predict
                                    state = build_52d_state(build_info, log_text)
                                    failure_prob = predictor.predict_failure_probability(state)
                                    logger.info(f"  ML Prediction: {failure_prob:.2%} failure probability")

                                    if failure_prob > 0.7:
                                        logger.warning(f"  🚨 High failure risk detected!")
                                        # TODO: Trigger healing action based on failure type
                                        # For now, just log the detection
                                        logger.info(f"  Recommended: Analyze failure pattern and apply fix")
                                except Exception as e:
                                    logger.error(f"  Prediction failed: {e}")

                    elif build_info.result == "SUCCESS":
                        logger.info("  ✅ Build successful, no action needed")
                else:
                    logger.warning("  Could not retrieve build information")

            # TODO: Collect Prometheus metrics
            # TODO: Collect Kubernetes pod status
            # TODO: Apply RL agent decision-making
            # TODO: Execute healing actions if needed

            cycle_duration = time.time() - cycle_start
            logger.info(f"Cycle completed in {cycle_duration:.2f}s")

        except Exception as e:
            logger.error(f"Error in orchestration cycle: {e}", exc_info=True)

        # Sleep before next cycle (interruptible)
        logger.info(f"Sleeping for {check_interval}s...")

        # Sleep in small increments to allow quick shutdown
        sleep_elapsed = 0
        while sleep_elapsed < check_interval and not shutdown_requested:
            time.sleep(min(1, check_interval - sleep_elapsed))
            sleep_elapsed += 1

    logger.info("="*60)
    logger.info("Worker service shutdown complete")
    logger.info(f"Total cycles completed: {cycle_count}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
