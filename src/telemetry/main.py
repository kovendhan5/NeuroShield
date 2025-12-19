#!/usr/bin/env python3
"""
NeuroShield Telemetry Collector - CLI Entry Point
Usage: python -m src.telemetry.main
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.telemetry.collector import TelemetryCollector
from src.telemetry.config import (
    JENKINS_URL, JENKINS_USERNAME, JENKINS_TOKEN, JENKINS_JOB,
    PROMETHEUS_URL, TELEMETRY_OUTPUT, POLL_INTERVAL, LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='NeuroShield Telemetry Collector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.telemetry.main
  python -m src.telemetry.main --output /tmp/telemetry.csv --interval 5
  python -m src.telemetry.main --jenkins-url http://jenkins:8080 --prometheus-url http://prometheus:9090
        """
    )
    
    parser.add_argument(
        '--jenkins-url',
        default=JENKINS_URL,
        help=f'Jenkins URL (default: {JENKINS_URL})'
    )
    parser.add_argument(
        '--jenkins-job',
        default=JENKINS_JOB,
        help=f'Jenkins job name (default: {JENKINS_JOB})'
    )
    parser.add_argument(
        '--jenkins-username',
        default=JENKINS_USERNAME,
        help='Jenkins username (optional)'
    )
    parser.add_argument(
        '--jenkins-token',
        default=JENKINS_TOKEN,
        help='Jenkins API token (optional)'
    )
    parser.add_argument(
        '--prometheus-url',
        default=PROMETHEUS_URL,
        help=f'Prometheus URL (default: {PROMETHEUS_URL})'
    )
    parser.add_argument(
        '--output',
        default=TELEMETRY_OUTPUT,
        help=f'Output CSV file (default: {TELEMETRY_OUTPUT})'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=POLL_INTERVAL,
        help=f'Poll interval in seconds (default: {POLL_INTERVAL})'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("NeuroShield Telemetry Collector Starting")
    logger.info("=" * 60)
    logger.info(f"Jenkins URL: {args.jenkins_url}")
    logger.info(f"Jenkins Job: {args.jenkins_job}")
    logger.info(f"Prometheus URL: {args.prometheus_url}")
    logger.info(f"Output CSV: {args.output}")
    logger.info(f"Poll Interval: {args.interval}s")
    logger.info("=" * 60)
    
    # Create and start collector
    collector = TelemetryCollector(
        jenkins_url=args.jenkins_url,
        prometheus_url=args.prometheus_url,
        output_csv=args.output,
        poll_interval=args.interval,
        jenkins_job=args.jenkins_job,
        username=args.jenkins_username,
        token=args.jenkins_token
    )
    
    try:
        collector.start()
    except Exception as e:
        logger.error(f"Collector error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
