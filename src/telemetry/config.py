"""
Configuration module for NeuroShield telemetry collector.
Reads from environment variables or uses defaults.
"""

import os
from dotenv import load_dotenv

# Load from .env if exists
load_dotenv()

# Jenkins Configuration
JENKINS_URL = os.getenv('JENKINS_URL', 'http://localhost:8080')
JENKINS_USERNAME = os.getenv('JENKINS_USERNAME')
JENKINS_TOKEN = os.getenv('JENKINS_TOKEN')
JENKINS_JOB = os.getenv('JENKINS_JOB', 'build-pipeline')

# Prometheus Configuration
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')

# Telemetry Configuration
TELEMETRY_OUTPUT = os.getenv('TELEMETRY_OUTPUT', 'data/telemetry.csv')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '10'))  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
