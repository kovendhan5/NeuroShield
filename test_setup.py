#!/usr/bin/env python3
"""Quick test to verify setup"""

from app import Orchestrator, Database
from app.connectors import JenkinsConnector, KubernetesConnector, PrometheusConnector
import yaml

# Try loading config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('All imports successful')
print('Config loaded:', config['system']['name'], config['system']['version'])

# Initialize components
db = Database(config['database']['path'])
connectors = {
    'jenkins': JenkinsConnector(config['connectors']['jenkins']),
    'kubernetes': KubernetesConnector(config['connectors']['kubernetes']),
    'prometheus': PrometheusConnector(config['connectors']['prometheus']),
}
orchestrator = Orchestrator(config, db, connectors)

print('Components initialized successfully')
print('Running test cycle...')

result = orchestrator.run_cycle()
print('First cycle completed:', result['success'])
print('Metrics collected:', list(result['metrics'].keys()))
