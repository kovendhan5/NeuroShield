# NeuroShield Telemetry Collector

Real-time telemetry collection from Jenkins API and Prometheus metrics.

## Features

- **Jenkins API Polling**: Collects build status, duration, and queue length
- **Prometheus Metrics**: Fetches CPU, memory, pod count, and error rates
- **CSV Export**: Saves all data with timestamps for analysis
- **Configurable**: Environment variables and CLI arguments for easy setup
- **Robust Error Handling**: Graceful fallbacks on connection failures

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and update with your Jenkins/Prometheus URLs:

```bash
cp .env.example .env
```

Edit `.env`:
```env
JENKINS_URL=http://localhost:8080
JENKINS_JOB=build-pipeline
PROMETHEUS_URL=http://localhost:9090
POLL_INTERVAL=10
```

### Running

```bash
# Using default config from .env
python -m src.telemetry.main

# With custom parameters
python -m src.telemetry.main --jenkins-url http://jenkins:8080 --interval 5

# With Jenkins authentication
python -m src.telemetry.main --jenkins-username admin --jenkins-token YOUR_TOKEN
```

## Output Format

Telemetry is saved to `telemetry.csv` with the following columns:

| Column | Description |
|--------|-------------|
| timestamp | ISO format timestamp of collection |
| jenkins_last_build_status | Last build result (SUCCESS/FAILURE/UNSTABLE/etc) |
| jenkins_last_build_duration | Build duration in milliseconds |
| jenkins_queue_length | Number of queued builds |
| prometheus_cpu_usage | CPU usage as percentage |
| prometheus_memory_usage | Memory usage as percentage |
| prometheus_pod_count | Number of running pods |
| prometheus_error_rate | HTTP 5xx error rate |

## Architecture

```
TelemetryCollector
├── JenkinsPoll
│   ├── get_last_build_status()
│   └── get_queue_length()
└── PrometheusPoll
    ├── get_cpu_usage()
    ├── get_memory_usage()
    ├── get_pod_count()
    └── get_error_rate()
```

## Testing

```bash
pytest tests/test_telemetry.py -v
```

## API Reference

### TelemetryCollector

```python
from src.telemetry import TelemetryCollector

collector = TelemetryCollector(
    jenkins_url="http://localhost:8080",
    prometheus_url="http://localhost:9090",
    output_csv="telemetry.csv",
    poll_interval=10,
    jenkins_job="build-pipeline"
)

collector.start()  # Start collection loop
```

### JenkinsPoll

```python
from src.telemetry import JenkinsPoll

jenkins = JenkinsPoll(
    jenkins_url="http://localhost:8080",
    username="admin",
    token="YOUR_API_TOKEN"
)

status = jenkins.get_last_build_status("job-name")
queue = jenkins.get_queue_length()
```

### PrometheusPoll

```python
from src.telemetry import PrometheusPoll

prometheus = PrometheusPoll("http://localhost:9090")

cpu = prometheus.get_cpu_usage()
memory = prometheus.get_memory_usage()
pods = prometheus.get_pod_count()
errors = prometheus.get_error_rate()
```

## Notes

- Polls every 10 seconds by default (configurable)
- Handles connection failures gracefully with logging
- Requires Jenkins API access and Prometheus endpoint
- Optional: Jenkins authentication via username/token for private instances
