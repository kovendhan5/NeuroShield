"""
NeuroShield Telemetry Collector
Polls Jenkins API and Prometheus metrics every 10s, saves to CSV.
"""

import time
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass, asdict
from requests.auth import HTTPBasicAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TelemetryData:
    """Dataclass for telemetry records."""
    timestamp: str
    jenkins_last_build_status: Optional[str] = None
    jenkins_last_build_duration: Optional[float] = None
    jenkins_queue_length: Optional[int] = None
    jenkins_last_build_log: Optional[str] = None
    prometheus_cpu_usage: Optional[float] = None
    prometheus_memory_usage: Optional[float] = None
    prometheus_pod_count: Optional[int] = None
    prometheus_error_rate: Optional[float] = None


class JenkinsPoll:
    """Handles Jenkins API polling."""
    
    def __init__(self, jenkins_url: str, username: str = None, token: str = None):
        """
        Initialize Jenkins poller.
        
        Args:
            jenkins_url: Jenkins server URL (e.g., http://localhost:8080)
            username: Jenkins username (optional)
            token: Jenkins API token (optional)
        """
        self.jenkins_url = jenkins_url.rstrip('/')
        self.auth = None
        if username and token:
            self.auth = HTTPBasicAuth(username, token)
        self.session = requests.Session()
        self.session.auth = self.auth
    
    def get_last_build_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """
        Get last build status and duration for a Jenkins job.
        
        Args:
            job_name: Jenkins job name
            
        Returns:
            Dict with status and duration, or None on error
        """
        try:
            url = f"{self.jenkins_url}/job/{job_name}/lastBuild/api/json"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return {
                'status': data.get('result'),  # SUCCESS, FAILURE, etc.
                'duration_ms': data.get('duration')
            }
        except Exception as e:
            logger.warning(f"Failed to fetch Jenkins build status for {job_name}: {e}")
            return None
    
    def get_queue_length(self) -> Optional[int]:
        """
        Get Jenkins queue length.
        
        Returns:
            Queue length or None on error
        """
        try:
            url = f"{self.jenkins_url}/queue/api/json"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return len(data.get('items', []))
        except Exception as e:
            logger.warning(f"Failed to fetch Jenkins queue: {e}")
            return None

    def get_last_build_log(self, job_name: str, max_chars: int = 2000) -> Optional[str]:
        """Fetch the last build console log (truncated).

        Args:
            job_name: Jenkins job name.
            max_chars: Maximum number of characters to return.

        Returns:
            Log text or None on error.
        """
        try:
            url = f"{self.jenkins_url}/job/{job_name}/lastBuild/consoleText"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            text = response.text or ""
            if len(text) > max_chars:
                text = text[-max_chars:]
            return text.replace("\r\n", "\n")
        except Exception as e:
            logger.warning(f"Failed to fetch Jenkins build log for {job_name}: {e}")
            return None


class PrometheusPoll:
    """Handles Prometheus metrics polling."""
    
    def __init__(self, prometheus_url: str):
        """
        Initialize Prometheus poller.
        
        Args:
            prometheus_url: Prometheus server URL (e.g., http://localhost:9090)
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.session = requests.Session()
    
    def query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a Prometheus query.
        
        Args:
            query: PromQL query string
            
        Returns:
            List of results or None on error
        """
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            response = self.session.get(url, params={'query': query}, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get('data', {}).get('result', [])
        except Exception as e:
            logger.warning(f"Prometheus query failed: {e}")
            return None
    
    def get_cpu_usage(self) -> Optional[float]:
        """Get average CPU usage percentage."""
        results = self.query('rate(container_cpu_usage_seconds_total[1m])')
        if results:
            try:
                return float(results[0]['value'][1]) * 100
            except (KeyError, ValueError):
                pass
        return None
    
    def get_memory_usage(self) -> Optional[float]:
        """Get average memory usage percentage."""
        results = self.query('(container_memory_working_set_bytes / container_spec_memory_limit_bytes) * 100')
        if results:
            try:
                return float(results[0]['value'][1])
            except (KeyError, ValueError):
                pass
        return None
    
    def get_pod_count(self) -> Optional[int]:
        """Get running pod count."""
        results = self.query('count(kube_pod_info)')
        if results:
            try:
                return int(float(results[0]['value'][1]))
            except (KeyError, ValueError):
                pass
        return None
    
    def get_error_rate(self) -> Optional[float]:
        """Get HTTP error rate."""
        results = self.query('rate(http_requests_total{status=~"5.."}[1m])')
        if results:
            try:
                return float(results[0]['value'][1])
            except (KeyError, ValueError):
                pass
        return None


class TelemetryCollector:
    """Main telemetry collector orchestrating Jenkins and Prometheus polling."""
    
    def __init__(
        self,
        jenkins_url: str,
        prometheus_url: str,
        output_csv: str = "data/telemetry.csv",
        poll_interval: int = 10,
        jenkins_job: str = "default-job",
        username: str = None,
        token: str = None
    ):
        """
        Initialize telemetry collector.
        
        Args:
            jenkins_url: Jenkins server URL
            prometheus_url: Prometheus server URL
            output_csv: Output CSV file path
            poll_interval: Polling interval in seconds (default: 10)
            jenkins_job: Jenkins job name to monitor
            username: Jenkins username (optional)
            token: Jenkins API token (optional)
        """
        self.jenkins = JenkinsPoll(jenkins_url, username, token)
        self.prometheus = PrometheusPoll(prometheus_url)
        self.output_csv = output_csv
        self.poll_interval = poll_interval
        self.jenkins_job = jenkins_job
        self.running = False
        
        # Initialize CSV with headers
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        try:
            Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TelemetryData.__dataclass_fields__.keys())
                writer.writeheader()
            logger.info(f"Initialized CSV: {self.output_csv}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}")
    
    def collect_once(self) -> TelemetryData:
        """
        Collect telemetry data once from Jenkins and Prometheus.
        
        Returns:
            TelemetryData object
        """
        timestamp = datetime.now().isoformat()
        
        # Fetch Jenkins data
        jenkins_build = self.jenkins.get_last_build_status(self.jenkins_job)
        jenkins_queue = self.jenkins.get_queue_length()
        jenkins_log = self.jenkins.get_last_build_log(self.jenkins_job)
        
        # Fetch Prometheus data
        cpu_usage = self.prometheus.get_cpu_usage()
        memory_usage = self.prometheus.get_memory_usage()
        pod_count = self.prometheus.get_pod_count()
        error_rate = self.prometheus.get_error_rate()
        
        telemetry = TelemetryData(
            timestamp=timestamp,
            jenkins_last_build_status=jenkins_build['status'] if jenkins_build else None,
            jenkins_last_build_duration=jenkins_build['duration_ms'] if jenkins_build else None,
            jenkins_queue_length=jenkins_queue,
            jenkins_last_build_log=jenkins_log,
            prometheus_cpu_usage=cpu_usage,
            prometheus_memory_usage=memory_usage,
            prometheus_pod_count=pod_count,
            prometheus_error_rate=error_rate
        )
        
        logger.debug(f"Collected telemetry: {telemetry}")
        return telemetry
    
    def save_record(self, data: TelemetryData):
        """
        Append telemetry record to CSV.
        
        Args:
            data: TelemetryData object to save
        """
        try:
            with open(self.output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TelemetryData.__dataclass_fields__.keys())
                writer.writerow(asdict(data))
            logger.info(f"Saved telemetry record: {data.timestamp}")
        except Exception as e:
            logger.error(f"Failed to save record: {e}")
    
    def start(self):
        """Start continuous telemetry collection loop."""
        self.running = True
        logger.info(f"Starting telemetry collection (interval: {self.poll_interval}s)")
        
        try:
            while self.running:
                try:
                    data = self.collect_once()
                    self.save_record(data)
                except Exception as e:
                    logger.error(f"Error during collection: {e}")
                
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Telemetry collection stopped by user")
            self.stop()
    
    def stop(self):
        """Stop telemetry collection loop."""
        self.running = False
        logger.info("Telemetry collector stopped")


if __name__ == "__main__":
    # Example usage
    collector = TelemetryCollector(
        jenkins_url="http://localhost:8080",
        prometheus_url="http://localhost:9090",
        output_csv="telemetry.csv",
        poll_interval=10,
        jenkins_job="build-pipeline"
    )
    
    # Collect continuously
    collector.start()
