"""Production Prometheus client with caching and error handling."""

import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import time
from src.config import get_config
import logging

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Production-grade Prometheus client."""

    def __init__(
        self,
        url: Optional[str] = None,
        timeout: int = 10,
        query_timeout: int = 30,
        cache_duration: int = 60,
    ):
        """Initialize Prometheus client.

        Args:
            url: Prometheus URL (defaults to config)
            timeout: Request timeout in seconds
            query_timeout: Prometheus query timeout
            cache_duration: Cache query results for N seconds
        """
        config = get_config()
        self.url = (url or config.get("prometheus", "url")).rstrip("/")
        self.timeout = timeout
        self.query_timeout = query_timeout
        self.cache_duration = cache_duration
        self._cache: Dict[str, tuple[Any, float]] = {}

    def _query(self, promql: str, time_arg: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Execute PromQL query.

        Args:
            promql: PromQL query string
            time_arg: Optional time argument

        Returns:
            Query results or None if failed
        """
        # Check cache
        cache_key = f"{promql}:{time_arg}"
        if cache_key in self._cache:
            result, cache_time = self._cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return result

        try:
            params = {
                "query": promql,
                "timeout": f"{self.query_timeout}s",
            }
            if time_arg:
                params["time"] = time_arg

            response = requests.get(
                f"{self.url}/api/v1/query",
                params=params,
                timeout=self.timeout,
                verify=False
            )
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                results = data.get("data", {}).get("result", [])
                self._cache[cache_key] = (results, time.time())
                return results
            else:
                logger.warning(f"Prometheus query error: {data.get('error')}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Prometheus query timeout: {promql}")
            return None
        except Exception as e:
            logger.error(f"Prometheus query error: {str(e)}")
            return None

    def get_cpu_usage(self, minutes: int = 5) -> Optional[float]:
        """Get average CPU usage percentage.

        Args:
            minutes: Time window in minutes

        Returns:
            CPU percentage (0-100) or None if unavailable
        """
        # Try node CPU formula first
        promql = f"avg(100 - (avg by (instance) (rate(node_cpu_seconds_total{{mode='idle'}}[{minutes}m]))) * 100) over ({minutes}m)"
        results = self._query(promql)

        if results and len(results) > 0:
            try:
                value = float(results[0]["value"][1])
                return max(0, min(100, value))
            except (KeyError, ValueError, IndexError):
                pass

        return None

    def get_memory_usage(self) -> Optional[float]:
        """Get average memory usage percentage.

        Returns:
            Memory percentage (0-100) or None if unavailable
        """
        promql = "avg((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100)"
        results = self._query(promql)

        if results and len(results) > 0:
            try:
                value = float(results[0]["value"][1])
                return max(0, min(100, value))
            except (KeyError, ValueError, IndexError):
                pass

        return None

    def get_pod_count(self) -> Optional[int]:
        """Get running pod count.

        Returns:
            Pod count or None if unavailable
        """
        promql = "count(kube_pod_info{pod_name=~'.*'})"
        results = self._query(promql)

        if results and len(results) > 0:
            try:
                return int(float(results[0]["value"][1]))
            except (KeyError, ValueError, IndexError):
                pass

        return None

    def get_error_rate(self, minutes: int = 5) -> Optional[float]:
        """Get error rate percentage.

        Returns:
            Error rate (0-100) or None if unavailable
        """
        promql = f"rate(http_requests_total{{status=~'5..'}}[{minutes}m]) / rate(http_requests_total[{minutes}m]) * 100"
        results = self._query(promql)

        if results and len(results) > 0:
            try:
                value = float(results[0]["value"][1])
                return max(0, min(100, value))
            except (KeyError, ValueError, IndexError):
                pass

        return 0  # No errors

    def get_pod_restart_count(self) -> Optional[int]:
        """Get total pod restart count.

        Returns:
            Restart count or None if unavailable
        """
        promql = "sum(kube_pod_container_status_restarts_total)"
        results = self._query(promql)

        if results and len(results) > 0:
            try:
                return int(float(results[0]["value"][1]))
            except (KeyError, ValueError, IndexError):
                pass

        return 0

    def get_disk_usage(self) -> Optional[float]:
        """Get disk usage percentage.

        Returns:
            Disk usage (0-100) or None if unavailable
        """
        promql = "avg((node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100)"
        results = self._query(promql)

        if results and len(results) > 0:
            try:
                value = float(results[0]["value"][1])
                return max(0, min(100, value))
            except (KeyError, ValueError, IndexError):
                pass

        return None

    def get_network_bytes(self, interface: str = "eth0") -> Optional[Dict[str, float]]:
        """Get network bytes in/out.

        Args:
            interface: Network interface name

        Returns:
            Dictionary with 'in' and 'out' bytes or None
        """
        promql_in = f"rate(node_network_receive_bytes_total{{device='{interface}'}}[5m])"
        promql_out = f"rate(node_network_transmit_bytes_total{{device='{interface}'}}[5m])"

        results_in = self._query(promql_in)
        results_out = self._query(promql_out)

        if results_in and results_out:
            try:
                return {
                    "in": float(results_in[0]["value"][1]),
                    "out": float(results_out[0]["value"][1]),
                }
            except (KeyError, ValueError, IndexError):
                pass

        return None

    def is_healthy(self) -> bool:
        """Check if Prometheus is healthy.

        Returns:
            True if Prometheus is accessible
        """
        try:
            response = requests.get(f"{self.url}/-/healthy", timeout=self.timeout, verify=False)
            return response.status_code == 200
        except Exception:
            return False

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get all current metrics (CPU, memory, pods, errors).

        Returns:
            Dictionary with all metrics
        """
        return {
            "cpu_percent": self.get_cpu_usage(),
            "memory_percent": self.get_memory_usage(),
            "disk_percent": self.get_disk_usage(),
            "pod_count": self.get_pod_count(),
            "error_rate": self.get_error_rate(),
            "pod_restarts": self.get_pod_restart_count(),
            "timestamp": datetime.now().__str__(),
        }
