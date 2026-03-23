"""Production Jenkins API client with retry logic and error handling."""

import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
from src.config import get_config
import logging

logger = logging.getLogger(__name__)


@dataclass
class Build:
    """Jenkins build information."""
    number: int
    result: str  # SUCCESS, FAILURE, UNSTABLE, ABORTED, RUNNING
    duration: int  # milliseconds
    timestamp: int  # unix timestamp
    url: str
    display_name: str
    log: Optional[str] = None


class JenkinsClient:
    """Production-grade Jenkins API client."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize Jenkins client.

        Args:
            url: Jenkins URL (defaults to config)
            username: Jenkins username (defaults to config)
            password: Jenkins password (defaults to config)
            token: Jenkins API token (defaults to config)
            timeout: Request timeout in seconds
            max_retries: Max retries for failed requests
        """
        config = get_config()
        self.url = (url or config.get("jenkins", "url")).rstrip("/")
        self.username = username or config.get("jenkins", "username")
        self.password = password or config.get("jenkins", "password")
        self.token = token or config.get("jenkins", "token")
        self.timeout = timeout
        self.max_retries = max_retries

        # Use token if available, otherwise password
        if self.token:
            self.auth = (self.username, self.token)
        elif self.password:
            self.auth = (self.username, self.password)
        else:
            self.auth = None

    def _request(
        self, method: str, endpoint: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc)
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            JSON response or None if failed
        """
        url = f"{self.url}/api/json/{endpoint}"
        kwargs.setdefault("auth", self.auth)
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", False)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                if attempt == self.max_retries:
                    logger.error(f"Jenkins timeout after {self.max_retries} retries: {endpoint}")
                    return None
                time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries:
                    logger.error(f"Jenkins connection error after {self.max_retries} retries")
                    return None
                time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as e:
                logger.error(f"Jenkins HTTP error: {e.response.status_code} {endpoint}")
                return None
            except Exception as e:
                logger.error(f"Jenkins request error: {str(e)}")
                return None

        return None

    def get_last_build_for_job(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get last build information for a job.

        Args:
            job_name: Jenkins job name

        Returns:
            Build information or None if not found
        """
        data = self._request("GET", f"job/{job_name}/lastBuild")
        if not data:
            return None

        return {
            "number": data.get("number", 0),
            "result": data.get("result", "UNKNOWN"),
            "duration": data.get("duration", 0),
            "timestamp": data.get("timestamp", 0),
            "url": data.get("url", ""),
            "display_name": data.get("displayName", ""),
        }

    def get_build_log(self, job_name: str, build_number: int) -> Optional[str]:
        """Get build log text.

        Args:
            job_name: Jenkins job name
            build_number: Build number

        Returns:
            Build log text or None if not found
        """
        try:
            url = f"{self.url}/job/{job_name}/{build_number}/consoleText"
            response = requests.get(url, auth=self.auth, timeout=self.timeout, verify=False)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching build log: {str(e)}")
            return None

    def get_queue_length(self) -> int:
        """Get queue length.

        Returns:
            Number of items in queue
        """
        data = self._request("GET", "queue")
        return len(data.get("items", [])) if data else 0

    def trigger_build(self, job_name: str, parameters: Optional[Dict[str, str]] = None) -> bool:
        """Trigger a build.

        Args:
            job_name: Jenkins job name
            parameters: Optional parameters for build

        Returns:
            True if triggered successfully
        """
        try:
            if parameters:
                url = f"{self.url}/job/{job_name}/buildWithParameters"
                response = requests.post(
                    url,
                    auth=self.auth,
                    data=parameters,
                    timeout=self.timeout,
                    verify=False
                )
            else:
                url = f"{self.url}/job/{job_name}/build"
                response = requests.post(url, auth=self.auth, timeout=self.timeout, verify=False)

            response.raise_for_status()
            logger.info(f"Build triggered: {job_name}")
            return True
        except Exception as e:
            logger.error(f"Build trigger failed: {str(e)}")
            return False

    def get_jobs(self) -> List[Dict[str, str]]:
        """Get all jobs.

        Returns:
            List of job information
        """
        data = self._request("GET", "")
        jobs = []

        if data:
            for job in data.get("jobs", []):
                jobs.append({
                    "name": job.get("name", ""),
                    "url": job.get("url", ""),
                    "color": job.get("color", ""),
                })

        return jobs

    def is_healthy(self) -> bool:
        """Check if Jenkins is healthy.

        Returns:
            True if Jenkins is accessible
        """
        return self._request("GET", "") is not None
