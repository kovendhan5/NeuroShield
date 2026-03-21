"""
NeuroShield Auto-Recovery
Self-healing for the orchestrator itself
"""

import subprocess
import time
from typing import Optional
from datetime import datetime, timedelta


class ServiceHealthCheck:
    """Check and recover services."""

    @staticmethod
    def check_kubernetes() -> bool:
        """Check if Kubernetes is responsive."""
        try:
            subprocess.run(
                "kubectl get nodes",
                shell=True,
                capture_output=True,
                timeout=5,
                check=True
            )
            return True
        except:
            return False

    @staticmethod
    def check_jenkins() -> bool:
        """Check Jenkins health."""
        try:
            import requests
            response = requests.get("http://localhost:8080/", timeout=5)
            return response.status_code < 400
        except:
            return False

    @staticmethod
    def check_prometheus() -> bool:
        """Check Prometheus health."""
        try:
            import requests
            response = requests.get("http://localhost:9090/-/healthy", timeout=5)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def check_orchestrator() -> bool:
        """Check orchestrator health."""
        try:
            import requests
            response = requests.get("http://localhost:8502/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def check_dashboard() -> bool:
        """Check dashboard health."""
        try:
            import requests
            response = requests.get("http://localhost:8501/", timeout=5)
            return response.status_code < 500
        except:
            return False


class AutoRecovery:
    """Auto-recovery mechanisms."""

    def __init__(self):
        self.last_check = {}
        self.failure_counts = {}
        self.disabled_services = set()

    def check_all_services(self) -> dict:
        """Check all services and auto-recover if needed."""
        checks = {
            "kubernetes": ServiceHealthCheck.check_kubernetes(),
            "jenkins": ServiceHealthCheck.check_jenkins(),
            "prometheus": ServiceHealthCheck.check_prometheus(),
            "orchestrator": ServiceHealthCheck.check_orchestrator(),
            "dashboard": ServiceHealthCheck.check_dashboard(),
        }

        # Attempt recovery for failed services
        for service, healthy in checks.items():
            if not healthy:
                self._attempt_recovery(service)

        return checks

    def _attempt_recovery(self, service: str):
        """Attempt to recover a failing service."""
        # Increment failure count
        self.failure_counts[service] = self.failure_counts.get(service, 0) + 1

        failure_count = self.failure_counts[service]

        # Progressive recovery
        if failure_count == 1:
            # First failure: wait and retry
            time.sleep(5)
            self._restart_service(service)

        elif failure_count == 2:
            # Second failure: restart all Docker services
            self._restart_all_docker_services()

        elif failure_count == 3:
            # Third failure: full system restart
            self._restart_full_system()

        elif failure_count >= 4:
            # Fourth+ failure: disable service and alert
            self.disabled_services.add(service)
            self._escalate_alert(service)

    def _restart_service(self, service: str):
        """Restart a specific service."""
        if service == "kubernetes":
            subprocess.run("minikube start", shell=True, capture_output=True)

        elif service == "jenkins":
            subprocess.run("docker restart neuroshield-jenkins", shell=True, capture_output=True)

        elif service == "prometheus":
            subprocess.run("docker restart neuroshield-prometheus", shell=True, capture_output=True)

        elif service == "orchestrator":
            subprocess.run("docker restart neuroshield-orchestrator", shell=True, capture_output=True)

        elif service == "dashboard":
            subprocess.run("docker restart neuroshield-dashboard", shell=True, capture_output=True)

    def _restart_all_docker_services(self):
        """Restart all Docker services."""
        subprocess.run("docker-compose -f docker-compose.yml restart", shell=True, capture_output=True)

    def _restart_full_system(self):
        """Full system restart."""
        subprocess.run("docker-compose -f docker-compose.yml down", shell=True, capture_output=True)
        time.sleep(5)
        subprocess.run("docker-compose -f docker-compose.yml up -d", shell=True, capture_output=True)

    def _escalate_alert(self, service: str):
        """Send escalation alert."""
        from src.logging_system import get_logger
        logger = get_logger()

        logger.critical(
            f"Service {service} failed to recover after multiple attempts",
            source="auto_recovery",
            context={"service": service, "failure_count": self.failure_counts[service]}
        )

    def reset_failure_count(self, service: str):
        """Reset failure count when service recovers."""
        self.failure_counts[service] = 0

    def is_service_disabled(self, service: str) -> bool:
        """Check if service is disabled."""
        return service in self.disabled_services

    def get_status(self) -> dict:
        """Get auto-recovery status."""
        return {
            "disabled_services": list(self.disabled_services),
            "failure_counts": self.failure_counts,
        }


class OrchestrationRecovery:
    """Recovery for orchestr orchestrator-specific issues."""

    @staticmethod
    def recover_from_crash(crash_info: dict):
        """Recover orchestrator from crash."""
        from src.state_manager import get_state_manager
        from src.logging_system import get_logger

        logger = get_logger()
        state_mgr = get_state_manager()

        # Save crash info
        logger.error(
            "Orchestrator crashed, attempting recovery",
            source="orchestration_recovery",
            context=crash_info
        )

        # Stop current process
        try:
            subprocess.run("pkill -f 'python.*main.py'", shell=True, capture_output=True)
        except:
            pass

        # Wait
        time.sleep(5)

        # Restart
        logger.info("Restarting orchestrator...", source="orchestration_recovery")
        subprocess.Popen("python src/orchestrator/main.py", shell=True)

        logger.info("Orchestrator restarted", source="orchestration_recovery")

    @staticmethod
    def recover_from_memory_leak(memory_usage_percent: float):
        """Recover from memory issues."""
        from src.logging_system import get_logger

        logger = get_logger()

        if memory_usage_percent > 90:
            logger.warn(
                "High memory usage detected, restarting component",
                source="memory_recovery",
                context={"memory_percent": memory_usage_percent}
            )

            # Kill oldest log entries
            try:
                from src.logging_system import get_logger as get_logger_fn
                logger_inst = get_logger_fn()
                logger_inst.clear_old_logs(days=1)
            except:
                pass

            # Restart if still critical
            if memory_usage_percent > 95:
                subprocess.run("docker restart neuroshield-orchestrator", shell=True)

    @staticmethod
    def recover_from_stuck_action(action_id: str, stuck_duration_seconds: int = 300):
        """Recover from stuck healing action."""
        from src.logging_system import get_logger

        logger = get_logger()

        if stuck_duration_seconds > 300:  # 5 minutes
            logger.warn(
                f"Action {action_id} appears stuck, forcing termination",
                source="action_recovery",
                context={"action_id": action_id, "duration": stuck_duration_seconds}
            )

            # Mark action as failed
            try:
                from src.state_manager import get_state_manager
                state_mgr = get_state_manager()
                # TODO: Mark action as failed in database
            except:
                pass


# Global instance
_auto_recovery = AutoRecovery()


def get_auto_recovery() -> AutoRecovery:
    """Get auto-recovery instance."""
    return _auto_recovery


# Background thread
def start_auto_recovery_monitor(check_interval_seconds: int = 60):
    """Start auto-recovery monitor in background thread."""
    import threading

    def monitor_loop():
        while True:
            try:
                checks = get_auto_recovery().check_all_services()
                # Log results
                healthy_count = sum(1 for v in checks.values() if v)
                if healthy_count < len(checks):
                    from src.logging_system import get_logger
                    logger = get_logger()
                    logger.warn(
                        f"Service health check: {healthy_count}/{len(checks)} healthy",
                        source="auto_recovery",
                        context=checks
                    )
            except Exception as e:
                pass

            time.sleep(check_interval_seconds)

    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    return thread
