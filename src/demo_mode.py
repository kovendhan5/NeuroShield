"""
NeuroShield Demo Mode
Deterministic, guaranteed-success scenarios for presentations
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path


class DemoScenario:
    """Single demo scenario with deterministic execution."""

    def __init__(self, name: str, description: str, expected_action: str, duration_seconds: int):
        self.name = name
        self.description = description
        self.expected_action = expected_action
        self.duration_seconds = duration_seconds
        self.start_time = None
        self.status = "pending"  # pending, running, complete, success, failure

    def start(self):
        """Start scenario execution."""
        self.status = "running"
        self.start_time = datetime.now()

    def complete(self, success: bool = True):
        """Complete scenario."""
        self.status = "success" if success else "failure"

    def elapsed_seconds(self) -> float:
        """Get elapsed time."""
        if not self.start_time:
            return 0
        return (datetime.now() - self.start_time).total_seconds()

    def is_complete(self) -> bool:
        """Check if scenario should be complete."""
        return self.elapsed_seconds() >= self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "expected_action": self.expected_action,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds(),
            "progress_percent": min(100, int((self.elapsed_seconds() / self.duration_seconds) * 100)),
        }


class DemoMode:
    """Demo mode manager - controlled scenarios for presentations."""

    SCENARIOS = {
        "pod_crash": DemoScenario(
            name="Pod Crash",
            description="A pod crashes unexpectedly. NeuroShield detects it and restarts the pod within 11.2 seconds.",
            expected_action="restart_pod",
            duration_seconds=12,
        ),
        "cpu_spike": DemoScenario(
            name="CPU Spike",
            description="CPU usage spikes to 85%. NeuroShield scales up the deployment immediately.",
            expected_action="scale_up",
            duration_seconds=15,
        ),
        "memory_pressure": DemoScenario(
            name="Memory Pressure",
            description="Memory usage reaches 92%. NeuroShield clears the cache and reduces load.",
            expected_action="clear_cache",
            duration_seconds=10,
        ),
        "build_failure": DemoScenario(
            name="Build Failure",
            description="Jenkins build fails. NeuroShield detects and retries. Second attempt succeeds.",
            expected_action="retry_build",
            duration_seconds=25,
        ),
        "rollback": DemoScenario(
            name="Bad Deployment",
            description="A bad deployment causes 40% error rate. NeuroShield rolls back to previous working version.",
            expected_action="rollback_deploy",
            duration_seconds=20,
        ),
    }

    def __init__(self):
        self.current_scenario: Optional[DemoScenario] = None
        self.scenario_history: List[DemoScenario] = []
        self.demo_data_dir = Path("demo_data")
        self.demo_data_dir.mkdir(exist_ok=True)

    def start_scenario(self, scenario_name: str) -> bool:
        """Start a demo scenario."""
        if scenario_name not in self.SCENARIOS:
            return False

        scenario = self.SCENARIOS[scenario_name]
        self.current_scenario = scenario
        scenario.start()
        self.scenario_history.append(scenario)

        return True

    def get_scenario_status(self) -> Optional[Dict[str, Any]]:
        """Get current scenario status."""
        if not self.current_scenario:
            return None

        status = self.current_scenario.to_dict()

        # Auto-complete when time expires
        if self.current_scenario.status == "running" and self.current_scenario.is_complete():
            self.current_scenario.complete(success=True)
            status["status"] = "success"

        return status

    def get_deterministic_metrics(self) -> Dict[str, Any]:
        """Get deterministic metrics for current scenario."""
        if not self.current_scenario:
            return {}

        name = self.current_scenario.name
        progress = self.current_scenario.elapsed_seconds() / self.current_scenario.duration_seconds

        # Pre-canned metric sequences
        metrics_by_scenario = {
            "Pod Crash": {
                0.0: {"cpu": 45, "memory": 62, "pod_restarts": 2, "app_health": 0, "error_rate": 0.8},
                0.3: {"cpu": 45, "memory": 62, "pod_restarts": 3, "app_health": 0, "error_rate": 0.95},
                0.6: {"cpu": 50, "memory": 65, "pod_restarts": 0, "app_health": 20, "error_rate": 0.5},
                0.8: {"cpu": 52, "memory": 68, "pod_restarts": 0, "app_health": 85, "error_rate": 0.1},
                1.0: {"cpu": 48, "memory": 65, "pod_restarts": 0, "app_health": 100, "error_rate": 0.0},
            },
            "CPU Spike": {
                0.0: {"cpu": 45, "memory": 62, "replicas": 1, "requests_queued": 100},
                0.3: {"cpu": 75, "memory": 70, "replicas": 1, "requests_queued": 850},
                0.5: {"cpu": 85, "memory": 75, "replicas": 1, "requests_queued": 1200},
                0.7: {"cpu": 62, "memory": 68, "replicas": 3, "requests_queued": 150},
                1.0: {"cpu": 45, "memory": 65, "replicas": 3, "requests_queued": 50},
            },
            "Memory Pressure": {
                0.0: {"memory": 62, "cache_usage": 180, "app_health": 100},
                0.4: {"memory": 85, "cache_usage": 980, "app_health": 40},
                0.6: {"memory": 92, "cache_usage": 1024, "app_health": 15},
                0.8: {"memory": 70, "cache_usage": 150, "app_health": 95},
                1.0: {"memory": 65, "cache_usage": 140, "app_health": 100},
            },
            "Build Failure": {
                0.0: {"build_status": "RUNNING", "tests_passed": 0},
                0.3: {"build_status": "FAILURE", "tests_passed": 0, "test_failures": 5},
                0.5: {"build_status": "RUNNING", "tests_passed": 0},
                0.75: {"build_status": "SUCCESS", "tests_passed": 156},
                1.0: {"build_status": "SUCCESS", "tests_passed": 156},
            },
            "Bad Deployment": {
                0.0: {"deployment_status": "NEW", "error_rate": 0.0, "latency_ms": 50},
                0.2: {"deployment_status": "ROLLING", "error_rate": 0.15, "latency_ms": 150},
                0.4: {"deployment_status": "COMPLETE", "error_rate": 0.4, "latency_ms": 2000},
                0.7: {"deployment_status": "ROLLBACK", "error_rate": 0.1, "latency_ms": 200},
                1.0: {"deployment_status": "STABLE", "error_rate": 0.0, "latency_ms": 50},
            },
        }

        scenario_metrics = metrics_by_scenario.get(name, {})

        # Find closest metric point
        closest_point = 0.0
        for point_progress in sorted(scenario_metrics.keys()):
            if point_progress <= progress:
                closest_point = point_progress
            else:
                break

        return scenario_metrics.get(closest_point, {})

    def get_deterministic_decision(self) -> Optional[Dict[str, Any]]:
        """Get the pre-determined decision stages."""
        if not self.current_scenario:
            return None

        progress = self.current_scenario.elapsed_seconds() / self.current_scenario.duration_seconds

        stages = {
            0.1: {  # Failure detected
                "stage": "detection",
                "message": "Failure detected via webhooks",
                "icon": "⚡",
            },
            0.3: {  # Data collection
                "stage": "collection",
                "message": "Telemetry collected from Jenkins, Prometheus, Kubernetes",
                "icon": "📊",
            },
            0.5: {  # Prediction
                "stage": "prediction",
                "message": "DistilBERT: Failure probability 92%",
                "icon": "🧠",
            },
            0.7: {  # Decision
                "stage": "decision",
                "message": f"PPO Agent decided: {self.current_scenario.expected_action}",
                "icon": "🤖",
                "confidence": 0.96,
            },
            1.0: {  # Execution
                "stage": "execution",
                "message": "Action executed successfully",
                "icon": "✓",
                "result": "success",
                "mttr_seconds": self.current_scenario.duration_seconds * 0.9,
            },
        }

        # Find current stage
        for stage_progress in sorted(stages.keys(), reverse=True):
            if progress >= stage_progress:
                return stages[stage_progress]

        return None

    def export_scenario_data(self):
        """Export deterministic data for dashboard consumption."""
        if not self.current_scenario:
            return

        data = {
            "scenario": self.current_scenario.to_dict(),
            "metrics": self.get_deterministic_metrics(),
            "decision_stage": self.get_deterministic_decision() or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Save for dashboard to read
        output_file = self.demo_data_dir / f"{self.current_scenario.name}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def get_all_scenarios() -> List[Dict[str, Any]]:
        """Get all available scenarios."""
        return [
            {
                "name": name,
                "description": scenario.description,
                "expected_action": scenario.expected_action,
                "duration_seconds": scenario.duration_seconds,
            }
            for name, scenario in DemoMode.SCENARIOS.items()
        ]


# Global instance
_demo_mode = DemoMode()


def get_demo_mode() -> DemoMode:
    """Get demo mode instance."""
    return _demo_mode


# Usage
if __name__ == "__main__":
    demo = get_demo_mode()

    # Start a scenario
    demo.start_scenario("pod_crash")

    # Simulate running through it
    for i in range(13):
        time.sleep(1)
        metrics = demo.get_deterministic_metrics()
        decision = demo.get_deterministic_decision()
        status = demo.get_scenario_status()

        print(f"\n[{i}s] {status}")
        print(f"Metrics: {metrics}")
        print(f"Decision: {decision}")

        if status["status"] == "success":
            print("\n✓ Demo completed successfully!")
            break
