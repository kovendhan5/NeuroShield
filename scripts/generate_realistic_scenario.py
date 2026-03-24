#!/usr/bin/env python3
"""
Realistic NeuroShield Data Generator
Generates realistic incident scenarios that mimic real CI/CD failures
"""

import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Incident types with realistic scenarios
INCIDENT_SCENARIOS = {
    "jenkins_failure": {
        "action": "retry_build",
        "base_probability": 0.15,
        "description": "Build test timeout or compilation error",
        "triggers": ["build timeout", "compilation failed", "unit test failed"],
    },
    "pod_crash": {
        "action": "restart_pod",
        "base_probability": 0.08,
        "description": "Pod crash due to high memory/OOM",
        "triggers": ["OOM", "memory limit exceeded", "container crashed"],
    },
    "high_cpu": {
        "action": "scale_up",
        "base_probability": 0.12,
        "description": "CPU spike from heavy processing",
        "triggers": ["cpu > 85%", "thread pool exhausted", "request queue full"],
    },
    "deployment_bad": {
        "action": "rollback_deploy",
        "base_probability": 0.06,
        "description": "Bad deployment causing errors",
        "triggers": ["500 error rate spike", "latency spike", "canary failed"],
    },
    "cache_miss": {
        "action": "clear_cache",
        "base_probability": 0.10,
        "description": "Cache corruption causing app issues",
        "triggers": ["stale cache", "cache consistency error", "session lost"],
    },
}

class RealisticDataGenerator:
    def __init__(self):
        self.base_time = datetime.utcnow() - timedelta(hours=24)
        self.current_time = self.base_time
        self.metrics_history: List[Dict[str, Any]] = []
        self.healing_events: List[Dict[str, Any]] = []

    def generate_realistic_metrics(self) -> Dict[str, float]:
        """Generate realistic system metrics that vary realistically"""
        # Simulate time-based patterns (higher load during business hours)
        hour = self.current_time.hour
        is_business_hours = 9 <= hour <= 18

        # Base metrics
        base_cpu = 40 if is_business_hours else 15
        base_memory = 50 if is_business_hours else 25

        # Add realistic variation
        cpu = base_cpu + random.gauss(0, 15)
        memory = base_memory + random.gauss(0, 12)

        # Occasional spikes
        if random.random() < 0.2:  # 20% chance of spike
            cpu += random.uniform(20, 50)
            memory += random.uniform(15, 40)

        return {
            "cpu": max(0, min(100, cpu)),
            "memory": max(0, min(100, memory)),
            "error_rate": random.uniform(0, 0.15) if random.random() < 0.3 else 0,
            "response_time_ms": 50 + random.gauss(100, 80) if is_business_hours else 40,
            "pod_restarts": random.randint(0, 5),
        }

    def should_trigger_incident(self, metrics: Dict[str, float]) -> bool:
        """Determine if incident should occur based on metrics"""
        cpu, memory = metrics["cpu"], metrics["memory"]
        error_rate = metrics["error_rate"]

        # Incidents more likely when metrics are high
        if cpu > 90 or memory > 85:
            return random.random() < 0.6
        if error_rate > 0.1:
            return random.random() < 0.5
        if metrics["pod_restarts"] > 3:
            return random.random() < 0.4

        # Random background incidents
        return random.random() < 0.15

    def generate_incident(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate realistic incident"""
        # Select incident type based on conditions
        cpu, memory = metrics["cpu"], metrics["memory"]

        if cpu > 85:
            incident_type = "high_cpu"
        elif memory > 85:
            incident_type = "pod_crash"
        elif metrics["error_rate"] > 0.1:
            incident_type = random.choice(["jenkins_failure", "deployment_bad"])
        else:
            incident_type = random.choice(list(INCIDENT_SCENARIOS.keys()))

        scenario = INCIDENT_SCENARIOS[incident_type]
        success = random.random() > (0.15 if cpu > 80 else 0.10)

        return {
            "timestamp": self.current_time.isoformat(),
            "action_id": hash(incident_type) % 10,
            "action_name": scenario["action"],
            "success": success,
            "duration_ms": random.randint(100, 2000),
            "detail": f"{scenario['description']}: {random.choice(scenario['triggers'])}",
            "context": {
                "affected_service": random.choice(["api-service", "frontend", "backend", "database", "cache"]),
                "cpu_usage": f"{metrics['cpu']:.1f}%",
                "memory_usage": f"{metrics['memory']:.1f}%",
                "failure_pattern": incident_type,
            }
        }

    def generate_24h_scenarios(self) -> tuple[List[Dict], List[Dict]]:
        """Generate realistic 24-hour scenario"""
        healing_events = []
        metrics_data = []

        # Generate data points every 15 minutes for 24 hours
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                self.current_time = self.base_time + timedelta(hours=hour, minutes=minute)

                metrics = self.generate_realistic_metrics()
                metrics_data.append({
                    "timestamp": self.current_time.isoformat(),
                    "cpu": metrics["cpu"],
                    "memory": metrics["memory"],
                    "error_rate": metrics["error_rate"],
                    "response_time": metrics["response_time_ms"],
                    "pod_restarts": metrics["pod_restarts"],
                })

                # 40% chance of incident each interval
                if self.should_trigger_incident(metrics):
                    incident = self.generate_incident(metrics)
                    healing_events.append(incident)

        return healing_events, metrics_data


def write_healing_log(events: List[Dict[str, Any]]) -> None:
    """Write events to healing_log.json (NDJSON format)"""
    path = Path("data/healing_log.json")
    path.parent.mkdir(exist_ok=True)

    # Append events
    with open(path, "a") as f:
        for event in sorted(events, key=lambda x: x["timestamp"]):
            f.write(json.dumps(event) + "\n")

    print(f"✓ Generated {len(events)} realistic healing events")

def write_metrics_file(metrics: List[Dict[str, Any]]) -> None:
    """Write metrics for dashboard analytics"""
    path = Path("data/system_metrics.json")
    path.parent.mkdir(exist_ok=True)

    with open(path, "w") as f:
        json.dump({
            "metrics": metrics,
            "generated_at": datetime.utcnow().isoformat(),
            "total_points": len(metrics),
        }, f, indent=2)

    print(f"✓ Generated {len(metrics)} system metrics data points")

if __name__ == "__main__":
    print("\n[GENERATION] Creating Realistic 24-Hour NeuroShield Scenario")
    print("=" * 70)

    generator = RealisticDataGenerator()
    healing_events, metrics = generator.generate_24h_scenarios()

    print(f"\n[DATA] Generated Data:")
    print(f"  + Healing events: {len(healing_events)}")
    print(f"  + Metrics intervals: {len(metrics)}")

    # Calculate statistics
    successful = sum(1 for e in healing_events if e["success"])
    success_rate = (successful / len(healing_events) * 100) if healing_events else 0

    actions = {}
    for e in healing_events:
        action = e["action_name"]
        actions[action] = actions.get(action, 0) + 1

    print(f"\n[STATS] Statistics:")
    print(f"  + Success rate: {success_rate:.1f}%")
    print(f"  + Actions breakdown:")
    for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {action}: {count} incidents")

    # Estimate cost saved
    mttr_per_incident = 87.5  # Average baseline (seconds)
    cost_per_hour = 50  # Rupees per hour
    hours_saved = (len(healing_events) * mttr_per_incident) / 3600
    cost_saved = hours_saved * cost_per_hour

    print(f"  + MTTR saved: {hours_saved:.1f} hours")
    print(f"  + Cost saved: Rs {cost_saved:.2f}")

    # Write files
    print(f"\n[WRITING] Writing data files...")
    write_healing_log(healing_events)
    write_metrics_file(metrics)

    print("\n[OK] Realistic scenario data generation complete!")
    print("     Start dashboard to see live realistic incidents\n")
