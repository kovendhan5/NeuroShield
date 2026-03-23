"""Tests for NeuroShield v3 Orchestrator"""

import pytest
import time
from app import Orchestrator, Database
from app.connectors import JenkinsConnector, KubernetesConnector, PrometheusConnector
import yaml
import os


@pytest.fixture
def config():
    """Load test configuration"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def db():
    """Create test database"""
    db_path = "data/test_neuroshield.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    test_db = Database(db_path)
    yield test_db
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def connectors():
    """Create mock connectors"""
    return {
        "jenkins": JenkinsConnector({"url": "http://localhost", "username": "test", "password": "test"}),
        "kubernetes": KubernetesConnector({"namespace": "default"}),
        "prometheus": PrometheusConnector({"url": "http://localhost"}),
    }


@pytest.fixture
def orchestrator(config, db, connectors):
    """Create orchestrator"""
    return Orchestrator(config, db, connectors)


class TestDetection:
    """Test anomaly detection"""

    def test_cpu_spike_detection(self, orchestrator):
        """CPU spike above threshold should be detected"""
        orchestrator.connectors["prometheus"].cpu_base = 85.0
        result = orchestrator.run_cycle()
        anomalies = [a for a in result["anomalies"] if "cpu" in a["type"]]
        assert len(anomalies) > 0

    def test_memory_pressure_detection(self, orchestrator):
        """Memory pressure above threshold should be detected"""
        orchestrator.connectors["prometheus"].memory_base = 90.0
        result = orchestrator.run_cycle()
        anomalies = [a for a in result["anomalies"] if "memory" in a["type"]]
        assert len(anomalies) > 0

    def test_pod_restart_detection(self, orchestrator):
        """Pod restart loop should be detected"""
        orchestrator.connectors["kubernetes"].pod_restarts = 5
        result = orchestrator.run_cycle()
        anomalies = [a for a in result["anomalies"] if "pod_restart" in a["type"]]
        assert len(anomalies) > 0

    def test_no_anomaly_when_healthy(self, orchestrator):
        """No anomalies when system is healthy"""
        result = orchestrator.run_cycle()
        assert len(result["anomalies"]) == 0


class TestDecision:
    """Test healing action decisions"""

    def test_pod_crash_action(self, orchestrator):
        """Pod crash should trigger restart action"""
        orchestrator.connectors["kubernetes"].pod_restarts = 5
        result = orchestrator.run_cycle()
        actions = [a for a in result["actions"] if a["type"] == "restart_pod"]
        assert len(actions) > 0

    def test_cpu_spike_action(self, orchestrator):
        """CPU spike should trigger scale-up"""
        orchestrator.connectors["prometheus"].cpu_base = 85.0
        result = orchestrator.run_cycle()
        actions = [a for a in result["actions"] if a["type"] == "scale_up"]
        assert len(actions) > 0

    def test_memory_action(self, orchestrator):
        """Memory pressure should trigger action"""
        orchestrator.connectors["prometheus"].memory_base = 90.0
        result = orchestrator.run_cycle()
        actions = [a for a in result["actions"]]
        assert any(a["type"] in ["clear_cache", "escalate_to_human"] for a in actions)


class TestExecution:
    """Test action execution"""

    def test_pod_restart_works(self, orchestrator):
        """Pod restart should execute successfully"""
        orchestrator.connectors["kubernetes"].pod_restarts = 5
        result = orchestrator.run_cycle()
        restart_actions = [a for a in result["actions_taken"] if a["action"] == "restart_pod"]
        assert len(restart_actions) > 0
        assert restart_actions[0]["success"] is True

    def test_scale_up_works(self, orchestrator):
        """Scale up should execute successfully"""
        orchestrator.connectors["prometheus"].cpu_base = 85.0
        result = orchestrator.run_cycle()
        scale_actions = [a for a in result["actions_taken"] if a["action"] == "scale_up"]
        assert len(scale_actions) > 0
        assert scale_actions[0]["success"] is True

    def test_action_timing(self, orchestrator):
        """Actions should complete quickly"""
        orchestrator.connectors["kubernetes"].pod_restarts = 5
        result = orchestrator.run_cycle()
        if result["actions_taken"]:
            for action in result["actions_taken"]:
                assert action["duration_ms"] < 10000


class TestDatabase:
    """Test persistence"""

    def test_event_logging(self, db):
        """Events should persist"""
        event_id = db.log_event(
            event_type="cpu_spike",
            severity="critical",
            component="prometheus",
            description="CPU at 85%",
        )
        assert event_id > 0
        events = db.get_recent_events(limit=1)
        assert len(events) > 0

    def test_action_logging(self, db):
        """Actions should persist"""
        action_id = db.log_action(
            action_type="restart_pod",
            status="executing",
            reason="Pod crashed",
        )
        assert action_id > 0

    def test_metrics_logging(self, db):
        """Metrics should persist"""
        db.save_metrics({"cpu_percent": 45.0, "memory_percent": 60.0})
        metrics = db.get_recent_metrics(limit=1)
        assert len(metrics) > 0
        assert metrics[0]["cpu_percent"] == 45.0


class TestIntegration:
    """End-to-end tests"""

    def test_full_cycle(self, orchestrator):
        """Full detect -> decide -> execute cycle"""
        orchestrator.connectors["kubernetes"].pod_restarts = 5
        orchestrator.connectors["prometheus"].cpu_base = 85.0
        result = orchestrator.run_cycle()

        assert len(result["anomalies"]) > 0
        assert len(result["actions"]) > 0
        assert len(result["actions_taken"]) > 0

    def test_cascading_failure(self, orchestrator):
        """Multiple simultaneous failures"""
        orchestrator.connectors["kubernetes"].pod_restarts = 5
        orchestrator.connectors["prometheus"].cpu_base = 85.0
        orchestrator.connectors["jenkins"].inject_failure()
        result = orchestrator.run_cycle()

        assert len(result["anomalies"]) >= 2
        assert len(result["actions_taken"]) > 0
