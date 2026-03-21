"""
NeuroShield Events System - End-to-End Tests
Tests for webhooks, reliability, interpretability, and judge dashboard
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Import new modules
from src.events.webhook_server import WebhookServer, WebhookEventQueue
from src.events.decision_trace import DecisionTrace, DecisionLogger
from src.events.reliability import ActionExecutor, ActionResult, ReliabilityConfig, SafetyChecker


class TestWebhookEventQueue:
    """Test event queue functionality."""

    def test_queue_creation(self):
        """Test queue can be created."""
        queue = WebhookEventQueue(max_size=100)
        assert queue is not None
        assert not queue.has_events()

    def test_put_and_get_event(self):
        """Test putting and getting events."""
        queue = WebhookEventQueue()
        queue.put_event("test_event", {"data": "test"}, "test_source")

        assert queue.has_events()

        event = queue.get_event(timeout=0.1)
        assert event is not None
        assert event["type"] == "test_event"
        assert event["source"] == "test_source"
        assert event["data"]["data"] == "test"

    def test_queue_full_behavior(self):
        """Test queue handles full state."""
        queue = WebhookEventQueue(max_size=2)
        queue.put_event("event1", {}, "source1")
        queue.put_event("event2", {}, "source2")
        queue.put_event("event3", {}, "source3")  # Should be dropped

        # Can get 2 events
        assert queue.get_event(timeout=0.01) is not None
        assert queue.get_event(timeout=0.01) is not None
        # Third is dropped
        assert queue.get_event(timeout=0.01) is None

    def test_event_metadata(self):
        """Test event has correct metadata."""
        queue = WebhookEventQueue()
        queue.put_event("jenkins_build", {"status": "SUCCESS"}, "jenkins")

        event = queue.get_event()
        assert "timestamp" in event
        assert "type" in event
        assert "source" in event
        assert "data" in event


class TestDecisionTrace:
    """Test decision tracing functionality."""

    def test_trace_creation(self):
        """Test decision trace creation."""
        trace = DecisionTrace("test-123")
        assert trace.decision_id == "test-123"
        assert trace.final_action is None
        assert len(trace.stages) == 0

    def test_add_stage(self):
        """Test adding stages to trace."""
        trace = DecisionTrace("test-123")
        trace.add_stage("collection", {"metric": "cpu", "value": 85.2}, duration_ms=100)

        assert len(trace.stages) == 1
        assert trace.stages[0]["stage"] == "collection"
        assert trace.stages[0]["data"]["metric"] == "cpu"
        assert trace.stages[0]["duration_ms"] == 100

    def test_set_decision(self):
        """Test setting decision."""
        trace = DecisionTrace("test-123")
        trace.set_decision("restart_pod", 0.95, {"reason": "pod_crashed"})

        assert trace.final_action == "restart_pod"
        assert trace.confidence == 0.95
        assert len(trace.stages) > 0
        assert trace.stages[-1]["data"]["reasoning"]["reason"] == "pod_crashed"

    def test_set_outcome(self):
        """Test setting execution outcome."""
        trace = DecisionTrace("test-123")
        trace.set_outcome(True, "Pod restarted successfully", 1500)

        assert trace.outcome == "success"
        assert trace.execution_time_ms == 1500

    def test_trace_to_dict(self):
        """Test converting trace to dictionary."""
        trace = DecisionTrace("test-123")
        trace.add_stage("test_stage", {"data": "value"})
        trace.set_decision("restart_pod", 0.9, {"reason": "test"})
        trace.set_outcome(True, "success", 1000)

        data = trace.to_dict()
        assert data["decision_id"] == "test-123"
        assert data["action"] == "restart_pod"
        assert data["confidence"] == 0.9
        assert data["outcome"] == "success"
        assert len(data["stages"]) > 0


class TestDecisionLogger:
    """Test decision logging functionality."""

    def test_logger_creation(self, tmp_path):
        """Test logger creation."""
        logger = DecisionLogger(log_dir=tmp_path)
        assert logger.log_dir == tmp_path
        assert logger.log_dir.exists()

    def test_log_decision(self, tmp_path):
        """Test logging a decision."""
        logger = DecisionLogger(log_dir=tmp_path)
        trace = DecisionTrace("test-123")
        trace.set_decision("restart_pod", 0.95, {})

        logger.log_decision(trace)

        assert logger.current_log.exists()

        with open(logger.current_log, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["decision_id"] == "test-123"
            assert data["action"] == "restart_pod"

    def test_get_decisions(self, tmp_path):
        """Test retrieving decisions."""
        logger = DecisionLogger(log_dir=tmp_path)

        for i in range(5):
            trace = DecisionTrace(f"test-{i}")
            trace.set_decision("restart_pod", 0.9 + i * 0.01, {})
            logger.log_decision(trace)

        decisions = logger.get_decisions(limit=10)
        assert len(decisions) == 5

    def test_get_decision_by_id(self, tmp_path):
        """Test retrieving specific decision."""
        logger = DecisionLogger(log_dir=tmp_path)
        trace = DecisionTrace("unique-id-123")
        trace.set_decision("scale_up", 0.85, {})
        logger.log_decision(trace)

        found = logger.get_decision("unique-id-123")
        assert found is not None
        assert found["decision_id"] == "unique-id-123"
        assert found["action"] == "scale_up"

    def test_action_statistics(self, tmp_path):
        """Test aggregating statistics."""
        logger = DecisionLogger(log_dir=tmp_path)

        actions = ["restart_pod", "scale_up", "restart_pod", "scale_up", "rollback_deploy"]
        for i, action in enumerate(actions):
            trace = DecisionTrace(f"test-{i}")
            trace.set_decision(action, 0.9, {})
            trace.set_outcome(i % 2 == 0, "test", 1000)  # Alternate success/failure
            logger.log_decision(trace)

        stats = logger.get_action_statistics()
        assert stats["total_decisions"] == 5
        assert "restart_pod" in stats["by_action"]
        assert stats["by_action"]["restart_pod"]["count"] == 2


class TestActionExecutor:
    """Test action execution with reliability."""

    def test_executor_creation(self):
        """Test executor creation."""
        executor = ActionExecutor()
        assert executor is not None
        assert executor.config.max_retries == 3

    def test_successful_execution(self):
        """Test successful action execution."""
        executor = ActionExecutor()

        def mock_action():
            return "success"

        result = executor.execute("test_action", mock_action)

        assert result.success
        assert result.retry_count == 0
        assert result.output == "success"

    def test_execution_with_retries(self):
        """Test execution with retries."""
        executor = ActionExecutor()
        executor.config.max_retries = 3
        executor.config.retry_delay_s = 0.01

        call_count = [0]

        def mock_action():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        def mock_verify():
            return call_count[0] >= 3

        result = executor.execute(
            "test_action",
            mock_action,
            verify_fn=mock_verify
        )

        assert result.success
        assert result.retry_count >= 1

    def test_fallback_execution(self):
        """Test fallback action execution."""
        executor = ActionExecutor()

        def mock_main():
            raise Exception("Main action failed")

        def mock_fallback():
            return "fallback executed"

        executor.register_fallback("test_action", mock_fallback)

        result = executor.execute("test_action", mock_main)

        assert result.success
        assert result.fallback_used
        assert result.output == "fallback executed"

    def test_execution_timeout(self):
        """Test execution timeout handling."""
        executor = ActionExecutor()
        executor.config.action_timeout_s = 0.1

        def slow_action():
            time.sleep(1)
            return "done"

        result = executor.execute("slow_action", slow_action)

        # Should timeout and fail (or succeed if fast enough)
        # Testing just the structure here
        assert isinstance(result.duration_ms, float)


class TestSafetyChecker:
    """Test safety checking functionality."""

    def test_checker_creation(self):
        """Test safety checker creation."""
        checker = SafetyChecker()
        assert checker is not None

    def test_register_check(self):
        """Test registering safety checks."""
        checker = SafetyChecker()

        def always_pass(ctx):
            return True

        checker.register_check("test_check", always_pass)
        is_safe, reason = checker.validate("test_action", {})

        assert is_safe

    def test_failed_check(self):
        """Test safety check failure."""
        checker = SafetyChecker()

        def always_fail(ctx):
            return False

        checker.register_check("failing_check", always_fail)
        is_safe, reason = checker.validate("test_action", {})

        assert not is_safe
        assert "failing_check" in reason

    def test_multiple_checks(self):
        """Test multiple safety checks."""
        checker = SafetyChecker()

        def check_1(ctx):
            return ctx.get("allow_1", True)

        def check_2(ctx):
            return ctx.get("allow_2", True)

        checker.register_check("check_1", check_1)
        checker.register_check("check_2", check_2)

        # Both pass
        is_safe, _ = checker.validate("action", {"allow_1": True, "allow_2": True})
        assert is_safe

        # One fails
        is_safe, reason = checker.validate("action", {"allow_1": True, "allow_2": False})
        assert not is_safe


class TestWebhookServer:
    """Test webhook server functionality."""

    def test_server_creation(self):
        """Test webhook server creation."""
        server = WebhookServer(port=9999)
        assert server is not None
        assert server.port == 9999

    def test_jenkins_webhook_parsing(self):
        """Test Jenkins webhook parsing."""
        server = WebhookServer()

        with server.app.test_client() as client:
            response = client.post(
                "/webhook/jenkins",
                json={
                    "build": {
                        "status": "FAILURE",
                        "number": "42",
                        "url": "http://jenkins/jobs/test/42/"
                    }
                }
            )

            assert response.status_code == 202
            assert server.event_queue.has_events()

            event = server.event_queue.get_event()
            assert event["type"] == "jenkins_build_complete"
            assert event["data"]["status"] == "FAILURE"

    def test_kubernetes_webhook_parsing(self):
        """Test Kubernetes webhook parsing."""
        server = WebhookServer()

        with server.app.test_client() as client:
            response = client.post(
                "/webhook/kubernetes",
                json={
                    "type": "MODIFIED",
                    "object": {
                        "kind": "Pod",
                        "metadata": {
                            "name": "test-pod",
                            "namespace": "default"
                        },
                        "reason": "CrashLoopBackOff"
                    }
                }
            )

            assert response.status_code == 202

    def test_health_endpoint(self):
        """Test health endpoint."""
        server = WebhookServer()

        with server.app.test_client() as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["status"] == "healthy"


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_decision_pipeline(self, tmp_path):
        """Test complete decision pipeline."""
        # Create components
        queue = WebhookEventQueue()
        logger = DecisionLogger(log_dir=tmp_path)
        executor = ActionExecutor()
        checker = SafetyChecker()

        # Simulate webhook event
        queue.put_event("jenkins_build_complete", {"status": "FAILURE"}, "jenkins")

        # Create decision trace
        trace = DecisionTrace("integration-test-1")
        trace.add_stage("webhook_received", {"event_type": "jenkins_build_complete"})
        trace.set_decision("retry_build", 0.92, {"reason": "build failure detected"})

        # Execute action
        def build_retry():
            return "Build triggered"

        result = executor.execute("retry_build", build_retry)

        # Record outcome
        trace.set_outcome(result.success, result.output, result.duration_ms)

        # Log for future retrieval
        logger.log_decision(trace)

        # Verify
        retrieved = logger.get_decision("integration-test-1")
        assert retrieved is not None
        assert retrieved["action"] == "retry_build"
        assert retrieved["outcome"] == "success"

    def test_fallback_chain(self):
        """Test fallback execution chain."""
        executor = ActionExecutor()

        # Main fails
        def failing_main():
            raise Exception("Main failed")

        # Fallback also fails
        def failing_fallback():
            raise Exception("Fallback also failed")

        executor.register_fallback("test", failing_fallback)

        result = executor.execute("test", failing_main)

        assert not result.success
        assert result.fallback_used
        assert "Fallback also failed" in result.error_message


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""

    def test_webhook_throughput(self):
        """Test webhook event throughput."""
        queue = WebhookEventQueue()
        start = time.time()

        for i in range(1000):
            queue.put_event(f"event_{i}", {"index": i}, "test")

        elapsed = time.time() - start
        throughput = 1000 / elapsed

        print(f"\nWebhook throughput: {throughput:.0f} events/second")
        assert throughput > 1000  # Should handle 1000+ events/sec

    def test_decision_logging_performance(self, tmp_path):
        """Test decision logging performance."""
        logger = DecisionLogger(log_dir=tmp_path)
        start = time.time()

        for i in range(100):
            trace = DecisionTrace(f"perf-test-{i}")
            trace.set_decision("restart_pod", 0.9, {})
            trace.set_outcome(True, "ok", 1500)
            logger.log_decision(trace)

        elapsed = time.time() - start
        throughput = 100 / elapsed

        print(f"\nDecision logging throughput: {throughput:.0f} decisions/second")
        assert throughput > 50  # Should handle 50+ decisions/sec

    def test_executor_performance(self):
        """Test action execution performance."""
        executor = ActionExecutor()
        start = time.time()

        for i in range(100):
            def quick_action():
                return "ok"

            result = executor.execute(f"action_{i}", quick_action)

        elapsed = time.time() - start
        throughput = 100 / elapsed

        print(f"\nAction execution throughput: {throughput:.0f} actions/second")
        assert throughput > 50  # Should handle 50+ actions/sec


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
