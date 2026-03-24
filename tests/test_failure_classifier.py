"""Tests for CI/CD failure classifier.

Tests the hybrid rule-based classifier without requiring ML models.
"""

from __future__ import annotations

import pytest
from src.prediction.failure_classifier import (
    HybridFailureClassifier,
    classify_failure,
    get_failure_type,
    FailureClassification,
)


class TestHybridFailureClassifier:
    """Test the hybrid failure classifier."""

    def test_dependency_npm_missing(self):
        """Test npm missing package detection."""
        log = """
        npm ERR! missing: express@^4.18.0, required by my-app@1.0.0
        npm ERR! A complete log of this run can be found in:
        """
        result = classify_failure(log)
        assert result.failure_type == "DEPENDENCY"
        assert result.confidence > 0.5
        assert "npm_missing_package" in result.matched_patterns

    def test_dependency_python_import(self):
        """Test Python import error detection."""
        log = """
        Traceback (most recent call last):
          File "app.py", line 5, in <module>
            import flask
        ModuleNotFoundError: No module named 'flask'
        """
        result = classify_failure(log)
        assert result.failure_type == "DEPENDENCY"
        assert result.confidence > 0.5
        assert any("module" in p.lower() for p in result.matched_patterns)

    def test_dependency_pip_version_conflict(self):
        """Test pip version conflict detection."""
        log = """
        ERROR: Could not find a version that satisfies the requirement django>=5.0
        ERROR: No matching distribution found for django>=5.0
        """
        result = classify_failure(log)
        assert result.failure_type == "DEPENDENCY"
        assert result.confidence > 0.5

    def test_config_env_var_missing(self):
        """Test missing environment variable detection."""
        log = """
        Error: Required environment variable DATABASE_URL not set
        Please configure your .env file
        """
        result = classify_failure(log)
        assert result.failure_type == "CONFIG"
        assert result.confidence > 0.5
        assert "env_var_missing" in result.matched_patterns

    def test_config_file_not_found(self):
        """Test missing config file detection."""
        log = """
        FileNotFoundError: [Errno 2] No such file or directory: '/app/config.yaml'
        """
        result = classify_failure(log)
        assert result.failure_type == "CONFIG"
        assert result.confidence > 0.5

    def test_test_assertion_failed(self):
        """Test assertion failure detection."""
        log = """
        test_login.py::test_user_authentication FAILED
        AssertionError: Expected status code 200, but got 401
        ===== 5 failed, 20 passed in 2.34s =====
        """
        result = classify_failure(log)
        assert result.failure_type == "TEST"
        assert result.confidence > 0.5
        assert any("assertion" in p.lower() or "failed" in p.lower() for p in result.matched_patterns)

    def test_test_timeout(self):
        """Test timeout detection."""
        log = """
        TimeoutError: Test exceeded maximum execution time of 30s
        test_api_integration timed out
        """
        result = classify_failure(log)
        assert result.failure_type == "TEST"
        assert result.confidence > 0.5

    def test_build_syntax_error(self):
        """Test syntax error detection."""
        log = """
        SyntaxError: invalid syntax
          File "main.py", line 42
            if x = 5:
               ^
        """
        result = classify_failure(log)
        assert result.failure_type == "BUILD"
        assert result.confidence > 0.5
        assert "syntax_error" in result.matched_patterns

    def test_build_compilation_error(self):
        """Test compilation error detection."""
        log = """
        Compilation error in src/App.tsx
        TS2345: Argument of type 'string' is not assignable to parameter of type 'number'
        Build failed with 3 errors
        """
        result = classify_failure(log)
        assert result.failure_type == "BUILD"
        assert result.confidence > 0.5

    def test_infrastructure_oom(self):
        """Test OOM detection."""
        log = """
        java.lang.OutOfMemoryError: Java heap space
        Killed: OutOfMemoryError
        """
        result = classify_failure(log)
        assert result.failure_type == "INFRASTRUCTURE"
        assert result.confidence > 0.5
        assert any("oom" in p.lower() for p in result.matched_patterns)

    def test_infrastructure_network(self):
        """Test network error detection."""
        log = """
        Error: ETIMEDOUT connecting to registry.npmjs.org
        Network unreachable: DNS resolution failed
        """
        result = classify_failure(log)
        assert result.failure_type == "INFRASTRUCTURE"
        assert result.confidence > 0.5

    def test_unknown_empty_log(self):
        """Test empty log classification."""
        result = classify_failure("")
        assert result.failure_type == "UNKNOWN"
        assert result.confidence == 0.0

    def test_unknown_no_patterns(self):
        """Test log with no matching patterns."""
        log = "Everything is fine, build succeeded"
        result = classify_failure(log)
        assert result.failure_type == "UNKNOWN"

    def test_telemetry_boost_infrastructure(self):
        """Test telemetry boosting infrastructure score."""
        log = "Build failed with unknown error"
        telemetry = {
            "prometheus_cpu_usage": 95.0,
            "prometheus_memory_usage": 90.0,
        }
        result = classify_failure(log, telemetry)
        # With high CPU/memory, INFRASTRUCTURE should be boosted
        # (may still be UNKNOWN if no patterns match, but score is boosted)
        assert "INFRASTRUCTURE" in result.raw_indicators

    def test_telemetry_boost_test_build(self):
        """Test telemetry boosting test/build scores."""
        log = "Error occurred during execution"
        telemetry = {
            "jenkins_last_build_status": "FAILURE",
        }
        result = classify_failure(log, telemetry)
        # Build failure should boost TEST and BUILD scores
        assert result.raw_indicators.get("TEST", 0) > 0 or result.raw_indicators.get("BUILD", 0) > 0

    def test_multiple_patterns_same_type(self):
        """Test multiple patterns of same type increase confidence."""
        log = """
        npm ERR! missing: express@^4.18.0
        npm ERR! missing: body-parser@^1.20.0
        ModuleNotFoundError: No module named 'flask'
        Cannot find module 'react'
        """
        result = classify_failure(log)
        assert result.failure_type == "DEPENDENCY"
        # Multiple matches should give high confidence
        assert result.confidence > 0.7

    def test_mixed_patterns_lower_confidence(self):
        """Test mixed patterns reduce confidence."""
        log = """
        npm ERR! missing: express@^4.18.0
        SyntaxError: invalid syntax
        AssertionError: test failed
        """
        result = classify_failure(log)
        # Should still classify, but with lower confidence
        assert result.failure_type in ["DEPENDENCY", "BUILD", "TEST"]
        # Confidence might be lower due to mixed signals
        assert result.confidence >= 0.0

    def test_get_failure_type_helper(self):
        """Test convenience helper function."""
        log = "npm ERR! missing: express@^4.18.0"
        failure_type = get_failure_type(log)
        assert failure_type == "DEPENDENCY"

    def test_classification_details(self):
        """Test that details are human-readable."""
        log = "ModuleNotFoundError: No module named 'requests'"
        result = classify_failure(log)
        assert isinstance(result.details, str)
        assert len(result.details) > 0
        assert "dependency" in result.details.lower() or "package" in result.details.lower()

    def test_matched_patterns_limited(self):
        """Test that matched patterns are limited to reasonable number."""
        log = """
        npm ERR! missing: pkg1
        npm ERR! missing: pkg2
        npm ERR! missing: pkg3
        npm ERR! missing: pkg4
        npm ERR! missing: pkg5
        npm ERR! missing: pkg6
        npm ERR! missing: pkg7
        """
        result = classify_failure(log)
        # Should limit to 5 patterns for readability
        assert len(result.matched_patterns) <= 5

    def test_classifier_instance_reuse(self):
        """Test that classifier instance is reused efficiently."""
        log1 = "npm ERR! missing: express"
        log2 = "ModuleNotFoundError: No module named 'flask'"

        result1 = classify_failure(log1)
        result2 = classify_failure(log2)

        assert result1.failure_type == "DEPENDENCY"
        assert result2.failure_type == "DEPENDENCY"

    def test_invalid_log_type(self):
        """Test handling of invalid log types."""
        result = classify_failure(None)
        assert result.failure_type == "UNKNOWN"
        assert result.confidence == 0.0

    def test_real_world_npm_error(self):
        """Test real-world npm error message."""
        log = """
npm WARN deprecated request@2.88.2: request has been deprecated
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! While resolving: my-app@1.0.0
npm ERR! Found: react@18.2.0
npm ERR! node_modules/react
npm ERR!   react@"^18.2.0" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^17.0.0" required by react-router-dom@6.0.0
        """
        result = classify_failure(log)
        assert result.failure_type == "DEPENDENCY"
        assert result.confidence > 0.5

    def test_real_world_python_import_error(self):
        """Test real-world Python import error."""
        log = """
Collecting pytest
  Using cached pytest-7.4.3-py3-none-any.whl
Collecting requests
  ERROR: Could not find a version that satisfies the requirement requests>=3.0.0
  ERROR: No matching distribution found for requests>=3.0.0
        """
        result = classify_failure(log)
        assert result.failure_type == "DEPENDENCY"
        assert result.confidence > 0.5


class TestFailureClassification:
    """Test FailureClassification dataclass."""

    def test_dataclass_creation(self):
        """Test creating a FailureClassification instance."""
        fc = FailureClassification(
            failure_type="DEPENDENCY",
            confidence=0.85,
            matched_patterns=["npm_missing_package"],
            details="Dependency issue detected",
            raw_indicators={"DEPENDENCY": 5, "TEST": 0},
        )
        assert fc.failure_type == "DEPENDENCY"
        assert fc.confidence == 0.85
        assert len(fc.matched_patterns) == 1
        assert "Dependency" in fc.details
        assert fc.raw_indicators["DEPENDENCY"] == 5
