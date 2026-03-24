"""Tests for CI/CD auto-fix engine."""

from __future__ import annotations

import pytest
from src.orchestrator.cicd_fixer import (
    CICDAutoFixer,
    fix_cicd_failure,
    FixResult,
)


class TestCICDAutoFixer:
    """Test the CI/CD auto-fix engine."""

    def test_dependency_npm_extraction(self):
        """Test extraction of npm package names."""
        fixer = CICDAutoFixer(dry_run=True)
        log = """
        npm ERR! missing: express@^4.18.0, required by my-app@1.0.0
        npm ERR! missing: body-parser@^1.20.0
        Cannot find module 'react'
        """
        packages = fixer._extract_npm_missing_packages(log)
        assert "express" in packages
        assert "body-parser" in packages
        assert "react" in packages

    def test_dependency_python_extraction(self):
        """Test extraction of Python package names."""
        fixer = CICDAutoFixer(dry_run=True)
        log = """
        ModuleNotFoundError: No module named 'flask'
        ImportError: cannot import name 'requests'
        """
        packages = fixer._extract_python_missing_packages(log)
        assert "flask" in packages
        assert "requests" in packages

    def test_npm_install_dry_run(self):
        """Test npm install in dry-run mode."""
        log = "npm ERR! missing: express@^4.18.0"
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        assert result.success is True
        assert result.fix_type == "npm_install"
        assert "[DRY-RUN]" in result.details
        assert result.reversible is True

    def test_pip_install_dry_run(self):
        """Test pip install in dry-run mode."""
        log = "ModuleNotFoundError: No module named 'flask'"
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        assert result.success is True
        assert result.fix_type == "pip_install"
        assert "[DRY-RUN]" in result.details
        assert result.reversible is True

    def test_npm_cache_clear_dry_run(self):
        """Test npm cache clear in dry-run mode."""
        log = "npm ERR! some cache error"
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        # Should default to cache clear if no specific packages found
        assert result.fix_type in ["npm_cache_clear", "npm_install"]
        assert result.reversible is True

    def test_pip_cache_clear_dry_run(self):
        """Test pip cache clear in dry-run mode."""
        log = "ImportError: some pip error"
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        # Should default to cache clear if no specific packages found
        assert result.fix_type in ["pip_cache_clear", "pip_install", "dependency_unknown"]
        assert result.reversible is True

    def test_config_recommendation(self):
        """Test config fix recommendations."""
        log = "Error: Required environment variable DATABASE_URL not set"
        result = fix_cicd_failure("CONFIG", log, "test-job", 123, dry_run=True)
        assert result.success is False  # Requires manual intervention
        assert result.fix_type == "config_recommendation"
        assert "environment variable" in result.details.lower()
        assert result.reversible is True

    def test_config_file_missing(self):
        """Test config file missing recommendation."""
        log = "FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'"
        result = fix_cicd_failure("CONFIG", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert result.fix_type == "config_recommendation"
        assert "file path" in result.details.lower()

    def test_test_flaky_recommendation(self):
        """Test flaky test recommendation."""
        log = "Test is flaky and failed intermittently"
        result = fix_cicd_failure("TEST", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert result.fix_type == "test_recommendation"
        assert "retry" in result.details.lower() or "flaky" in result.details.lower()

    def test_test_timeout_recommendation(self):
        """Test timeout recommendation."""
        log = "TimeoutError: Test exceeded maximum execution time"
        result = fix_cicd_failure("TEST", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert result.fix_type == "test_recommendation"
        assert "timeout" in result.details.lower()

    def test_build_recommendation(self):
        """Test build error recommendation."""
        log = "SyntaxError: invalid syntax at line 42"
        result = fix_cicd_failure("BUILD", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert result.fix_type == "build_recommendation"
        assert "syntax" in result.details.lower() or "compilation" in result.details.lower()

    def test_infrastructure_no_fix(self):
        """Test infrastructure failures don't have auto-fix."""
        log = "OutOfMemoryError: Java heap space"
        result = fix_cicd_failure("INFRASTRUCTURE", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert result.fix_type == "none"
        assert "no automated fix" in result.details.lower()

    def test_unknown_no_fix(self):
        """Test unknown failures don't have auto-fix."""
        log = "Some mysterious error"
        result = fix_cicd_failure("UNKNOWN", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert result.fix_type == "none"

    def test_safe_package_name_validation(self):
        """Test package name safety validation."""
        fixer = CICDAutoFixer(dry_run=True)
        assert fixer._is_safe_package_name("express") is True
        assert fixer._is_safe_package_name("@types/node") is True
        assert fixer._is_safe_package_name("lodash.get") is True
        assert fixer._is_safe_package_name("flask_cors") is True
        # Unsafe names
        assert fixer._is_safe_package_name("rm -rf /") is False
        assert fixer._is_safe_package_name("express; rm -rf") is False
        assert fixer._is_safe_package_name("../../etc/passwd") is False

    def test_fix_result_structure(self):
        """Test FixResult dataclass structure."""
        result = FixResult(
            success=True,
            fix_type="test_fix",
            duration_ms=123.45,
            details="Test details",
            actions_taken=["action1", "action2"],
            reversible=True,
        )
        assert result.success is True
        assert result.fix_type == "test_fix"
        assert result.duration_ms == 123.45
        assert result.details == "Test details"
        assert len(result.actions_taken) == 2
        assert result.reversible is True

    def test_multiple_packages_limit(self):
        """Test that package extraction is limited."""
        fixer = CICDAutoFixer(dry_run=True)
        log = """
        npm ERR! missing: pkg1
        npm ERR! missing: pkg2
        npm ERR! missing: pkg3
        npm ERR! missing: pkg4
        npm ERR! missing: pkg5
        npm ERR! missing: pkg6
        npm ERR! missing: pkg7
        """
        packages = fixer._extract_npm_missing_packages(log)
        # Should be limited to 5 packages
        assert len(packages) <= 5

    def test_dry_run_flag_preserved(self):
        """Test dry-run flag is preserved across calls."""
        log = "npm ERR! missing: express"
        result1 = fix_cicd_failure("DEPENDENCY", log, dry_run=True)
        result2 = fix_cicd_failure("DEPENDENCY", log, dry_run=True)
        assert "[DRY-RUN]" in result1.details
        assert "[DRY-RUN]" in result2.details

    def test_real_npm_install_log(self):
        """Test with real npm install failure log."""
        log = """
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! While resolving: my-app@1.0.0
npm ERR! Found: react@18.2.0
npm ERR! node_modules/react
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^17.0.0" required by react-router-dom@6.0.0
        """
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        # Should attempt some kind of fix
        assert result.fix_type in ["npm_cache_clear", "npm_install"]

    def test_real_python_import_error(self):
        """Test with real Python import error."""
        log = """
Traceback (most recent call last):
  File "app.py", line 5, in <module>
    import flask
ModuleNotFoundError: No module named 'flask'
        """
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        assert result.fix_type == "pip_install"
        assert result.success is True  # In dry-run mode

    def test_fixer_instance_creation(self):
        """Test fixer instance is created correctly."""
        fixer = CICDAutoFixer(dry_run=True, log_dir="test_logs")
        assert fixer.dry_run is True
        assert "test_logs" in str(fixer.log_dir)

    def test_telemetry_passed_to_fix(self):
        """Test telemetry is passed to fix methods."""
        log = "Some error"
        telemetry = {"prometheus_cpu_usage": 95.0}
        result = fix_cicd_failure("TEST", log, "test-job", 123, telemetry, dry_run=True)
        # Should complete without error
        assert result.fix_type in ["test_recommendation", "test_manual"]

    def test_duration_tracking(self):
        """Test duration is tracked in result."""
        log = "npm ERR! missing: express"
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        assert result.duration_ms >= 0.0

    def test_actions_taken_logging(self):
        """Test actions_taken list is populated."""
        log = "npm ERR! missing: express"
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        assert len(result.actions_taken) > 0
        assert any("[DRY-RUN]" in action for action in result.actions_taken)

    def test_reversibility_flag(self):
        """Test all operations are marked as reversible."""
        test_cases = [
            ("DEPENDENCY", "npm ERR! missing: express"),
            ("CONFIG", "Error: environment variable not set"),
            ("TEST", "Test timeout"),
            ("BUILD", "SyntaxError"),
            ("INFRASTRUCTURE", "OutOfMemoryError"),
        ]
        for failure_type, log in test_cases:
            result = fix_cicd_failure(failure_type, log, "test", 1, dry_run=True)
            assert result.reversible is True  # All operations should be reversible

    def test_dependency_unknown_package_manager(self):
        """Test dependency fix with unknown package manager."""
        log = "Some dependency error with no package manager keywords"
        result = fix_cicd_failure("DEPENDENCY", log, "test-job", 123, dry_run=True)
        # Should handle gracefully
        assert result.fix_type == "dependency_unknown"
        assert result.success is False

    def test_config_generic_error(self):
        """Test config fix with generic error."""
        log = "Configuration is invalid"
        result = fix_cicd_failure("CONFIG", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert "manual" in result.fix_type or "recommendation" in result.fix_type

    def test_test_generic_failure(self):
        """Test test fix with generic test failure."""
        log = "Tests failed"
        result = fix_cicd_failure("TEST", log, "test-job", 123, dry_run=True)
        assert result.success is False
        assert "test" in result.fix_type.lower()
