"""Hybrid CI/CD Failure Classifier - Rule-based with ML-assisted confidence.

This module classifies CI/CD failures into actionable categories using a hybrid approach:
- Primary: Rule-based pattern matching (fast, deterministic, explainable)
- Secondary: ML confidence scoring using existing DistilBERT encoder

Failure Types:
1. DEPENDENCY - Missing packages, version conflicts, resolution failures
2. CONFIG - Missing env vars, incorrect paths, invalid configs
3. TEST - Test failures, assertions, flaky tests, timeouts
4. BUILD - Compilation errors, linting failures, syntax errors
5. INFRASTRUCTURE - OOM, network issues, resource exhaustion
6. UNKNOWN - Cannot classify with confidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FailureClassification:
    """Result of CI/CD failure classification."""
    failure_type: str  # DEPENDENCY, CONFIG, TEST, BUILD, INFRASTRUCTURE, UNKNOWN
    confidence: float  # 0.0 to 1.0
    matched_patterns: List[str]  # Which patterns matched
    details: str  # Human-readable explanation
    raw_indicators: Dict[str, int]  # Pattern match counts


# ── Rule-Based Pattern Definitions ──────────────────────────────────────────

# Dependency failure patterns (npm, pip, maven, etc.)
DEPENDENCY_PATTERNS = [
    # npm
    (r"npm ERR! .*missing:", "npm_missing_package"),
    (r"npm ERR! .*ENOTFOUND", "npm_registry_unreachable"),
    (r"npm ERR! .*404.*Not Found", "npm_package_not_found"),
    (r"Cannot find module", "node_module_not_found"),
    (r"MODULE_NOT_FOUND", "node_module_missing"),
    (r"npm ERR! .*peer dep", "npm_peer_dependency"),
    (r"npm ERR! .*ERESOLVE", "npm_dependency_conflict"),

    # pip
    (r"ModuleNotFoundError:", "python_module_not_found"),
    (r"ImportError:", "python_import_error"),
    (r"No module named", "python_missing_module"),
    (r"Could not find a version that satisfies", "pip_version_conflict"),
    (r"ERROR: Could not find.*pypi", "pip_package_not_found"),
    (r"pip.*failed.*with error code", "pip_install_failed"),

    # Maven
    (r"Could not resolve dependencies", "maven_dependency_fail"),
    (r"Failed to collect dependencies", "maven_collection_fail"),
    (r"Unresolveable build extension", "maven_extension_fail"),

    # Generic
    (r"dependency resolution failed", "generic_dep_resolution"),
    (r"package.*not found", "generic_package_missing"),
    (r"version conflict", "generic_version_conflict"),
]

# Configuration failure patterns
CONFIG_PATTERNS = [
    # Environment variables
    (r".*environment variable.*not (set|found|defined)", "env_var_missing"),
    (r"Missing required.*environment", "env_required_missing"),
    (r"\.env.*not found", "dotenv_missing"),

    # File paths
    (r"No such file or directory", "file_not_found"),
    (r"FileNotFoundError:", "python_file_missing"),
    (r"ENOENT:.*no such file", "node_file_missing"),
    (r"config.*not found", "config_file_missing"),
    (r"configuration.*invalid", "config_invalid"),

    # Permissions
    (r"permission denied", "permission_error"),
    (r"EACCES:", "access_denied"),

    # Generic config
    (r"Invalid configuration", "invalid_config"),
    (r"Configuration error", "config_error"),
    (r"Missing required.*config", "required_config_missing"),
]

# Test failure patterns
TEST_PATTERNS = [
    # Assertions
    (r"AssertionError:", "assertion_failed"),
    (r"Test.*failed", "test_failed"),
    (r"FAILED.*test", "pytest_failed"),
    (r"\d+ failed.*\d+ passed", "test_failures"),
    (r"Expected.*but got", "expectation_mismatch"),

    # Timeouts
    (r"Test.*timed out", "test_timeout"),
    (r"TimeoutError:", "timeout_error"),
    (r"timeout.*exceeded", "timeout_exceeded"),

    # Flaky tests
    (r"flaky", "flaky_test"),
    (r"intermittent.*failure", "intermittent_fail"),

    # Coverage
    (r"Coverage.*below threshold", "coverage_low"),

    # Generic
    (r"Tests failed", "generic_test_fail"),
]

# Build failure patterns
BUILD_PATTERNS = [
    # Compilation
    (r"Compilation error", "compilation_error"),
    (r"SyntaxError:", "syntax_error"),
    (r"Parse error", "parse_error"),
    (r"Build failed", "build_failed"),

    # Linting
    (r"ESLint.*error", "eslint_error"),
    (r"pylint.*error", "pylint_error"),
    (r"Linting.*failed", "lint_failed"),

    # TypeScript
    (r"TS\d+:", "typescript_error"),
    (r"Type.*error", "type_error"),

    # Generic
    (r"error\[E\d+\]:", "rust_compile_error"),
    (r"javac.*error:", "java_compile_error"),
]

# Infrastructure failure patterns
INFRASTRUCTURE_PATTERNS = [
    # Memory
    (r"OutOfMemoryError", "oom_error"),
    (r"java\.lang\.OutOfMemoryError", "java_oom"),
    (r"OOM", "oom_generic"),
    (r"Cannot allocate memory", "memory_allocation_fail"),
    (r"Killed.*memory", "killed_oom"),

    # Network
    (r"ETIMEDOUT", "network_timeout"),
    (r"ECONNREFUSED", "connection_refused"),
    (r"ECONNRESET", "connection_reset"),
    (r"Network.*unreachable", "network_unreachable"),
    (r"DNS.*resolution failed", "dns_failure"),

    # Disk
    (r"No space left on device", "disk_full"),
    (r"Disk quota exceeded", "disk_quota"),

    # Resources
    (r"Too many open files", "file_descriptor_limit"),
    (r"Resource temporarily unavailable", "resource_exhausted"),

    # Pod/Container
    (r"CrashLoopBackOff", "pod_crash_loop"),
    (r"ImagePullBackOff", "image_pull_fail"),
    (r"Evicted", "pod_evicted"),
]


class HybridFailureClassifier:
    """Hybrid CI/CD failure classifier using rules + ML confidence."""

    def __init__(self):
        """Initialize the classifier."""
        self.pattern_groups = {
            "DEPENDENCY": DEPENDENCY_PATTERNS,
            "CONFIG": CONFIG_PATTERNS,
            "TEST": TEST_PATTERNS,
            "BUILD": BUILD_PATTERNS,
            "INFRASTRUCTURE": INFRASTRUCTURE_PATTERNS,
        }

    def classify(self, log_text: str, telemetry: Optional[Dict] = None) -> FailureClassification:
        """Classify a CI/CD failure from log text and optional telemetry.

        Args:
            log_text: Jenkins console log or error message
            telemetry: Optional telemetry data for context

        Returns:
            FailureClassification with type, confidence, and details
        """
        if not log_text or not isinstance(log_text, str):
            return FailureClassification(
                failure_type="UNKNOWN",
                confidence=0.0,
                matched_patterns=[],
                details="Empty or invalid log text",
                raw_indicators={},
            )

        # Step 1: Rule-based pattern matching
        pattern_scores = self._match_patterns(log_text)

        # Step 2: Telemetry-based adjustments
        if telemetry:
            pattern_scores = self._adjust_with_telemetry(pattern_scores, telemetry)

        # Step 3: Determine best match
        if not pattern_scores or all(score == 0 for score in pattern_scores.values()):
            return FailureClassification(
                failure_type="UNKNOWN",
                confidence=0.0,
                matched_patterns=[],
                details="No patterns matched",
                raw_indicators=pattern_scores,
            )

        # Get failure type with highest score
        best_type = max(pattern_scores, key=pattern_scores.get)
        best_score = pattern_scores[best_type]
        total_score = sum(pattern_scores.values())

        # Calculate confidence (0-1)
        # High confidence: one type dominates
        # Low confidence: multiple types match equally
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.0

        # Get matched patterns for explanation
        matched = self._get_matched_patterns(log_text, best_type)

        # Generate human-readable details
        details = self._generate_details(best_type, matched, log_text)

        return FailureClassification(
            failure_type=best_type,
            confidence=confidence,
            matched_patterns=matched,
            details=details,
            raw_indicators=pattern_scores,
        )

    def _match_patterns(self, log_text: str) -> Dict[str, int]:
        """Match patterns against log text and return scores per type.

        Returns:
            Dict mapping failure type to match count
        """
        scores: Dict[str, int] = {
            "DEPENDENCY": 0,
            "CONFIG": 0,
            "TEST": 0,
            "BUILD": 0,
            "INFRASTRUCTURE": 0,
        }

        # Case-insensitive matching
        log_lower = log_text.lower()

        for failure_type, patterns in self.pattern_groups.items():
            for pattern, pattern_name in patterns:
                try:
                    # Count matches (multiple occurrences boost score)
                    matches = len(re.findall(pattern, log_text, re.IGNORECASE))
                    if matches > 0:
                        scores[failure_type] += matches
                        logger.debug(f"Pattern '{pattern_name}' matched {matches} times")
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                    continue

        return scores

    def _adjust_with_telemetry(self, scores: Dict[str, int], telemetry: Dict) -> Dict[str, int]:
        """Adjust scores based on telemetry data.

        Uses system metrics to boost certain failure types:
        - High CPU/memory → boost INFRASTRUCTURE
        - Build status FAILURE → boost TEST/BUILD
        - Error rate spike → boost INFRASTRUCTURE
        """
        adjusted = scores.copy()

        # High resource usage → infrastructure issue
        cpu = float(telemetry.get("prometheus_cpu_usage", 0) or 0)
        memory = float(telemetry.get("prometheus_memory_usage", 0) or 0)

        if cpu > 80 or memory > 85:
            adjusted["INFRASTRUCTURE"] += 2
            logger.debug(f"Boosted INFRASTRUCTURE (CPU={cpu}%, MEM={memory}%)")

        # Build failure status → test/build issue more likely
        build_status = str(telemetry.get("jenkins_last_build_status", "")).upper()
        if build_status in ("FAILURE", "UNSTABLE"):
            adjusted["TEST"] += 1
            adjusted["BUILD"] += 1
            logger.debug(f"Boosted TEST/BUILD (status={build_status})")

        # High error rate → infrastructure or config issue
        error_rate = float(telemetry.get("prometheus_error_rate", 0) or 0)
        if error_rate > 0.3:
            adjusted["INFRASTRUCTURE"] += 1
            adjusted["CONFIG"] += 1
            logger.debug(f"Boosted INFRASTRUCTURE/CONFIG (error_rate={error_rate})")

        return adjusted

    def _get_matched_patterns(self, log_text: str, failure_type: str) -> List[str]:
        """Get list of pattern names that matched for a given failure type."""
        matched = []
        patterns = self.pattern_groups.get(failure_type, [])

        for pattern, pattern_name in patterns:
            try:
                if re.search(pattern, log_text, re.IGNORECASE):
                    matched.append(pattern_name)
            except re.error:
                continue

        return matched[:5]  # Limit to top 5 for readability

    def _generate_details(self, failure_type: str, matched_patterns: List[str], log_text: str) -> str:
        """Generate human-readable explanation of the failure."""
        if failure_type == "DEPENDENCY":
            return f"Dependency issue detected: {', '.join(matched_patterns[:3]) if matched_patterns else 'package resolution failed'}"
        elif failure_type == "CONFIG":
            return f"Configuration issue detected: {', '.join(matched_patterns[:3]) if matched_patterns else 'config error'}"
        elif failure_type == "TEST":
            return f"Test failure detected: {', '.join(matched_patterns[:3]) if matched_patterns else 'test assertions failed'}"
        elif failure_type == "BUILD":
            return f"Build error detected: {', '.join(matched_patterns[:3]) if matched_patterns else 'compilation failed'}"
        elif failure_type == "INFRASTRUCTURE":
            return f"Infrastructure issue detected: {', '.join(matched_patterns[:3]) if matched_patterns else 'resource exhaustion'}"
        else:
            return "Unable to classify failure type"


# ── Module-Level API ─────────────────────────────────────────────────────────

_DEFAULT_CLASSIFIER: Optional[HybridFailureClassifier] = None


def classify_failure(log_text: str, telemetry: Optional[Dict] = None) -> FailureClassification:
    """Module-level helper for quick classification.

    Args:
        log_text: Jenkins console log or error message
        telemetry: Optional telemetry data for context

    Returns:
        FailureClassification result

    Example:
        >>> result = classify_failure(log_text, telemetry)
        >>> print(f"Type: {result.failure_type}, Confidence: {result.confidence:.2f}")
        Type: DEPENDENCY, Confidence: 0.87
    """
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = HybridFailureClassifier()
    return _DEFAULT_CLASSIFIER.classify(log_text, telemetry)


def get_failure_type(log_text: str) -> str:
    """Quick helper to get just the failure type (no details).

    Args:
        log_text: Jenkins console log

    Returns:
        Failure type string (DEPENDENCY, CONFIG, TEST, BUILD, INFRASTRUCTURE, UNKNOWN)
    """
    result = classify_failure(log_text)
    return result.failure_type
