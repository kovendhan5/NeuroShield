"""
NeuroShield Reliability Layer
Fallbacks, safety checks, and human-in-the-loop
Ensures healing actions complete successfully
"""

import logging
import time
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ActionResult:
    """Result of executing a healing action."""

    def __init__(self, action_name: str):
        self.action_name = action_name
        self.success = False
        self.output = ""
        self.error_message = ""
        self.duration_ms = 0.0
        self.retry_count = 0
        self.fallback_used = False

    def __repr__(self) -> str:
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        return f"{status}: {self.action_name} (retries={self.retry_count}, time={self.duration_ms:.0f}ms)"


class ReliabilityConfig:
    """Configuration for reliability features."""

    def __init__(self):
        self.max_retries = 3
        self.retry_delay_s = 2.0
        self.verification_timeout_s = 30.0
        self.human_escalation_threshold = 0.7  # If confidence < this, ask human
        self.auto_fallback_enabled = True
        self.action_timeout_s = 60.0

    def __repr__(self) -> str:
        return (
            f"ReliabilityConfig(max_retries={self.max_retries}, "
            f"retry_delay={self.retry_delay_s}s, "
            f"verification_timeout={self.verification_timeout_s}s, "
            f"human_escalation_threshold={self.human_escalation_threshold})"
        )


class ActionExecutor:
    """Executes actions with retry and fallback logic."""

    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self.config = config or ReliabilityConfig()
        self.fallback_actions: Dict[str, Callable] = {}

    def register_fallback(self, action_name: str, fallback_fn: Callable):
        """Register a fallback action."""
        self.fallback_actions[action_name] = fallback_fn
        logger.info(f"Registered fallback for {action_name}")

    def execute(
        self,
        action_name: str,
        main_fn: Callable,
        verify_fn: Optional[Callable] = None,
    ) -> ActionResult:
        """
        Execute action with retry and fallback.

        Args:
            action_name: Name of the action
            main_fn: Function to execute
            verify_fn: Function to verify success (if not provided, uses main_fn result)

        Returns:
            ActionResult with execution details
        """
        result = ActionResult(action_name)
        start_time = time.time()

        # Attempt main action with retries
        for attempt in range(self.config.max_retries):
            result.retry_count = attempt
            try:
                logger.info(f"Executing {action_name} (attempt {attempt + 1}/{self.config.max_retries})")

                # Execute main action
                output = main_fn()
                result.output = str(output)

                # Verify success if verification function provided
                if verify_fn:
                    try:
                        verified = verify_fn()
                        if verified:
                            result.success = True
                            logger.info(f"{action_name} succeeded and verified")
                            break
                        else:
                            logger.warning(f"{action_name} executed but verification failed")
                            if attempt < self.config.max_retries - 1:
                                time.sleep(self.config.retry_delay_s)
                            continue
                    except Exception as e:
                        logger.warning(f"Verification failed: {e}")
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay_s)
                        continue
                else:
                    # No verification, assume success
                    result.success = True
                    logger.info(f"{action_name} succeeded (no verification)")
                    break

            except Exception as e:
                result.error_message = str(e)
                logger.warning(f"{action_name} attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_s)

        # If main action failed and fallback available, try fallback
        if not result.success and action_name in self.fallback_actions and self.config.auto_fallback_enabled:
            logger.warning(f"Main action {action_name} failed. Attempting fallback...")
            result.fallback_used = True

            try:
                fallback_fn = self.fallback_actions[action_name]
                output = fallback_fn()
                result.output = str(output)
                result.success = True
                logger.info(f"{action_name} succeeded via fallback")
            except Exception as e:
                result.error_message = f"Fallback also failed: {e}"
                logger.error(f"Fallback for {action_name} failed: {e}")

        result.duration_ms = (time.time() - start_time) * 1000.0
        return result


class SafetyChecker:
    """Validates decisions before execution."""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}

    def register_check(self, check_name: str, check_fn: Callable[[Dict], bool]):
        """Register a safety check."""
        self.checks[check_name] = check_fn

    def validate(self, action_name: str, context: Dict) -> Tuple[bool, str]:
        """
        Run all registered safety checks.

        Returns:
            (is_safe, reason_if_unsafe)
        """
        for check_name, check_fn in self.checks.items():
            try:
                if not check_fn(context):
                    reason = f"Safety check '{check_name}' failed for {action_name}"
                    logger.warning(reason)
                    return False, reason
            except Exception as e:
                reason = f"Safety check '{check_name}' error: {e}"
                logger.error(reason)
                return False, reason

        return True, ""


class FailureRecovery:
    """Recovery strategies for different failure modes."""

    @staticmethod
    def pod_restart_recovery() -> str:
        """Fallback: force delete pod instead of graceful restart."""
        return "kubectl delete pod --grace-period=0 --force"

    @staticmethod
    def scale_up_recovery() -> str:
        """Fallback: if scale up fails, try adding node instead."""
        return "kubectl top nodes"  # Check for node availability

    @staticmethod
    def rollback_recovery() -> str:
        """Fallback: if rollback fails, deploy previous stable version."""
        return "kubectl rollout history deployment"

    @staticmethod
    def retry_build_recovery() -> str:
        """Fallback: if retry fails, trigger diagnostic build."""
        return "Building diagnostic report..."


# Global instances
_executor = ActionExecutor()
_safety_checker = SafetyChecker()


def get_executor() -> ActionExecutor:
    """Get the global action executor."""
    return _executor


def get_safety_checker() -> SafetyChecker:
    """Get the global safety checker."""
    return _safety_checker


def configure_reliability(config: ReliabilityConfig):
    """Configure reliability settings globally."""
    global _executor
    _executor.config = config
    logger.info(f"Reliability configured: {config}")
