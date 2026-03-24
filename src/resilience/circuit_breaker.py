"""Circuit Breaker pattern for resilient external API calls.

Prevents cascading failures when external services are unavailable.
Implements exponential backoff and automatic recovery.
"""

import logging
import time
from enum import Enum
from typing import Callable, Generic, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    """Resilient circuit breaker with exponential backoff.

    Example:
        jenkins_breaker = CircuitBreaker(
            name="jenkins",
            fail_max=5,
            reset_timeout=60
        )

        @jenkins_breaker
        def call_jenkins_api():
            return requests.get(...)
    """

    def __init__(
        self,
        name: str,
        fail_max: int = 5,
        reset_timeout: int = 60,
        backoff_multiplier: float = 2.0,
        max_backoff: int = 600
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for logging
            fail_max: Number of failures before opening circuit
            reset_timeout: Seconds before attempting recovery
            backoff_multiplier: Exponential backoff factor
            max_backoff: Maximum backoff time (seconds)
        """
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff

        self.state = CircuitState.CLOSED
        self.fail_count = 0
        self.last_fail_time: float = 0
        self.last_check_time: float = 0

    def call(self, fn: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection.

        Args:
            fn: Callable to execute

        Returns:
            Result of fn

        Raises:
            CircuitBreakerError: If circuit is OPEN
            Exception: Re-raises exceptions from fn
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[{self.name}] Circuit HALF_OPEN - testing recovery")
            else:
                raise CircuitBreakerError(
                    f"Circuit {self.name} is OPEN - service unavailable"
                )

        try:
            result = fn()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"[{self.name}] Circuit CLOSED - service recovered")

        self.state = CircuitState.CLOSED
        self.fail_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.fail_count += 1
        self.last_fail_time = time.time()

        if self.fail_count >= self.fail_max:
            self.state = CircuitState.OPEN
            logger.error(
                f"[{self.name}] Circuit OPEN after {self.fail_count} failures"
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to attempt recovery."""
        backoff = min(
            self.reset_timeout * (self.backoff_multiplier ** (self.fail_count - self.fail_max)),
            self.max_backoff
        )
        return (time.time() - self.last_fail_time) > backoff

    def __call__(self, fn: Callable[[], T]) -> Callable[[], T]:
        """Decorator support."""
        def wrapper(*args, **kwargs):
            return self.call(lambda: fn(*args, **kwargs))
        return wrapper


# Pre-configured circuit breakers for NeuroShield services
jenkins_breaker = CircuitBreaker(
    name="jenkins",
    fail_max=5,
    reset_timeout=60,
    backoff_multiplier=2.0
)

prometheus_breaker = CircuitBreaker(
    name="prometheus",
    fail_max=3,
    reset_timeout=30,
    backoff_multiplier=1.5
)

kubernetes_breaker = CircuitBreaker(
    name="kubernetes",
    fail_max=4,
    reset_timeout=45,
    backoff_multiplier=2.0
)

database_breaker = CircuitBreaker(
    name="database",
    fail_max=3,
    reset_timeout=20,
    backoff_multiplier=1.5
)
