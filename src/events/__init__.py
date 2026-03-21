"""NeuroShield Events Module

Provides:
- Webhook event server (sub-second detection)
- Decision interpretability (full audit trail)
- Reliability and safety layer (fallbacks)
"""

from src.events.webhook_server import WebhookServer, get_event_queue, start_webhook_server
from src.events.decision_trace import DecisionLogger, DecisionTrace, get_decision_logger, trace_decision
from src.events.reliability import (
    ActionExecutor,
    ActionResult,
    ReliabilityConfig,
    SafetyChecker,
    FailureRecovery,
    get_executor,
    get_safety_checker,
    configure_reliability,
)

__all__ = [
    # Webhooks
    "WebhookServer",
    "get_event_queue",
    "start_webhook_server",
    # Decision Tracing
    "DecisionLogger",
    "DecisionTrace",
    "get_decision_logger",
    "trace_decision",
    # Reliability
    "ActionExecutor",
    "ActionResult",
    "ReliabilityConfig",
    "SafetyChecker",
    "FailureRecovery",
    "get_executor",
    "get_safety_checker",
    "configure_reliability",
]
