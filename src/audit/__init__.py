"""Structured audit logging for NeuroShield.

Provides compliance-ready audit trails for all critical operations:
- User actions (API calls, CLI)
- System events (service restarts, config changes)
- Security events (failed auth, permission denials)
- Healing actions (what changed and why)
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import structlog


class AuditCategory(Enum):
    """Audit event categories for compliance."""
    USER_ACTION = "USER_ACTION"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    SECURITY_EVENT = "SECURITY_EVENT"
    HEALING_ACTION = "HEALING_ACTION"
    DATA_ACCESS = "DATA_ACCESS"


class AuditResult(Enum):
    """Audit event outcomes."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    DENIED = "DENIED"


@dataclass
class AuditEvent:
    """Audit event record."""
    timestamp: str  # ISO 8601
    category: str
    action: str
    actor: str  # User ID or service name
    resource: str  # What was affected
    result: str  # SUCCESS, FAILURE, ACCEPTED, DENIED
    details: Dict[str, Any]
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    ip_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


# Configure structlog for audit logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Get audit logger
audit_logger = structlog.get_logger("audit")


def log_audit_event(
    category: AuditCategory,
    action: str,
    actor: str,
    resource: str,
    result: AuditResult,
    details: Dict[str, Any],
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    ip_address: Optional[str] = None
) -> None:
    """Log an audit event.

    Args:
        category: Type of event
        action: What happened (e.g., "CREATE", "DELETE", "FAILED_AUTH")
        actor: Who did it (username, service name, system)
        resource: What was affected (e.g., "healing_action", "config")
        result: Outcome (SUCCESS, FAILURE, DENIED)
        details: Contextual details (JSON-serializable dict)
        session_id: For grouping related events
        correlation_id: For distributed tracing
        ip_address: If from external request

    Example:
        log_audit_event(
            category=AuditCategory.HEALING_ACTION,
            action="RESTART_POD",
            actor="orchestrator",
            resource="dummy-app-7c5f6d9b88-x9kfz",
            result=AuditResult.SUCCESS,
            details={
                "reason": "OOM detected",
                "mttr_seconds": 45.3,
                "pod_restarts_before": 2,
            }
        )
    """
    event = AuditEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        category=category.value,
        action=action,
        actor=actor,
        resource=resource,
        result=result.value,
        details=details,
        session_id=session_id,
        correlation_id=correlation_id,
        ip_address=ip_address,
    )

    # Log to structured logger
    audit_logger.info(
        "audit_event",
        **event.to_dict()
    )

    # Push to real-time audit streaming (async-safe)
    try:
        import asyncio
        from src.api.routers.audit import push_audit_event

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(push_audit_event(event.to_dict()))
        else:
            loop.run_until_complete(push_audit_event(event.to_dict()))
    except Exception:
        pass  # Don't break audit logging if streaming fails


def log_healing_action(
    action_name: str,
    failure_type: str,
    mttr_seconds: float,
    success: bool,
    pod_name: str,
    reason: str,
    model_confidence: float
) -> None:
    """Convenience function for logging healing actions.

    Args:
        action_name: restart_pod, scale_up, retry_build, rollback_deploy
        failure_type: OOM, FlakyTest, Timeout, etc.
        mttr_seconds: Time to recovery
        success: Did healing succeed?
        pod_name: Affected Kubernetes pod
        reason: Why was this action chosen?
        model_confidence: ML model confidence (0-1)
    """
    log_audit_event(
        category=AuditCategory.HEALING_ACTION,
        action=f"EXECUTE_{action_name.upper()}",
        actor="orchestrator",
        resource=pod_name,
        result=AuditResult.SUCCESS if success else AuditResult.FAILURE,
        details={
            "action_name": action_name,
            "failure_type": failure_type,
            "mttr_seconds": mttr_seconds,
            "reason": reason,
            "model_confidence": model_confidence,
        }
    )


def log_api_request(
    endpoint: str,
    method: str,
    user_id: str,
    status_code: int,
    response_time_ms: float,
    ip_address: str
) -> None:
    """Log API request for audit trail."""
    log_audit_event(
        category=AuditCategory.USER_ACTION,
        action=f"{method} {endpoint}",
        actor=user_id,
        resource=endpoint,
        result=AuditResult.SUCCESS if 200 <= status_code < 300 else AuditResult.FAILURE,
        details={
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
        },
        ip_address=ip_address
    )


def log_config_change(
    actor: str,
    config_key: str,
    old_value: Any,
    new_value: Any,
    reason: str
) -> None:
    """Log configuration change for compliance."""
    log_audit_event(
        category=AuditCategory.CONFIG_CHANGE,
        action="UPDATE_CONFIG",
        actor=actor,
        resource=config_key,
        result=AuditResult.SUCCESS,
        details={
            "old_value": str(old_value),  # Sanitized
            "new_value": str(new_value),  # Sanitized
            "reason": reason,
        }
    )


def log_security_event(
    action: str,
    result: AuditResult,
    details: Dict[str, Any],
    ip_address: Optional[str] = None
) -> None:
    """Log security-related event."""
    log_audit_event(
        category=AuditCategory.SECURITY_EVENT,
        action=action,
        actor="security_system",
        resource="system",
        result=result,
        details=details,
        ip_address=ip_address
    )


def log_failed_auth(
    username: str,
    reason: str,
    ip_address: str
) -> None:
    """Log failed authentication attempt."""
    log_security_event(
        action="FAILED_AUTH",
        result=AuditResult.DENIED,
        details={
            "username": username,
            "reason": reason,
        },
        ip_address=ip_address
    )


def log_data_access(
    actor: str,
    resource: str,
    access_type: str,  # READ, WRITE, DELETE
    query: str,
    rows_affected: int
) -> None:
    """Log database/data access."""
    log_audit_event(
        category=AuditCategory.DATA_ACCESS,
        action=access_type,
        actor=actor,
        resource=resource,
        result=AuditResult.SUCCESS,
        details={
            "access_type": access_type,
            "query": query[:200],  # Truncate for safety
            "rows_affected": rows_affected,
        }
    )


# FastAPI middleware integration
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic audit logging of API calls."""

    async def dispatch(self, request: Request, call_next):
        import time

        # Skip health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000  # ms

        # Extract user from JWT token if present
        user_id = "anonymous"
        if "authorization" in request.headers:
            try:
                from src.security.auth import verify_token

                auth_header = request.headers["authorization"]
                if auth_header.startswith("Bearer "):
                    token = auth_header.split(" ")[1]
                    user_id = verify_token(token)
            except Exception:
                user_id = "unauthorized"

        log_api_request(
            endpoint=str(request.url.path),
            method=request.method,
            user_id=user_id,
            status_code=response.status_code,
            response_time_ms=process_time,
            ip_address=request.client.host if request.client else "unknown"
        )

        return response
