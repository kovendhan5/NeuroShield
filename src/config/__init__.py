"""NeuroShield centralized configuration.

Single source of truth for action names, MTTR baselines, and shared constants.
Import from here rather than duplicating across modules.
"""

from __future__ import annotations

import os
import re
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Healing action definitions (shared across orchestrator, RL env, dashboard)
# ---------------------------------------------------------------------------

ACTION_NAMES: Dict[int, str] = {
    0: "restart_pod",
    1: "scale_up",
    2: "retry_build",
    3: "rollback_deploy",
    4: "clear_cache",
    5: "escalate_to_human",
}

ACTION_IDS: Dict[str, int] = {v: k for k, v in ACTION_NAMES.items()}

NUM_ACTIONS = len(ACTION_NAMES)

# MTTR baselines (seconds) — manual remediation without NeuroShield
MTTR_BASELINES: Dict[str, float] = {
    "restart_pod": 90.0,
    "scale_up": 60.0,
    "retry_build": 70.0,
    "rollback_deploy": 120.0,
    "clear_cache": 45.0,
    "escalate_to_human": 300.0,
}

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

# Kubernetes name rules: lowercase alphanumeric + hyphens, max 253 chars
_K8S_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,251}[a-z0-9])?$")


def validate_k8s_name(value: str, label: str = "name") -> str:
    """Validate that a value is a legal Kubernetes resource name.

    Raises ValueError if invalid.
    """
    if not value or not _K8S_NAME_RE.match(value):
        raise ValueError(
            f"Invalid Kubernetes {label}: {value!r}. "
            "Must be lowercase alphanumeric with hyphens, 1-253 chars."
        )
    return value


def validate_positive_int(value: int, label: str = "value") -> int:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value
