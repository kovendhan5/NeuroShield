"""NeuroShield centralized configuration.

Single source of truth for action names, MTTR baselines, shared constants, and YAML configuration.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# YAML Configuration Management
# ---------------------------------------------------------------------------

@dataclass
class ConfigSection:
    """Wrapper for config section with dot-notation access."""

    _data: Dict[str, Any]

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigSection({self._data})"


class Config:
    """Central configuration management - singleton pattern."""

    _instance = None
    _config_data = {}
    _initialized = False

    def __new__(cls, config_file: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file: Optional[str] = None):
        # Only load once
        if Config._initialized:
            return

        config_path = Path(config_file or os.getenv('CONFIG_FILE', 'config/neuroshield.yaml'))

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.load(config_path)
        Config._initialized = True

    @staticmethod
    def load(config_path):
        """Load YAML configuration."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")

        with open(config_path, 'r') as f:
            Config._config_data = yaml.safe_load(f) or {}

    @staticmethod
    def get(path: str, default: Any = None) -> Any:
        """Get config value using dot notation.

        Args:
            path: Dot-separated path (e.g., "orchestrator.poll_interval_seconds")
            default: Default value if not found
        """
        keys = path.split('.')
        value = Config._config_data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    @staticmethod
    def section(name: str) -> ConfigSection:
        """Get config section as object with dot notation."""
        return ConfigSection(Config._config_data.get(name, {}))

    @staticmethod
    def all() -> Dict[str, Any]:
        """Get entire config."""
        return Config._config_data

    @staticmethod
    def reload(config_path):
        """Reload configuration."""
        Config._config_data = {}
        Config.load(config_path)

    @staticmethod
    def validate() -> bool:
        """Validate configuration has required fields."""
        required_sections = [
            'application',
            'orchestrator',
            'kubernetes',
            'jenkins',
            'prometheus',
        ]

        for section in required_sections:
            if section not in Config._config_data:
                return False

        return True

    @staticmethod
    def to_env_vars() -> Dict[str, str]:
        """Convert YAML config to environment variables."""
        env_vars = {}

        def flatten(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    flatten(value, new_key.upper())
            else:
                env_vars[prefix] = str(obj)

        flatten(Config._config_data)
        return env_vars


# Convenience functions
def get_config(path: str, default: Any = None) -> Any:
    """Get config value."""
    return Config.get(path, default)


def get_section(name: str) -> ConfigSection:
    """Get config section."""
    return Config.section(name)
