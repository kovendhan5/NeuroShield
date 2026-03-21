"""
NeuroShield Configuration Management
Loads and manages centralized YAML configuration
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


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
    """Central configuration management."""

    _instance = None
    _config_data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file: Optional[str] = None):
        if self._config_data:
            return  # Already loaded

        config_path = Path(config_file or os.getenv('CONFIG_FILE', 'config/neuroshield.yaml'))

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.load(config_path)

    @staticmethod
    def load(config_path: Path):
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
    def reload(config_path: Path):
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


# Usage examples:
if __name__ == "__main__":
    config = Config("config/neuroshield.yaml")

    # Direct access
    print(f"App name: {config.get('application.name')}")
    print(f"Log level: {config.get('application.log_level')}")

    # Section access
    orchestrator = config.section('orchestrator')
    print(f"Poll interval: {orchestrator.poll_interval_seconds}")

    # Full config
    print(json.dumps(config.all(), indent=2))
