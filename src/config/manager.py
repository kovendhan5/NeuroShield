"""NeuroShield Configuration Management.

Loads configuration from YAML files and environment variables.
Environment variables override YAML settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Application configuration with environment overrides."""

    def __init__(self, config_path: Optional[str | Path] = None):
        """Load configuration.

        Args:
            config_path: Path to config.yaml. If None, uses default locations.

        Raises:
            FileNotFoundError: If config file not found in any default location.
        """
        self.config_path = self._find_config(config_path)
        self._config = self._load_yaml()
        self._apply_env_overrides()

    def _find_config(self, config_path: Optional[str | Path]) -> Path:
        """Find configuration file.

        Tries (in order):
        1. Provided path
        2. ./config/neuroshield.yaml
        3. ./config.yaml
        4. Environment variable NEUROSHIELD_CONFIG
        """
        if config_path:
            p = Path(config_path)
            if p.exists():
                return p

        default_paths = [
            Path("config/neuroshield.yaml"),
            Path("config.yaml"),
            Path(os.environ.get("NEUROSHIELD_CONFIG", "")),
        ]

        for path in default_paths:
            if path and path.exists():
                return path

        raise FileNotFoundError(
            "No neuroshield.yaml found. Create one in ./config/ or ./config.yaml"
        )

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f) or {}
        return config

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides.

        Format: NEUROSHIELD_{SECTION}_{KEY}
        Example: NEUROSHIELD_JENKINS_URL=http://localhost:8080
        """
        for env_var, value in os.environ.items():
            if not env_var.startswith("NEUROSHIELD_"):
                continue

            # Parse NEUROSHIELD_JENKINS_URL -> jenkins.url
            parts = env_var[12:].lower().split("_")  # Remove "NEUROSHIELD_"
            section = parts[0]
            key = "_".join(parts[1:])

            if section not in self._config:
                self._config[section] = {}

            self._config[section][key] = value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            section: Configuration section (e.g., "jenkins", "prometheus")
            key: Configuration key (e.g., "url", "username")
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        return self._config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Dictionary of all settings in section
        """
        return self._config.get(section, {})

    def to_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation (safe, doesn't expose secrets)."""
        return f"<Config from {self.config_path}>"


# Global config instance
_config: Optional[Config] = None


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """Load or reload global configuration.

    Args:
        config_path: Optional path to config file

    Returns:
        Config instance
    """
    global _config
    _config = Config(config_path)
    return _config


def get_config() -> Config:
    """Get global configuration instance.

    Loads default config if not already loaded.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
