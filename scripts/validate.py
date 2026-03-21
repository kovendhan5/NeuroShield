#!/usr/bin/env python3
"""
NeuroShield Configuration & Validation Tool
============================================

Usage:
    python scripts/validate.py              # Run all checks
    python scripts/validate.py --fix        # Auto-fix issues
    python scripts/validate.py --config     # Interactive configuration
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

COLORS = {
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "YELLOW": "\033[93m",
    "CYAN": "\033[96m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}

class Validator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues: List[Tuple[str, str]] = []  # (severity, message)
        self.fixes: List[str] = []

    def check(self, condition: bool, msg: str, severity: str = "INFO"):
        """Record a check result."""
        if not condition:
            self.issues.append((severity, msg))
            return False
        return True

    def print_banner(self, text: str):
        """Print section header."""
        print(f"\n{COLORS['CYAN']}{COLORS['BOLD']}{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}{COLORS['RESET']}\n")

    def print_result(self, severity: str, msg: str):
        """Print result with color."""
        colors = {
            "OK": COLORS["GREEN"],
            "WARN": COLORS["YELLOW"],
            "ERROR": COLORS["RED"],
        }
        color = colors.get(severity, COLORS["CYAN"])
        print(f"  {color}[{severity:5}]{COLORS['RESET']} {msg}")

    def validate_environment(self) -> bool:
        """Validate .env file and configuration."""
        self.print_banner("Environment Configuration")

        env_file = self.project_root / ".env"
        if not env_file.exists():
            self.print_result("WARN", ".env file not found - using defaults")
            return False

        env_vars = {
            "JENKINS_URL": "http://localhost:8080",
            "JENKINS_USERNAME": "admin",
            "JENKINS_PASSWORD": "admin123",
            "PROMETHEUS_URL": "http://localhost:9090",
            "DUMMY_APP_URL": "http://localhost:5000",
        }

        with open(env_file) as f:
            current_env = dict(line.split("=", 1) for line in f if "=" in line and not line.startswith("#"))

        all_ok = True
        for key, default in env_vars.items():
            if key in current_env:
                self.print_result("OK", f"{key:30} = {current_env[key]}")
            else:
                self.print_result("WARN", f"{key:30} not set (using {default})")
                all_ok = False

        return all_ok

    def validate_directories(self) -> bool:
        """Validate required directories."""
        self.print_banner("Directory Structure")

        required = [
            "src",
            "src/orchestrator",
            "src/telemetry",
            "src/dashboard",
            "src/api",
            "src/prediction",
            "src/rl_agent",
            "scripts",
            "tests",
            "data",
            "models",
            "neuroshield-pro",
        ]

        all_ok = True
        for dir_name in required:
            path = self.project_root / dir_name
            if path.exists():
                self.print_result("OK", f"{dir_name:40} exists")
            else:
                self.print_result("ERROR", f"{dir_name:40} MISSING")
                all_ok = False

        return all_ok

    def validate_models(self) -> bool:
        """Validate ML models."""
        self.print_banner("Machine Learning Models")

        models = [
            "models/failure_predictor.pth",
            "models/log_pca.joblib",
            "models/ppo_policy.zip",
        ]

        all_ok = True
        for model_path in models:
            path = self.project_root / model_path
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.print_result("OK", f"{model_path:40} {size_mb:.1f}MB")
            else:
                self.print_result("ERROR", f"{model_path:40} MISSING")
                all_ok = False

        return all_ok

    def validate_dependencies(self) -> bool:
        """Validate Python dependencies."""
        self.print_banner("Python Dependencies")

        critical_deps = [
            "torch",
            "transformers",
            "stable_baselines3",
            "streamlit",
            "fastapi",
            "requests",
            "pandas",
        ]

        all_ok = True
        for dep in critical_deps:
            try:
                __import__(dep.replace("-", "_"))
                self.print_result("OK", f"{dep:35} installed")
            except ImportError:
                self.print_result("ERROR", f"{dep:35} NOT INSTALLED")
                all_ok = False

        return all_ok

    def validate_services(self) -> bool:
        """Check if services can be reached."""
        self.print_banner("Service Connectivity")

        services = {
            "Jenkins": "http://localhost:8080",
            "Prometheus": "http://localhost:9090",
            "Dummy App": "http://localhost:5000",
            "Dashboard": "http://localhost:8501",
            "API": "http://localhost:8502",
        }

        import requests

        all_ok = True
        for name, url in services.items():
            try:
                response = requests.head(url, timeout=2)
                self.print_result("OK", f"{name:35} responding")
            except (requests.ConnectionError, requests.Timeout):
                self.print_result("WARN", f"{name:35} not responding (may be normal)")
                # Not an error - services may not be running

        return True

    def run_all_checks(self) -> int:
        """Run all validation checks."""
        self.print_banner("NeuroShield Project Validation")

        checks = [
            ("Environment", self.validate_environment),
            ("Directories", self.validate_directories),
            ("Models", self.validate_models),
            ("Dependencies", self.validate_dependencies),
            ("Services", self.validate_services),
        ]

        results = []
        for name, check_func in checks:
            try:
                result = check_func()
                results.append((name, result))
            except Exception as e:
                print(f"  {COLORS['RED']}[ERROR]{COLORS['RESET']} Check '{name}' failed: {e}")
                results.append((name, False))

        self.print_banner("Validation Summary")

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for name, result in results:
            status = f"{COLORS['GREEN']}PASS{COLORS['RESET']}" if result else f"{COLORS['RED']}FAIL{COLORS['RESET']}"
            print(f"  {name:30} {status}")

        print(f"\n  Total: {passed}/{total} checks passed")

        if len(self.issues) > 0:
            print(f"\n{COLORS['YELLOW']}Issues found:{COLORS['RESET']}")
            for severity, msg in self.issues:
                color = COLORS["RED"] if severity == "ERROR" else COLORS["YELLOW"]
                print(f"  {color}[{severity}]{COLORS['RESET']} {msg}")

        return 0 if passed == total else 1

    def main(self):
        """Main entry point."""
        if "--help" in sys.argv:
            print(__doc__)
            return 0

        return self.run_all_checks()

if __name__ == "__main__":
    validator = Validator()
    sys.exit(validator.main())
