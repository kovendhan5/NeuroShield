#!/usr/bin/env python3
"""NeuroShield Health Check — verifies all services, models, and data files."""

import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def _http_ok(url: str, timeout: int = 5) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_env() -> None:
    env_path = REPO_ROOT / ".env"
    _check(".env file exists", env_path.exists())


def check_models() -> None:
    models_dir = REPO_ROOT / "models"
    for name in ("failure_predictor.pth", "log_pca.joblib", "ppo_policy.zip"):
        path = models_dir / name
        _check(f"Model: {name}", path.exists(), f"{path.stat().st_size:,} bytes" if path.exists() else "missing")


def check_data() -> None:
    csv_path = REPO_ROOT / "data" / "telemetry.csv"
    if csv_path.exists():
        lines = sum(1 for _ in csv_path.open(encoding="utf-8", errors="replace"))
        _check("Data: telemetry.csv", True, f"{lines:,} rows")
    else:
        _check("Data: telemetry.csv", False, "missing")


def check_imports() -> None:
    critical = ["torch", "transformers", "stable_baselines3", "streamlit", "plotly", "requests", "pandas"]
    for mod in critical:
        try:
            __import__(mod)
            _check(f"Import: {mod}", True)
        except ImportError:
            _check(f"Import: {mod}", False, "not installed")


def check_services() -> None:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    jenkins_url = os.getenv("JENKINS_URL", "http://localhost:8080")
    prom_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    app_url = os.getenv("DUMMY_APP_URL", "http://localhost:5000")

    _check("Service: Jenkins", _http_ok(jenkins_url), jenkins_url)
    _check("Service: Prometheus", _http_ok(prom_url), prom_url)
    _check("Service: Dummy App", _http_ok(app_url), app_url)


def check_source_files() -> None:
    key_files = [
        "src/orchestrator/main.py",
        "src/dashboard/app.py",
        "src/prediction/predictor.py",
        "src/telemetry/collector.py",
        "src/rl_agent/env.py",
    ]
    for rel in key_files:
        path = REPO_ROOT / rel
        _check(f"Source: {rel}", path.exists())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n  NeuroShield Health Check")
    print("  " + "=" * 40 + "\n")

    check_env()
    check_models()
    check_data()
    check_source_files()
    check_imports()
    check_services()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)

    print(f"\n  {'=' * 40}")
    print(f"  Results: {passed}/{total} passed, {failed} failed\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
