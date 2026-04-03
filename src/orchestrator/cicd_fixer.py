"""CI/CD Auto-Fix Engine - Safe, deterministic fixes for common failures.

This module provides SAFE automated fixes for CI/CD failures:
- Dependency installation (npm, pip)
- Workspace/cache cleanup
- Build retry with modifications
- Timeout adjustments

SAFETY RULES:
- NO source code modifications
- NO risky config changes
- ALL fixes are reversible
- ALL actions are logged
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FixResult:
    """Result of an auto-fix attempt."""
    success: bool
    fix_type: str  # dependency_install, cache_clear, etc.
    duration_ms: float
    details: str
    actions_taken: List[str]
    reversible: bool


class CICDAutoFixer:
    """Safe auto-fix engine for CI/CD failures."""

    def __init__(self, dry_run: bool = False, log_dir: str = "data/auto_fix_logs"):
        """Initialize auto-fixer.

        Args:
            dry_run: If True, log actions but don't execute them
            log_dir: Directory for fix logs
        """
        self.dry_run = dry_run
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def fix_failure(
        self,
        failure_type: str,
        log_text: str,
        job_name: str,
        build_number: int,
        telemetry: Optional[Dict] = None,
    ) -> FixResult:
        """Attempt to fix a CI/CD failure.

        Args:
            failure_type: Type from classifier (DEPENDENCY, CONFIG, TEST, BUILD, INFRASTRUCTURE)
            log_text: Jenkins console log
            job_name: Jenkins job name
            build_number: Build number
            telemetry: Optional telemetry data

        Returns:
            FixResult with success status and details
        """
        start_time = time.time()

        logger.info(f"Attempting fix for {failure_type} failure in {job_name} #{build_number}")

        # Dispatch to appropriate fixer
        if failure_type == "DEPENDENCY":
            result = self._fix_dependency(log_text, job_name, build_number)
        elif failure_type == "CONFIG":
            result = self._fix_config(log_text, job_name, build_number)
        elif failure_type == "TEST":
            result = self._fix_test(log_text, job_name, build_number, telemetry)
        elif failure_type == "BUILD":
            result = self._fix_build(log_text, job_name, build_number)
        else:
            # INFRASTRUCTURE and UNKNOWN don't have automated fixes
            result = FixResult(
                success=False,
                fix_type="none",
                duration_ms=0.0,
                details=f"No automated fix available for {failure_type}",
                actions_taken=[],
                reversible=True,
            )

        result.duration_ms = (time.time() - start_time) * 1000
        self._log_fix_attempt(failure_type, result, job_name, build_number)

        return result

    # ── Dependency Fixes ─────────────────────────────────────────────────────

    def _fix_dependency(self, log_text: str, job_name: str, build_number: int) -> FixResult:
        """Fix dependency-related failures.

        Safe fixes:
        - Install missing npm packages
        - Install missing pip packages
        - Clear npm cache
        - Clear pip cache
        """
        actions_taken = []

        # Detect package manager
        if "npm ERR!" in log_text or "Cannot find module" in log_text:
            # npm dependency issue
            missing_packages = self._extract_npm_missing_packages(log_text)
            if missing_packages:
                return self._install_npm_packages(missing_packages, job_name, build_number)
            else:
                # Generic npm cache clear
                return self._clear_npm_cache(job_name, build_number)

        elif "ModuleNotFoundError" in log_text or "ImportError" in log_text:
            # Python dependency issue
            missing_packages = self._extract_python_missing_packages(log_text)
            if missing_packages:
                return self._install_pip_packages(missing_packages, job_name, build_number)
            else:
                # Generic pip cache clear
                return self._clear_pip_cache(job_name, build_number)

        elif "Could not resolve dependencies" in log_text:
            # Maven dependency issue
            return self._clear_maven_cache(job_name, build_number)

        else:
            return FixResult(
                success=False,
                fix_type="dependency_unknown",
                duration_ms=0.0,
                details="Could not determine specific dependency fix",
                actions_taken=[],
                reversible=True,
            )

    def _extract_npm_missing_packages(self, log_text: str) -> List[str]:
        """Extract missing npm package names from log."""
        import re
        packages = []

        # Pattern: npm ERR! missing: express@^4.18.0
        # Capture package name before @version
        matches = re.findall(r"npm ERR! missing:\s+([a-z0-9@/_-]+?)@", log_text, re.IGNORECASE)
        packages.extend(matches)

        # Pattern: Cannot find module 'express'
        matches = re.findall(r"Cannot find module\s+['\"]([a-z0-9@/_-]+)['\"]", log_text, re.IGNORECASE)
        packages.extend(matches)

        # Pattern: MODULE_NOT_FOUND for 'react'
        matches = re.findall(r"MODULE_NOT_FOUND.*['\"]([a-z0-9@/_-]+)['\"]", log_text, re.IGNORECASE)
        packages.extend(matches)

        # Deduplicate and limit
        unique_packages = list(set(packages))[:5]  # Max 5 packages to avoid abuse
        return unique_packages

    def _extract_python_missing_packages(self, log_text: str) -> List[str]:
        """Extract missing Python package names from log."""
        import re
        packages = []

        # Pattern: ModuleNotFoundError: No module named 'flask'
        matches = re.findall(r"No module named\s+['\"]([a-z0-9_-]+)['\"]", log_text, re.IGNORECASE)
        packages.extend(matches)

        # Pattern: ImportError: cannot import name 'Flask'
        matches = re.findall(r"ImportError:.*['\"]([a-z0-9_-]+)['\"]", log_text, re.IGNORECASE)
        packages.extend(matches)

        # Deduplicate and limit
        unique_packages = list(set(packages))[:5]  # Max 5 packages
        return unique_packages

    def _install_npm_packages(self, packages: List[str], job_name: str, build_number: int) -> FixResult:
        """Install missing npm packages."""
        actions = []

        if self.dry_run:
            actions.append(f"[DRY-RUN] Would install npm packages: {', '.join(packages)}")
            logger.info(f"Dry-run: npm install {' '.join(packages)}")
            return FixResult(
                success=True,
                fix_type="npm_install",
                duration_ms=0.0,
                details=f"[DRY-RUN] Would install {len(packages)} npm packages",
                actions_taken=actions,
                reversible=True,
            )

        # Real execution
        try:
            # SAFETY: Only install packages that look valid (alphanumeric + @/-)
            safe_packages = [p for p in packages if self._is_safe_package_name(p)]
            if not safe_packages:
                return FixResult(
                    success=False,
                    fix_type="npm_install",
                    duration_ms=0.0,
                    details="No safe package names to install",
                    actions_taken=[],
                    reversible=True,
                )

            cmd = ["npm", "install", "--save"] + safe_packages
            actions.append(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                actions.append(f"Successfully installed: {', '.join(safe_packages)}")
                return FixResult(
                    success=True,
                    fix_type="npm_install",
                    duration_ms=0.0,
                    details=f"Installed {len(safe_packages)} npm packages",
                    actions_taken=actions,
                    reversible=True,  # Can be reverted via npm uninstall
                )
            else:
                actions.append(f"npm install failed: {result.stderr[:200]}")
                return FixResult(
                    success=False,
                    fix_type="npm_install",
                    duration_ms=0.0,
                    details=f"npm install failed: {result.stderr[:200]}",
                    actions_taken=actions,
                    reversible=True,
                )

        except subprocess.TimeoutExpired:
            actions.append("npm install timed out after 5 minutes")
            return FixResult(
                success=False,
                fix_type="npm_install",
                duration_ms=0.0,
                details="npm install timeout",
                actions_taken=actions,
                reversible=True,
            )
        except Exception as e:
            actions.append(f"npm install error: {str(e)[:200]}")
            return FixResult(
                success=False,
                fix_type="npm_install",
                duration_ms=0.0,
                details=f"Error: {str(e)[:200]}",
                actions_taken=actions,
                reversible=True,
            )

    def _install_pip_packages(self, packages: List[str], job_name: str, build_number: int) -> FixResult:
        """Install missing pip packages."""
        actions = []

        if self.dry_run:
            actions.append(f"[DRY-RUN] Would install pip packages: {', '.join(packages)}")
            logger.info(f"Dry-run: pip install {' '.join(packages)}")
            return FixResult(
                success=True,
                fix_type="pip_install",
                duration_ms=0.0,
                details=f"[DRY-RUN] Would install {len(packages)} pip packages",
                actions_taken=actions,
                reversible=True,
            )

        # Real execution
        try:
            # SAFETY: Only install packages that look valid
            safe_packages = [p for p in packages if self._is_safe_package_name(p)]
            if not safe_packages:
                return FixResult(
                    success=False,
                    fix_type="pip_install",
                    duration_ms=0.0,
                    details="No safe package names to install",
                    actions_taken=[],
                    reversible=True,
                )

            cmd = ["pip", "install"] + safe_packages
            actions.append(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                actions.append(f"Successfully installed: {', '.join(safe_packages)}")
                return FixResult(
                    success=True,
                    fix_type="pip_install",
                    duration_ms=0.0,
                    details=f"Installed {len(safe_packages)} pip packages",
                    actions_taken=actions,
                    reversible=True,  # Can be reverted via pip uninstall
                )
            else:
                actions.append(f"pip install failed: {result.stderr[:200]}")
                return FixResult(
                    success=False,
                    fix_type="pip_install",
                    duration_ms=0.0,
                    details=f"pip install failed: {result.stderr[:200]}",
                    actions_taken=actions,
                    reversible=True,
                )

        except subprocess.TimeoutExpired:
            actions.append("pip install timed out after 5 minutes")
            return FixResult(
                success=False,
                fix_type="pip_install",
                duration_ms=0.0,
                details="pip install timeout",
                actions_taken=actions,
                reversible=True,
            )
        except Exception as e:
            actions.append(f"pip install error: {str(e)[:200]}")
            return FixResult(
                success=False,
                fix_type="pip_install",
                duration_ms=0.0,
                details=f"Error: {str(e)[:200]}",
                actions_taken=actions,
                reversible=True,
            )

    def _is_safe_package_name(self, package: str) -> bool:
        """Check if package name looks safe (no shell injection or path traversal)."""
        import re
        # Reject path traversal attempts
        if ".." in package or "/" in package.replace("@/", ""):
            # Allow @types/node but reject ../../etc/passwd
            if not package.startswith("@"):
                return False
        # Allow alphanumeric, @, /, _, -, .
        return bool(re.match(r'^[a-zA-Z0-9@/._-]+$', package))

    # ── Cache Clearing ───────────────────────────────────────────────────────

    def _clear_npm_cache(self, job_name: str, build_number: int) -> FixResult:
        """Clear npm cache."""
        actions = []

        if self.dry_run:
            actions.append("[DRY-RUN] Would clear npm cache")
            return FixResult(
                success=True,
                fix_type="npm_cache_clear",
                duration_ms=0.0,
                details="[DRY-RUN] Would clear npm cache",
                actions_taken=actions,
                reversible=True,
            )

        try:
            cmd = ["npm", "cache", "clean", "--force"]
            actions.append(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                actions.append("npm cache cleared successfully")
                return FixResult(
                    success=True,
                    fix_type="npm_cache_clear",
                    duration_ms=0.0,
                    details="npm cache cleared",
                    actions_taken=actions,
                    reversible=True,
                )
            else:
                return FixResult(
                    success=False,
                    fix_type="npm_cache_clear",
                    duration_ms=0.0,
                    details=f"Failed: {result.stderr[:200]}",
                    actions_taken=actions,
                    reversible=True,
                )
        except Exception as e:
            return FixResult(
                success=False,
                fix_type="npm_cache_clear",
                duration_ms=0.0,
                details=f"Error: {str(e)[:200]}",
                actions_taken=actions,
                reversible=True,
            )

    def _clear_pip_cache(self, job_name: str, build_number: int) -> FixResult:
        """Clear pip cache."""
        actions = []

        if self.dry_run:
            actions.append("[DRY-RUN] Would clear pip cache")
            return FixResult(
                success=True,
                fix_type="pip_cache_clear",
                duration_ms=0.0,
                details="[DRY-RUN] Would clear pip cache",
                actions_taken=actions,
                reversible=True,
            )

        try:
            cmd = ["pip", "cache", "purge"]
            actions.append(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                actions.append("pip cache cleared successfully")
                return FixResult(
                    success=True,
                    fix_type="pip_cache_clear",
                    duration_ms=0.0,
                    details="pip cache cleared",
                    actions_taken=actions,
                    reversible=True,
                )
            else:
                return FixResult(
                    success=False,
                    fix_type="pip_cache_clear",
                    duration_ms=0.0,
                    details=f"Failed: {result.stderr[:200]}",
                    actions_taken=actions,
                    reversible=True,
                )
        except Exception as e:
            return FixResult(
                success=False,
                fix_type="pip_cache_clear",
                duration_ms=0.0,
                details=f"Error: {str(e)[:200]}",
                actions_taken=actions,
                reversible=True,
            )

    def _clear_maven_cache(self, job_name: str, build_number: int) -> FixResult:
        """Clear maven cache."""
        actions = []

        if self.dry_run:
            actions.append("[DRY-RUN] Would clear maven cache")
            return FixResult(
                success=True,
                fix_type="maven_cache_clear",
                duration_ms=0.0,
                details="[DRY-RUN] Would clear maven .m2 cache",
                actions_taken=actions,
                reversible=True,
            )

        try:
            # Maven cache must resolve inside ~/.m2/repository (canonical path check)
            expected_root = (Path.home() / ".m2").resolve()
            expected_repo = (expected_root / "repository").resolve()

            if expected_repo.exists():
                resolved_repo = expected_repo.resolve()
                if not resolved_repo.is_relative_to(expected_root):
                    return FixResult(
                        success=False,
                        fix_type="maven_cache_clear",
                        duration_ms=0.0,
                        details=f"Unsafe maven cache path rejected: {resolved_repo}",
                        actions_taken=actions,
                        reversible=True,
                    )

                actions.append(f"Clearing maven cache at {resolved_repo}")
                import shutil

                shutil.rmtree(resolved_repo, ignore_errors=True)
                actions.append("maven cache cleared")
                return FixResult(
                    success=True,
                    fix_type="maven_cache_clear",
                    duration_ms=0.0,
                    details="maven cache cleared",
                    actions_taken=actions,
                    reversible=False,  # Cache deletion is not reversible
                )
            else:
                return FixResult(
                    success=False,
                    fix_type="maven_cache_clear",
                    duration_ms=0.0,
                    details="maven cache directory not found",
                    actions_taken=actions,
                    reversible=True,
                )
        except Exception as e:
            return FixResult(
                success=False,
                fix_type="maven_cache_clear",
                duration_ms=0.0,
                details=f"Error: {str(e)[:200]}",
                actions_taken=actions,
                reversible=True,
            )

    # ── Config Fixes ─────────────────────────────────────────────────────────

    def _fix_config(self, log_text: str, job_name: str, build_number: int) -> FixResult:
        """Fix configuration-related failures.

        SAFE fixes only:
        - Log recommendation for missing env vars (don't modify .env)
        - Suggest config file restoration (don't auto-restore)
        """
        actions = []

        # For config issues, we recommend manual intervention
        # Auto-fixing configs is too risky

        if "environment variable" in log_text.lower() and "not" in log_text.lower():
            return FixResult(
                success=False,
                fix_type="config_recommendation",
                duration_ms=0.0,
                details="Recommendation: Check .env file for missing environment variables",
                actions_taken=["Detected missing environment variable"],
                reversible=True,
            )

        elif "no such file" in log_text.lower() or "filenotfound" in log_text.lower():
            return FixResult(
                success=False,
                fix_type="config_recommendation",
                duration_ms=0.0,
                details="Recommendation: Verify file paths in configuration",
                actions_taken=["Detected missing file"],
                reversible=True,
            )

        else:
            return FixResult(
                success=False,
                fix_type="config_manual",
                duration_ms=0.0,
                details="Configuration issues require manual intervention",
                actions_taken=[],
                reversible=True,
            )

    # ── Test Fixes ───────────────────────────────────────────────────────────

    def _fix_test(self, log_text: str, job_name: str, build_number: int, telemetry: Optional[Dict]) -> FixResult:
        """Fix test-related failures.

        SAFE fixes:
        - Recommend retry (flaky tests)
        - Suggest timeout increase
        - Recommend running subset of tests
        """
        actions = []

        if "flaky" in log_text.lower() or "intermittent" in log_text.lower():
            return FixResult(
                success=False,
                fix_type="test_recommendation",
                duration_ms=0.0,
                details="Recommendation: Retry build (flaky test detected)",
                actions_taken=["Detected flaky test"],
                reversible=True,
            )

        elif "timeout" in log_text.lower() or "timed out" in log_text.lower():
            return FixResult(
                success=False,
                fix_type="test_recommendation",
                duration_ms=0.0,
                details="Recommendation: Increase test timeout configuration",
                actions_taken=["Detected test timeout"],
                reversible=True,
            )

        else:
            return FixResult(
                success=False,
                fix_type="test_manual",
                duration_ms=0.0,
                details="Test failures require investigation",
                actions_taken=[],
                reversible=True,
            )

    # ── Build Fixes ──────────────────────────────────────────────────────────

    def _fix_build(self, log_text: str, job_name: str, build_number: int) -> FixResult:
        """Fix build-related failures.

        SAFE fixes:
        - Clear build artifacts
        - Recommend clean build
        """
        actions = []

        # For build errors (syntax, compilation), manual fix is required
        # We can only recommend clean build

        return FixResult(
            success=False,
            fix_type="build_recommendation",
            duration_ms=0.0,
            details="Recommendation: Review syntax/compilation errors and retry with clean build",
            actions_taken=["Detected build error"],
            reversible=True,
        )

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_fix_attempt(
        self,
        failure_type: str,
        result: FixResult,
        job_name: str,
        build_number: int,
    ) -> None:
        """Log fix attempt to file for audit trail."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_name": job_name,
            "build_number": build_number,
            "failure_type": failure_type,
            "fix_type": result.fix_type,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "details": result.details,
            "actions_taken": result.actions_taken,
            "reversible": result.reversible,
        }

        log_file = self.log_dir / f"fix_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(
            f"Fix attempt logged: {failure_type} -> {result.fix_type} "
            f"(success={result.success}, duration={result.duration_ms:.0f}ms)"
        )


# ── Module-Level API ─────────────────────────────────────────────────────────

_DEFAULT_FIXER: Optional[CICDAutoFixer] = None


def fix_cicd_failure(
    failure_type: str,
    log_text: str,
    job_name: str = "unknown",
    build_number: int = 0,
    telemetry: Optional[Dict] = None,
    dry_run: bool = False,
) -> FixResult:
    """Module-level helper for quick fix attempts.

    Args:
        failure_type: Type from classifier
        log_text: Jenkins console log
        job_name: Jenkins job name
        build_number: Build number
        telemetry: Optional telemetry data
        dry_run: If True, log actions but don't execute

    Returns:
        FixResult with success status and details
    """
    global _DEFAULT_FIXER
    if _DEFAULT_FIXER is None or _DEFAULT_FIXER.dry_run != dry_run:
        _DEFAULT_FIXER = CICDAutoFixer(dry_run=dry_run)
    return _DEFAULT_FIXER.fix_failure(failure_type, log_text, job_name, build_number, telemetry)
