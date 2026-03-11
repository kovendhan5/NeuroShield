#!/usr/bin/env python3
"""Inject, fix, or check a simulated dependency conflict.

Writes a broken requirements file into the Jenkins container so that
Stage 1 (Dependency Install) of the upgraded job fails.

Usage:
    python scripts/inject_dep_conflict.py --inject   # create conflict
    python scripts/inject_dep_conflict.py --fix       # remove conflict
    python scripts/inject_dep_conflict.py --status    # show current state
"""

import argparse
import subprocess
import sys

from colorama import Fore, Style, init as colorama_init

colorama_init()

CONTAINER = "neuroshield-jenkins"
BROKEN_FILE = "/tmp/demo_requirements_broken.txt"

BROKEN_CONTENT = """\
# AUTO-GENERATED — simulated dependency conflict
# NeuroShield demo: these versions cannot coexist
numpy==1.21.0
numpy==2.1.0
scipy==1.7.0          # requires numpy>=1.22
torch==2.2.0          # requires numpy<2.0
pandas==2.0.0         # requires numpy>=1.23
scikit-learn==1.0.0   # requires scipy>=1.3,<1.8
"""


def _docker(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", CONTAINER, "sh", "-c", cmd],
        capture_output=True, text=True, timeout=15,
    )


def inject() -> int:
    print(f"\n{Fore.YELLOW}[INJECT]{Style.RESET_ALL} Writing broken deps into Jenkins container...")
    # Use printf to avoid shell interpretation issues
    escaped = BROKEN_CONTENT.replace("'", "'\\''")
    r = _docker(f"printf '%s' '{escaped}' > {BROKEN_FILE}")
    if r.returncode != 0:
        print(f"{Fore.RED}  [FAIL]{Style.RESET_ALL} {r.stderr.strip()}")
        return 1

    # Verify
    check = _docker(f"cat {BROKEN_FILE}")
    if "numpy==1.21.0" in check.stdout:
        print(f"{Fore.GREEN}  [OK]{Style.RESET_ALL} Conflict injected — next Jenkins build will fail at Stage 1")
        print(f"{Fore.CYAN}       File: {BROKEN_FILE} (inside {CONTAINER}){Style.RESET_ALL}")
        return 0

    print(f"{Fore.RED}  [FAIL]{Style.RESET_ALL} File not written correctly")
    return 1


def fix() -> int:
    print(f"\n{Fore.YELLOW}[FIX]{Style.RESET_ALL} Removing broken deps from Jenkins container...")
    r = _docker(f"rm -f {BROKEN_FILE}")
    if r.returncode != 0:
        print(f"{Fore.RED}  [FAIL]{Style.RESET_ALL} {r.stderr.strip()}")
        return 1

    check = _docker(f"test -f {BROKEN_FILE} && echo exists || echo gone")
    if "gone" in check.stdout:
        print(f"{Fore.GREEN}  [OK]{Style.RESET_ALL} Conflict removed — next Jenkins build will pass Stage 1")
        return 0

    print(f"{Fore.RED}  [FAIL]{Style.RESET_ALL} File still present")
    return 1


def status() -> int:
    print(f"\n{Fore.CYAN}[STATUS]{Style.RESET_ALL} Checking dependency conflict state...")
    check = _docker(f"test -f {BROKEN_FILE} && echo exists || echo gone")

    if "exists" in check.stdout:
        print(f"{Fore.RED}  INJECTED{Style.RESET_ALL} — {BROKEN_FILE} is present")
        content = _docker(f"cat {BROKEN_FILE}")
        for line in content.stdout.strip().splitlines():
            print(f"    {Style.DIM}{line}{Style.RESET_ALL}")
        print(f"\n  Next build will {Fore.RED}FAIL{Style.RESET_ALL} at Stage 1 (Dependency Install)")
        return 0

    print(f"{Fore.GREEN}  CLEAN{Style.RESET_ALL} — no conflict file present")
    print(f"  Next build will {Fore.GREEN}PASS{Style.RESET_ALL} Stage 1 (if deps are OK)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroShield Dependency Conflict Injector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inject", action="store_true", help="Inject broken dependencies")
    group.add_argument("--fix", action="store_true", help="Remove broken dependencies")
    group.add_argument("--status", action="store_true", help="Check current state")
    args = parser.parse_args()

    if args.inject:
        return inject()
    if args.fix:
        return fix()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
