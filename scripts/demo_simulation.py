#!/usr/bin/env python3
"""NeuroShield Demo Simulation — Two real-world self-healing scenarios."""

import sys
import time
from datetime import datetime

from colorama import Fore, Style, init

init(autoreset=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CYAN = Fore.CYAN
GREEN = Fore.GREEN
RED = Fore.RED
YELLOW = Fore.YELLOW
WHITE = Fore.WHITE
BRIGHT = Style.BRIGHT
RESET = Style.RESET_ALL


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def step(color: str, msg: str, delay: float = 2.0) -> None:
    print(f"  {WHITE}[{ts()}]{RESET} {color}{msg}{RESET}")
    time.sleep(delay)


def header(title: str) -> None:
    width = 60
    print()
    print(f"{CYAN}{BRIGHT}{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}{RESET}")
    print()


def subheader(title: str) -> None:
    print(f"\n  {CYAN}{BRIGHT}--- {title} ---{RESET}\n")


# ---------------------------------------------------------------------------
# Scenario 1: Build Failure → Auto-Heal
# ---------------------------------------------------------------------------

def scenario_build_failure() -> None:
    header("SCENARIO 1: Developer pushes code — Build fails and auto-heals")

    step(WHITE, "Developer pushed code to main branch")
    step(WHITE, "Jenkins detected new commit — starting build #42")
    step(RED + BRIGHT, "Build FAILED — dependency download timed out")
    step(CYAN, "NeuroShield detected failure (confidence: 87%)")
    step(WHITE, "Telemetry collected — CPU: 45%, Memory: 62%, Error rate: 0.8")
    step(YELLOW, "DistilBERT encoding build log...")
    step(YELLOW, "PPO Agent analyzing 52-dimensional state vector...")
    step(CYAN + BRIGHT, "Decision: retry_build (confidence: 91%)")
    step(YELLOW, "Executing: Retrying Jenkins build #42...")
    step(WHITE, "Build #43 started automatically")
    step(GREEN + BRIGHT, "Build #43 SUCCESS ✓")
    step(GREEN, "MTTR: 18 seconds | Baseline: 32 seconds | Reduction: 43.75%")
    step(GREEN + BRIGHT, "Developer never notified — issue resolved silently")


# ---------------------------------------------------------------------------
# Scenario 2: Pod Crash → Restart → Escalate
# ---------------------------------------------------------------------------

def scenario_pod_crash() -> None:
    header("SCENARIO 2: Pod crashes — NeuroShield restarts it, then escalates")

    step(WHITE, "Kubernetes pod dummy-app-7f4d8ddfc7 status: Running")
    step(RED, "Prometheus alert: Memory usage spike — 91%")
    step(CYAN, "NeuroShield detected anomaly (confidence: 94%)")
    step(WHITE, "Telemetry collected — CPU: 78%, Memory: 91%, Pod restarts: 3")
    step(YELLOW, "DistilBERT encoding system logs...")
    step(YELLOW, "PPO Agent analyzing 52-dimensional state vector...")
    step(CYAN + BRIGHT, "Decision: restart_pod (confidence: 88%)")
    step(YELLOW, "Executing: kubectl rollout restart deployment/dummy-app")
    step(YELLOW, "Pod restarting... waiting for healthy status")
    step(GREEN + BRIGHT, "Pod restarted successfully ✓")
    step(GREEN, "Memory back to normal: 34%")
    step(GREEN, "MTTR: 22 seconds | Baseline: 41 seconds | Reduction: 46.3%")

    subheader("Monitoring for recurrence...")
    time.sleep(3)

    step(RED + BRIGHT, "ALERT: Pod crashed again within 5 minutes — pattern detected")
    step(CYAN + BRIGHT, "Decision: escalate_to_human (this needs investigation)")
    step(YELLOW, "Generating diagnosis report...")
    step(RED + BRIGHT, "ESCALATION SENT: Slack/Email notification with full report")
    step(WHITE, "Report includes: crash history, memory profile, suggested fix")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    print()
    print(f"{CYAN}{BRIGHT}╔══════════════════════════════════════════════════════╗")
    print(f"║           NEUROSHIELD DEMO SUMMARY                   ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║ {GREEN}Scenario 1: Build Failure     → AUTO-HEALED ✓{CYAN}        ║")
    print(f"║ {GREEN}Scenario 2: Pod Crash         → AUTO-HEALED ✓{CYAN}        ║")
    print(f"║ {RED}Scenario 2: Repeated Crash    → ESCALATED TO HUMAN ✓{CYAN} ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║ {WHITE}Total MTTR Reduction: 44%{CYAN}                            ║")
    print(f"║ {WHITE}Human Interventions Needed: 1 out of 3 incidents{CYAN}     ║")
    print(f"║ {WHITE}Issues Resolved Silently: 2 out of 3 incidents{CYAN}       ║")
    print(f"╚══════════════════════════════════════════════════════╝{RESET}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"\n{CYAN}{BRIGHT}{'=' * 60}")
    print(f"       NEUROSHIELD — AI-Powered Self-Healing CI/CD Demo")
    print(f"{'=' * 60}{RESET}\n")
    print(f"  {WHITE}Starting simulation at {ts()}...")
    print(f"  Two real-world scenarios will play out below.{RESET}\n")
    time.sleep(2)

    scenario_build_failure()
    time.sleep(2)
    scenario_pod_crash()
    print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
