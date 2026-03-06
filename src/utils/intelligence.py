"""NeuroShield Intelligence Utilities.

Predictive healing (early warning detection) and explainable AI (decision explanation).
These are used by both the orchestrator and the dashboard.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def detect_early_warning(
    telemetry_history: List[Dict],
) -> Tuple[Optional[str], float]:
    """Detect warning signs BEFORE a failure occurs by analyzing metric trends.

    Args:
        telemetry_history: List of telemetry dicts (most recent last), need >= 5.

    Returns:
        (action_name, confidence) if a warning trend is detected, else (None, 0.0)
    """
    if len(telemetry_history) < 5:
        return None, 0.0

    recent = telemetry_history[-5:]

    # Trend 1: Build duration increasing (sign of flakiness, slowness)
    durations = [
        float(r.get("jenkins_last_build_duration", 0) or 0)
        for r in recent
    ]
    durations = [d for d in durations if d > 0]
    if len(durations) >= 3:
        trend = np.polyfit(range(len(durations)), durations, 1)[0]
        if trend > 500:  # increasing by 500ms per build
            return "retry_build", 0.60

    # Trend 2: Memory creeping up (sign of memory leak / cache bloat)
    memories = [
        float(r.get("prometheus_memory_usage", 0) or 0)
        for r in recent
    ]
    memories = [m for m in memories if m > 0]
    if len(memories) >= 3:
        mem_trend = np.polyfit(range(len(memories)), memories, 1)[0]
        if mem_trend > 2.0:  # memory growing 2% per cycle
            return "clear_cache", 0.55

    # Trend 3: Any non-zero error rate (sign of deployment issue)
    error_rates = [
        float(r.get("prometheus_error_rate", 0) or 0)
        for r in recent
    ]
    if any(e > 0 for e in error_rates) and sum(error_rates) > 0:
        return "rollback_deploy", 0.58

    # Trend 4: CPU consistently elevated
    cpus = [
        float(r.get("prometheus_cpu_usage", 0) or 0)
        for r in recent
    ]
    cpus = [c for c in cpus if c > 0]
    if len(cpus) >= 3:
        cpu_trend = np.polyfit(range(len(cpus)), cpus, 1)[0]
        if cpu_trend > 3.0 and max(cpus) > 60:  # CPU growing toward saturation
            return "scale_up", 0.52

    return None, 0.0


def explain_decision(
    telemetry: Dict,
    action: str,
    prob: float,
) -> Dict:
    """Generate a human-readable explanation of why the AI chose an action.

    Args:
        telemetry: Current telemetry dict with build/resource metrics.
        action: The chosen healing action name.
        prob: Failure probability (0.0 – 1.0).

    Returns:
        Dict with keys: action, confidence, reasons (list of str), model.
    """
    reasons: List[str] = []

    build_status = str(telemetry.get("jenkins_last_build_status", "") or "").upper()
    cpu = float(telemetry.get("prometheus_cpu_usage", 0) or 0)
    memory = float(telemetry.get("prometheus_memory_usage", 0) or 0)
    restarts = float(telemetry.get("pod_restart_count", 0) or 0)
    error_rate = float(telemetry.get("prometheus_error_rate", 0) or 0)

    if build_status in ("FAILURE", "UNSTABLE", "ABORTED"):
        reasons.append(f"Build status: {build_status} (+0.525 probability)")
    if cpu > 80:
        reasons.append(f"CPU spike: {cpu:.0f}% (threshold: 80%)")
    elif cpu > 60:
        reasons.append(f"CPU elevated: {cpu:.0f}%")
    if memory > 85:
        reasons.append(f"Memory critical: {memory:.0f}% (threshold: 85%)")
    elif memory > 70:
        reasons.append(f"Memory elevated: {memory:.0f}%")
    if restarts >= 3:
        reasons.append(f"Pod restart loop: {int(restarts)} restarts detected")
    if error_rate > 0.3:
        reasons.append(f"High error rate: {error_rate:.3f} req/s failing")
    elif error_rate > 0:
        reasons.append(f"Non-zero error rate detected: {error_rate:.4f}")

    if not reasons:
        reasons.append(f"Pattern anomaly detected (prob={prob:.3f})")

    return {
        "action": action,
        "confidence": f"{prob:.1%}",
        "reasons": reasons,
        "model": "PPO + DistilBERT",
    }
