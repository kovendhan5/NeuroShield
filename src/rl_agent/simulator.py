"""Synthetic CI/CD pipeline simulator for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import random

FAILURE_MTTR: Dict[str, float] = {
    "oom": 420.0,
    "flaky_test": 300.0,
    "network_timeout": 240.0,
    "config_error": 360.0,
    "disk_full": 480.0,
    "healthy": 60.0,
}

ACTION_COST: Dict[int, float] = {
    0: 0.1,  # Retry
    1: 0.25,  # Scale pods
    2: 0.2,  # Rollback
    3: 0.0,  # No-op
}


@dataclass
class SimulationResult:
    """Results of a simulated pipeline action."""

    success: bool
    mttr: float
    cost: float


def simulate_action(failure_type: str, action: int, rng: random.Random) -> SimulationResult:
    """Simulate action outcome for a given failure type.

    Args:
        failure_type: Failure category or healthy.
        action: Discrete action id.
        rng: Random generator.

    Returns:
        SimulationResult with success flag, MTTR, and cost.
    """
    base_mttr = FAILURE_MTTR.get(failure_type, 300.0)
    cost = ACTION_COST.get(action, 0.0)

    if failure_type == "healthy":
        return SimulationResult(success=True, mttr=base_mttr, cost=cost)

    success_prob = 0.6
    mttr = base_mttr

    if action == 0:  # Retry
        if failure_type == "flaky_test":
            success_prob = 0.8
            mttr *= 0.7
        elif failure_type == "network_timeout":
            success_prob = 0.65
            mttr *= 0.85
        else:
            success_prob = 0.45
            mttr *= 1.05
    elif action == 1:  # Scale pods
        if failure_type == "oom":
            success_prob = 0.85
            mttr *= 0.6
        else:
            success_prob = 0.55
            mttr *= 0.9
    elif action == 2:  # Rollback
        if failure_type in {"config_error", "disk_full"}:
            success_prob = 0.8
            mttr *= 0.75
        else:
            success_prob = 0.5
            mttr *= 0.95
    elif action == 3:  # No-op
        success_prob = 0.2
        mttr *= 1.2

    success = rng.random() < success_prob
    if not success:
        mttr *= 1.2

    return SimulationResult(success=success, mttr=mttr, cost=cost)
