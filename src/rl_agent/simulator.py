"""Synthetic CI/CD pipeline simulator for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import random
import numpy as np

FAILURE_TYPES = [
    "OOM",
    "FlakyTest",
    "DependencyConflict",
    "NetworkLatency",
    "Healthy",
]

BASELINE_MTTR_MINUTES: Dict[str, float] = {
    "OOM": 14.2,
    "FlakyTest": 8.5,
    "DependencyConflict": 15.1,
    "NetworkLatency": 11.3,
    "Healthy": 0.5,
}

OPTIMAL_ACTION: Dict[str, int] = {
    "OOM": 1,
    "FlakyTest": 0,
    "DependencyConflict": 2,
    "NetworkLatency": 0,
    "Healthy": 3,
}

OPTIMAL_MTTR_MINUTES: Dict[str, float] = {
    "OOM": 7.5,
    "FlakyTest": 4.3,
    "DependencyConflict": 9.8,
    "NetworkLatency": 6.2,
    "Healthy": 0.5,
}


@dataclass
class SimulationResult:
    """Results of a simulated pipeline action."""

    success: bool
    mttr: float
    cost: float
    state: np.ndarray


def sample_state(rng: random.Random) -> np.ndarray:
    """Generate a synthetic 24D state vector.

    Args:
        rng: Random generator.

    Returns:
        24D float32 vector.
    """
    values = [rng.normalvariate(0.0, 1.0) for _ in range(24)]
    return np.array(values, dtype=np.float32)


def simulate_mttr(failure_type: str, action: int) -> float:
    """Return MTTR (minutes) based on optimal action table.

    Args:
        failure_type: Failure category.
        action: Discrete action id.

    Returns:
        MTTR in minutes. Optimal action yields optimal MTTR; otherwise baseline.
    """
    baseline = BASELINE_MTTR_MINUTES.get(failure_type, 12.4)
    optimal = OPTIMAL_MTTR_MINUTES.get(failure_type, baseline)
    return optimal if action == OPTIMAL_ACTION.get(failure_type, 3) else baseline


def simulate_action(failure_type: str, action: int, rng: random.Random) -> SimulationResult:
    """Simulate action outcome for a given failure type.

    Deterministic MTTR values based on paper Table 1.

    Args:
        failure_type: Failure category or Healthy.
        action: Discrete action id.
        rng: Random generator.

    Returns:
        SimulationResult with success flag, MTTR (minutes), cost, and state.
    """
    baseline = BASELINE_MTTR_MINUTES.get(failure_type, 12.4)
    optimal = OPTIMAL_MTTR_MINUTES.get(failure_type, baseline)
    mttr = simulate_mttr(failure_type, action)
    success = float(mttr <= (optimal + 1.0)) == 1.0
    cost = 1.0 if action in (1, 2) else 0.0
    state = sample_state(rng)

    return SimulationResult(success=success, mttr=mttr, cost=cost, state=state)
