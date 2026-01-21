"""Synthetic CI/CD pipeline simulator for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import random
import numpy as np

FAILURE_TYPES = ["OOM", "FlakyTest", "Dependency", "Healthy"]

BASELINE_MTTR_MINUTES: Dict[str, float] = {
    "OOM": 14.2,
    "FlakyTest": 8.5,
    "Dependency": 15.1,
    "Healthy": 2.0,
}

ACTION_MTTR_MINUTES: Dict[Tuple[str, int], float] = {
    ("OOM", 1): 7.5,
    ("FlakyTest", 0): 4.3,
    ("Dependency", 2): 9.8,
    ("Healthy", 3): 2.0,
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
    mttr = ACTION_MTTR_MINUTES.get((failure_type, action), baseline)
    success = (failure_type, action) in ACTION_MTTR_MINUTES or failure_type == "Healthy"
    cost = 1.0 if action in (1, 2) else 0.0
    state = sample_state(rng)

    return SimulationResult(success=success, mttr=mttr, cost=cost, state=state)
