"""Gymnasium environment for NeuroShield CI/CD decision making."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator import FAILURE_TYPES, BASELINE_MTTR_MINUTES, OPTIMAL_MTTR_MINUTES, simulate_action


class NeuroShieldEnv(gym.Env):
    """Custom environment for CI/CD recovery action selection."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42, max_steps: int = 100) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )
        self.rng = random.Random(seed)
        self.current_failure_type: Optional[str] = None
        self.state: Optional[np.ndarray] = None
        self.steps = 0
        self.max_steps = max_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng.seed(seed)
        self.steps = 0
        if self.rng.random() < 0.3:
            self.current_failure_type = "Healthy"
        else:
            self.current_failure_type = self.rng.choice([t for t in FAILURE_TYPES if t != "Healthy"])
        result = simulate_action(self.current_failure_type, 3, self.rng)
        self.state = result.state.astype(np.float32)
        return self.state, {"failure_type": self.current_failure_type}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.state is None or self.current_failure_type is None:
            raise RuntimeError("Environment must be reset before stepping.")

        result = simulate_action(self.current_failure_type, int(action), self.rng)
        baseline_mttr = BASELINE_MTTR_MINUTES.get(self.current_failure_type, 12.4)
        reward = (
            0.6 * (1.0 - result.mttr / baseline_mttr)
            + 0.3 * float(result.success)
            - 0.1 * result.cost
        )
        info = {
            "failure_type": self.current_failure_type,
            "mttr": result.mttr,
            "baseline_mttr": baseline_mttr,
            "action_taken": int(action),
        }
        self.state = result.state.astype(np.float32)
        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False
        return self.state, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Minimal render for debugging."""
        if self.current_failure_type is None:
            print("Environment not initialized.")
        else:
            print(f"Failure: {self.current_failure_type} | Step: {self.steps}")


# Backwards-compatible alias
CICDEnv = NeuroShieldEnv
