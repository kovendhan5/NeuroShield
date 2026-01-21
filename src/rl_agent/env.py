"""Gymnasium environment for NeuroShield CI/CD decision making."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.prediction.data_generator import generate_sample
from src.prediction.predictor import FailurePredictor
from .simulator import simulate_action


class CICDEnv(gym.Env):
    """Custom environment for CI/CD recovery action selection."""

    metadata = {"render_modes": []}

    def __init__(self, model_dir: str = "models", seed: int = 42) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )
        self.rng = random.Random(seed)
        self.predictor = FailurePredictor(model_dir=model_dir)
        self.current_failure_type: Optional[str] = None
        self.state: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng.seed(seed)
        sample = generate_sample(self.rng)
        self.current_failure_type = sample.failure_type
        _, state_vector = self.predictor.predict_with_state(sample.log_text, sample.telemetry)
        self.state = state_vector.astype(np.float32)
        return self.state, {"failure_type": self.current_failure_type}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.state is None or self.current_failure_type is None:
            raise RuntimeError("Environment must be reset before stepping.")

        result = simulate_action(self.current_failure_type, int(action), self.rng)
        mttr_norm = min(result.mttr / 600.0, 1.0)
        reward = 0.5 * (1.0 - mttr_norm) + 0.3 * float(result.success) - 0.2 * result.cost
        info = {
            "success": result.success,
            "mttr": result.mttr,
            "cost": result.cost,
            "failure_type": self.current_failure_type,
        }
        terminated = True
        truncated = False
        return self.state, float(reward), terminated, truncated, info
