"""Gymnasium environment for NeuroShield CI/CD decision making.

Observation space (52D):
  [ 0:10]  Build Metrics          – build_duration, passed_tests, failed_tests,
           retry_count, stage_failure_rate, build_number, queue_time,
           artifact_size, test_coverage, change_set_size
  [10:22]  Resource Metrics       – cpu_avg_5m, memory_avg_5m, memory_max,
           pod_restarts, throttle_events, network_latency, disk_io,
           cpu_limit_pct, memory_limit_pct, node_count, pending_pods,
           evicted_pods
  [22:38]  Log Embeddings (16D)   – PCA-reduced DistilBERT output
  [38:52]  Dependency Signals     – dep_version_drifts, cache_hit_ratio,
           cache_miss_ratio, new_deps_count, outdated_deps, pkg_manager_npm,
           pkg_manager_maven, pkg_manager_pip, dep_resolution_time,
           lock_file_changed, transitive_dep_count, dep_conflict_count,
           registry_latency, dep_download_failures

Action space (6 discrete):
  0: retry_stage           1: clean_and_rerun
  2: regenerate_config     3: reallocate_resources
  4: trigger_safe_rollback 5: escalate_to_human

Reward:
  R = 0.6 * mttr_reduction + 0.3 * resource_efficiency
      - 0.1 * false_positive_penalty
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator import (
    FAILURE_TYPES,
    BASELINE_MTTR_MINUTES,
    OPTIMAL_MTTR_MINUTES,
    NUM_ACTIONS,
    OBS_DIM,
    simulate_action,
)


class NeuroShieldEnv(gym.Env):
    """Custom environment for CI/CD recovery action selection."""

    metadata = {"render_modes": []}

    # --- Action labels (for logging / human readability) ---
    ACTION_NAMES = {
        0: "retry_stage",
        1: "clean_and_rerun",
        2: "regenerate_config",
        3: "reallocate_resources",
        4: "trigger_safe_rollback",
        5: "escalate_to_human",
    }

    def __init__(self, seed: int = 42, max_steps: int = 100) -> None:
        super().__init__()
        # 6 discrete healing actions
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # 52-dimensional continuous observation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
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
        # Use escalate_to_human (5) as the reset action – neutral observation
        result = simulate_action(self.current_failure_type, 5, self.rng)
        self.state = result.state.astype(np.float32)
        return self.state, {"failure_type": self.current_failure_type}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.state is None or self.current_failure_type is None:
            raise RuntimeError("Environment must be reset before stepping.")

        result = simulate_action(self.current_failure_type, int(action), self.rng)
        baseline_mttr = BASELINE_MTTR_MINUTES.get(self.current_failure_type, 12.4)

        # --- Paper reward function ---
        # R = 0.6 * mttr_reduction + 0.3 * resource_efficiency
        #     - 0.1 * false_positive_penalty
        mttr_reduction = 1.0 - result.mttr / baseline_mttr
        reward = (
            0.6 * mttr_reduction
            + 0.3 * result.resource_efficiency
            - 0.1 * result.false_positive
        )

        info = {
            "failure_type": self.current_failure_type,
            "mttr": result.mttr,
            "baseline_mttr": baseline_mttr,
            "action_taken": int(action),
            "action_name": self.ACTION_NAMES.get(int(action), "unknown"),
            "resource_efficiency": result.resource_efficiency,
            "false_positive": result.false_positive,
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
