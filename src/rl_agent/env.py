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

Action space (6 discrete) — aligned with orchestrator:
  0: restart_pod        1: scale_up
  2: retry_build        3: rollback_deploy
  4: clear_cache        5: escalate_to_human

Reward:
  Context-aware shaping that encourages the agent to pick the right
  action for the observed failure type.
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

    # --- Action labels (aligned with orchestrator) ---
    ACTION_NAMES = {
        0: "restart_pod",
        1: "scale_up",
        2: "retry_build",
        3: "rollback_deploy",
        4: "clear_cache",
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

        # --- Context-aware reward shaping ---
        mttr_reduction = 1.0 - result.mttr / baseline_mttr
        base_reward = (
            0.6 * mttr_reduction
            + 0.3 * result.resource_efficiency
            - 0.1 * result.false_positive
        )

        # Extract state features for context-aware bonuses
        pod_restarts = float(self.state[13])  # resource_metrics[3]
        cpu_avg = float(self.state[10])        # resource_metrics[0]
        mem_avg = float(self.state[11])        # resource_metrics[1]
        failed_tests = float(self.state[2])    # build_metrics[2]
        stage_failure_rate = float(self.state[4])  # build_metrics[4]

        bonus = 0.0
        action = int(action)
        ft = self.current_failure_type

        if action == 0:  # restart_pod — good when pod_restarts high
            if pod_restarts > 3 or ft == "OOM":
                bonus = 0.2
        elif action == 1:  # scale_up — good when cpu/memory high
            if cpu_avg > 0.7 or mem_avg > 0.7 or ft == "NetworkLatency":
                bonus = 0.2
        elif action == 2:  # retry_build — good when build failed (flaky)
            if failed_tests > 0 or stage_failure_rate > 0.1 or ft == "FlakyTest":
                bonus = 0.2
        elif action == 3:  # rollback_deploy — good when error_rate high
            if ft == "DependencyConflict" or stage_failure_rate > 0.5:
                bonus = 0.2
        elif action == 4:  # clear_cache — good when memory high + build ok
            if mem_avg > 0.7 and failed_tests == 0:
                bonus = 0.15
        elif action == 5:  # escalate_to_human
            if ft == "Healthy":
                bonus = 0.3  # correct to not act on healthy
            else:
                # Small positive only when very uncertain
                prob_proxy = stage_failure_rate
                bonus = 0.05 if prob_proxy > 0.85 else -0.1

        reward = base_reward + bonus

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
