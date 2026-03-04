"""Tests for the NeuroShield RL agent (environment + simulator).

All tests run without GPU or real Kubernetes — purely synthetic data.
"""

from __future__ import annotations

import random
import numpy as np
import pytest

from src.rl_agent.simulator import (
    FAILURE_TYPES,
    BASELINE_MTTR_MINUTES,
    OPTIMAL_MTTR_MINUTES,
    OPTIMAL_ACTION,
    NUM_ACTIONS,
    OBS_DIM,
    SimulationResult,
    sample_state,
    simulate_action,
    simulate_mttr,
)
from src.rl_agent.env import NeuroShieldEnv


# ── Simulator ─────────────────────────────────────────────────────────────────


class TestSimulatorConstants:
    def test_obs_dim(self):
        assert OBS_DIM == 52

    def test_num_actions(self):
        assert NUM_ACTIONS == 6

    def test_failure_types_list(self):
        assert "OOM" in FAILURE_TYPES
        assert "FlakyTest" in FAILURE_TYPES
        assert "Healthy" in FAILURE_TYPES
        assert len(FAILURE_TYPES) == 5


class TestSampleState:
    @pytest.mark.parametrize("failure_type", FAILURE_TYPES)
    def test_shape_and_dtype(self, failure_type):
        state = sample_state(random.Random(0), failure_type)
        assert state.shape == (52,)
        assert state.dtype == np.float32

    def test_deterministic_with_seed(self):
        s1 = sample_state(random.Random(42), "OOM")
        s2 = sample_state(random.Random(42), "OOM")
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        s1 = sample_state(random.Random(0), "OOM")
        s2 = sample_state(random.Random(999), "OOM")
        assert not np.array_equal(s1, s2)


class TestSimulateMttr:
    @pytest.mark.parametrize("failure_type", [ft for ft in FAILURE_TYPES if ft != "Healthy"])
    def test_optimal_action_yields_lower_mttr(self, failure_type):
        optimal_act = OPTIMAL_ACTION[failure_type]
        mttr_optimal = simulate_mttr(failure_type, optimal_act)
        mttr_suboptimal = simulate_mttr(failure_type, 5)  # escalate is never optimal for failures
        assert mttr_optimal <= mttr_suboptimal

    def test_healthy_escalate(self):
        mttr = simulate_mttr("Healthy", 5)
        assert mttr == pytest.approx(0.5)


class TestSimulateAction:
    def test_returns_simulation_result(self):
        result = simulate_action("OOM", 3, random.Random(0))
        assert isinstance(result, SimulationResult)

    def test_result_fields(self):
        result = simulate_action("FlakyTest", 0, random.Random(1))
        assert isinstance(result.success, bool)
        assert isinstance(result.mttr, float)
        assert isinstance(result.cost, float)
        assert isinstance(result.resource_efficiency, float)
        assert isinstance(result.false_positive, (int, float))
        assert result.state.shape == (52,)

    def test_false_positive_on_healthy(self):
        # Acting on Healthy with a non-escalate action → false positive
        result = simulate_action("Healthy", 0, random.Random(0))
        assert result.false_positive == 1.0

    def test_no_false_positive_on_failure(self):
        result = simulate_action("OOM", 3, random.Random(0))
        assert result.false_positive == 0.0

    def test_resource_efficiency_range(self):
        for action in range(NUM_ACTIONS):
            result = simulate_action("OOM", action, random.Random(0))
            assert 0.0 <= result.resource_efficiency <= 1.0


# ── NeuroShieldEnv ────────────────────────────────────────────────────────────


class TestNeuroShieldEnv:
    @pytest.fixture()
    def env(self):
        e = NeuroShieldEnv(seed=42)
        yield e
        e.close()

    def test_observation_space(self, env):
        assert env.observation_space.shape == (52,)

    def test_action_space(self, env):
        assert env.action_space.n == 6

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs.shape == (52,)
        assert obs.dtype == np.float32
        assert "failure_type" in info

    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (52,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "action_name" in info

    @pytest.mark.parametrize("action", range(6))
    def test_all_actions_valid(self, env, action):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (52,)

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_episode_terminates(self):
        env = NeuroShieldEnv(seed=0, max_steps=5)
        env.reset()
        terminated = False
        for _ in range(10):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                break
        assert terminated
        env.close()

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reward_is_finite(self, env):
        env.reset()
        for _ in range(5):
            _, reward, _, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
