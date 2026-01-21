"""Train PPO policy for NeuroShield recovery actions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed

from .env import NeuroShieldEnv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train PPO policy for CI/CD recovery.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training timesteps")
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    return parser.parse_args()


def _make_env(seed: int) -> NeuroShieldEnv:
    """Create a seeded NeuroShieldEnv."""
    env = NeuroShieldEnv(seed=seed)
    return env


def _evaluate(model: PPO, episodes: int = 50) -> None:
    """Run evaluation episodes and print metrics."""
    eval_env = _make_env(seed=42)
    action_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    mttr_reductions: List[float] = []
    successes = 0
    optimal_mttr = {
        "OOM": 7.5,
        "FlakyTest": 4.3,
        "DependencyConflict": 9.8,
        "NetworkLatency": 6.2,
        "Healthy": 0.5,
    }

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1
        _, _, _, _, info = eval_env.step(action)

        mttr = float(info.get("mttr", 12.4))
        failure_type = str(info.get("failure_type", "Healthy"))
        mttr_reduction = max(0.0, (12.4 - mttr) / 12.4) * 100.0
        mttr_reductions.append(mttr_reduction)
        threshold = optimal_mttr.get(failure_type, 12.4) + 1.0
        if failure_type == "Healthy":
            threshold = 1.5
        successes += int(mttr <= threshold)

    avg_reduction = float(np.mean(mttr_reductions)) if mttr_reductions else 0.0
    success_rate = successes / max(episodes, 1)

    action_percentages = [
        action_counts[0] / max(episodes, 1) * 100.0,
        action_counts[1] / max(episodes, 1) * 100.0,
        action_counts[2] / max(episodes, 1) * 100.0,
        action_counts[3] / max(episodes, 1) * 100.0,
    ]

    print(f"=== Final Evaluation ({episodes} episodes) ===")
    print(f"Avg MTTR Reduction: {avg_reduction:.1f}%")
    print(f"Success Rate: {success_rate * 100.0:.0f}%")
    print(
        "Action Distribution: ["
        f"{action_percentages[0]:.1f}%, "
        f"{action_percentages[1]:.1f}%, "
        f"{action_percentages[2]:.1f}%, "
        f"{action_percentages[3]:.1f}%]"
    )


def main() -> None:
    """Train PPO and save policy."""
    args = parse_args()
    try:
        set_random_seed(42)
        env = DummyVecEnv([lambda: Monitor(_make_env(seed=42))])
        env = VecMonitor(env)

        eval_env = DummyVecEnv([lambda: Monitor(_make_env(seed=123))])
        eval_env = VecMonitor(eval_env)

        eval_callback = EvalCallback(
            eval_env,
            eval_freq=1000,
            n_eval_episodes=5,
            deterministic=True,
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            batch_size=64,
            verbose=1,
        )
        model.learn(total_timesteps=args.timesteps, callback=eval_callback)

        output_dir = Path(args.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(output_dir / "ppo_policy.zip"))

        _evaluate(model, episodes=50)
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc


if __name__ == "__main__":
    main()
