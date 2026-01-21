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
    parser.add_argument("--timesteps", type=int, default=10_000, help="Training timesteps")
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    return parser.parse_args()


def _make_env(seed: int) -> NeuroShieldEnv:
    """Create a seeded NeuroShieldEnv."""
    env = NeuroShieldEnv(seed=seed)
    return env


def _evaluate(model: PPO, episodes: int = 10) -> None:
    """Run evaluation episodes and print metrics."""
    eval_env = _make_env(seed=42)
    action_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    mttr_reductions: List[float] = []
    successes = 0

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_counts[action] += 1
        _, _, _, _, info = eval_env.step(action)

        mttr = float(info.get("mttr", 12.4))
        mttr_reduction = max(0.0, (12.4 - mttr) / 12.4) * 100.0
        mttr_reductions.append(mttr_reduction)
        successes += int(bool(info.get("success", False)))

    avg_reduction = float(np.mean(mttr_reductions)) if mttr_reductions else 0.0
    success_rate = successes / max(episodes, 1)

    print("Evaluation Results")
    print(f"Average MTTR reduction (%): {avg_reduction:.2f}")
    print(f"Success rate: {success_rate:.2f}")
    print("Action distribution:")
    for action, count in action_counts.items():
        print(f"  Action {action}: {count}")


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

        _evaluate(model, episodes=10)
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc


if __name__ == "__main__":
    main()
