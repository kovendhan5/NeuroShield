"""Train PPO policy for NeuroShield recovery actions."""

from __future__ import annotations

import argparse
from pathlib import Path
from stable_baselines3 import PPO

from .env import CICDEnv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train PPO policy for CI/CD recovery.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Training timesteps")
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    return parser.parse_args()


def main() -> None:
    """Train PPO and save policy."""
    args = parse_args()
    env = CICDEnv(model_dir=args.model_dir)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    output_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "ppo_policy.zip")


if __name__ == "__main__":
    main()
