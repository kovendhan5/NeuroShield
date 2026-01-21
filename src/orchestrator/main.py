"""NeuroShield end-to-end orchestrator demo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
import random

from stable_baselines3 import PPO

from src.prediction.predictor import FailurePredictor
from src.rl_agent.simulator import FAILURE_TYPES, simulate_action

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ACTION_NAMES: Dict[int, str] = {
    0: "Retry Stage",
    1: "Scale pods (+20%)",
    2: "Rollback",
    3: "No-op",
}


def _generate_log(failure_type: str) -> str:
    """Generate a raw log string based on failure type."""
    if failure_type == "OOM":
        return "[ERROR] OutOfMemoryError: Java heap space at step compile"
    if failure_type == "FlakyTest":
        return "[ERROR] TestLoginFlow failed intermittently; retry suggested"
    if failure_type == "Dependency":
        return "[ERROR] Dependency resolution failed; rollback required"
    return "[INFO] Build completed successfully"


def _load_predictor(model_dir: Path) -> FailurePredictor | None:
    """Load the failure predictor with graceful error handling."""
    try:
        return FailurePredictor(model_dir=model_dir)
    except Exception as exc:
        logger.error("Failed to load failure predictor: %s", exc)
        return None


def _load_policy(model_dir: Path) -> PPO | None:
    """Load the PPO policy with graceful error handling."""
    policy_path = model_dir / "ppo_policy.zip"
    if not policy_path.exists():
        logger.error("PPO policy not found at %s", policy_path)
        return None
    try:
        return PPO.load(policy_path)
    except Exception as exc:
        logger.error("Failed to load PPO policy: %s", exc)
        return None


def run_demo(runs: int = 100, seed: int = 42) -> None:
    """Run the end-to-end NeuroShield evaluation demo."""
    rng = random.Random(seed)
    model_dir = Path("models")

    predictor = _load_predictor(model_dir)
    policy = _load_policy(model_dir)
    if predictor is None or policy is None:
        logger.error("Missing models. Train predictor and PPO policy before running demo.")
        return

    total_mttr_neuro = 0.0
    total_mttr_baseline = 0.0
    successes = 0

    for _ in range(runs):
        failure_type = rng.choice(FAILURE_TYPES)
        log_text = _generate_log(failure_type)

        failure_prob, state_vector = predictor.predict_with_state(log_text, telemetry=None)

        if failure_prob > 0.5:
            action, _ = policy.predict(state_vector, deterministic=True)
            action = int(action)
            result = simulate_action(failure_type, action, rng)
            mttr = result.mttr
            success = result.success
        else:
            action = 3
            mttr = 0.5
            success = True

        baseline_result = simulate_action(failure_type, 0, rng)

        total_mttr_neuro += mttr
        total_mttr_baseline += baseline_result.mttr
        successes += int(success)

        logger.debug(
            "Run: failure=%s action=%s mttr=%.2f baseline=%.2f",
            failure_type,
            ACTION_NAMES.get(action, str(action)),
            mttr,
            baseline_result.mttr,
        )

    mttr_reduction = 0.0
    if total_mttr_baseline > 0:
        mttr_reduction = (total_mttr_baseline - total_mttr_neuro) / total_mttr_baseline * 100.0
    success_rate = successes / max(runs, 1) * 100.0

    print(f"=== NeuroShield Evaluation ({runs} runs) ===")
    print(f"Baseline MTTR: {total_mttr_baseline:.1f} min")
    print(f"NeuroShield MTTR: {total_mttr_neuro:.1f} min")
    print(f"MTTR Reduction: {mttr_reduction:.1f}%")
    print(f"Success Rate: {success_rate:.0f}%")


if __name__ == "__main__":
    run_demo()
