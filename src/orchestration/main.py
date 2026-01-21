"""Orchestrator for NeuroShield MVP flow."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import logging
import random
import pandas as pd
from stable_baselines3 import PPO

from src.prediction.data_generator import generate_sample
from src.prediction.predictor import FailurePredictor
from src.rl_agent.simulator import simulate_action

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ACTION_NAMES: Dict[int, str] = {
    0: "Retry Stage",
    1: "Scale Pods (+20%)",
    2: "Rollback",
    3: "No-op",
}


def load_policy(model_dir: Path) -> PPO | None:
    """Load PPO policy if available."""
    policy_path = model_dir / "ppo_policy.zip"
    if policy_path.exists():
        return PPO.load(policy_path)
    return None


def load_latest_telemetry(csv_path: Path) -> Dict[str, object] | None:
    """Load the latest telemetry record if the CSV exists."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def run_once(model_dir: str = "models") -> None:
    """Run one end-to-end simulated decision."""
    model_path = Path(model_dir)
    predictor = FailurePredictor(model_dir=model_path)
    policy = load_policy(model_path)

    telemetry_record = load_latest_telemetry(Path("data/telemetry.csv"))
    if telemetry_record and telemetry_record.get("jenkins_last_build_log"):
        log_text = str(telemetry_record.get("jenkins_last_build_log"))
        failure_prob, state_vector = predictor.predict_with_state(log_text, telemetry_record)
        failure_type = "unknown"
    else:
        sample = generate_sample(random.Random(42))
        log_text = sample.log_text
        failure_prob, state_vector = predictor.predict_with_state(sample.log_text, sample.telemetry)
        failure_type = sample.failure_type

    if policy is None:
        action = 0
        logger.warning("PPO policy not found. Falling back to Retry action.")
    else:
        action, _ = policy.predict(state_vector, deterministic=True)
        action = int(action)

    result = simulate_action(failure_type, action, random.Random(123))
    baseline = simulate_action(failure_type, 0, random.Random(456))

    mttr_reduction = max(0.0, (baseline.mttr - result.mttr) / max(baseline.mttr, 1.0))

    logger.info("Failure probability: %.3f", failure_prob)
    logger.info("Chosen action: %s", ACTION_NAMES.get(action, str(action)))
    logger.info("Simulated MTTR: %.1fs (baseline retry %.1fs)", result.mttr, baseline.mttr)
    logger.info("MTTR reduction: %.1f%%", mttr_reduction * 100.0)


if __name__ == "__main__":
    run_once()
