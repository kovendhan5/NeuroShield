"""Training script for the NeuroShield failure predictor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .data_generator import generate_dataset, split_logs_and_telemetry
from .log_encoder import LogEncoder
from .predictor import telemetry_to_vector


def _build_feature_matrix(log_embeddings: np.ndarray, telemetry: List[dict]) -> np.ndarray:
    """Concatenate PCA log embeddings with telemetry features."""
    telemetry_vectors = np.vstack([telemetry_to_vector(item) for item in telemetry])
    return np.hstack([log_embeddings, telemetry_vectors])


def train_model(
    num_samples: int,
    model_dir: Path,
    test_size: float,
    seed: int,
) -> float:
    """Train the classifier and return F1 score."""
    df = generate_dataset(num_samples=num_samples, seed=seed)
    logs, telemetry, labels = split_logs_and_telemetry(df)

    encoder = LogEncoder()
    embeddings = encoder.encode_texts(logs)
    encoder.fit_pca(embeddings, n_components=16)
    log_embeddings = encoder.transform_embeddings(embeddings)

    features = _build_feature_matrix(log_embeddings, telemetry)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(x_train, y_train)

    preds = classifier.predict(x_test)
    f1 = f1_score(y_test, preds)

    model_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pca(model_dir / "log_pca.joblib")
    dump(classifier, model_dir / "failure_classifier.joblib")

    return f1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train NeuroShield failure prediction model.")
    parser.add_argument("--num-samples", type=int, default=800, help="Number of synthetic samples")
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    f1 = train_model(
        num_samples=args.num_samples,
        model_dir=Path(args.model_dir),
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"F1-score: {f1:.3f}")


if __name__ == "__main__":
    main()
