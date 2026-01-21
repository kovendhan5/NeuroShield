"""Training script for the NeuroShield failure predictor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .data_generator import generate_dataset, split_logs_and_telemetry
from .log_encoder import LogEncoder
from .model import FailureClassifier
from .predictor import telemetry_to_vector


def _build_feature_matrix(log_embeddings: np.ndarray, telemetry: List[dict]) -> np.ndarray:
    """Concatenate PCA log embeddings with telemetry features."""
    telemetry_vectors = np.vstack([telemetry_to_vector(item) for item in telemetry])
    return np.hstack([log_embeddings, telemetry_vectors])


def _prepare_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    """Split features into train/test TensorDatasets."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    return train_ds, test_ds


def train_model(
    num_samples: int,
    model_dir: Path,
    test_size: float,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> float:
    """Train the classifier and return F1 score."""
    df = generate_dataset(num_samples=num_samples, seed=seed)
    logs, telemetry, labels = split_logs_and_telemetry(df)

    encoder = LogEncoder()
    embeddings = encoder.encode_texts(logs)
    encoder.fit_pca(embeddings, n_components=16)
    log_embeddings = encoder.transform_embeddings(embeddings)

    features = _build_feature_matrix(log_embeddings, telemetry)
    train_ds, test_ds = _prepare_data(features, labels, test_size, seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FailureClassifier(input_dim=features.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    test_x, test_y = next(iter(DataLoader(test_ds, batch_size=len(test_ds))))
    with torch.no_grad():
        logits = model(test_x.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(test_y.numpy(), preds)

    model_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pca(model_dir / "log_pca.joblib")
    torch.save(model.state_dict(), model_dir / "failure_predictor.pth")

    return f1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train NeuroShield failure prediction model.")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of synthetic samples")
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    f1 = train_model(
        num_samples=args.num_samples,
        model_dir=Path(args.model_dir),
        test_size=args.test_size,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(f"F1-score: {f1:.3f}")


if __name__ == "__main__":
    main()
