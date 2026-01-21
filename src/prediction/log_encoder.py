"""Log encoding utilities using DistilBERT and PCA."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from joblib import dump, load


class LogEncoder:
    """Encodes CI/CD logs with DistilBERT and reduces embeddings with PCA."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 128,
    ) -> None:
        """Initialize the encoder.

        Args:
            model_name: Hugging Face model name.
            device: Torch device override.
            max_length: Tokenization max length.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.pca: Optional[PCA] = None

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool token embeddings."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_texts(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        """Encode texts into DistilBERT embeddings.

        Args:
            texts: Iterable of log strings.
            batch_size: Batch size for encoding.

        Returns:
            NumPy array of shape (n_samples, hidden_size).
        """
        embeddings: List[np.ndarray] = []
        texts_list = list(texts)

        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
            pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"]).cpu().numpy()
            embeddings.append(pooled)

        return np.vstack(embeddings)

    def fit_pca(self, embeddings: np.ndarray, n_components: int = 16) -> PCA:
        """Fit PCA on embeddings.

        Args:
            embeddings: DistilBERT embeddings.
            n_components: PCA output dimensions.

        Returns:
            Fitted PCA instance.
        """
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(embeddings)
        self.pca = pca
        return pca

    def transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings with PCA.

        Args:
            embeddings: DistilBERT embeddings.

        Returns:
            PCA-reduced embeddings.
        """
        if self.pca is None:
            raise ValueError("PCA model is not loaded. Call fit_pca() or load_pca().")
        return self.pca.transform(embeddings)

    def encode_logs(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        """Encode logs and reduce to PCA dimensions.

        Args:
            texts: Iterable of log strings.
            batch_size: Batch size for encoding.

        Returns:
            Reduced embeddings of shape (n_samples, n_components).
        """
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        return self.transform_embeddings(embeddings)

    def save_pca(self, path: str | Path) -> None:
        """Persist PCA model to disk."""
        if self.pca is None:
            raise ValueError("PCA model is not fitted.")
        dump(self.pca, Path(path))

    def load_pca(self, path: str | Path) -> None:
        """Load PCA model from disk."""
        self.pca = load(Path(path))
