"""Tests for the NeuroShield prediction pipeline.

All tests run without GPU / real DistilBERT — heavy models are mocked.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.prediction.data_generator import generate_dataset, generate_sample
from src.prediction.model import FailureClassifier
from src.prediction.predictor import build_52d_state


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_log_encoder(monkeypatch):
    """Replace the real LogEncoder with a lightweight stub that returns
    random 16-D vectors (no DistilBERT download required).
    """
    from src.prediction import log_encoder as le_mod

    class _StubEncoder:
        def __init__(self, *_a, **_kw):
            pass

        def encode_texts(self, texts, **_kw):
            return np.random.default_rng(0).standard_normal((len(list(texts)), 768))

        def encode_logs(self, texts):
            return np.random.default_rng(0).standard_normal((len(list(texts)), 16)).astype(np.float32)

        def fit_pca(self, embeddings, n_components=16):
            pass

        def transform_pca(self, embeddings):
            return np.random.default_rng(0).standard_normal((len(embeddings), 16)).astype(np.float32)

    monkeypatch.setattr(le_mod, "LogEncoder", _StubEncoder)
    return _StubEncoder()


# ── Data Generator ────────────────────────────────────────────────────────────


class TestDataGenerator:
    def test_data_generator_shape(self):
        df = generate_dataset(num_samples=50, seed=0)
        assert len(df) == 50
        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})

    def test_data_generator_balance(self):
        df = generate_dataset(num_samples=200, seed=1)
        counts = df["label"].value_counts()
        assert counts.get(0, 0) > 0
        assert counts.get(1, 0) > 0

    def test_generate_sample_returns_dataclass(self):
        import random

        sample = generate_sample(random.Random(42))
        assert hasattr(sample, "log_text")
        assert hasattr(sample, "label")
        assert sample.label in (0, 1)


# ── Log Encoder ───────────────────────────────────────────────────────────────


class TestLogEncoder:
    def test_log_encoder_output_shape(self, mock_log_encoder):
        output = mock_log_encoder.encode_logs(["OOM error in stage build"])
        assert output.shape == (1, 16)
        assert output.dtype == np.float32


# ── Failure Classifier ────────────────────────────────────────────────────────


class TestFailureClassifier:
    def test_forward_shape(self):
        model = FailureClassifier(input_dim=24)
        x = torch.randn(4, 24)
        out = model(x)
        # Model output is (batch,) after squeeze
        assert out.shape == (4,)

    def test_forward_single_sample(self):
        model = FailureClassifier(input_dim=24)
        x = torch.randn(1, 24)
        out = model(x)
        assert out.shape == (1,) or out.dim() <= 1


# ── 52-D State Builder ───────────────────────────────────────────────────────


class TestBuild52dState:
    def test_returns_correct_shape(self):
        state = build_52d_state(
            jenkins_data={"build_duration": 120, "passed_tests": 45, "failed_tests": 3},
            prometheus_data={"cpu_avg_5m": 0.75, "memory_avg_5m": 0.6},
            log_text="Build failed: OOM error",
        )
        assert state.shape == (52,)
        assert state.dtype in (np.float32, np.float64)
        assert np.all(state >= -10.0) and np.all(state <= 10.0)

    def test_empty_inputs(self):
        state = build_52d_state(jenkins_data={}, prometheus_data={}, log_text="")
        assert state.shape == (52,)
        # All zeros when no data provided and no encoder
        assert np.allclose(state, 0.0)

    def test_with_mock_encoder(self, mock_log_encoder):
        state = build_52d_state(
            jenkins_data={"build_duration": 200},
            prometheus_data={"cpu_avg_5m": 0.9},
            log_text="ERROR: test failure",
            encoder=mock_log_encoder,
        )
        assert state.shape == (52,)
        # Log embed slots (22:38) should be non-zero when encoder is used
        assert not np.allclose(state[22:38], 0.0)

    def test_clipping(self):
        state = build_52d_state(
            jenkins_data={"build_duration": 99999},
            prometheus_data={},
            log_text="",
        )
        assert np.all(state >= -10.0) and np.all(state <= 10.0)
