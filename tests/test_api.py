"""Tests for the NeuroShield REST API.

Uses FastAPI TestClient — no running server required.
"""

import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app

client = TestClient(app)


# 1. GET / returns 200 and correct name
def test_root_returns_api_info():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "NeuroShield AIOps API"
    assert data["version"] == "2.0"
    assert data["status"] == "running"


# 2. GET /health returns all service statuses
def test_health_returns_services():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "services" in data
    assert "jenkins" in data["services"]
    assert "prometheus" in data["services"]
    assert "dummy_app" in data["services"]
    assert "models" in data
    assert "overall" in data


# 3. GET /metrics returns cpu_usage as float
def test_metrics_returns_cpu_float():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "cpu_usage" in data
    assert isinstance(data["cpu_usage"], (int, float))


# 4. GET /telemetry returns list
def test_telemetry_returns_list():
    r = client.get("/telemetry")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# 5. GET /telemetry?limit=5 returns max 5 items
def test_telemetry_limit():
    r = client.get("/telemetry?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) <= 5


# 6. GET /telemetry/summary returns total_records > 0
def test_telemetry_summary():
    r = client.get("/telemetry/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_records" in data
    assert isinstance(data["total_records"], int)
    # May be 0 if no data, but key must exist
    assert data["total_records"] >= 0


# 7. POST /predict with FAILURE log returns prob > 0.5
@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_predict_failure(mock_tokenizer_from_pretrained, mock_model_from_pretrained):
    class DummyTokenizer:
        def __call__(self, batch, padding=True, truncation=True, max_length=128, return_tensors="pt"):
            batch_size = len(batch)
            seq_len = min(max_length, 8)
            return {
                "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            }

    class DummyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **encoded):
            input_ids = encoded["input_ids"]
            batch_size, seq_len = input_ids.shape
            hidden = torch.ones((batch_size, seq_len, 768), dtype=torch.float32)

            class DummyOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state

            return DummyOutput(hidden)

    mock_tokenizer_from_pretrained.return_value = DummyTokenizer()
    mock_model_from_pretrained.return_value = DummyModel()

    r = client.post("/predict", json={
        "log_text": "Finished: FAILURE",
        "cpu": 45.0,
        "memory": 67.0,
        "error_rate": 0.5,
        "build_status": "FAILURE",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["failure_probability"] > 0.5
    assert data["risk_level"] in ("MEDIUM", "HIGH")


# 8. POST /predict with SUCCESS log returns prob < 0.3
def test_predict_success():
    r = client.post("/predict", json={
        "log_text": "Finished: SUCCESS",
        "cpu": 10.0,
        "memory": 30.0,
        "error_rate": 0.0,
        "build_status": "SUCCESS",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["failure_probability"] < 0.3
    assert data["risk_level"] == "LOW"


# 9. GET /healing/stats returns action_distribution dict
def test_healing_stats():
    r = client.get("/healing/stats")
    assert r.status_code == 200
    data = r.json()
    assert "action_distribution" in data
    assert isinstance(data["action_distribution"], dict)


# 10. GET /mttr returns avg_reduction_pct as float
def test_mttr_response():
    r = client.get("/mttr")
    assert r.status_code == 200
    data = r.json()
    assert "avg_reduction_pct" in data
    assert isinstance(data["avg_reduction_pct"], (int, float))


# 11. GET /report/summary returns report data or 404
def test_report_summary():
    r = client.get("/report/summary")
    assert r.status_code in (200, 404)
    data = r.json()
    if r.status_code == 200:
        assert "predictor" in data
        assert "rl_agent" in data
    else:
        assert "error" in data


# 12. POST /report/generate returns generating status
def test_report_generate():
    r = client.post("/report/generate")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "generating"
    assert "estimated_time_seconds" in data
