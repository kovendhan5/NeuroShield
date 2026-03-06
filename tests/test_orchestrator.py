"""Tests for the NeuroShield orchestrator module.

All tests run without Jenkins, Kubernetes, or GPU.
External HTTP calls and subprocess calls are fully mocked.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.main import (
    ACTION_NAMES,
    BuildInfo,
    _append_csv,
    _is_failure,
    _log_action_history,
    _parse_kubectl_top_nodes,
    detect_failure_pattern,
    execute_healing_action,
    get_build_log,
    get_latest_build_info,
    retry_call,
)


# ── retry_call ────────────────────────────────────────────────────────────────


class TestRetryCall:
    def test_success_first_try(self):
        result = retry_call(lambda: 42, max_attempts=3, delay=0)
        assert result == 42

    def test_success_after_retries(self):
        attempt = {"n": 0}

        def flaky():
            attempt["n"] += 1
            if attempt["n"] < 3:
                raise ValueError("transient")
            return "ok"

        result = retry_call(flaky, max_attempts=3, delay=0)
        assert result == "ok"
        assert attempt["n"] == 3

    def test_all_retries_exhausted(self):
        with pytest.raises(ValueError, match="always fails"):
            retry_call(lambda: (_ for _ in ()).throw(ValueError("always fails")), max_attempts=2, delay=0)


# ── detect_failure_pattern ────────────────────────────────────────────────────


class TestDetectFailurePattern:
    @pytest.mark.parametrize(
        "log_text, expected_type",
        [
            ("java.lang.OutOfMemoryError: Java heap space", "OOM"),
            ("oom kill detected in container", "OOM"),
            ("flaky test com.example.FooTest", "FlakyTest"),
            ("dependency resolution failed", "Dependency"),
            ("connection timed out", "Timeout"),
            ("everything looks fine", "Unknown"),
        ],
    )
    def test_patterns(self, log_text, expected_type):
        pattern_type, _ = detect_failure_pattern(log_text)
        assert pattern_type == expected_type


# ── _parse_kubectl_top_nodes ──────────────────────────────────────────────────


class TestParseKubectlTopNodes:
    def test_normal_output(self):
        output = (
            "NAME       CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%\n"
            "node-1     250m         12%    1024Mi          25%\n"
            "node-2     500m         25%    2048Mi          50%\n"
        )
        cpu, mem = _parse_kubectl_top_nodes(output)
        assert cpu == pytest.approx(375.0)
        assert mem == pytest.approx(1536.0)

    def test_empty_output(self):
        cpu, mem = _parse_kubectl_top_nodes("")
        assert cpu == 0.0
        assert mem == 0.0


# ── _is_failure ───────────────────────────────────────────────────────────────


class TestIsFailure:
    @pytest.mark.parametrize("result", ["FAILURE", "UNSTABLE", "ABORTED", "failure"])
    def test_true(self, result):
        assert _is_failure(result) is True

    @pytest.mark.parametrize("result", ["SUCCESS", "success", "RUNNING"])
    def test_false(self, result):
        assert _is_failure(result) is False


# ── _append_csv ───────────────────────────────────────────────────────────────


class TestAppendCsv:
    def test_creates_file_and_appends(self, tmp_path):
        csv_path = str(tmp_path / "sub" / "test.csv")
        _append_csv(csv_path, {"a": "1", "b": "2"})
        _append_csv(csv_path, {"a": "3", "b": "4"})

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["a"] == "1"
        assert rows[1]["b"] == "4"


# ── _log_action_history ──────────────────────────────────────────────────────


class TestLogActionHistory:
    def test_logs_to_csv(self, tmp_path, monkeypatch):
        csv_file = tmp_path / "action_history.csv"
        monkeypatch.setattr(
            "src.orchestrator.main._append_csv",
            lambda path, row: _append_csv(str(csv_file), row),
        )
        _log_action_history(action_id=0, success=True, duration_ms=123.0)
        rows = list(csv.DictReader(open(csv_file, encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["action_name"] == "restart_pod"
        assert rows[0]["success"] == "True"


# ── BuildInfo ─────────────────────────────────────────────────────────────────


class TestBuildInfo:
    def test_end_time(self):
        b = BuildInfo(number=1, timestamp_ms=1000, duration_ms=500, result="SUCCESS", url="http://x")
        assert b.end_time_ms == 1500


# ── get_latest_build_info ─────────────────────────────────────────────────────


class TestGetLatestBuildInfo:
    @patch("src.orchestrator.main.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "number": 42,
                "timestamp": 1_000_000,
                "duration": 5000,
                "result": "SUCCESS",
                "url": "http://jenkins/job/1/42/",
            },
        )
        info = get_latest_build_info("http://jenkins", "job1", "admin", "tok")
        assert info is not None
        assert info.number == 42
        assert info.result == "SUCCESS"

    @patch("src.orchestrator.main.requests.get")
    def test_returns_none_on_error(self, mock_get):
        mock_get.side_effect = ConnectionError("unreachable")
        info = get_latest_build_info("http://jenkins", "job1", "admin", "tok")
        assert info is None


# ── get_build_log ─────────────────────────────────────────────────────────────


class TestGetBuildLog:
    @patch("src.orchestrator.main.requests.get")
    def test_returns_log_text(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="[ERROR] OOM kill")
        log = get_build_log("http://jenkins", "job1", 10, "admin", "tok")
        assert "OOM" in log


# ── execute_healing_action ────────────────────────────────────────────────────


class TestExecuteHealingAction:
    """Test each of the 6 healing actions with mocked externals."""

    @pytest.fixture(autouse=True)
    def _patch_csv(self, monkeypatch, tmp_path):
        """Redirect all CSV writes to tmp_path."""
        self._tmp = tmp_path

        def _fake_append(path, row):
            fname = Path(path).name
            _append_csv(str(tmp_path / fname), row)

        monkeypatch.setattr("src.orchestrator.main._append_csv", _fake_append)

    @pytest.fixture(autouse=True)
    def _patch_env(self, monkeypatch):
        monkeypatch.setenv("JENKINS_URL", "http://jenkins")
        monkeypatch.setenv("JENKINS_JOB", "test-job")
        monkeypatch.setenv("JENKINS_USERNAME", "admin")
        monkeypatch.setenv("JENKINS_TOKEN", "secret-token")
        monkeypatch.setenv("K8S_NAMESPACE", "ns")
        monkeypatch.setenv("AFFECTED_SERVICE", "svc")

    @patch("src.orchestrator.main.requests.post")
    def test_action_0_retry_stage(self, mock_post):
        mock_post.return_value = MagicMock(status_code=201)
        ok = execute_healing_action(0, {"build_number": "99"})
        assert ok is True
        mock_post.assert_called_once()

    @patch("src.orchestrator.main.requests.post")
    def test_action_1_clean_and_rerun(self, mock_post):
        mock_post.return_value = MagicMock(status_code=201)
        ok = execute_healing_action(1, {"build_number": "100"})
        assert ok is True

    def test_action_2_regenerate_config(self):
        ok = execute_healing_action(2, {"build_number": "101", "failure_pattern": "dep"})
        assert ok is True
        rows = list(csv.DictReader(open(self._tmp / "config_regen_log.csv", encoding="utf-8")))
        assert len(rows) == 1

    @patch("src.orchestrator.main.subprocess.run")
    def test_action_3_reallocate_resources(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        ok = execute_healing_action(3, {})
        assert ok is True
        mock_run.assert_called_once()

    @patch("src.orchestrator.main.subprocess.run")
    def test_action_4_trigger_safe_rollback(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        ok = execute_healing_action(4, {})
        assert ok is True
        assert mock_run.call_count == 2  # undo + status

    def test_action_5_escalate_to_human(self):
        ok = execute_healing_action(5, {"failure_prob": "0.9"})
        assert ok is True
        rows = list(csv.DictReader(open(self._tmp / "escalation_log.csv", encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["status"] == "PENDING_HUMAN_REVIEW"


# ── ACTION_NAMES ──────────────────────────────────────────────────────────────


class TestActionNames:
    def test_all_six_present(self):
        assert len(ACTION_NAMES) == 6
        assert ACTION_NAMES[0] == "retry_stage"
        assert ACTION_NAMES[5] == "escalate_to_human"
