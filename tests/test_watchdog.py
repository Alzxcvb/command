"""Tests for WatchdogAgent — mission alignment monitor."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestrator.watchdog import WatchdogAgent

TASK = "Write a competitive analysis report on three SaaS pricing models"
OUTPUTS = [
    {"agent_id": "agent_abc", "status": "completed", "final_text": "Section 1 done: flat-rate pricing analysis complete."},
]


def _make_provider(response: str):
    provider = MagicMock()
    provider.call_raw.return_value = (response, 0.01)
    return provider


def _aligned_response():
    return json.dumps({"aligned": True, "drift_note": None, "recurring_issue": False, "memory_note": "Keep sections under 500 words."})


def _drift_response():
    return json.dumps({"aligned": False, "drift_note": "Agent shifted to feature comparison instead of pricing.", "recurring_issue": False, "memory_note": None})


def _recurring_response():
    return json.dumps({"aligned": True, "drift_note": None, "recurring_issue": True, "memory_note": "Agents consistently over-write intro sections."})


class TestWatchdogCheckAlignment:
    def test_aligned_response_returns_aligned_true(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = _make_provider(_aligned_response())
            result = w.check_alignment("job_1", TASK, OUTPUTS)
        assert result["aligned"] is True
        assert result["drift_note"] is None
        assert result["memory_note"] == "Keep sections under 500 words."

    def test_drift_detected_returns_aligned_false_with_note(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = _make_provider(_drift_response())
            result = w.check_alignment("job_2", TASK, OUTPUTS)
        assert result["aligned"] is False
        assert "pricing" in result["drift_note"]

    def test_recurring_issue_flagged(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = _make_provider(_recurring_response())
            result = w.check_alignment("job_3", TASK, OUTPUTS)
        assert result["recurring_issue"] is True

    def test_model_failure_does_not_raise(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        provider = MagicMock()
        provider.call_raw.side_effect = RuntimeError("network error")
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = provider
            result = w.check_alignment("job_4", TASK, OUTPUTS)
        assert result["aligned"] is True

    def test_bad_json_does_not_raise(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = _make_provider("not json at all")
            result = w.check_alignment("job_5", TASK, OUTPUTS)
        assert result["aligned"] is True

    def test_observation_written_to_jsonl(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = _make_provider(_aligned_response())
            w.check_alignment("job_6", TASK, OUTPUTS)
        lines = (tmp_path / "observations.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["job_id"] == "job_6"
        assert entry["aligned"] is True
        assert entry["memory_note"] == "Keep sections under 500 words."

    def test_multiple_observations_accumulate(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = _make_provider(_aligned_response())
            w.check_alignment("job_7a", TASK, OUTPUTS)
            w.check_alignment("job_7b", TASK, OUTPUTS)
        lines = (tmp_path / "observations.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_empty_outputs_does_not_crash(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.return_value = _make_provider(_aligned_response())
            result = w.check_alignment("job_8", TASK, [])
        assert result["aligned"] is True

    def test_provider_import_failure_defaults_to_aligned(self, tmp_path):
        w = WatchdogAgent(state_dir=str(tmp_path))
        with patch("router.providers.AnthropicProvider") as MockProv:
            MockProv.side_effect = ImportError("no module")
            result = w.check_alignment("job_9", TASK, OUTPUTS)
        assert result["aligned"] is True
