"""Tests for judge_spawn_plan — planning judge gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import orchestrator.job as job
from orchestrator.breakdown import SpawnInstruction


def _make_provider(responses: list[str]):
    provider = MagicMock()
    provider.call_raw.side_effect = [(r, 0.01) for r in responses]
    return provider


def _approved(flags=None, confidence=0.95):
    return json.dumps({"approved": True, "flags": flags or [], "confidence": confidence})


def _rejected(flags, confidence=0.6):
    return json.dumps({"approved": False, "flags": flags, "confidence": confidence})


SPAWNS = [
    SpawnInstruction(task="Write intro section", runtime="claude_code", budget_tokens=3000),
    SpawnInstruction(task="Write body section", runtime="claude_code", budget_tokens=5000),
]
GOAL = "Write a 2-part article on climate change"


class TestJudgeSpawnPlan:
    def test_approved_response_returns_approved_true(self):
        with patch("router.providers.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider([_approved()])
            result = job.judge_spawn_plan(GOAL, SPAWNS, {})
        assert result["approved"] is True
        assert result["flags"] == []
        assert result["confidence"] == 0.95

    def test_rejected_response_returns_approved_false_with_flags(self):
        flags = ["Sub-tasks share a database dependency — not safely parallel"]
        with patch("router.providers.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider([_rejected(flags)])
            result = job.judge_spawn_plan(GOAL, SPAWNS, {})
        assert result["approved"] is False
        assert len(result["flags"]) == 1
        assert "parallel" in result["flags"][0]

    def test_bad_json_defaults_to_approved_true(self):
        with patch("router.providers.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider(["not json at all"])
            result = job.judge_spawn_plan(GOAL, SPAWNS, {})
        assert result["approved"] is True
        assert result["flags"] == []

    def test_provider_error_defaults_to_approved_true(self):
        provider = MagicMock()
        provider.call_raw.side_effect = RuntimeError("network timeout")
        with patch("router.providers.OpenRouterProvider") as MockProv:
            MockProv.return_value = provider
            result = job.judge_spawn_plan(GOAL, SPAWNS, {})
        assert result["approved"] is True

    def test_missing_provider_key_defaults_to_approved_true(self):
        with patch("router.providers.OpenRouterProvider") as MockProv:
            MockProv.side_effect = ValueError("No OpenRouter API key")
            result = job.judge_spawn_plan(GOAL, SPAWNS, {})
        assert result["approved"] is True

    def test_fenced_json_is_stripped_correctly(self):
        fenced = "```json\n" + json.dumps({"approved": True, "flags": [], "confidence": 0.8}) + "\n```"
        with patch("router.providers.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider([fenced])
            result = job.judge_spawn_plan(GOAL, SPAWNS, {})
        assert result["approved"] is True
        assert result["confidence"] == 0.8

    def test_empty_spawn_plan_does_not_crash(self):
        with patch("router.providers.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider([_approved()])
            result = job.judge_spawn_plan(GOAL, [], {})
        assert result["approved"] is True
