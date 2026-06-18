"""Tests for the dual-council prompt pipeline (Phase 8)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.prompt_pipeline import (
    PipelineResult,
    _call_judge,
    _strip_think,
    run_prompt_pipeline,
)


class TestStripThink:
    def test_removes_think_block(self):
        raw = "<think>\nsome reasoning\n</think>\n{\"approved\": true}"
        assert _strip_think(raw) == '{"approved": true}'

    def test_passthrough_when_no_think(self):
        raw = '{"approved": true, "quality_score": 8}'
        assert _strip_think(raw) == raw

    def test_multiline_think(self):
        raw = "<think>\nline1\nline2\n</think>\nclean output"
        assert _strip_think(raw) == "clean output"


class TestCallJudge:
    def _make_provider(self, response: str):
        p = MagicMock()
        p.call_raw.return_value = (response, 50.0)
        return p

    def test_returns_parsed_dict(self):
        payload = json.dumps({
            "approved": True,
            "quality_score": 8,
            "drift_flags": [],
            "revised_prompt": "refined ask",
        })
        provider = self._make_provider(payload)
        result = _call_judge(provider, "some-model", "raw ask", "optimized ask")
        assert result["approved"] is True
        assert result["quality_score"] == 8

    def test_handles_provider_error(self):
        provider = MagicMock()
        provider.call_raw.side_effect = RuntimeError("network error")
        result = _call_judge(provider, "some-model", "raw ask", "optimized ask")
        assert result["approved"] is False
        assert "_error" in result

    def test_strips_think_block_before_parse(self):
        payload = '<think>reasoning here</think>\n' + json.dumps({
            "approved": True, "quality_score": 9, "drift_flags": [], "revised_prompt": "p",
        })
        provider = self._make_provider(payload)
        result = _call_judge(provider, "deepseek/deepseek-r1", "ask", "opt")
        assert result["quality_score"] == 9


class TestRunPromptPipeline:
    def _architect_response(self) -> str:
        return json.dumps({
            "optimized_prompt": "Optimized version of the ask",
            "rationale": "Added specificity",
        })

    def _judge_response(self, approved: bool, score: int, flags: list = None) -> str:
        return json.dumps({
            "approved": approved,
            "quality_score": score,
            "drift_flags": flags or [],
            "revised_prompt": "Revised prompt",
        })

    def _make_provider(self, responses: list[str]):
        provider = MagicMock()
        provider.call_raw.side_effect = [(r, 10.0) for r in responses]
        return provider

    def test_both_judges_approve(self):
        responses = [
            self._architect_response(),
            self._judge_response(True, 9),   # Codex
            self._judge_response(True, 8),   # DeepSeek R1
        ]
        with patch("orchestrator.prompt_pipeline.OpenRouterProvider") as MockProv:
            MockProv.return_value = self._make_provider(responses)
            result = run_prompt_pipeline("raw ask")

        assert result.approved is True
        assert result.judges_approved == 2
        assert result.optimized_prompt == "Optimized version of the ask"
        assert result.quality_score == 8  # min(9, 8)

    def test_one_judge_rejects_uses_revision(self):
        responses = [
            self._architect_response(),
            self._judge_response(True, 8),                    # Codex approves
            self._judge_response(False, 5, flags=["1", "7"]), # DeepSeek R1 rejects
        ]
        with patch("orchestrator.prompt_pipeline.OpenRouterProvider") as MockProv:
            MockProv.return_value = self._make_provider(responses)
            result = run_prompt_pipeline("raw ask")

        assert result.approved is True          # 1/2 still approves
        assert result.judges_approved == 1
        assert result.optimized_prompt == "Revised prompt"  # stricter judge's revision
        assert "1" in result.drift_flags
        assert "7" in result.drift_flags

    def test_both_judges_reject(self):
        responses = [
            self._architect_response(),
            self._judge_response(False, 4),  # Codex rejects
            self._judge_response(False, 3),  # DeepSeek R1 rejects
        ]
        with patch("orchestrator.prompt_pipeline.OpenRouterProvider") as MockProv:
            MockProv.return_value = self._make_provider(responses)
            result = run_prompt_pipeline("raw ask")

        assert result.approved is False
        assert result.judges_approved == 0
        assert result.quality_score == 3  # min(4, 3)

    def test_drift_flags_merged_and_deduplicated(self):
        responses = [
            self._architect_response(),
            self._judge_response(True, 7, flags=["2", "5"]),
            self._judge_response(True, 7, flags=["5", "8"]),
        ]
        with patch("orchestrator.prompt_pipeline.OpenRouterProvider") as MockProv:
            MockProv.return_value = self._make_provider(responses)
            result = run_prompt_pipeline("raw ask")

        assert sorted(result.drift_flags) == ["2", "5", "8"]

    def test_council_verdicts_include_model_names(self):
        responses = [
            self._architect_response(),
            self._judge_response(True, 8),
            self._judge_response(True, 7),
        ]
        with patch("orchestrator.prompt_pipeline.OpenRouterProvider") as MockProv:
            MockProv.return_value = self._make_provider(responses)
            result = run_prompt_pipeline("raw ask")

        models = [v["model"] for v in result.council_verdicts]
        assert any("codex" in m.lower() for m in models)
        assert any("deepseek" in m.lower() for m in models)

    def test_auto_mode_false_returns_unapproved(self):
        responses = [
            self._architect_response(),
            self._judge_response(True, 9),
            self._judge_response(True, 8),
        ]
        with patch("orchestrator.prompt_pipeline.OpenRouterProvider") as MockProv:
            MockProv.return_value = self._make_provider(responses)
            result = run_prompt_pipeline("raw ask", auto_mode=False)

        assert result.approved is False

    def test_architect_failure_raises_pipeline_error(self):
        from orchestrator.prompt_pipeline import PipelineError
        with patch("orchestrator.prompt_pipeline.OpenRouterProvider") as MockProv:
            MockProv.return_value.call_raw.side_effect = RuntimeError("api down")
            with pytest.raises(PipelineError):
                run_prompt_pipeline("raw ask")
