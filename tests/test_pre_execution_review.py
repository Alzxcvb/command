"""Tests for the pre-execution decomposition review gate (Phase 8)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from orchestrator.pre_execution_review import PreReviewResult, run_pre_execution_review


def _make_provider(responses: list[str]):
    provider = MagicMock()
    provider.call_raw.side_effect = [(r, 10.0) for r in responses]
    return provider


def _judge_ok(concerns: list = None, recommendation: str = "Looks good.") -> str:
    return json.dumps({
        "proceed": True,
        "concerns": concerns or [],
        "recommendation": recommendation,
    })


def _judge_flag(concerns: list, recommendation: str = "Fix this.") -> str:
    return json.dumps({
        "proceed": False,
        "concerns": concerns,
        "recommendation": recommendation,
    })


class TestRunPreExecutionReview:
    GOAL = "Write a 3-part blog post on polycrisis"
    BREAKDOWN = "Task 1: intro\nTask 2: body\nTask 3: conclusion"

    def test_both_judges_clear_returns_proceed_true(self):
        responses = [_judge_ok(), _judge_ok()]
        with patch("orchestrator.pre_execution_review.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider(responses)
            result = run_pre_execution_review(self.GOAL, self.BREAKDOWN)

        assert result.proceed is True
        assert result.concerns == []

    def test_one_judge_flags_returns_proceed_false(self):
        responses = [
            _judge_ok(),
            _judge_flag(["Goal clause 2 not covered by any subtask"]),
        ]
        with patch("orchestrator.pre_execution_review.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider(responses)
            result = run_pre_execution_review(self.GOAL, self.BREAKDOWN)

        assert result.proceed is False
        assert len(result.concerns) == 1
        assert "Goal clause 2" in result.concerns[0]

    def test_both_judges_flag_merges_concerns(self):
        responses = [
            _judge_flag(["Missing citations task"], "Add a citations step."),
            _judge_flag(["Budget too low for task 3"], "Increase budget."),
        ]
        with patch("orchestrator.pre_execution_review.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider(responses)
            result = run_pre_execution_review(self.GOAL, self.BREAKDOWN)

        assert result.proceed is False
        concern_text = " ".join(result.concerns)
        assert "citations" in concern_text
        assert "Budget" in concern_text

    def test_recommendation_joins_both_judges(self):
        responses = [_judge_ok(recommendation="All good."), _judge_ok(recommendation="LGTM.")]
        with patch("orchestrator.pre_execution_review.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider(responses)
            result = run_pre_execution_review(self.GOAL, self.BREAKDOWN)

        assert "All good." in result.recommendation
        assert "LGTM." in result.recommendation

    def test_judge_error_defaults_to_proceed_true(self):
        """A failed judge call should not block the job."""
        provider = MagicMock()
        provider.call_raw.side_effect = RuntimeError("timeout")
        with patch("orchestrator.pre_execution_review.OpenRouterProvider") as MockProv:
            MockProv.return_value = provider
            result = run_pre_execution_review(self.GOAL, self.BREAKDOWN)

        assert result.proceed is True  # errors default to proceed=True per _call_review_judge

    def test_verdict_dicts_stored_on_result(self):
        responses = [_judge_ok(), _judge_flag(["concern"])]
        with patch("orchestrator.pre_execution_review.OpenRouterProvider") as MockProv:
            MockProv.return_value = _make_provider(responses)
            result = run_pre_execution_review(self.GOAL, self.BREAKDOWN)

        assert result.judge_1_verdict.get("proceed") is True
        assert result.judge_2_verdict.get("proceed") is False
