"""Tests for the benchmark runner and evaluator (Phase 2)."""

import json

import pytest

from benchmarks.runner import load_prompts, get_models_for_task
from benchmarks.evaluator import _parse_eval, load_prompts_map


class TestBenchmarkRunner:
    def test_load_all_prompts(self):
        prompts = load_prompts()
        assert len(prompts) >= 15
        for p in prompts:
            assert "id" in p
            assert "text" in p
            assert "expected_type" in p
            assert "difficulty" in p
            assert "eval_criteria" in p

    def test_load_specific_prompt(self):
        prompts = load_prompts("code-001")
        assert len(prompts) == 1
        assert prompts[0]["id"] == "code-001"
        assert prompts[0]["expected_type"] == "code"

    def test_load_nonexistent_prompt_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_prompts("nonexistent-999")

    def test_get_models_for_task(self):
        models = get_models_for_task("code")
        assert len(models) > 0

    def test_get_models_top_n(self):
        models = get_models_for_task("writing", top_n=2)
        assert len(models) == 2

    def test_all_prompt_types_are_valid(self):
        """Every prompt's expected_type should map to a valid task type with models."""
        prompts = load_prompts()
        for p in prompts:
            models = get_models_for_task(p["expected_type"])
            assert len(models) > 0, f"No models for task type: {p['expected_type']}"


class TestEvaluator:
    def test_parse_valid_eval(self):
        score, reasoning = _parse_eval(
            '{"score": 8, "reasoning": "Good response with clear structure"}'
        )
        assert score == 8.0
        assert "clear structure" in reasoning

    def test_parse_markdown_fenced(self):
        score, reasoning = _parse_eval(
            '```json\n{"score": 9, "reasoning": "Excellent"}\n```'
        )
        assert score == 9.0

    def test_parse_invalid_json_defaults(self):
        score, reasoning = _parse_eval("Not valid JSON")
        assert score == 5.0
        assert "Failed to parse" in reasoning

    def test_score_clamped_to_range(self):
        score, _ = _parse_eval('{"score": 15, "reasoning": "Off the charts"}')
        assert score == 10.0

        score, _ = _parse_eval('{"score": -3, "reasoning": "Terrible"}')
        assert score == 1.0

    def test_load_prompts_map(self):
        pmap = load_prompts_map()
        assert "code-001" in pmap
        assert "writing-001" in pmap
        assert pmap["code-001"]["expected_type"] == "code"
