"""Tests for the AI Model Router classification and routing logic."""

import pytest

from router.classifier import classify
from router.models import get_best_model_for_task, get_ranked_models, MODELS
from router.rules import classify_by_rules
from router.types import TaskType


# --- Classification tests ---

class TestRulesClassifier:
    def test_code_prompt(self):
        result = classify_by_rules("Write a Python function to sort a list")
        assert result.task_type == TaskType.CODE
        assert result.confidence > 0.4

    def test_writing_prompt(self):
        result = classify_by_rules("Write me a persuasive essay on climate change")
        assert result.task_type == TaskType.WRITING

    def test_reasoning_prompt(self):
        result = classify_by_rules("Calculate the probability of rolling two sixes")
        assert result.task_type == TaskType.REASONING

    def test_summarization_prompt(self):
        result = classify_by_rules("Summarize the key points of this article")
        assert result.task_type == TaskType.SUMMARIZATION

    def test_translation_prompt(self):
        result = classify_by_rules("Translate this paragraph to Spanish")
        assert result.task_type == TaskType.TRANSLATION

    def test_data_prompt(self):
        result = classify_by_rules("Parse this CSV and create a chart of sales by month")
        assert result.task_type == TaskType.DATA

    def test_research_prompt(self):
        result = classify_by_rules("What is quantum computing and compare the pros and cons")
        assert result.task_type == TaskType.RESEARCH

    def test_fallback_to_conversation(self):
        result = classify_by_rules("Hello there!")
        assert result.task_type == TaskType.CONVERSATION
        assert result.confidence < 0.5

    def test_multiple_keywords_boost_confidence(self):
        result = classify_by_rules(
            "Write a Python function to implement a sorting algorithm and debug the runtime error"
        )
        assert result.task_type == TaskType.CODE
        assert result.confidence > 0.5

    def test_keywords_matched_populated(self):
        result = classify_by_rules("Debug this Python function")
        assert result.task_type == TaskType.CODE
        assert len(result.keywords_matched) > 0
        assert "debug" in result.keywords_matched or "python" in result.keywords_matched


class TestClassifier:
    def test_delegates_to_rules(self):
        """classify() should use rules engine in v0.1."""
        result = classify("Write a unit test for this class")
        assert result.task_type == TaskType.CODE


# --- Model selection tests ---

class TestModelSelection:
    def test_best_budget_returns_highest_score(self):
        model = get_best_model_for_task(TaskType.WRITING, budget="best")
        # Best writing model should score 10
        assert model.scores.get(TaskType.WRITING.value, 0) == 10.0

    def test_cheap_budget_returns_affordable_model(self):
        model = get_best_model_for_task(TaskType.CODE, budget="cheap")
        # Should pick a cheap model that still scores >= 7
        assert model.cost_per_million_input < 1.0

    def test_balanced_budget(self):
        model = get_best_model_for_task(TaskType.CODE, budget="balanced")
        # Should return something — exact model depends on score/cost ratio
        assert model is not None
        assert model.id in MODELS

    def test_ranked_models_descending(self):
        ranked = get_ranked_models(TaskType.CODE)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_all_models_have_all_task_scores(self):
        for model_id, model in MODELS.items():
            for task_type in TaskType:
                assert task_type.value in model.scores or task_type in model.scores, (
                    f"{model_id} missing score for {task_type}"
                )


# --- Integration tests (no API call) ---

class TestRouterDryRun:
    def test_code_routes_to_strong_code_model(self):
        """End-to-end: code prompt → code classification → code-strong model."""
        classification = classify("Write a Python quicksort implementation")
        assert classification.task_type == TaskType.CODE
        model = get_best_model_for_task(classification.task_type)
        assert model.scores.get(TaskType.CODE.value, 0) >= 9.0

    def test_writing_routes_to_strong_writing_model(self):
        classification = classify("Write a persuasive essay on climate change")
        assert classification.task_type == TaskType.WRITING
        model = get_best_model_for_task(classification.task_type)
        assert model.scores.get(TaskType.WRITING.value, 0) >= 9.0

    def test_math_routes_to_reasoning_model(self):
        classification = classify("What is 15% of 847? Calculate step by step")
        assert classification.task_type == TaskType.REASONING
        model = get_best_model_for_task(classification.task_type)
        assert model.scores.get(TaskType.REASONING.value, 0) >= 8.0
