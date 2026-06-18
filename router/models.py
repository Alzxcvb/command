"""Model registry — capabilities, costs, and scores per task type."""

from __future__ import annotations

from .types import ModelInfo, TaskType

# Model registry with scores based on published benchmarks + community consensus.
# Scores are 0-10 where 10 = best-in-class for that task type.
# Costs are USD per 1M tokens (as of June 2026 via OpenRouter).
# Verify exact slugs and pricing at openrouter.ai/models before changing providers.

MODELS: dict[str, ModelInfo] = {
    # --- Anthropic (current Claude 4.x / Fable 5 family) ---
    "anthropic/claude-fable-5": ModelInfo(
        id="anthropic/claude-fable-5",
        name="Claude Fable 5",
        provider="anthropic",
        scores={
            TaskType.CODE: 9.5,
            TaskType.WRITING: 10.0,
            TaskType.REASONING: 10.0,
            TaskType.SUMMARIZATION: 10.0,
            TaskType.CONVERSATION: 10.0,
            TaskType.RESEARCH: 10.0,
            TaskType.TRANSLATION: 9.0,
            TaskType.DATA: 9.5,
        },
        cost_per_million_input=15.0,   # verify at openrouter.ai/models
        cost_per_million_output=75.0,
        max_context=200_000,
        supports_images=True,
    ),
    "anthropic/claude-opus-4-8": ModelInfo(
        id="anthropic/claude-opus-4-8",
        name="Claude Opus 4.8",
        provider="anthropic",
        scores={
            TaskType.CODE: 9.5,
            TaskType.WRITING: 9.5,
            TaskType.REASONING: 10.0,
            TaskType.SUMMARIZATION: 9.5,
            TaskType.CONVERSATION: 9.5,
            TaskType.RESEARCH: 9.5,
            TaskType.TRANSLATION: 9.0,
            TaskType.DATA: 9.5,
        },
        cost_per_million_input=15.0,   # verify at openrouter.ai/models
        cost_per_million_output=75.0,
        max_context=200_000,
        supports_images=True,
    ),
    "anthropic/claude-sonnet-4-6": ModelInfo(
        id="anthropic/claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        scores={
            TaskType.CODE: 9.0,
            TaskType.WRITING: 9.5,
            TaskType.REASONING: 9.0,
            TaskType.SUMMARIZATION: 9.0,
            TaskType.CONVERSATION: 9.0,
            TaskType.RESEARCH: 9.0,
            TaskType.TRANSLATION: 8.5,
            TaskType.DATA: 8.5,
        },
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
        max_context=200_000,
        supports_images=True,
    ),
    "anthropic/claude-haiku-4-5": ModelInfo(
        id="anthropic/claude-haiku-4-5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        scores={
            TaskType.CODE: 7.5,
            TaskType.WRITING: 7.5,
            TaskType.REASONING: 7.5,
            TaskType.SUMMARIZATION: 8.0,
            TaskType.CONVERSATION: 8.0,
            TaskType.RESEARCH: 7.5,
            TaskType.TRANSLATION: 7.5,
            TaskType.DATA: 7.5,
        },
        cost_per_million_input=0.80,
        cost_per_million_output=4.0,
        max_context=200_000,
        supports_images=True,
    ),
    "openai/gpt-4o": ModelInfo(
        id="openai/gpt-4o",
        name="GPT-4o",
        provider="openai",
        scores={
            TaskType.CODE: 9.0,
            TaskType.WRITING: 8.0,
            TaskType.REASONING: 9.0,
            TaskType.SUMMARIZATION: 8.0,
            TaskType.CONVERSATION: 8.5,
            TaskType.RESEARCH: 8.5,
            TaskType.TRANSLATION: 9.0,
            TaskType.DATA: 9.0,
        },
        cost_per_million_input=2.5,
        cost_per_million_output=10.0,
        max_context=128_000,
        supports_images=True,
    ),
    "google/gemini-2.0-flash-001": ModelInfo(
        id="google/gemini-2.0-flash-001",
        name="Gemini 2.0 Flash",
        provider="google",
        scores={
            TaskType.CODE: 7.0,
            TaskType.WRITING: 7.0,
            TaskType.REASONING: 7.0,
            TaskType.SUMMARIZATION: 8.0,
            TaskType.CONVERSATION: 7.5,
            TaskType.RESEARCH: 7.5,
            TaskType.TRANSLATION: 7.5,
            TaskType.DATA: 7.5,
        },
        cost_per_million_input=0.1,
        cost_per_million_output=0.4,
        max_context=1_000_000,
        supports_images=True,
    ),
    "deepseek/deepseek-chat-v3": ModelInfo(
        id="deepseek/deepseek-chat-v3",
        name="DeepSeek V3",
        provider="deepseek",
        scores={
            TaskType.CODE: 9.0,
            TaskType.WRITING: 6.0,
            TaskType.REASONING: 8.5,
            TaskType.SUMMARIZATION: 7.0,
            TaskType.CONVERSATION: 6.5,
            TaskType.RESEARCH: 7.0,
            TaskType.TRANSLATION: 7.0,
            TaskType.DATA: 8.5,
        },
        cost_per_million_input=0.27,
        cost_per_million_output=1.10,
        max_context=128_000,
        supports_images=False,
    ),
    "meta-llama/llama-3.3-70b-instruct": ModelInfo(
        id="meta-llama/llama-3.3-70b-instruct",
        name="Llama 3.3 70B",
        provider="meta",
        scores={
            TaskType.CODE: 7.5,
            TaskType.WRITING: 7.0,
            TaskType.REASONING: 7.5,
            TaskType.SUMMARIZATION: 7.5,
            TaskType.CONVERSATION: 7.0,
            TaskType.RESEARCH: 7.0,
            TaskType.TRANSLATION: 7.0,
            TaskType.DATA: 7.0,
        },
        cost_per_million_input=0.40,
        cost_per_million_output=0.40,
        max_context=128_000,
        supports_images=False,
    ),
    "qwen/qwen-2.5-72b-instruct": ModelInfo(
        id="qwen/qwen-2.5-72b-instruct",
        name="Qwen 2.5 72B",
        provider="qwen",
        scores={
            TaskType.CODE: 8.0,
            TaskType.WRITING: 7.0,
            TaskType.REASONING: 8.0,
            TaskType.SUMMARIZATION: 7.5,
            TaskType.CONVERSATION: 7.0,
            TaskType.RESEARCH: 7.0,
            TaskType.TRANSLATION: 8.5,
            TaskType.DATA: 8.0,
        },
        cost_per_million_input=0.35,
        cost_per_million_output=0.40,
        max_context=128_000,
        supports_images=False,
    ),
    # Kimi K2 — strong coder at very low cost; verify slug at openrouter.ai/models
    "moonshotai/kimi-k2": ModelInfo(
        id="moonshotai/kimi-k2",
        name="Kimi K2",
        provider="moonshot",
        scores={
            TaskType.CODE: 9.0,
            TaskType.WRITING: 7.5,
            TaskType.REASONING: 8.5,
            TaskType.SUMMARIZATION: 7.5,
            TaskType.CONVERSATION: 7.5,
            TaskType.RESEARCH: 7.5,
            TaskType.TRANSLATION: 7.5,
            TaskType.DATA: 8.5,
        },
        cost_per_million_input=0.15,   # verify at openrouter.ai/models
        cost_per_million_output=2.50,
        max_context=128_000,
        supports_images=False,
    ),
    # DeepSeek R1 — strong reasoning at very low cost
    "deepseek/deepseek-r1": ModelInfo(
        id="deepseek/deepseek-r1",
        name="DeepSeek R1",
        provider="deepseek",
        scores={
            TaskType.CODE: 9.0,
            TaskType.WRITING: 7.0,
            TaskType.REASONING: 9.5,
            TaskType.SUMMARIZATION: 7.5,
            TaskType.CONVERSATION: 7.0,
            TaskType.RESEARCH: 8.5,
            TaskType.TRANSLATION: 7.0,
            TaskType.DATA: 9.0,
        },
        cost_per_million_input=0.55,   # verify at openrouter.ai/models
        cost_per_million_output=2.19,
        max_context=128_000,
        supports_images=False,
    ),
}


def get_model(model_id: str) -> ModelInfo | None:
    """Look up a model by ID."""
    return MODELS.get(model_id)


def get_best_model_for_task(
    task_type: TaskType,
    *,
    budget: str = "best",
) -> ModelInfo:
    """Return the best model for a given task type.

    Args:
        task_type: The classified task type.
        budget: "best" = highest score regardless of cost,
                "balanced" = good score with reasonable cost,
                "cheap" = prioritize low cost.
    """
    task_key = task_type.value if isinstance(task_type, TaskType) else task_type

    if budget == "cheap":
        # Pick the cheapest model with a score >= 7
        candidates = [
            m for m in MODELS.values()
            if m.scores.get(task_key, m.scores.get(task_type, 0)) >= 7.0
        ]
        if candidates:
            return min(candidates, key=lambda m: m.cost_per_million_input)

    if budget == "balanced":
        # Score-to-cost ratio
        def ratio(m: ModelInfo) -> float:
            score = m.scores.get(task_key, m.scores.get(task_type, 0))
            cost = m.cost_per_million_input + m.cost_per_million_output
            return score / max(cost, 0.01)

        return max(MODELS.values(), key=ratio)

    # "best" — highest raw score
    return max(
        MODELS.values(),
        key=lambda m: m.scores.get(task_key, m.scores.get(task_type, 0)),
    )


def get_ranked_models(task_type: TaskType) -> list[tuple[ModelInfo, float]]:
    """Return all models ranked by score for a task type, descending."""
    task_key = task_type.value if isinstance(task_type, TaskType) else task_type
    ranked = [
        (m, m.scores.get(task_key, m.scores.get(task_type, 0)))
        for m in MODELS.values()
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
