"""Core routing logic — classify prompt, select model, call provider."""

from __future__ import annotations

from .classifier import classify
from .models import get_best_model_for_task, get_ranked_models
from .providers import OpenRouterProvider, get_provider_for
from .types import Complexity, RouterResponse, RoutingDecision


class Router:
    """Routes prompts to the best model for each task."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        budget: str = "best",
        classifier_method: str = "rules",
    ):
        self.provider = OpenRouterProvider(api_key=api_key)
        self.budget = budget
        self.classifier_method = classifier_method

    def route(self, prompt: str, *, dry_run: bool = False) -> RouterResponse | RoutingDecision:
        """Classify a prompt, pick the best model, and (optionally) call it.

        Args:
            prompt: The user's input prompt.
            dry_run: If True, return only the routing decision without calling the model.

        Returns:
            RouterResponse with the model's output and metadata,
            or RoutingDecision if dry_run=True.
        """
        # Step 1: Classify the prompt
        classification = classify(
            prompt,
            method=self.classifier_method,
            provider=self.provider if self.classifier_method == "llm" else None,
        )

        # Step 2: Select budget — downgrade for low-complexity prompts to save cost
        effective_budget = self.budget
        if self.budget == "best" and classification.complexity == Complexity.LOW:
            effective_budget = "balanced"

        # Step 3: Select the best model
        model = get_best_model_for_task(classification.task_type, budget=effective_budget)
        ranked = get_ranked_models(classification.task_type)
        score = model.scores.get(
            classification.task_type.value,
            model.scores.get(classification.task_type, 0),
        )

        # Build routing decision
        alternatives = [(m, s) for m, s in ranked if m.id != model.id][:3]

        reasoning_parts = [
            f"Classified as '{classification.task_type.value}' "
            f"(confidence: {classification.confidence}, method: {classification.method})",
        ]
        if classification.method == "llm":
            reasoning_parts.append(
                f"complexity: {classification.complexity.value}, "
                f"needs_reasoning: {classification.needs_reasoning}, "
                f"needs_creativity: {classification.needs_creativity}"
            )
        else:
            reasoning_parts.append(f"keywords: {classification.keywords_matched}")

        if effective_budget != self.budget:
            reasoning_parts.append(
                f"budget downgraded {self.budget} → {effective_budget} (low complexity)"
            )

        reasoning_parts.append(
            f"Selected {model.name} with score {score}/10 (budget: {effective_budget})"
        )

        decision = RoutingDecision(
            model=model,
            task_type=classification.task_type,
            score=score,
            reasoning=". ".join(reasoning_parts) + ".",
            alternatives=alternatives,
        )

        if dry_run:
            return decision

        # Step 4: Call the model via the best provider (direct if keyed, OpenRouter fallback)
        inference_provider = get_provider_for(model)
        content, latency_ms = inference_provider.call(model, prompt)

        # Estimate cost (rough: assume ~prompt_len/4 input tokens, ~response_len/4 output tokens)
        est_input_tokens = len(prompt) / 4
        est_output_tokens = len(content) / 4
        estimated_cost = (
            (est_input_tokens / 1_000_000) * model.cost_per_million_input
            + (est_output_tokens / 1_000_000) * model.cost_per_million_output
        )

        return RouterResponse(
            content=content,
            decision=decision,
            latency_ms=round(latency_ms, 1),
            estimated_cost=round(estimated_cost, 6),
        )
