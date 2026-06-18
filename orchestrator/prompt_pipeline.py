from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from router.providers import OpenRouterProvider  # noqa: E402

_ARCHITECT_MODEL = "anthropic/claude-haiku-4-5"
_REVIEWER_MODEL = "anthropic/claude-sonnet-4-6"

_ARCHITECT_SYSTEM = (
    "You are a prompt engineer. Given a raw user ask and optional context, "
    "rewrite it into a precise, unambiguous prompt. "
    "Output raw JSON only (no markdown fences) with exactly two keys: "
    '"optimized_prompt" (string) and "rationale" (string).'
)

_REVIEWER_SYSTEM = (
    "You are a prompt quality reviewer. Compare an original ask against an optimized prompt. "
    "Check for scope drift, dropped constraints, and changed audience. "
    "Output raw JSON only (no markdown fences) with exactly four keys: "
    '"approved" (bool), "quality_score" (integer 1-10), '
    '"drift_flags" (array of strings), "revised_prompt" (string).'
)


class PipelineError(Exception):
    pass


@dataclass
class PipelineResult:
    original_ask: str
    optimized_prompt: str
    quality_score: int
    drift_flags: list[str] = field(default_factory=list)
    approved: bool = False
    architect_rationale: str = ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()
    return text


def _parse_json(raw: str) -> dict:
    return json.loads(_strip_fences(raw))


def run_prompt_pipeline(
    raw_ask: str,
    *,
    memory_context: str = "",
    system_rules: str = "",
    auto_mode: bool = True,
    quality_threshold: int = 7,
) -> PipelineResult:
    provider = OpenRouterProvider()

    # Stage 1: Architect
    architect_parts = ["Raw ask:", raw_ask]
    if memory_context:
        architect_parts += ["Memory context:", memory_context]
    if system_rules:
        architect_parts += ["System rules:", system_rules]
    architect_prompt = "\n\n".join(architect_parts)

    try:
        architect_raw, _ = provider.call_raw(
            _ARCHITECT_MODEL,
            architect_prompt,
            system_prompt=_ARCHITECT_SYSTEM,
            max_tokens=1024,
        )
        architect_data = _parse_json(architect_raw)
        optimized_prompt: str = architect_data["optimized_prompt"]
        architect_rationale: str = architect_data["rationale"]
    except Exception as exc:
        raise PipelineError(f"Stage 1 (Architect) failed: {exc}") from exc

    # Stage 2: Reviewer
    reviewer_prompt = (
        f"Original ask:\n{raw_ask}\n\n"
        f"Optimized prompt:\n{optimized_prompt}"
    )

    try:
        reviewer_raw, _ = provider.call_raw(
            _REVIEWER_MODEL,
            reviewer_prompt,
            system_prompt=_REVIEWER_SYSTEM,
            max_tokens=1024,
        )
        reviewer_data = _parse_json(reviewer_raw)
        reviewer_approved: bool = bool(reviewer_data["approved"])
        quality_score: int = int(reviewer_data["quality_score"])
        drift_flags: list[str] = list(reviewer_data.get("drift_flags", []))
        revised_prompt: str = reviewer_data.get("revised_prompt", optimized_prompt)
    except Exception:
        return PipelineResult(
            original_ask=raw_ask,
            optimized_prompt=raw_ask,
            quality_score=0,
            drift_flags=[],
            approved=False,
            architect_rationale=architect_rationale,
        )

    # Stage 3: Gate (no LLM call)
    if auto_mode:
        if quality_score >= quality_threshold and reviewer_approved:
            final_prompt = optimized_prompt
            approved = True
        elif quality_score >= quality_threshold and not reviewer_approved:
            final_prompt = revised_prompt
            approved = True
        else:
            final_prompt = revised_prompt
            approved = True
    else:
        final_prompt = optimized_prompt if reviewer_approved else revised_prompt
        approved = False

    return PipelineResult(
        original_ask=raw_ask,
        optimized_prompt=final_prompt,
        quality_score=quality_score,
        drift_flags=drift_flags,
        approved=approved,
        architect_rationale=architect_rationale,
    )
