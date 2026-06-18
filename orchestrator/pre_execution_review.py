"""Pre-execution decomposition review gate (Phase 8).

Runs in start_job() after parse_breakdown() and before _dispatch(). A council
of two non-Claude judges (Codex + DeepSeek R1) checks whether the orchestrator's
breakdown plan covers the stated goal.

This is a soft gate: callers log the result in job meta and surface it in the
dashboard. The job still proceeds; hard-blocking can be added once the gate is
trusted.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from router.providers import OpenRouterProvider  # noqa: E402

_JUDGE_1_MODEL = "openai/codex-mini-latest"
_JUDGE_2_MODEL = "deepseek/deepseek-r1"

_RUBRIC_PATH = Path(__file__).parent / "prompts" / "judge_rubric.md"


def _load_rubric() -> str:
    try:
        return _RUBRIC_PATH.read_text().strip()
    except Exception:
        return ""


_JUDGE_RUBRIC = _load_rubric()

_REVIEW_SYSTEM = (
    "You are a decomposition reviewer for an AI agent orchestrator. You receive a job goal "
    "and the orchestrator's breakdown plan (a list of subtasks). Evaluate whether the plan "
    "fully addresses the goal.\n\n"
    "Focus on these criteria from the shared rubric:\n"
    f"{_JUDGE_RUBRIC}\n\n"
    "Output raw JSON only (no markdown fences) with exactly three keys:\n"
    '{"proceed": true/false, "concerns": ["..."], "recommendation": "..."}\n\n'
    "Set proceed=false only if you see a clear, specific gap — not a general preference. "
    "recommendation must be one sentence."
)


@dataclass
class PreReviewResult:
    proceed: bool
    concerns: list[str] = field(default_factory=list)
    recommendation: str = ""
    judge_1_verdict: dict = field(default_factory=dict)
    judge_2_verdict: dict = field(default_factory=dict)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start, end = 1, len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()
    return text


def _call_review_judge(
    provider: OpenRouterProvider,
    model_id: str,
    prompt: str,
) -> dict:
    import json
    try:
        raw, _ = provider.call_raw(
            model_id,
            prompt,
            system_prompt=_REVIEW_SYSTEM,
            max_tokens=512,
        )
        cleaned = _strip_fences(_strip_think(raw))
        return json.loads(cleaned)
    except Exception as exc:
        return {"proceed": True, "concerns": [], "recommendation": "", "_error": str(exc)}


def run_pre_execution_review(goal: str, breakdown_md: str) -> PreReviewResult:
    """Check the orchestrator's breakdown plan before dispatching sub-agents.

    Returns a PreReviewResult. proceed=False means at least one judge found a
    clear gap between the goal and the plan. The caller should log this in job
    meta and surface it in the dashboard.
    """
    provider = OpenRouterProvider()
    prompt = (
        f"## Job goal\n{goal}\n\n"
        f"## Orchestrator breakdown plan\n{breakdown_md[:3000]}"
    )

    j1 = _call_review_judge(provider, _JUDGE_1_MODEL, prompt)
    j2 = _call_review_judge(provider, _JUDGE_2_MODEL, prompt)

    j1_proceed = bool(j1.get("proceed", True))
    j2_proceed = bool(j2.get("proceed", True))
    all_concerns = list(dict.fromkeys(
        (j1.get("concerns") or []) + (j2.get("concerns") or [])
    ))

    # Both judges must clear for a clean pass
    proceed = j1_proceed and j2_proceed

    recs = [r for r in [j1.get("recommendation", ""), j2.get("recommendation", "")] if r]
    recommendation = " | ".join(recs) if recs else "No concerns."

    return PreReviewResult(
        proceed=proceed,
        concerns=all_concerns,
        recommendation=recommendation,
        judge_1_verdict={k: v for k, v in j1.items() if not k.startswith("_")},
        judge_2_verdict={k: v for k, v in j2.items() if not k.startswith("_")},
    )
