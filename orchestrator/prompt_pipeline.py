"""Three-stage prompt optimization pipeline (Phase 7/8).

Stage 1 — Architect (Haiku): rewrites the raw ask into a precise prompt.
Stage 2 — Council (Codex + DeepSeek R1): two non-Claude judges review in parallel.
           Neither judge is from the same provider as the architect, preventing
           self-grading bias.
Stage 3 — Gate: majority vote (1/2 or 2/2) determines final prompt and approval.

Council majority rules:
  2/2 approve  → approved=True, use optimized_prompt
  1/2 approve  → approved=True, use the rejector's revised_prompt (stricter edit wins)
  0/2 approve  → approved=False, use whichever judge scored higher
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

_ARCHITECT_MODEL = "anthropic/claude-haiku-4-5"
_JUDGE_1_MODEL = "openai/codex-mini-latest"   # non-Claude judge 1
_JUDGE_2_MODEL = "deepseek/deepseek-r1"        # non-Claude judge 2


def _load_rubric() -> str:
    p = Path(__file__).parent / "prompts" / "judge_rubric.md"
    try:
        return p.read_text().strip()
    except Exception:
        return ""


_JUDGE_RUBRIC = _load_rubric()

_ARCHITECT_SYSTEM = (
    "You are a prompt engineer. Given a raw user ask and optional context, "
    "rewrite it into a precise, unambiguous prompt. "
    "Output raw JSON only (no markdown fences) with exactly two keys: "
    '"optimized_prompt" (string) and "rationale" (string).'
)

_REVIEWER_SYSTEM = (
    "You are a prompt quality reviewer. Compare an original ask against an optimized prompt. "
    "Check for scope drift, dropped constraints, and changed audience.\n\n"
    "Apply these evaluation criteria:\n"
    f"{_JUDGE_RUBRIC}\n\n"
    "Output raw JSON only (no markdown fences) with exactly four keys: "
    '"approved" (bool), "quality_score" (integer 1-10), '
    '"drift_flags" (array of strings, one per criterion number that failed, e.g. ["1","7"]), '
    '"revised_prompt" (string, your improved version).'
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
    judges_approved: int = 0          # 0, 1, or 2
    council_verdicts: list[dict] = field(default_factory=list)


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


def _strip_think(text: str) -> str:
    """Remove DeepSeek R1 chain-of-thought <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_json(raw: str) -> dict:
    import json
    return json.loads(_strip_fences(_strip_think(raw)))


def _call_judge(
    provider: OpenRouterProvider,
    model_id: str,
    original_ask: str,
    optimized_prompt: str,
) -> dict:
    """Call one judge model. Returns parsed dict or empty dict on failure."""
    prompt = f"Original ask:\n{original_ask}\n\nOptimized prompt:\n{optimized_prompt}"
    try:
        raw, _ = provider.call_raw(
            model_id,
            prompt,
            system_prompt=_REVIEWER_SYSTEM,
            max_tokens=1024,
        )
        return _parse_json(raw)
    except Exception as exc:
        return {"_error": str(exc), "approved": False, "quality_score": 0,
                "drift_flags": [], "revised_prompt": optimized_prompt}


def run_prompt_pipeline(
    raw_ask: str,
    *,
    memory_context: str = "",
    system_rules: str = "",
    auto_mode: bool = True,
    quality_threshold: int = 7,
) -> PipelineResult:
    """Run the 3-stage pipeline: Architect → Council (Codex + DeepSeek R1) → Gate."""
    provider = OpenRouterProvider()

    # Stage 1: Architect (Haiku)
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

    # Stage 2: Council — Codex then DeepSeek R1
    j1 = _call_judge(provider, _JUDGE_1_MODEL, raw_ask, optimized_prompt)
    j2 = _call_judge(provider, _JUDGE_2_MODEL, raw_ask, optimized_prompt)

    j1_approved = bool(j1.get("approved", False)) and "_error" not in j1
    j2_approved = bool(j2.get("approved", False)) and "_error" not in j2
    j1_score = int(j1.get("quality_score", 0))
    j2_score = int(j2.get("quality_score", 0))
    j1_flags: list[str] = j1.get("drift_flags") or []
    j2_flags: list[str] = j2.get("drift_flags") or []
    j1_revised: str = j1.get("revised_prompt") or optimized_prompt
    j2_revised: str = j2.get("revised_prompt") or optimized_prompt

    judges_approved = sum(1 for ok in [j1_approved, j2_approved] if ok)
    all_flags = sorted(set(j1_flags + j2_flags))
    quality_score = min(j1_score, j2_score) if j1_score and j2_score else max(j1_score, j2_score)

    council_verdicts = [
        {"model": _JUDGE_1_MODEL, **{k: v for k, v in j1.items() if not k.startswith("_")}},
        {"model": _JUDGE_2_MODEL, **{k: v for k, v in j2.items() if not k.startswith("_")}},
    ]

    # Stage 3: Gate — choose final prompt by majority
    if judges_approved == 2:
        final_prompt = optimized_prompt
        approved = True
    elif j1_approved and not j2_approved:
        # Stricter judge (j2) rejected — take its revision
        final_prompt = j2_revised
        approved = True
    elif j2_approved and not j1_approved:
        # Stricter judge (j1) rejected — take its revision
        final_prompt = j1_revised
        approved = True
    else:
        # 0/2: both rejected — take revision from whichever scored higher
        final_prompt = j1_revised if j1_score >= j2_score else j2_revised
        approved = False

    if not auto_mode:
        approved = False

    return PipelineResult(
        original_ask=raw_ask,
        optimized_prompt=final_prompt,
        quality_score=quality_score,
        drift_flags=all_flags,
        approved=approved,
        architect_rationale=architect_rationale,
        judges_approved=judges_approved,
        council_verdicts=council_verdicts,
    )
