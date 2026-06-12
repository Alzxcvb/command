"""Static mid-turn model routing — Phase 3a of tasks/mid-turn-model-routing-and-prompt-flywheel.md.

v1 rule table: route each turn by the tools involved in it. Destructive turns
(code edits, shell) always get opus; planning gets opus; web summarization gets
sonnet; pure lookups get haiku; anything else inherits the previous turn's
model to avoid thrash.

NOT yet wired into a live loop: Command's runtimes spawn the claude CLI as a
subprocess and do not own per-turn API calls, so route_turn() has no caller in
production yet (see docs/BLOCKERS.md). The rule table, ledger logging, and the
dashboard savings widget are ready for when the harness loop lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

OPUS = "opus"
SONNET = "sonnet"
HAIKU = "haiku"

DESTRUCTIVE_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit", "Bash"}
LOOKUP_TOOLS = {"Read", "Grep", "Glob"}
WEB_TOOLS = {"WebSearch", "WebFetch"}
PLANNING_MARKERS = ("plan", "architect", "design", "tradeoff", "trade-off", "strategy")

# Per-MTok USD (input, output). Source: claude-api skill pricing table, 2026-06-04.
MODEL_COSTS_PER_MTOK = {
    OPUS: (5.00, 25.00),
    SONNET: (3.00, 15.00),
    HAIKU: (1.00, 5.00),
}
# Turn logs carry one combined token count, so savings math uses a blended
# rate. Agent transcripts are input-heavy; assume 80% input / 20% output.
INPUT_FRACTION = 0.8


@dataclass
class Turn:
    turn_id: str
    tools: list[str] = field(default_factory=list)
    text: str = ""
    model: str = ""  # model that served this turn (read when inheriting)


@dataclass
class RouterConfig:
    pin_model: str = ""  # per-conversation override: always wins
    default_model: str = SONNET  # used when there is no history to inherit


def route_turn(turn: Turn, history: list[Turn], config: Optional[RouterConfig] = None) -> str:
    """Pick the model tier for one turn. Returns 'opus' | 'sonnet' | 'haiku' or the pinned model."""
    cfg = config or RouterConfig()
    if cfg.pin_model:
        return cfg.pin_model

    tools = set(turn.tools)
    if tools & DESTRUCTIVE_TOOLS:
        return OPUS  # code correctness matters
    if _is_planning(turn.text):
        return OPUS  # multi-step reasoning
    if tools & WEB_TOOLS:
        return SONNET  # summarizing pages, not reasoning
    if tools & LOOKUP_TOOLS:
        return HAIKU  # pure lookup, deterministic

    # Default: inherit the previous turn's model to avoid thrash
    for prev in reversed(history):
        if prev.model:
            return prev.model
    return cfg.default_model


def _is_planning(text: str) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in PLANNING_MARKERS)


def model_tier(model: str) -> str:
    """Collapse a model string ('claude-sonnet-4-6', 'sonnet', …) to its pricing tier."""
    m = (model or "").lower()
    for tier in (OPUS, SONNET, HAIKU):
        if tier in m:
            return tier
    return OPUS  # unknown models priced conservatively


def blended_rate_usd_per_mtok(model: str) -> float:
    inp, out = MODEL_COSTS_PER_MTOK[model_tier(model)]
    return inp * INPUT_FRACTION + out * (1 - INPUT_FRACTION)


def estimate_turn_cost_usd(model: str, tokens: int) -> float:
    return round(tokens / 1_000_000 * blended_rate_usd_per_mtok(model), 6)


def savings_vs_all_opus(turns: list[dict]) -> dict:
    """Aggregate routed turns [{model, tokens}, …] against an all-Opus baseline.

    Returns {turns, downgraded_pct, actual_usd, all_opus_usd, saved_usd}.
    """
    total = len(turns)
    downgraded = sum(1 for t in turns if model_tier(t.get("model", "")) != OPUS)
    actual = sum(estimate_turn_cost_usd(t.get("model", ""), int(t.get("tokens", 0))) for t in turns)
    baseline = sum(estimate_turn_cost_usd(OPUS, int(t.get("tokens", 0))) for t in turns)
    return {
        "turns": total,
        "downgraded_pct": round(100.0 * downgraded / total, 1) if total else 0.0,
        "actual_usd": round(actual, 6),
        "all_opus_usd": round(baseline, 6),
        "saved_usd": round(baseline - actual, 6),
    }
