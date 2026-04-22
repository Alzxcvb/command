"""
Benchmark: Does extended thinking improve reasoning accuracy on hard tasks?

Tests claude-sonnet-4-6 at three thinking budget levels (none / medium / high)
across 5 tasks that require multi-step reasoning.  N=1 per cell — this is a
*sample* evaluation to calibrate routing heuristics, not a statistical study.

Setup:
    Set OPENROUTER_API_KEY (or ANTHROPIC_API_KEY) in your env or a .env file.
    pip install openai python-dotenv  (already in requirements.txt)

Run:
    python -m benchmarks.thinking_levels
    python -m benchmarks.thinking_levels --dry-run   # skips API calls
    python -m benchmarks.thinking_levels --no-preflight  # skip smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "anthropic/claude-sonnet-4-6"

# Sonnet 4.6 pricing via OpenRouter (USD per 1M tokens)
COST_INPUT_PER_M = 3.00
COST_OUTPUT_PER_M = 15.00

THINKING_LEVELS = [
    {"name": "none",   "budget_tokens": 0},
    {"name": "medium", "budget_tokens": 5_000},
    {"name": "high",   "budget_tokens": 16_000},
]

# ---------------------------------------------------------------------------
# Task definitions  (each answer field is what the model should produce)
# ---------------------------------------------------------------------------

def _match_number(answer: str):
    """Returns a checker that finds the answer number as a whole word."""
    def check(response: str) -> bool:
        return bool(re.search(rf"\b{re.escape(answer)}\b", response[:200]))
    return check


TASKS = [
    {
        "id": "count-001",
        "type": "Inclusion-exclusion counting",
        "difficulty": "hard",
        "prompt": (
            "How many integers from 1 to 1000 (inclusive) are NOT divisible by "
            "any of 3, 5, or 7?\n"
            "Think carefully and show your work. State your final answer as a plain integer."
        ),
        "answer": "457",
        "check": _match_number("457"),
    },
    {
        "id": "mod-001",
        "type": "Modular arithmetic",
        "difficulty": "medium",
        "prompt": (
            "What is the last digit (units digit) of 7^100?\n"
            "State your final answer as a single digit."
        ),
        "answer": "1",
        "check": lambda r: bool(re.search(r"\b1\b", r[:100])) and "391" not in r[:20],
    },
    {
        "id": "trace-001",
        "type": "Algorithm trace",
        "difficulty": "medium",
        "prompt": (
            "What does this Python function return when called as mystery([2, 3, 1, 4, 2])?\n\n"
            "def mystery(arr):\n"
            "    seen = {}\n"
            "    for i, x in enumerate(arr):\n"
            "        if x in seen:\n"
            "            return i - seen[x]\n"
            "        seen[x] = i\n"
            "    return -1\n\n"
            "State your final answer as a single integer."
        ),
        "answer": "4",
        "check": lambda r: bool(re.search(r"\b4\b", r[:80])),
    },
    {
        "id": "perm-001",
        "type": "Combinatorics",
        "difficulty": "medium",
        "prompt": (
            "How many distinct arrangements exist for all letters in the word BANANA?\n"
            "State your final answer as a plain integer."
        ),
        "answer": "60",
        "check": _match_number("60"),
    },
    {
        "id": "time-001",
        "type": "Multi-step time zone reasoning",
        "difficulty": "medium",
        "prompt": (
            "Island A is exactly 3 hours ahead of Island B. "
            "A ferry departs Island A at 11:00 PM local Island-A time. "
            "The crossing takes 90 minutes. "
            "What time does it arrive at Island B, expressed in Island B's local time?\n"
            "Answer in 12-hour format, e.g. '9:30 PM'."
        ),
        "answer": "9:30 PM",
        "check": lambda r: bool(re.search(r"9:30\s*(pm|PM|p\.m\.)", r)),
    },
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    task_type: str
    difficulty: str
    thinking_level: str
    budget_tokens: int
    answer: str
    response_excerpt: str
    correct: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _make_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print(
            "ERROR: No API key found.\n"
            "Set OPENROUTER_API_KEY (or ANTHROPIC_API_KEY) in your environment or .env file.\n"
            "  echo 'OPENROUTER_API_KEY=sk-or-...' >> command/.env"
        )
        sys.exit(1)

    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(base_url="https://api.anthropic.com/v1", api_key=key)

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def call(client: OpenAI, prompt: str, budget_tokens: int) -> tuple[str, float, int, int]:
    """(response_text, latency_ms, input_tokens, output_tokens)"""
    max_tokens = max(2048, budget_tokens + 2048)
    kwargs: dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if budget_tokens > 0:
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}

    start = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return text, latency_ms, usage.prompt_tokens, usage.completion_tokens


# ---------------------------------------------------------------------------
# Preflight: verify thinking parameter actually gets through
# ---------------------------------------------------------------------------

def preflight(client: OpenAI) -> bool:
    print("Preflight: verifying thinking passthrough...")
    probe = "What is 293 * 47? Show your work then state the final answer."
    try:
        _, _, _, out_no = call(client, probe, 0)
        _, ms_think, _, out_think = call(client, probe, 5000)
        passed = out_think > out_no + 50
        token_delta = out_think - out_no
        print(
            f"  No-think out_tokens={out_no}  |  Think-5k out_tokens={out_think} "
            f"({'+' if token_delta>=0 else ''}{token_delta})  "
            f"latency={ms_think:.0f}ms"
        )
        if passed:
            print("  PASS — thinking tokens are flowing through.\n")
        else:
            print(
                "  WARN — token count did not increase significantly.\n"
                "  OpenRouter may not be passing 'thinking' to Anthropic.\n"
                "  Set ANTHROPIC_API_KEY instead to use Anthropic directly.\n"
            )
        return passed
    except Exception as e:
        print(f"  Preflight failed with error: {e}")
        return False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(dry_run: bool = False, skip_preflight: bool = False) -> list[TaskResult]:
    client = None if dry_run else _make_client()

    if not dry_run and not skip_preflight:
        ok = preflight(client)
        if not ok:
            ans = input("Thinking passthrough unconfirmed. Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborted.")
                sys.exit(0)

    results: list[TaskResult] = []

    for task in TASKS:
        for level in THINKING_LEVELS:
            label = f"{task['id']} / {level['name']}"
            print(f"  Running {label}...", end="", flush=True)

            if dry_run:
                r = TaskResult(
                    task_id=task["id"], task_type=task["type"],
                    difficulty=task["difficulty"],
                    thinking_level=level["name"], budget_tokens=level["budget_tokens"],
                    answer=task["answer"], response_excerpt="[DRY RUN]",
                    correct=False, latency_ms=0, input_tokens=0, output_tokens=0, cost_usd=0,
                )
            else:
                text, latency_ms, in_tok, out_tok = call(client, task["prompt"], level["budget_tokens"])
                correct = task["check"](text)
                cost = (in_tok / 1e6 * COST_INPUT_PER_M) + (out_tok / 1e6 * COST_OUTPUT_PER_M)
                r = TaskResult(
                    task_id=task["id"], task_type=task["type"],
                    difficulty=task["difficulty"],
                    thinking_level=level["name"], budget_tokens=level["budget_tokens"],
                    answer=task["answer"],
                    response_excerpt=text[:120].replace("\n", " "),
                    correct=correct,
                    latency_ms=latency_ms,
                    input_tokens=in_tok, output_tokens=out_tok,
                    cost_usd=cost,
                )
                print(f" {'PASS' if correct else 'FAIL'}  {latency_ms:.0f}ms  "
                      f"out={out_tok}tok  ${cost*1000:.3f}m", flush=True)

            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[TaskResult]) -> None:
    print("\n" + "=" * 72)
    print("RESULTS: Thinking level vs. reasoning accuracy")
    print("=" * 72)

    # Per-level summary
    print(f"\n{'Level':<10} {'Correct':>8} {'Avg tokens':>12} {'Avg latency':>12} {'Total cost':>12}")
    print("-" * 60)
    for level in THINKING_LEVELS:
        name = level["name"]
        subset = [r for r in results if r.thinking_level == name]
        if not subset:
            continue
        n_correct = sum(1 for r in subset if r.correct)
        avg_out = sum(r.output_tokens for r in subset) / len(subset)
        avg_lat = sum(r.latency_ms for r in subset) / len(subset)
        total_cost = sum(r.cost_usd for r in subset)
        print(f"{name:<10} {n_correct}/{len(subset):>6} {avg_out:>12.0f} "
              f"{avg_lat:>10.0f}ms  ${total_cost*1000:>8.2f}m")

    # Per-task breakdown
    print(f"\n{'Task':<14} {'Type':<32} ", end="")
    for level in THINKING_LEVELS:
        print(f"{level['name']:>8}", end="")
    print()
    print("-" * 72)
    for task in TASKS:
        tid = task["id"]
        subset = [r for r in results if r.task_id == tid]
        row = f"{tid:<14} {task['type'][:30]:<32} "
        for r in subset:
            row += f"{'PASS':>8}" if r.correct else f"{'fail':>8}"
        print(row)

    # Raw responses for failed cases
    fails = [r for r in results if not r.correct]
    if fails:
        print(f"\nFailed responses ({len(fails)} cases):")
        for r in fails:
            print(f"  [{r.task_id} / {r.thinking_level}] expected={r.answer!r}")
            print(f"    response: {r.response_excerpt[:100]!r}")

    total_cost = sum(r.cost_usd for r in results)
    print(f"\nTotal cost: ${total_cost*1000:.2f}m  ({len(results)} API calls)")
    print(
        "\nNote: N=1 per cell. Use results directionally, not as ground truth.\n"
        "Rerun several times or expand task set before updating routing weights."
    )


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list[TaskResult]) -> Path:
    results_dir = ROOT / "benchmarks" / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"thinking_levels_{ts}.json"
    payload = {
        "benchmark": "thinking_levels",
        "model": MODEL,
        "timestamp": ts,
        "thinking_levels": THINKING_LEVELS,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    parser.add_argument("--no-preflight", action="store_true", help="Skip smoke test")
    args = parser.parse_args()

    print(f"Benchmark: thinking levels on {MODEL}")
    print(f"Tasks: {len(TASKS)} | Levels: {len(THINKING_LEVELS)} | Total calls: {len(TASKS)*len(THINKING_LEVELS)}\n")

    results = run_benchmark(dry_run=args.dry_run, skip_preflight=args.no_preflight)

    if not args.dry_run:
        saved = save_results(results)
        print(f"\nResults saved to: {saved.relative_to(ROOT)}")

    print_report(results)


if __name__ == "__main__":
    main()
