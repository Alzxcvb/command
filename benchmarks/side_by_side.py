"""
Benchmark: Claude Code (haiku/sonnet/opus) vs Codex (low/medium/high effort)
on 5 hard reasoning tasks.

No API keys required — uses Pro/Plus subscription CLIs only.

Run:
    python -m benchmarks.side_by_side
    python -m benchmarks.side_by_side --provider claude   # Claude only
    python -m benchmarks.side_by_side --provider codex   # Codex only
    python -m benchmarks.side_by_side --dry-run          # No CLI calls
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLAUDE_BIN = os.environ.get("COMMAND_CLAUDE_BIN") or "claude"
CODEX_BIN  = os.environ.get("COMMAND_CODEX_BIN")  or "codex"
TIMEOUT    = 180  # seconds per call

# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

CLAUDE_VARIANTS = [
    {"provider": "claude", "name": "claude:haiku",  "model": "haiku"},
    {"provider": "claude", "name": "claude:sonnet", "model": "sonnet"},
    {"provider": "claude", "name": "claude:opus",   "model": "opus"},
]

CODEX_VARIANTS = [
    {"provider": "codex", "name": "codex:low",    "effort": "low"},
    {"provider": "codex", "name": "codex:medium", "effort": "medium"},
    {"provider": "codex", "name": "codex:high",   "effort": "high"},
]

# ---------------------------------------------------------------------------
# Tasks  (verified correct answers)
# ---------------------------------------------------------------------------

def _match(pattern: str):
    return lambda r: bool(re.search(pattern, r[:300]))

TASKS = [
    {
        "id": "count-001",
        "type": "Inclusion-exclusion",
        "difficulty": "hard",
        "prompt": (
            "How many integers from 1 to 1000 (inclusive) are NOT divisible by "
            "any of 3, 5, or 7? "
            "Think step by step, then state your final answer as a plain integer."
        ),
        "answer": "457",
        "check": _match(r"\b457\b"),
    },
    {
        "id": "mod-001",
        "type": "Modular arithmetic",
        "difficulty": "medium",
        "prompt": (
            "What is the units digit (last digit) of 7^100? "
            "Think step by step, then state your final answer as a single digit."
        ),
        "answer": "1",
        "check": lambda r: bool(re.search(r"\b1\b", r[:150])) and "391" not in r[:30],
    },
    {
        "id": "trace-001",
        "type": "Algorithm trace",
        "difficulty": "medium",
        "prompt": (
            "What does this Python function return for mystery([2, 3, 1, 4, 2])?\n\n"
            "def mystery(arr):\n"
            "    seen = {}\n"
            "    for i, x in enumerate(arr):\n"
            "        if x in seen:\n"
            "            return i - seen[x]\n"
            "        seen[x] = i\n"
            "    return -1\n\n"
            "Trace through it step by step, then state your final answer as a single integer."
        ),
        "answer": "4",
        "check": _match(r"\b4\b"),
    },
    {
        "id": "perm-001",
        "type": "Combinatorics",
        "difficulty": "medium",
        "prompt": (
            "How many distinct arrangements exist for all 6 letters in the word BANANA? "
            "(Letters: B=1, A=3, N=2.) "
            "Show your work, then state your final answer as a plain integer."
        ),
        "answer": "60",
        "check": _match(r"\b60\b"),
    },
    {
        "id": "time-001",
        "type": "Time-zone reasoning",
        "difficulty": "medium",
        "prompt": (
            "Island A is exactly 3 hours ahead of Island B. "
            "A ferry departs Island A at 11:00 PM local time. "
            "The crossing takes 90 minutes. "
            "What time does it arrive at Island B, in Island B's local time? "
            "Answer in 12-hour format, e.g. '9:30 PM'."
        ),
        "answer": "9:30 PM",
        "check": _match(r"9:30\s*(pm|PM|p\.m\.)"),
    },
]

# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------

def run_claude(model: str, prompt: str, work_dir: Path) -> tuple[str, float]:
    """Returns (response_text, latency_ms). Raises on timeout or error."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        prompt,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(work_dir), timeout=TIMEOUT,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Extract final result from stream-json
    result_text = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "result":
            result_text = evt.get("result", "") or ""
            break
        if evt.get("type") == "assistant":
            for block in evt.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    result_text += block.get("text", "")

    if not result_text and proc.stderr:
        raise RuntimeError(proc.stderr[:300])
    return result_text, latency_ms


def run_codex(effort: str, prompt: str, work_dir: Path) -> tuple[str, float]:
    """Returns (response_text, latency_ms). Raises on timeout or error."""
    cmd = [
        CODEX_BIN, "exec",
        "--json",
        "--skip-git-repo-check",
        "-c", f"model_reasoning_effort={effort}",
        prompt,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(work_dir), timeout=TIMEOUT,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    result_text = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "item.completed":
            item = evt.get("item") or {}
            if item.get("type") == "agent_message":
                result_text += item.get("text", "")

    if not result_text and proc.returncode != 0:
        raise RuntimeError(proc.stderr[:300])
    return result_text, latency_ms


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Result:
    task_id: str
    task_type: str
    difficulty: str
    variant: str
    provider: str
    answer: str
    response: str
    correct: bool
    latency_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    providers: list[str],
    dry_run: bool = False,
) -> list[Result]:
    variants = []
    if "claude" in providers:
        variants += CLAUDE_VARIANTS
    if "codex" in providers:
        variants += CODEX_VARIANTS

    results: list[Result] = []

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for task in TASKS:
            for v in variants:
                label = f"{task['id']} / {v['name']}"
                print(f"  {label:<35}", end="", flush=True)

                if dry_run:
                    results.append(Result(
                        task_id=task["id"], task_type=task["type"],
                        difficulty=task["difficulty"],
                        variant=v["name"], provider=v["provider"],
                        answer=task["answer"], response="[DRY RUN]",
                        correct=False, latency_ms=0,
                    ))
                    print("DRY RUN")
                    continue

                try:
                    if v["provider"] == "claude":
                        text, ms = run_claude(v["model"], task["prompt"], work_dir)
                    else:
                        text, ms = run_codex(v["effort"], task["prompt"], work_dir)

                    correct = task["check"](text)
                    results.append(Result(
                        task_id=task["id"], task_type=task["type"],
                        difficulty=task["difficulty"],
                        variant=v["name"], provider=v["provider"],
                        answer=task["answer"],
                        response=text[:200].replace("\n", " "),
                        correct=correct, latency_ms=ms,
                    ))
                    print(f"{'PASS' if correct else 'FAIL'}  {ms:.0f}ms")

                except subprocess.TimeoutExpired:
                    results.append(Result(
                        task_id=task["id"], task_type=task["type"],
                        difficulty=task["difficulty"],
                        variant=v["name"], provider=v["provider"],
                        answer=task["answer"], response="",
                        correct=False, latency_ms=TIMEOUT * 1000,
                        error="TIMEOUT",
                    ))
                    print("TIMEOUT")
                except Exception as e:
                    results.append(Result(
                        task_id=task["id"], task_type=task["type"],
                        difficulty=task["difficulty"],
                        variant=v["name"], provider=v["provider"],
                        answer=task["answer"], response="",
                        correct=False, latency_ms=0,
                        error=str(e)[:200],
                    ))
                    print(f"ERROR: {e!s:.60}")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[Result], variants: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("RESULTS: Accuracy and latency by variant")
    print("=" * 72)

    # Per-variant summary
    print(f"\n{'Variant':<20} {'Correct':>9} {'Avg latency':>14}")
    print("-" * 46)
    for v in variants:
        name = v["name"]
        sub = [r for r in results if r.variant == name and not r.error]
        total = [r for r in results if r.variant == name]
        if not total:
            continue
        n_ok = sum(1 for r in sub if r.correct)
        avg_ms = sum(r.latency_ms for r in sub) / max(len(sub), 1)
        n_err = sum(1 for r in total if r.error)
        err_note = f"  ({n_err} errors)" if n_err else ""
        print(f"{name:<20} {n_ok}/{len(total):>7} {avg_ms:>12.0f}ms{err_note}")

    # Per-task grid
    col_names = [v["name"] for v in variants]
    header = f"\n{'Task':<12} {'Type':<24} " + "".join(f"{n:>14}" for n in col_names)
    print(header)
    print("-" * (36 + 14 * len(variants)))
    for task in TASKS:
        row = f"{task['id']:<12} {task['type'][:22]:<24} "
        for v in variants:
            r = next((x for x in results if x.task_id == task["id"] and x.variant == v["name"]), None)
            if r is None:
                row += f"{'—':>14}"
            elif r.error:
                row += f"{'ERR':>14}"
            else:
                row += f"{'PASS':>14}" if r.correct else f"{'fail':>14}"
        print(row)

    # Failures detail
    fails = [r for r in results if not r.correct and not r.error]
    if fails:
        print(f"\nFailed responses ({len(fails)}):")
        for r in fails:
            print(f"  [{r.variant} / {r.task_id}] expected={r.answer!r}")
            print(f"    {r.response[:120]!r}")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors:
            print(f"  [{r.variant} / {r.task_id}] {r.error}")

    print(
        f"\nNote: N=1 per cell ({len(TASKS)} tasks × {len(variants)} variants). "
        "Use directionally — rerun for confidence."
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(results: list[Result], variants: list[dict]) -> Path:
    out_dir = ROOT / "benchmarks" / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"side_by_side_{ts}.json"
    path.write_text(json.dumps({
        "benchmark": "side_by_side",
        "timestamp": ts,
        "variants": variants,
        "tasks": [{"id": t["id"], "type": t["type"], "answer": t["answer"]} for t in TASKS],
        "results": [asdict(r) for r in results],
    }, indent=2))
    return path


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code vs Codex side-by-side benchmark")
    parser.add_argument("--provider", choices=["claude", "codex", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    providers = ["claude", "codex"] if args.provider == "both" else [args.provider]
    variants = []
    if "claude" in providers:
        variants += CLAUDE_VARIANTS
    if "codex" in providers:
        variants += CODEX_VARIANTS

    n_calls = len(TASKS) * len(variants)
    print(f"Side-by-side benchmark: {', '.join(v['name'] for v in variants)}")
    print(f"Tasks: {len(TASKS)}  |  Variants: {len(variants)}  |  Total calls: {n_calls}")
    print(f"Timeout: {TIMEOUT}s per call  |  Est. time: ~{n_calls * 30 // 60}–{n_calls * 60 // 60} min\n")

    results = run_benchmark(providers, dry_run=args.dry_run)

    if not args.dry_run:
        saved = save(results, variants)
        print(f"\nSaved: {saved.relative_to(ROOT)}")

    print_report(results, variants)


if __name__ == "__main__":
    main()
