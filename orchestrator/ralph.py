"""Ralph improver: iterates the orchestrator guide using failure signal only.

The flow:
    1. Pull failure samples since the last improvement guide_sha.
    2. If there are enough, spawn a claude_code sub-agent with:
       - the current orchestrator.md
       - the failure buffer (drafted tasks + outcomes + fail_reasons)
       - instructions to return a revised orchestrator.md between markers
    3. Extract the revised guide, overwrite orchestrator.md.
    4. Commit the change (so the guide_sha actually moves forward).
    5. Record last_improvement.json.

Why the overwrite format (not a unified diff): we want the improver to have
full rewrite authority on a small file. Diffs introduce a parsing step that
fails noisily; overwriting a ~2KB file is cheap and robust.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.lifecycle import spawn_agent  # noqa: E402
from agents.registry import get_agent, get_result  # noqa: E402
from orchestrator import observations  # noqa: E402

GUIDE_PATH = observations.GUIDE_PATH
CHANGELOG_PATH = _REPO_ROOT / "state" / "ralph" / "changelog.md"

MIN_SAMPLES_TO_TRIGGER = 3
DEFAULT_RALPH_BUDGET = 15000

REVISED_START = "<<<REVISED_ORCHESTRATOR_MD>>>"
REVISED_END = "<<<END_REVISED_ORCHESTRATOR_MD>>>"


def _sample_card(sample: dict) -> str:
    disp = sample["dispatch"]
    out = sample.get("outcome") or {}
    retries = sample.get("retries") or []
    thumbs = sample.get("thumbs") or []
    fail_reasons = ", ".join(sample.get("fail_reasons", []))
    lines = [
        f"### Sample {sample['agent_id']}",
        f"- runtime: {disp.get('runtime')} / {disp.get('model') or '—'}",
        f"- budget_tokens: {disp.get('budget_tokens')}  (used: {out.get('tokens_used', 0)})",
        f"- fail_reasons: {fail_reasons}",
        "",
        "Drafted task:",
        "```",
        disp.get("drafted_task", "")[:1500],
        "```",
    ]
    if out.get("final_text"):
        lines += ["", "Agent output (truncated):", "```", out["final_text"][:800], "```"]
    if retries:
        lines += ["", f"Retries: {len(retries)}. Sample rationale: {retries[0].get('rationale', '(none)')}"]
    if thumbs:
        t = thumbs[0]
        lines += ["", f"Thumb {t.get('direction')}: {t.get('comment', '(no comment)')}"]
    return "\n".join(lines)


def build_improver_prompt(samples: list[dict], current_guide: str) -> str:
    sample_md = "\n\n".join(_sample_card(s) for s in samples)
    return (
        "You are Ralph — the guide improver for the Command orchestrator.\n\n"
        "Your job: read the current orchestrator.md and the failure buffer below, "
        "then produce a revised orchestrator.md that would have avoided these failures. "
        "Change routing rules, examples, or constraints — whatever the failure pattern tells you.\n\n"
        "## Current orchestrator.md\n"
        f"```markdown\n{current_guide}\n```\n\n"
        "## Failure buffer\n"
        f"These are sub-agent prompts that failed, went over budget, were retried, or were thumbed down. "
        f"Successful prompts were filtered out — you don't see them.\n\n{sample_md}\n\n"
        "## Output format — mandatory\n"
        f"First, a 1-3 sentence summary of what change you're making and why. "
        f"Then the full revised orchestrator.md between these markers:\n\n"
        f"{REVISED_START}\n...full revised file content here...\n{REVISED_END}\n\n"
        "If no change is warranted, still output the markers with the file unchanged, "
        "and say so in the summary.\n"
    )


def _extract_revised(text: str) -> Optional[str]:
    if REVISED_START not in text or REVISED_END not in text:
        return None
    start = text.index(REVISED_START) + len(REVISED_START)
    end = text.index(REVISED_END)
    return text[start:end].strip("\n")


def _git_commit_guide(summary: str) -> bool:
    try:
        subprocess.run(
            ["git", "add", str(GUIDE_PATH)],
            cwd=str(_REPO_ROOT), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"ralph: {summary[:80]}"],
            cwd=str(_REPO_ROOT), check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _append_changelog(entry: str) -> None:
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHANGELOG_PATH.open("a") as f:
        f.write(entry + "\n\n---\n\n")


def pending_failure_count() -> int:
    last_ts = observations.last_improvement_ts()
    samples = observations.failure_samples(since_ts=last_ts)
    return len(samples)


def improve(
    *,
    model: str = "sonnet",
    budget_tokens: int = DEFAULT_RALPH_BUDGET,
    force: bool = False,
    min_samples: int = MIN_SAMPLES_TO_TRIGGER,
    wait: bool = True,
    timeout: float = 1800,
) -> dict:
    """Run one Ralph pass. Returns a status dict."""
    last_ts = observations.last_improvement_ts()
    samples = observations.failure_samples(since_ts=last_ts)
    if len(samples) < min_samples and not force:
        return {
            "status": "skipped",
            "reason": f"only {len(samples)} failure samples since last improvement (need {min_samples})",
            "samples": len(samples),
        }

    current_guide = GUIDE_PATH.read_text() if GUIDE_PATH.exists() else ""
    prompt = build_improver_prompt(samples, current_guide)

    agent_id = spawn_agent(
        task=prompt,
        runtime_name="claude_code",
        system_prompt="",
        budget_tokens=budget_tokens,
        model=model,
        parent_job_id=None,
    )

    if not wait:
        return {"status": "spawned", "agent_id": agent_id, "samples": len(samples)}

    start = time.time()
    while True:
        meta = get_agent(agent_id) or {}
        if meta.get("status") not in ("starting", "running"):
            break
        if time.time() - start > timeout:
            return {"status": "timeout", "agent_id": agent_id, "samples": len(samples)}
        time.sleep(1.0)

    # Compute max dispatch timestamp from this batch so future calls skip these samples.
    processed_through_ts: Optional[str] = None
    if samples:
        processed_through_ts = max(
            s["dispatch"].get("timestamp", "") for s in samples
        ) or None

    result = get_result(agent_id) or {}
    output = result.get("final_text", "") or ""
    revised = _extract_revised(output)
    if not revised:
        return {
            "status": "no_diff",
            "agent_id": agent_id,
            "samples": len(samples),
            "reason": "improver did not emit revised-file markers",
        }

    if revised == current_guide:
        new_sha = observations.current_guide_sha()
        observations.record_improvement(new_sha, agent_id, len(samples),
                                        processed_through_ts=processed_through_ts)
        return {
            "status": "no_change",
            "agent_id": agent_id,
            "samples": len(samples),
            "guide_sha": new_sha,
        }

    GUIDE_PATH.write_text(revised)
    new_sha = observations.current_guide_sha()

    # Summary = everything before the first marker.
    summary = output.split(REVISED_START, 1)[0].strip() or "guide revision"
    committed = _git_commit_guide(summary.splitlines()[0][:100])
    observations.record_improvement(new_sha, agent_id, len(samples),
                                    processed_through_ts=processed_through_ts)
    _append_changelog(
        f"## {new_sha}  ({len(samples)} samples, agent {agent_id})\n\n{summary}"
    )

    return {
        "status": "applied",
        "agent_id": agent_id,
        "samples": len(samples),
        "guide_sha": new_sha,
        "committed": committed,
        "summary": summary,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="run even if below min_samples")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES_TO_TRIGGER)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--budget", type=int, default=DEFAULT_RALPH_BUDGET)
    args = ap.parse_args()
    result = improve(
        model=args.model,
        budget_tokens=args.budget,
        force=args.force,
        min_samples=args.min_samples,
    )
    print(json.dumps(result, indent=2))
