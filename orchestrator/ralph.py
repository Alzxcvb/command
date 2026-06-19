"""Ralph improver: iterates the orchestrator guide using failure signal only.

The flow:
    1. Pull failure samples since the last improvement guide_sha.
    2. If there are enough, spawn a claude_code sub-agent with:
       - the current orchestrator.md
       - the failure buffer (drafted tasks + outcomes + fail_reasons)
       - instructions to return a revised orchestrator.md between markers
    3. Extract the revised guide, run the 2-model judge council.
    4. If both judges approve, overwrite orchestrator.md and git-commit.
    5. If the council rejects, log the rejection and return without committing.
    6. Record last_improvement.json.

Why the overwrite format (not a unified diff): we want the improver to have
full rewrite authority on a small file. Diffs introduce a parsing step that
fails noisily; overwriting a ~2KB file is cheap and robust.

Judge council: one model judging its own kind of work has self-enhancement bias.
A 2-model council (Haiku 4.5 fast-cheap + Opus 4.8 authoritative) must both
approve before any guide change is committed. Rejections are logged to
state/ralph/council_rejections.jsonl for review.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
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
COUNCIL_REJECTIONS_PATH = _REPO_ROOT / "state" / "ralph" / "council_rejections.jsonl"

MIN_SAMPLES_TO_TRIGGER = 3
DEFAULT_RALPH_BUDGET = 15000

REVISED_START = "<<<REVISED_ORCHESTRATOR_MD>>>"
REVISED_END = "<<<END_REVISED_ORCHESTRATOR_MD>>>"

# Judge models: Haiku 4.5 (fast, cheap) + Opus 4.8 (authoritative).
# Both must approve before any guide revision is committed.
HAIKU_JUDGE_MODEL = "anthropic/claude-haiku-4-5"
OPUS_JUDGE_MODEL = "anthropic/claude-opus-4-8"

JUDGE_RUBRIC = (
    "You are reviewing a proposed change to an AI orchestrator guide.\n"
    "Score each criterion 1-5:\n"
    "1. Does the proposed change address the failure patterns shown?\n"
    "2. Does it preserve the core orchestration intent of the original?\n"
    "3. Does it avoid introducing ambiguous or over-broad instructions?\n"
    "4. Is it likely to reduce over-budget or failed agent runs?\n"
    "Verdict: APPROVE if all criteria >= 3, otherwise REJECT. "
    "State the weakest criterion."
)


def _sample_card(sample: dict) -> str:
    disp = sample["dispatch"]
    out = sample.get("outcome") or {}
    retries = sample.get("retries") or []
    thumbs = sample.get("thumbs") or []
    fail_reasons = ", ".join(sample.get("fail_reasons", []))
    lines = [
        f"### Sample {sample['agent_id']}",
        f"- runtime: {disp.get('runtime')} / {disp.get('model') or chr(8212)}",
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


def _build_judge_prompt(
    original_guide: str, proposed_guide: str, failure_samples: list[dict]
) -> str:
    """Build the judge prompt with rubric, both guide versions, and failure context."""
    sample_md = "\n\n".join(_sample_card(s) for s in failure_samples[:10])
    return (
        f"{JUDGE_RUBRIC}\n\n"
        "## Original orchestrator.md\n"
        f"```markdown\n{original_guide}\n```\n\n"
        "## Proposed revised orchestrator.md\n"
        f"```markdown\n{proposed_guide}\n```\n\n"
        "## Failure samples that motivated this change\n"
        f"{sample_md}\n"
    )


def _call_judge(model_id: str, prompt: str) -> str:
    """Call a judge model via the project router, falling back to direct Anthropic or OpenRouter.

    Dispatch chain (mirrors the project's routing philosophy):
      1. Router dispatch: respects COMMAND_PROVIDER env override, then model's native
         provider (Anthropic direct if keyed), then OpenRouter fallback.
      2. If the router chain raises ValueError (no keys at all), try direct Anthropic.
      3. If that also fails, try OpenRouter directly.
      4. If all three fail, raise a clear RuntimeError — never silently skip the council.
    """
    from router.models import get_model
    from router.providers import AnthropicProvider, OpenRouterProvider, get_provider_for

    errors: list[str] = []

    # 1. Try via the project router (COMMAND_PROVIDER -> native provider -> OpenRouter)
    model = get_model(model_id)
    if model is not None:
        try:
            provider = get_provider_for(model)
            content, _ = provider.call_raw(model_id, prompt, max_tokens=1024)
            return content
        except ValueError as exc:
            errors.append(f"router: {exc}")

    # 2. Direct Anthropic fallback (covers model not in registry, or router had no keys)
    try:
        provider = AnthropicProvider()
        content, _ = provider.call_raw(model_id, prompt, max_tokens=1024)
        return content
    except ValueError as exc:
        errors.append(f"direct_anthropic: {exc}")

    # 3. OpenRouter final fallback
    try:
        provider = OpenRouterProvider()
        content, _ = provider.call_raw(model_id, prompt, max_tokens=1024)
        return content
    except ValueError as exc:
        errors.append(f"openrouter: {exc}")
        raise RuntimeError(
            f"No API key available to call judge model {model_id!r}. "
            f"Tried: {'; '.join(errors)}. "
            "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY."
        )


def _parse_verdict(text: str) -> tuple[str, str]:
    """Return (verdict, reason) where verdict is 'APPROVE' or 'REJECT'.

    Looks for a line containing 'Verdict:'. If none is found, falls back to
    scanning the full text for the last occurrence of either keyword.
    Defaults to 'REJECT' if neither is found — fail-closed is correct for a
    commit gate.
    """
    verdict = "REJECT"  # fail-closed default
    verdict_found = False

    for line in text.splitlines():
        upper = line.upper()
        if "VERDICT:" in upper:
            if "APPROVE" in upper:
                verdict = "APPROVE"
            else:
                verdict = "REJECT"
            verdict_found = True
            break

    if not verdict_found:
        # Scan entire text; last explicit keyword wins
        upper_text = text.upper()
        approve_pos = upper_text.rfind("APPROVE")
        reject_pos = upper_text.rfind("REJECT")
        if approve_pos > reject_pos:
            verdict = "APPROVE"
        # else stays REJECT

    return verdict, text.strip()


def judge_guide_revision(
    original_guide: str,
    proposed_guide: str,
    failure_samples: list[dict],
) -> dict:
    """Run the 2-model judge council on a proposed guide revision.

    Haiku 4.5 (fast, cheap) is called first for a quick screen, then
    Opus 4.8 (authoritative) for a deeper review. Both must emit APPROVE
    before `council_approved` is True.

    Args:
        original_guide: The current orchestrator.md content before any changes.
        proposed_guide: The revised orchestrator.md proposed by the improver.
        failure_samples: The failure buffer that motivated the revision.

    Returns:
        dict with keys: haiku_verdict, opus_verdict, haiku_reason,
        opus_reason, council_approved.
    """
    prompt = _build_judge_prompt(original_guide, proposed_guide, failure_samples)

    haiku_text = _call_judge(HAIKU_JUDGE_MODEL, prompt)
    haiku_verdict, haiku_reason = _parse_verdict(haiku_text)

    opus_text = _call_judge(OPUS_JUDGE_MODEL, prompt)
    opus_verdict, opus_reason = _parse_verdict(opus_text)

    council_approved = haiku_verdict == "APPROVE" and opus_verdict == "APPROVE"

    return {
        "haiku_verdict": haiku_verdict,
        "opus_verdict": opus_verdict,
        "haiku_reason": haiku_reason,
        "opus_reason": opus_reason,
        "council_approved": council_approved,
    }


def _log_council_rejection(council: dict, proposed: str) -> None:
    """Append a JSONL entry to the council rejections log."""
    COUNCIL_REJECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    proposed_sha = hashlib.sha256(proposed.encode()).hexdigest()[:12]
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "proposed_sha": proposed_sha,
        "haiku_verdict": council["haiku_verdict"],
        "opus_verdict": council["opus_verdict"],
        "haiku_reason": council["haiku_reason"],
        "opus_reason": council["opus_reason"],
    }
    with COUNCIL_REJECTIONS_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


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
    """Run one Ralph pass. Returns a status dict.

    The --force flag bypasses the min_samples gate but the judge council still
    gates the final commit. A force-run with a council rejection logs to
    state/ralph/council_rejections.jsonl and returns without committing.
    """
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

    # Run the judge council before committing any change to the guide.
    # Both models must approve; force=True still passes through this gate.
    council = judge_guide_revision(current_guide, revised, samples)
    if not council["council_approved"]:
        _log_council_rejection(council, revised)
        return {
            "status": "council_rejected",
            "agent_id": agent_id,
            "samples": len(samples),
            "haiku_verdict": council["haiku_verdict"],
            "opus_verdict": council["opus_verdict"],
            "haiku_reason": council["haiku_reason"],
            "opus_reason": council["opus_reason"],
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
