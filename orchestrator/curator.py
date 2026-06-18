"""Post-job memory curator (the Toby watchdog pattern).

After every job completes, this module reads the job goal, final output, and
relevant observations, then asks a cheap LLM three questions:
  - Did the output actually answer the original goal?
  - What failed or got retried, and why?
  - What should be remembered for next time?

Results are written to two append-only JSONL files:
  state/curator/patterns.jsonl      -- recurring issues flagged across this job
  state/curator/memory_queue.jsonl  -- suggested memory updates pending human approval
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from router.providers import OpenRouterProvider  # noqa: E402

STATE_ROOT = _REPO_ROOT / "state"
JOBS_DIR = STATE_ROOT / "jobs"
OBSERVATIONS_PATH = STATE_ROOT / "ralph" / "observations.jsonl"
CURATOR_DIR = STATE_ROOT / "curator"
PATTERNS_PATH = CURATOR_DIR / "patterns.jsonl"
MEMORY_QUEUE_PATH = CURATOR_DIR / "memory_queue.jsonl"

CURATOR_MODEL = "anthropic/claude-haiku-4-5"
MAX_OBSERVATIONS = 10

CURATOR_SYSTEM_PROMPT = """\
You are a post-job memory curator for an agent orchestration platform.

You will receive a summary of a completed job: its goal, final status, child agent
outcomes, and a sample of observations (dispatches, completions, retries).

Answer three questions and return ONLY raw JSON with no markdown fences, no preamble,
and no trailing text. The JSON must match this exact shape:

{
  "answered_goal": true,
  "drift_summary": "...",
  "recurring_issues": ["..."],
  "memory_updates": [
    {"file": "project-command.md", "note": "..."}
  ]
}

Field rules:
- answered_goal: true if the job output clearly addressed the stated goal, false otherwise.
- drift_summary: a short sentence describing how the job strayed from the goal, or an
  empty string if there was no meaningful drift.
- recurring_issues: a list of short phrases naming patterns that caused problems across
  multiple agents in this job (e.g. "budget overrun on code tasks", "retry loop on tool call").
  Empty list if no issues.
- memory_updates: a list of objects with "file" (a project memory filename) and "note"
  (a one-sentence update worth saving). Empty list if nothing is worth saving.

Be concise. Do not pad the response.
"""


@dataclass
class CuratorResult:
    job_id: str
    answered_goal: bool
    drift_summary: str
    recurring_issues: list[str] = field(default_factory=list)
    memory_updates: list[dict] = field(default_factory=list)
    ts: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_meta(job_dir: Path) -> dict:
    p = job_dir / "meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _load_observations(job_id: str) -> list[dict]:
    """Read up to MAX_OBSERVATIONS dispatch and outcome events for this job."""
    if not OBSERVATIONS_PATH.exists():
        return []
    matches: list[dict] = []
    try:
        with OBSERVATIONS_PATH.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if obj.get("job_id") != job_id:
                    continue
                if obj.get("kind") in ("dispatch", "outcome"):
                    matches.append(obj)
                if len(matches) >= MAX_OBSERVATIONS:
                    break
    except Exception:
        pass
    return matches


def _format_observations(obs: list[dict]) -> str:
    """Turn raw observation dicts into readable text for the LLM prompt."""
    lines: list[str] = []
    for o in obs:
        kind = o.get("kind", "unknown")
        if kind == "dispatch":
            lines.append(
                f"DISPATCH agent={o.get('agent_id', '?')} "
                f"runtime={o.get('runtime', '?')} "
                f"budget={o.get('budget_tokens', 0)} "
                f"task={repr((o.get('drafted_task') or '')[:120])}"
            )
        elif kind == "outcome":
            lines.append(
                f"OUTCOME agent={o.get('agent_id', '?')} "
                f"status={o.get('status', '?')} "
                f"tokens={o.get('tokens_used', 0)} "
                f"overrun={o.get('budget_overrun', False)} "
                f"fail_reasons={o.get('fail_reasons', [])}"
            )
    return "\n".join(lines) if lines else "(no observations found)"


def _load_drift_flags(job_dir: Path) -> list[str]:
    """Return drift_flags from prompt_pipeline.json if it exists."""
    p = job_dir / "prompt_pipeline.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        flags = data.get("drift_flags", [])
        return flags if isinstance(flags, list) else []
    except Exception:
        return []


def _build_user_prompt(meta: dict, obs: list[dict], drift_flags: list[str]) -> str:
    goal = meta.get("goal") or "(no goal recorded)"
    status = meta.get("status") or "unknown"
    summary = meta.get("children_summary") or {}

    parts: list[str] = [
        f"## Job goal\n{goal}",
        f"## Final status\n{status}",
        f"## Children summary\n{json.dumps(summary, indent=2)}",
        f"## Observations (up to {MAX_OBSERVATIONS})\n{_format_observations(obs)}",
    ]

    if drift_flags:
        parts.append(f"## Prompt pipeline drift flags\n" + "\n".join(f"- {f}" for f in drift_flags))

    return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
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


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def run_curator(job_id: str) -> CuratorResult | None:
    """Run the memory curator for a completed job.

    Returns None silently if the job directory does not exist.
    """
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        return None

    meta = _load_meta(job_dir)
    obs = _load_observations(job_id)
    drift_flags = _load_drift_flags(job_dir)

    user_prompt = _build_user_prompt(meta, obs, drift_flags)

    try:
        provider = OpenRouterProvider()
        raw_response, _ = provider.call_raw(
            CURATOR_MODEL,
            user_prompt,
            system_prompt=CURATOR_SYSTEM_PROMPT,
            max_tokens=1024,
        )
    except Exception as e:
        print(f"[curator {job_id}] LLM call failed: {e!r}")
        return None

    cleaned = _strip_fences(raw_response)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[curator {job_id}] JSON parse error: {e!r}  raw={cleaned[:200]!r}")
        return None

    ts = _now_iso()

    answered_goal: bool = bool(parsed.get("answered_goal", False))
    drift_summary: str = parsed.get("drift_summary") or ""
    recurring_issues: list[str] = parsed.get("recurring_issues") or []
    memory_updates: list[dict] = parsed.get("memory_updates") or []

    for issue in recurring_issues:
        _append_jsonl(PATTERNS_PATH, {"job_id": job_id, "issue": issue, "ts": ts})

    for update in memory_updates:
        _append_jsonl(
            MEMORY_QUEUE_PATH,
            {
                "job_id": job_id,
                "file": update.get("file", ""),
                "note": update.get("note", ""),
                "ts": ts,
                "status": "pending",
            },
        )

    result = CuratorResult(
        job_id=job_id,
        answered_goal=answered_goal,
        drift_summary=drift_summary,
        recurring_issues=recurring_issues,
        memory_updates=memory_updates,
        ts=ts,
    )

    print(
        f"[curator {job_id}] answered_goal={answered_goal}"
        f"  issues={len(recurring_issues)}"
        f"  memory_updates={len(memory_updates)}"
    )

    return result
