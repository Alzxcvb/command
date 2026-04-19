"""Observation log: every orchestrator dispatch + its eventual outcome.

Every `<spawn/>` drafted by the orchestrator appends a row here with the task
string, chosen runtime/model/budget, and the guide-version SHA under which it
was drafted. When the child completes (or is retried or thumbed-down), an
outcome event is appended to the same file.

Ralph's improver reads this log and keeps only the samples that signal a
*prompt failure* — retried, failed, over-budget, or explicitly thumbed-down.
Successful prompts never enter Ralph's buffer. That's the filter.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
OBSERVATIONS_PATH = _REPO_ROOT / "state" / "ralph" / "observations.jsonl"
GUIDE_PATH = _REPO_ROOT / "orchestrator" / "prompts" / "orchestrator.md"

ObsKind = Literal["dispatch", "outcome", "retry", "thumb"]
FailReason = Literal[
    "failed",
    "killed_over_budget",
    "completed_over_budget",
    "retried",
    "thumb_down",
]


@dataclass
class DispatchObs:
    kind: Literal["dispatch"] = "dispatch"
    timestamp: str = ""
    job_id: str = ""
    agent_id: str = ""
    parent_agent_id: Optional[str] = None
    goal: str = ""
    drafted_task: str = ""
    runtime: str = ""
    model: Optional[str] = None
    budget_tokens: int = 0
    estimated_tokens: int = 0
    guide_sha: str = ""


@dataclass
class OutcomeObs:
    kind: Literal["outcome"] = "outcome"
    timestamp: str = ""
    job_id: str = ""
    agent_id: str = ""
    status: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    budget_overrun: bool = False
    final_text: str = ""
    fail_reasons: list[str] = field(default_factory=list)


@dataclass
class RetryObs:
    kind: Literal["retry"] = "retry"
    timestamp: str = ""
    job_id: str = ""
    failed_agent_id: str = ""
    replacement_agent_id: str = ""
    old_task: str = ""
    new_task: str = ""
    rationale: str = ""


@dataclass
class ThumbObs:
    kind: Literal["thumb"] = "thumb"
    timestamp: str = ""
    job_id: str = ""
    agent_id: str = ""
    direction: Literal["up", "down"] = "down"
    comment: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir() -> None:
    OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)


def current_guide_sha() -> str:
    """Content hash of orchestrator.md at the moment a dispatch is logged.

    Using a content hash (not git sha) lets Ralph attribute observations to
    exactly the guide text that was in effect, even between commits.
    """
    if not GUIDE_PATH.exists():
        return ""
    return hashlib.sha256(GUIDE_PATH.read_bytes()).hexdigest()[:12]


def append(obs) -> None:
    _ensure_dir()
    with OBSERVATIONS_PATH.open("a") as f:
        f.write(json.dumps(asdict(obs), default=str) + "\n")


def read_all() -> list[dict]:
    if not OBSERVATIONS_PATH.exists():
        return []
    out = []
    for line in OBSERVATIONS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def failure_samples(since_guide_sha: Optional[str] = None) -> list[dict]:
    """Build Ralph's buffer: failure-tagged samples only.

    A sample is a dispatch joined to its outcome / retry / thumb, kept only
    if at least one fail signal is present. This is the whole point of the
    filter — successful prompts are dropped here and never consume Ralph's
    tokens.
    """
    events = read_all()
    dispatches: dict[str, dict] = {}
    outcomes: dict[str, dict] = {}
    retries_for: dict[str, list[dict]] = {}
    thumbs_for: dict[str, list[dict]] = {}

    for e in events:
        kind = e.get("kind")
        if kind == "dispatch":
            dispatches[e["agent_id"]] = e
        elif kind == "outcome":
            outcomes[e["agent_id"]] = e
        elif kind == "retry":
            retries_for.setdefault(e["failed_agent_id"], []).append(e)
        elif kind == "thumb":
            thumbs_for.setdefault(e["agent_id"], []).append(e)

    samples = []
    for agent_id, disp in dispatches.items():
        if since_guide_sha and disp.get("guide_sha") == since_guide_sha:
            # Already-seen samples when filtering for "new since last improvement"
            continue
        out = outcomes.get(agent_id)
        retries = retries_for.get(agent_id, [])
        thumbs = thumbs_for.get(agent_id, [])
        fail_reasons = []
        if out:
            if out.get("status") == "failed":
                fail_reasons.append("failed")
            if out.get("status") == "killed_over_budget":
                fail_reasons.append("killed_over_budget")
            if out.get("budget_overrun"):
                fail_reasons.append("completed_over_budget")
        if retries:
            fail_reasons.append("retried")
        if any(t.get("direction") == "down" for t in thumbs):
            fail_reasons.append("thumb_down")
        if not fail_reasons:
            continue  # <<< filter: successful prompts never enter the buffer
        samples.append({
            "agent_id": agent_id,
            "dispatch": disp,
            "outcome": out,
            "retries": retries,
            "thumbs": thumbs,
            "fail_reasons": fail_reasons,
        })
    return samples


def last_improvement_guide_sha() -> Optional[str]:
    """Read the guide SHA at the time of the last Ralph improvement commit."""
    marker = _REPO_ROOT / "state" / "ralph" / "last_improvement.json"
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text()).get("guide_sha")
    except Exception:
        return None


def record_improvement(guide_sha: str, improvement_agent_id: str, n_samples: int) -> None:
    _ensure_dir()
    (_REPO_ROOT / "state" / "ralph" / "last_improvement.json").write_text(json.dumps({
        "guide_sha": guide_sha,
        "improvement_agent_id": improvement_agent_id,
        "n_samples_consumed": n_samples,
        "timestamp": _now_iso(),
    }, indent=2))


def log_dispatch(*, job_id: str, agent_id: str, goal: str, drafted_task: str,
                 runtime: str, model: Optional[str], budget_tokens: int,
                 estimated_tokens: int = 0, parent_agent_id: Optional[str] = None) -> None:
    append(DispatchObs(
        timestamp=_now_iso(),
        job_id=job_id,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
        goal=goal,
        drafted_task=drafted_task,
        runtime=runtime,
        model=model,
        budget_tokens=budget_tokens,
        estimated_tokens=estimated_tokens,
        guide_sha=current_guide_sha(),
    ))


def log_outcome(*, job_id: str, agent_id: str, status: str,
                tokens_used: int, cost_usd: float,
                budget_overrun: bool = False, final_text: str = "") -> None:
    fail_reasons = []
    if status == "failed":
        fail_reasons.append("failed")
    if status == "killed_over_budget":
        fail_reasons.append("killed_over_budget")
    if budget_overrun:
        fail_reasons.append("completed_over_budget")
    append(OutcomeObs(
        timestamp=_now_iso(),
        job_id=job_id,
        agent_id=agent_id,
        status=status,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        budget_overrun=budget_overrun,
        final_text=final_text[:2000],
        fail_reasons=fail_reasons,
    ))


def log_retry(*, job_id: str, failed_agent_id: str, replacement_agent_id: str,
              old_task: str, new_task: str, rationale: str = "") -> None:
    append(RetryObs(
        timestamp=_now_iso(),
        job_id=job_id,
        failed_agent_id=failed_agent_id,
        replacement_agent_id=replacement_agent_id,
        old_task=old_task,
        new_task=new_task,
        rationale=rationale,
    ))


def log_thumb(*, job_id: str, agent_id: str, direction: str, comment: str = "") -> None:
    append(ThumbObs(
        timestamp=_now_iso(),
        job_id=job_id,
        agent_id=agent_id,
        direction=direction,
        comment=comment,
    ))
