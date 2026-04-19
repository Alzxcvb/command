"""Retry a failed sub-agent: spawn a replacement and log the failure signal.

A retry is the strongest failure signal Ralph sees. If the orchestrator (or
the user) decided the original output wasn't good enough to keep, the prompt
that produced it is a candidate for improvement.

The replacement inherits the *original task text* by default — because we're
testing whether re-running the same instruction helps, not whether a reworded
instruction helps. If the caller wants to rewrite the task, they pass
`new_task` explicitly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.lifecycle import spawn_agent  # noqa: E402
from agents.registry import get_agent  # noqa: E402
from orchestrator import observations  # noqa: E402

STATE_ROOT = _REPO_ROOT / "state"
AGENTS_DIR = STATE_ROOT / "agents"


def _agent_meta_path(agent_id: str) -> Path:
    return AGENTS_DIR / agent_id / "meta.json"


def _read_meta(agent_id: str) -> Optional[dict]:
    p = _agent_meta_path(agent_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _write_meta(agent_id: str, meta: dict) -> None:
    _agent_meta_path(agent_id).write_text(json.dumps(meta, indent=2))


def retry_agent(
    failed_agent_id: str,
    new_task: Optional[str] = None,
    rationale: str = "",
) -> Optional[str]:
    """Spawn a replacement for a failed agent.

    Returns the new agent_id, or None if the original is missing / can't be retried.
    """
    orig = _read_meta(failed_agent_id)
    if not orig:
        return None

    task = new_task or orig.get("task", "")
    replacement_id = spawn_agent(
        task=task,
        runtime_name=orig.get("runtime", "claude_code"),
        system_prompt="",
        budget_tokens=orig.get("budget_tokens") or 5000,
        model=orig.get("model"),
        parent_job_id=orig.get("parent_job_id"),
        estimated_tokens=orig.get("estimated_tokens") or 0,
        metered_cap_usd=orig.get("metered_cap_usd") or 0.0,
    )

    # Cross-link metas so the dashboard and job view can follow the chain.
    new_meta = _read_meta(replacement_id) or {}
    new_meta["retry_of"] = failed_agent_id
    _write_meta(replacement_id, new_meta)
    orig["retried_as"] = replacement_id
    _write_meta(failed_agent_id, orig)

    observations.log_retry(
        job_id=orig.get("parent_job_id", ""),
        failed_agent_id=failed_agent_id,
        replacement_agent_id=replacement_id,
        old_task=orig.get("task", ""),
        new_task=task,
        rationale=rationale,
    )
    return replacement_id
