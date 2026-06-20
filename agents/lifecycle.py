"""Spawn, monitor, kill agents. State persisted to state/agents/<id>/."""
from __future__ import annotations

import json
import os
import re
import signal
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .ids import agent_dir as _validated_agent_dir

from .runtimes.base import (
    CheckpointMarker,
    Done,
    Error,
    Runtime,
    TextChunk,
    TokenUsage,
    ToolCall,
)
from .runtimes.claude_code import ClaudeCodeRuntime
from .runtimes.codex import CodexRuntime
from .runtimes.ollama import OllamaRuntime
from .runtimes.opencode import OpenCodeRuntime

STATE_ROOT = Path(__file__).resolve().parent.parent / "state"
AGENTS_DIR = STATE_ROOT / "agents"

_RUNTIMES: dict[str, type[Runtime]] = {
    "claude_code": ClaudeCodeRuntime,
    "codex": CodexRuntime,
    "opencode": OpenCodeRuntime,
    "ollama": OllamaRuntime,
}


def register_runtime(name: str, cls: type[Runtime]) -> None:
    _RUNTIMES[name] = cls


def available_runtimes() -> list[str]:
    return list(_RUNTIMES.keys())


def runtime_is_metered(name: str) -> bool:
    runtime_cls = _RUNTIMES.get(name)
    return bool(runtime_cls and getattr(runtime_cls, "metered", False))


_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9\-_]{20,})", re.IGNORECASE),          # OpenAI / generic sk- keys
    re.compile(r"(AKIA[0-9A-Z]{16})", re.IGNORECASE),                  # AWS access key IDs
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9\-_]{20,})", re.IGNORECASE),
]


def _redact(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: m.group(0)[:4] + "***REDACTED***", text)
    return text


def _redact_state(value):
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_state(item) for key, item in value.items()}
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_dir(agent_id: str) -> Path:
    return _validated_agent_dir(AGENTS_DIR, agent_id)


def _read_meta(agent_id: str) -> dict:
    return json.loads((_agent_dir(agent_id) / "meta.json").read_text())


def _write_meta(agent_id: str, meta: dict) -> None:
    target = _agent_dir(agent_id) / "meta.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(json.dumps(_redact_state(meta), indent=2))
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _update_meta(agent_id: str, **changes) -> dict:
    meta = _read_meta(agent_id)
    meta.update(changes)
    meta["updated_at"] = _now_iso()
    _write_meta(agent_id, meta)
    return meta


def _append_log(agent_id: str, line: str, stream: str = "stdout") -> None:
    with (_agent_dir(agent_id) / f"{stream}.log").open("a") as f:
        f.write(_redact(line) + "\n")


def spawn_agent(
    task: str,
    runtime_name: str = "claude_code",
    system_prompt: str = "",
    budget_tokens: int = 10000,
    model: Optional[str] = None,
    parent_job_id: Optional[str] = None,
    estimated_tokens: int = 0,
    metered_cap_usd: float = 0.0,
) -> str:
    """Spawn an agent. Returns agent_id immediately. Monitor runs in background thread."""
    if runtime_name not in _RUNTIMES:
        raise ValueError(f"Unknown runtime: {runtime_name}. Available: {available_runtimes()}")

    agent_id = f"agt_{uuid.uuid4().hex[:10]}"
    work_dir = _agent_dir(agent_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    runtime_cls = _RUNTIMES[runtime_name]
    runtime = runtime_cls(model=model) if model else runtime_cls()

    meta = {
        "agent_id": agent_id,
        "task": task,
        "runtime": runtime_name,
        "model": getattr(runtime, "model", None),
        "status": "starting",
        "tokens_used": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "budget_tokens": budget_tokens,
        "estimated_tokens": estimated_tokens,
        "metered": runtime.metered,
        "metered_cap_usd": metered_cap_usd,
        "parent_job_id": parent_job_id,
        "system_prompt_preview": (system_prompt or "")[:200],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "pid": None,
        "injected_messages": [],
    }
    _write_meta(agent_id, meta)
    (work_dir / "checkpoint.md").write_text(
        f"# Agent {agent_id} Checkpoint\n\n## Task\n{_redact(task)}\n\n## Status\nstarting\n\n## Notes\n"
    )

    proc = runtime.spawn(task=task, system_prompt=system_prompt, agent_id=agent_id, work_dir=work_dir)
    _update_meta(agent_id, status="running", pid=proc.pid)

    t = threading.Thread(
        target=_monitor,
        args=(agent_id, runtime, proc, budget_tokens),
        daemon=True,
        name=f"monitor-{agent_id}",
    )
    t.start()
    return agent_id


def _write_result(agent_id: str, status: str, final_text: str = "", error: Optional[str] = None) -> None:
    ts = _now_iso()
    data = json.dumps({
        "status": status,
        "final_text": _redact(final_text),
        "error": _redact(error) if error else None,
        "completed_at": ts,
    }, indent=2)
    target = _agent_dir(agent_id) / "result.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    cp = _agent_dir(agent_id) / "checkpoint.md"
    if cp.exists():
        snippet = _redact(final_text or error or "")[:300].replace("\n", " ")
        cp.write_text(cp.read_text() + f"\n[{ts}] status → **{status}**\n> {snippet}\n")


def _monitor(agent_id: str, runtime: Runtime, proc, budget_tokens: int) -> None:
    try:
        for evt in runtime.stream_events(proc):
            if isinstance(evt, TextChunk):
                _append_log(agent_id, evt.text, "stdout")
            elif isinstance(evt, ToolCall):
                _append_log(agent_id, f"[tool] {evt.name}: {json.dumps(evt.input)[:200]}", "stdout")
            elif isinstance(evt, TokenUsage):
                meta = _read_meta(agent_id)
                meta["input_tokens"] += evt.input_tokens
                meta["output_tokens"] += evt.output_tokens
                meta["tokens_used"] = meta["input_tokens"] + meta["output_tokens"]
                meta["cost_usd"] = round(meta["cost_usd"] + evt.cost_usd, 6)
                meta["updated_at"] = _now_iso()
                _write_meta(agent_id, meta)
                if runtime.metered and evt.cost_usd > 0:
                    try:
                        from core.metered_ledger import record_metered_spend
                        record_metered_spend(agent_id, evt.cost_usd)
                    except Exception as e:
                        _append_log(agent_id, f"[ledger error] {e}", "stderr")
                # Budget handling: soft overage flags for Ralph review but keeps the
                # work; hard overage (>=2.5x estimate, or 3x budget if no estimate)
                # kills because the estimator is almost certainly wrong.
                used = meta["tokens_used"]
                est = meta.get("estimated_tokens") or 0
                hard_cap = int(2.5 * est) if est > 0 else (3 * budget_tokens if budget_tokens > 0 else 0)
                if budget_tokens > 0 and used >= budget_tokens and not meta.get("budget_overrun"):
                    _append_log(
                        agent_id,
                        f"[budget] tokens_used={used:,} exceeded cap={budget_tokens:,} — "
                        f"NOT killing (preserving work). Flagging for Ralph review.",
                        "stderr",
                    )
                    _update_meta(agent_id, budget_overrun=True)
                if hard_cap > 0 and used >= hard_cap:
                    _append_log(
                        agent_id,
                        f"[budget] tokens_used={used:,} >= hard_cap={hard_cap:,} "
                        f"(2.5x est={est} / 3x budget={budget_tokens}); killing — "
                        f"likely estimator bug or runaway loop.",
                        "stderr",
                    )
                    runtime.kill(proc)
                    _update_meta(agent_id, status="killed_over_budget", budget_overrun=True)
                    _write_result(
                        agent_id, "killed_over_budget",
                        error=f"hard cap {hard_cap} tokens ({used:,} used)",
                    )
                    return
            elif isinstance(evt, CheckpointMarker):
                _append_log(agent_id, f"[checkpoint] {evt.note}", "stdout")
            elif isinstance(evt, Done):
                final_status = evt.status
                meta = _read_meta(agent_id)
                if meta.get("budget_overrun") and evt.status == "completed":
                    final_status = "completed_over_budget"
                _update_meta(agent_id, status=final_status)
                _write_result(agent_id, final_status, evt.final_text, evt.error)
                return
            elif isinstance(evt, Error):
                _append_log(agent_id, evt.message, "stderr")
                _update_meta(agent_id, status="failed")
                _write_result(agent_id, "failed", error=evt.message)
                return
    except Exception as e:
        _append_log(agent_id, f"[monitor exception] {e!r}", "stderr")
        _update_meta(agent_id, status="failed")
        _write_result(agent_id, "failed", error=str(e))


def kill_agent(agent_id: str) -> bool:
    try:
        meta = _read_meta(agent_id)
    except FileNotFoundError:
        return False
    pid = meta.get("pid")
    if not pid or meta.get("status") not in ("running", "starting"):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _update_meta(agent_id, status="killed")
        _write_result(agent_id, "killed", error="killed by user")
        return True
    except ProcessLookupError:
        _update_meta(agent_id, status="killed")
        return True
    except Exception:
        return False


def inject_message(agent_id: str, message: str) -> bool:
    try:
        meta = _read_meta(agent_id)
    except FileNotFoundError:
        return False
    redacted_message = _redact(message)
    meta.setdefault("injected_messages", []).append({
        "message": redacted_message,
        "timestamp": _now_iso(),
        "consumed": False,
    })
    _write_meta(agent_id, meta)
    cp = _agent_dir(agent_id) / "checkpoint.md"
    if cp.exists():
        cp.write_text(cp.read_text() + f"\n[{_now_iso()}] /btw queued: {redacted_message}\n")
    return True


def consume_pending_injects(agent_id: str) -> list[str]:
    """Return all pending /btw messages and mark them consumed. Older entries first."""
    try:
        meta = _read_meta(agent_id)
    except FileNotFoundError:
        return []
    pending: list[str] = []
    for entry in meta.get("injected_messages", []):
        if not entry.get("consumed"):
            pending.append(entry["message"])
            entry["consumed"] = True
            entry["consumed_at"] = _now_iso()
    if pending:
        _write_meta(agent_id, meta)
    return pending


def continue_agent(agent_id: str, additional_message: str = "") -> Optional[str]:
    """Spawn a follow-up agent that inherits the original task + previous result + queued /btw msgs.

    Returns the new agent_id, or None if the source agent is missing.
    """
    try:
        meta = _read_meta(agent_id)
    except FileNotFoundError:
        return None

    if additional_message:
        inject_message(agent_id, additional_message)

    pending = consume_pending_injects(agent_id)

    prev_result = ""
    res_path = _agent_dir(agent_id) / "result.json"
    if res_path.exists():
        try:
            prev_result = json.loads(res_path.read_text()).get("final_text", "") or ""
        except Exception:
            prev_result = ""

    parts = [
        f"## Original task\n{meta['task']}",
    ]
    if prev_result:
        parts.append("## Your previous response\n" + prev_result[:4000])
    if pending:
        parts.append("## Updates from the user (apply these now)\n" + "\n\n".join(f"- {m}" for m in pending))
    if not pending and not additional_message:
        parts.append("## Continue\nThe user has asked you to continue or refine. Improve on your previous response.")
    parts.append("Now produce an updated response that incorporates the updates.")
    new_task = "\n\n".join(parts)

    new_id = spawn_agent(
        task=new_task,
        runtime_name=meta["runtime"],
        system_prompt="",
        budget_tokens=meta.get("budget_tokens") or 5000,
        model=meta.get("model"),
        parent_job_id=meta.get("parent_job_id"),
    )

    # Cross-link
    new_meta = _read_meta(new_id)
    new_meta["continued_from"] = agent_id
    _write_meta(new_id, new_meta)
    meta["continued_as"] = new_id
    _write_meta(agent_id, meta)
    cp = _agent_dir(agent_id) / "checkpoint.md"
    if cp.exists():
        cp.write_text(cp.read_text() + f"\n[{_now_iso()}] continued as {new_id} (consumed {len(pending)} msg)\n")
    return new_id
