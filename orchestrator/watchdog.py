"""Toby watchdog — mission alignment monitor for active jobs.

Sits beside the orchestrator (not in the execution chain). After each child agent
completes, checks whether the work so far is still aligned with the original task,
flags recurring failure patterns, and writes a memory note for future runs.

Never modifies job state, agent outputs, or the orchestrator guide.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_WATCHDOG_MODEL = "claude-sonnet-4-6"
_MAX_PAST_OBSERVATIONS = 5
_MAX_OUTPUT_CHARS = 400


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summarize_outputs(completed_outputs: list[dict]) -> str:
    parts = []
    for o in completed_outputs:
        agent_id = o.get("agent_id", "?")
        status = o.get("status", "?")
        text = (o.get("final_text") or "")[:_MAX_OUTPUT_CHARS]
        parts.append(f"- {agent_id} ({status}): {text}")
    return "\n".join(parts) if parts else "(none yet)"


class WatchdogAgent:
    """Lightweight Sonnet monitor that checks goal alignment after each child completes."""

    def __init__(self, state_dir: Optional[str] = None) -> None:
        self.obs_path = Path(state_dir or (_REPO_ROOT / "state" / "watchdog")) / "observations.jsonl"
        self.obs_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_past_observations(self, task_summary: str) -> list[dict]:
        if not self.obs_path.exists():
            return []
        lines = self.obs_path.read_text().strip().splitlines()
        relevant = []
        for line in lines[-50:]:
            try:
                entry = json.loads(line)
                if entry.get("task_summary", "")[:50] == task_summary[:50]:
                    relevant.append(entry)
            except Exception:
                continue
        return relevant[-_MAX_PAST_OBSERVATIONS:]

    def _call_model(self, prompt: str) -> str:
        try:
            from router.providers import AnthropicProvider
            provider = AnthropicProvider()
            raw, _ = provider.call_raw(
                _WATCHDOG_MODEL,
                prompt,
                system_prompt=(
                    "You are a watchdog agent monitoring an AI agent workforce. "
                    "Your job is to check alignment and flag issues — not to do work. "
                    "Be concise. Return valid JSON only."
                ),
                max_tokens=256,
            )
            return raw.strip()
        except Exception:
            return ""

    def check_alignment(
        self,
        job_id: str,
        original_task: str,
        completed_outputs: list[dict],
    ) -> dict:
        """Check alignment after a child completes. Writes one JSONL entry. Never raises."""
        task_summary = original_task[:100]
        past = self._load_past_observations(task_summary)
        past_text = (
            "\n".join(f"- {p.get('memory_note', '')}" for p in past if p.get("memory_note"))
            or "(no prior observations)"
        )

        prompt = (
            f"Original task: {original_task[:800]}\n\n"
            f"Work completed so far:\n{_summarize_outputs(completed_outputs)}\n\n"
            f"Past watchdog observations for similar tasks:\n{past_text}\n\n"
            "Answer in JSON:\n"
            '{"aligned": bool, "drift_note": string_or_null, '
            '"recurring_issue": bool, "memory_note": string_or_null}'
        )

        result: dict = {
            "aligned": True,
            "drift_note": None,
            "recurring_issue": False,
            "memory_note": None,
        }

        raw = self._call_model(prompt)
        if raw:
            try:
                parsed = json.loads(raw)
                result = {
                    "aligned": bool(parsed.get("aligned", True)),
                    "drift_note": parsed.get("drift_note") or None,
                    "recurring_issue": bool(parsed.get("recurring_issue", False)),
                    "memory_note": parsed.get("memory_note") or None,
                }
            except Exception:
                pass

        entry = {
            "ts": _now_iso(),
            "job_id": job_id,
            "task_summary": task_summary,
            "aligned": result["aligned"],
            "drift_note": result["drift_note"],
            "recurring_issue": result["recurring_issue"],
            "memory_note": result["memory_note"],
        }

        try:
            with self.obs_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        return result
