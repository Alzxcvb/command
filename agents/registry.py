"""Read agent state from disk."""
from __future__ import annotations

import json
from typing import Optional

from .lifecycle import AGENTS_DIR


def list_agents() -> list[dict]:
    if not AGENTS_DIR.exists():
        return []
    out = []
    for d in sorted(AGENTS_DIR.iterdir()):
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                out.append(json.loads(meta_path.read_text()))
            except json.JSONDecodeError:
                continue
    return out


def get_agent(agent_id: str) -> Optional[dict]:
    meta_path = AGENTS_DIR / agent_id / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def get_checkpoint(agent_id: str) -> Optional[str]:
    cp = AGENTS_DIR / agent_id / "checkpoint.md"
    return cp.read_text() if cp.exists() else None


def get_result(agent_id: str) -> Optional[dict]:
    r = AGENTS_DIR / agent_id / "result.json"
    if not r.exists():
        return None
    return json.loads(r.read_text())


def get_log(agent_id: str, stream: str = "stdout", tail_chars: int = 4000) -> str:
    p = AGENTS_DIR / agent_id / f"{stream}.log"
    if not p.exists():
        return ""
    txt = p.read_text()
    return txt[-tail_chars:] if len(txt) > tail_chars else txt
