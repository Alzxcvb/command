"""Centralized agent and job ID validation with path containment."""
from __future__ import annotations

import re
from pathlib import Path

AGENT_ID_RE = re.compile(r"^agt_[0-9a-f]{10}$")
JOB_ID_RE = re.compile(r"^job_[0-9a-f]{10}$")


def _safe_path(base_dir: Path, id_str: str, id_re: re.Pattern, label: str) -> Path:
    if not id_re.fullmatch(id_str):
        raise ValueError(f"invalid {label}: {id_str!r}")
    path = (base_dir / id_str).resolve()
    root = base_dir.resolve()
    if root not in path.parents:
        raise ValueError(f"{label} path escaped state root: {id_str!r}")
    return path


def agent_dir(base_dir: Path, agent_id: str) -> Path:
    return _safe_path(base_dir, agent_id, AGENT_ID_RE, "agent_id")


def job_dir(base_dir: Path, job_id: str) -> Path:
    return _safe_path(base_dir, job_id, JOB_ID_RE, "job_id")
