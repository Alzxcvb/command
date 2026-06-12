"""Analytics jsonl writer: one line per completed job."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import orchestrator.job as job


def test_append_analytics_writes_one_line_per_job(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "STATE_ROOT", tmp_path)
    summary = {"total": 3, "completed": 3, "failed": 0, "tokens_used": 42_000, "cost_usd": 0.12}

    job._append_analytics("job_aaa", "polycrisis", summary)
    job._append_analytics("job_bbb", "", summary)

    lines = (tmp_path / "analytics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first == {
        "job_id": "job_aaa",
        "project_hint": "polycrisis",
        "task_count": 3,
        "total_tokens": 42_000,
        "cost_usd": 0.12,
        "ts": first["ts"],
    }
    assert first["ts"]  # timestamp present

    second = json.loads(lines[1])
    assert second["project_hint"] == "unlabeled"


def test_append_analytics_never_raises(monkeypatch):
    # Pointing STATE_ROOT at a non-writable location must not break start_job's return path
    monkeypatch.setattr(job, "STATE_ROOT", Path("/nonexistent-root-dir"))
    job._append_analytics("job_ccc", "x", {})  # swallows the error, prints a warning
