"""Estimator integration in orchestrator/job.py:_dispatch().

The estimator acts as a floor on per-task budgets: it raises under-drafted
orchestrator budgets but never lowers them. budget_source records who won.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.estimator import EstimatorResult
from orchestrator.breakdown import Breakdown, SpawnInstruction
from orchestrator.job import _dispatch


def _estimator_result(estimated_tokens: int) -> EstimatorResult:
    return EstimatorResult(
        task_type="code",
        recommended_runtime="claude_code",
        recommended_model="sonnet",
        estimated_tokens=estimated_tokens,
        requires_metered=False,
    )


def _run_dispatch(drafted_budget: int, estimated_tokens: int):
    """Run _dispatch with one spawn; return (spawn kwargs, agent meta written, dispatch log kwargs)."""
    bd = Breakdown(spawns=[SpawnInstruction(
        task="write a parser",
        runtime="claude_code",
        budget_tokens=drafted_budget,
    )])
    captured = {}

    def fake_spawn(**kwargs):
        captured["spawn"] = kwargs
        return "agt_test000001"

    def fake_log_dispatch(**kwargs):
        captured["log"] = kwargs

    def fake_write_agent_meta(agent_id, meta):
        captured["agent_meta"] = (agent_id, meta)

    with patch("orchestrator.job.estimate", return_value=_estimator_result(estimated_tokens)), \
         patch("orchestrator.job.spawn_agent", side_effect=fake_spawn), \
         patch("orchestrator.job._read_meta", return_value={"goal": "g", "orchestrator_agent_id": "agt_orch"}), \
         patch("orchestrator.job._read_agent_meta", return_value={"agent_id": "agt_test000001"}), \
         patch("orchestrator.job._write_agent_meta", side_effect=fake_write_agent_meta), \
         patch("orchestrator.job.observations.log_dispatch", side_effect=fake_log_dispatch), \
         patch("orchestrator.job.get_agent", return_value={"status": "completed"}):
        ids = _dispatch(bd, "job_test")

    assert ids == ["agt_test000001"]
    return captured


def test_estimator_floor_raises_low_draft():
    captured = _run_dispatch(drafted_budget=5000, estimated_tokens=12000)
    assert captured["spawn"]["budget_tokens"] == 12000
    assert captured["spawn"]["estimated_tokens"] == 12000
    aid, meta = captured["agent_meta"]
    assert aid == "agt_test000001"
    assert meta["budget_source"] == "estimator"
    assert captured["log"]["estimated_tokens"] == 12000


def test_orchestrator_draft_kept_when_higher():
    captured = _run_dispatch(drafted_budget=20000, estimated_tokens=12000)
    assert captured["spawn"]["budget_tokens"] == 20000
    assert captured["spawn"]["estimated_tokens"] == 12000
    aid, meta = captured["agent_meta"]
    assert meta["budget_source"] == "orchestrator"


def test_estimator_failure_falls_back_silently():
    bd = Breakdown(spawns=[SpawnInstruction(
        task="write a parser",
        runtime="claude_code",
        budget_tokens=5000,
    )])
    captured = {}

    def fake_spawn(**kwargs):
        captured["spawn"] = kwargs
        return "agt_test000002"

    with patch("orchestrator.job.estimate", side_effect=RuntimeError("config missing")), \
         patch("orchestrator.job.spawn_agent", side_effect=fake_spawn), \
         patch("orchestrator.job._read_meta", return_value={"goal": "g", "orchestrator_agent_id": None}), \
         patch("orchestrator.job._read_agent_meta", return_value={}), \
         patch("orchestrator.job._write_agent_meta") as write_meta, \
         patch("orchestrator.job.observations.log_dispatch"), \
         patch("orchestrator.job.get_agent", return_value={"status": "completed"}):
        ids = _dispatch(bd, "job_test")

    assert ids == ["agt_test000002"]
    assert captured["spawn"]["budget_tokens"] == 5000
    assert captured["spawn"]["estimated_tokens"] == 0
    _, meta = write_meta.call_args[0]
    assert meta["budget_source"] == "orchestrator"
