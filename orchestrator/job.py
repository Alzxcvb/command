"""Job orchestration: spawn an orchestrator agent, parse its breakdown, fan out sub-agents."""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from agents.lifecycle import spawn_agent  # noqa: E402
from agents.lifecycle import _read_meta as _read_agent_meta  # noqa: E402
from agents.lifecycle import _write_meta as _write_agent_meta  # noqa: E402
from agents.registry import get_agent, get_result  # noqa: E402
from core.estimator import estimate  # noqa: E402
from orchestrator import observations  # noqa: E402
from orchestrator.breakdown import Breakdown, parse_breakdown, validate  # noqa: E402

STATE_ROOT = _REPO_ROOT / "state"
JOBS_DIR = STATE_ROOT / "jobs"
PROMPT_PATH = _REPO_ROOT / "orchestrator" / "prompts" / "orchestrator.md"
BUDGET_LIMITS_PATH = _REPO_ROOT / "config" / "budget_limits.yaml"

_PLANNING_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"

_PLANNING_JUDGE_SYSTEM = (
    "You are reviewing an AI agent spawn plan before execution. "
    "The original task is shown, along with the proposed sub-tasks.\n"
    "Evaluate:\n"
    "1. Coverage: does the plan address all parts of the original task?\n"
    "2. Decomposition: are sub-tasks independent enough to run in parallel, "
    "or do hidden dependencies exist?\n"
    "3. Budget realism: do the estimated token budgets match the complexity "
    "of each sub-task?\n"
    "4. Completeness: will the combined outputs, if successful, actually "
    "fulfill the original task?\n"
    'Return JSON only (no markdown fences): {"approved": bool, '
    '"flags": [list of issue strings], "confidence": 0.0-1.0}'
)


def _strip_judge_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start, end = 1, len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()
    return text


def judge_spawn_plan(
    original_task: str,
    spawn_plan: list,
    estimated_budgets: dict,
) -> dict:
    """Evaluate a spawn plan before agents fire. Returns approved/flags/confidence.

    Defaults to approved=True on any error so the job is never hard-blocked.
    """
    _default: dict = {"approved": True, "flags": [], "confidence": 0.0}
    try:
        from router.providers import OpenRouterProvider
        provider = OpenRouterProvider()
    except Exception:
        return _default

    spawns_text = ""
    for i, s in enumerate(spawn_plan, 1):
        task_str = getattr(s, "task", "")
        est = estimated_budgets.get(task_str)
        est_str = f", estimated={est}" if est is not None else ""
        spawns_text += (
            f"{i}. runtime={getattr(s, 'runtime', '?')} "
            f"budget={getattr(s, 'budget_tokens', '?')}{est_str}\n"
            f"   task: {task_str[:300]}\n"
        )

    prompt = (
        f"## Original task\n{original_task[:1000]}\n\n"
        f"## Proposed spawn plan ({len(spawn_plan)} sub-agents)\n{spawns_text}"
    )

    try:
        raw, _ = provider.call_raw(
            _PLANNING_JUDGE_MODEL,
            prompt,
            system_prompt=_PLANNING_JUDGE_SYSTEM,
            max_tokens=512,
        )
        cleaned = _strip_judge_fences(raw)
        result = json.loads(cleaned)
        return {
            "approved": bool(result.get("approved", True)),
            "flags": list(result.get("flags", [])),
            "confidence": float(result.get("confidence", 0.0)),
        }
    except Exception:
        return _default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _write_meta(job_id: str, meta: dict) -> None:
    (_job_dir(job_id) / "meta.json").write_text(json.dumps(meta, indent=2))


def _read_meta(job_id: str) -> dict:
    return json.loads((_job_dir(job_id) / "meta.json").read_text())


def _update_meta(job_id: str, **changes) -> dict:
    meta = _read_meta(job_id)
    meta.update(changes)
    meta["updated_at"] = _now_iso()
    _write_meta(job_id, meta)
    return meta


def _max_concurrent() -> int:
    try:
        cfg = yaml.safe_load(BUDGET_LIMITS_PATH.read_text())
        return int(cfg.get("max_concurrent_agents", 4))
    except Exception:
        return 4


def _wait_for_agent(agent_id: str, poll: float = 1.0, timeout: float = 1800) -> dict:
    start = time.time()
    while True:
        meta = get_agent(agent_id)
        if meta and meta["status"] not in ("starting", "running"):
            return meta
        if time.time() - start > timeout:
            return meta or {"status": "timeout", "agent_id": agent_id}
        time.sleep(poll)


def _build_orchestrator_prompt(goal: str, total_budget: int, metered_cap: float) -> str:
    return (
        f"## Job goal\n{goal}\n\n"
        f"## Job budget\n- total tokens across all sub-agents: {total_budget}\n"
        f"- total metered USD cap (opencode only): ${metered_cap:.4f}\n\n"
        f"Now emit your <spawn> tags and <done/>."
    )


def start_job(
    goal: str,
    total_budget_tokens: int = 100_000,
    total_metered_cap_usd: float = 0.50,
    orchestrator_model: str = "sonnet",
    orchestrator_budget: int = 8000,
    dry_run: bool = False,
    project_hint: str = "",
    model_pin: str = "",
    use_prompt_pipeline: bool = False,
) -> str:
    """Spawn orchestrator → parse breakdown → fan out sub-agents → return job_id.

    Blocks until all sub-agents complete (or fail). Use dry_run=True to stop after parsing.
    project_hint groups jobs in historical analytics (e.g. "polycrisis", "mdac").
    model_pin overrides routing for any spawn with no explicit model in its breakdown tag.
    use_prompt_pipeline runs the 3-stage Architect/Reviewer/Gate pipeline on the goal first.
    """
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Optional: refine goal through the prompt pipeline before orchestrating
    pipeline_result = None
    effective_goal = goal
    if use_prompt_pipeline:
        try:
            from orchestrator.prompt_pipeline import run_prompt_pipeline
            pipeline_result = run_prompt_pipeline(goal)
            if pipeline_result.approved:
                effective_goal = pipeline_result.optimized_prompt
            (job_dir / "prompt_pipeline.json").write_text(
                json.dumps({
                    "original_goal": goal,
                    "optimized_prompt": pipeline_result.optimized_prompt,
                    "quality_score": pipeline_result.quality_score,
                    "drift_flags": pipeline_result.drift_flags,
                    "approved": pipeline_result.approved,
                    "rationale": pipeline_result.architect_rationale,
                }, indent=2)
            )
            print(f"[job {job_id}] prompt pipeline: score={pipeline_result.quality_score} approved={pipeline_result.approved}")
        except Exception as e:
            print(f"[job {job_id}] prompt pipeline skipped: {e!r}")

    meta = {
        "job_id": job_id,
        "goal": goal,
        "effective_goal": effective_goal,
        "project_hint": project_hint,
        "model_pin": model_pin,
        "use_prompt_pipeline": use_prompt_pipeline,
        "status": "orchestrating",
        "total_budget_tokens": total_budget_tokens,
        "total_metered_cap_usd": total_metered_cap_usd,
        "orchestrator_agent_id": None,
        "child_agent_ids": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _write_meta(job_id, meta)

    system_prompt = PROMPT_PATH.read_text()
    user_prompt = _build_orchestrator_prompt(effective_goal, total_budget_tokens, total_metered_cap_usd)

    orch_id = spawn_agent(
        task=user_prompt,
        runtime_name="claude_code",
        system_prompt=system_prompt,
        budget_tokens=orchestrator_budget,
        model=orchestrator_model,
        parent_job_id=job_id,
    )
    _update_meta(job_id, orchestrator_agent_id=orch_id)
    print(f"[job {job_id}] orchestrator spawned: {orch_id}")

    orch_meta = _wait_for_agent(orch_id)
    if orch_meta.get("status") != "completed":
        _update_meta(job_id, status=f"orchestrator_{orch_meta.get('status', 'failed')}")
        print(f"[job {job_id}] orchestrator did not complete: {orch_meta.get('status')}")
        return job_id

    result = get_result(orch_id) or {}
    output = result.get("final_text", "")
    (job_dir / "breakdown.md").write_text(output)

    bd = parse_breakdown(output)
    problems = validate(bd, total_budget_tokens=total_budget_tokens,
                        total_metered_cap_usd=total_metered_cap_usd)
    if bd.parse_errors:
        problems = problems + [f"parse: {e}" for e in bd.parse_errors]

    (job_dir / "tasks.md").write_text(_render_tasks_md(goal, bd, problems))

    if problems:
        print(f"[job {job_id}] breakdown problems:")
        for p in problems:
            print(f"  - {p}")
        if not bd.spawns:
            _update_meta(job_id, status="breakdown_invalid")
            return job_id

    if dry_run:
        _update_meta(job_id, status="dry_run", spawn_count=len(bd.spawns))
        print(f"[job {job_id}] dry-run: parsed {len(bd.spawns)} spawn(s); not dispatched.")
        return job_id

    # Phase 8: pre-execution decomposition review (soft gate — logs concerns, never blocks)
    try:
        from orchestrator.pre_execution_review import run_pre_execution_review
        pre_review = run_pre_execution_review(effective_goal, output)
        _update_meta(
            job_id,
            pre_review={
                "proceed": pre_review.proceed,
                "concerns": pre_review.concerns,
                "recommendation": pre_review.recommendation,
            },
        )
        if not pre_review.proceed:
            print(
                f"[job {job_id}] pre-execution review flagged concerns: {pre_review.concerns}"
            )
        else:
            print(f"[job {job_id}] pre-execution review passed")
    except Exception as e:
        print(f"[job {job_id}] pre-execution review skipped: {e!r}")

    # Planning judge gate: lightweight plan quality check before agents fire (soft gate — logs, never blocks)
    try:
        judge_result = judge_spawn_plan(effective_goal, bd.spawns, {})
        _update_meta(job_id, planning_judge=judge_result)
        if not judge_result.get("approved", True):
            flags = judge_result.get("flags", [])
            print(f"[job {job_id}] planning judge flagged: {flags}")
            rejections_dir = STATE_ROOT / "planning"
            rejections_dir.mkdir(parents=True, exist_ok=True)
            rejection_entry = {
                "ts": _now_iso(),
                "job_id": job_id,
                "task_summary": effective_goal[:200],
                "flags": flags,
                "confidence": judge_result.get("confidence", 0.0),
            }
            with (rejections_dir / "rejections.jsonl").open("a") as f:
                f.write(json.dumps(rejection_entry) + "\n")
        else:
            print(f"[job {job_id}] planning judge approved (confidence={judge_result.get('confidence', 0.0):.2f})")
    except Exception as e:
        print(f"[job {job_id}] planning judge skipped: {e!r}")

    print(f"[job {job_id}] dispatching {len(bd.spawns)} sub-agent(s) (max concurrent {_max_concurrent()})")
    child_ids = _dispatch(bd, job_id)
    _update_meta(job_id, status="running", child_agent_ids=child_ids, spawn_count=len(child_ids))

    final = _wait_for_children(child_ids, job_id=job_id, original_task=effective_goal)
    summary = _summarize(final)
    _update_meta(
        job_id,
        status="completed" if all(m.get("status") == "completed" for m in final) else "completed_with_failures",
        children_summary=summary,
    )
    (job_dir / "tasks.md").write_text(_render_tasks_md(goal, bd, problems, final))
    print(f"[job {job_id}] done. summary: {summary}")
    _append_analytics(job_id, project_hint, summary)
    _maybe_auto_improve(job_id)
    _maybe_run_curator(job_id)
    return job_id


def _append_analytics(job_id: str, project_hint: str, summary: dict) -> None:
    """Append one line per completed job to state/analytics.jsonl for the dashboard."""
    try:
        line = {
            "job_id": job_id,
            "project_hint": project_hint or "unlabeled",
            "task_count": int(summary.get("total", 0)),
            "total_tokens": int(summary.get("tokens_used", 0)),
            "cost_usd": float(summary.get("cost_usd", 0.0)),
            "ts": _now_iso(),
        }
        with (STATE_ROOT / "analytics.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception as e:
        print(f"[analytics error] {e!r}")


def _maybe_run_curator(job_id: str) -> None:
    """Run the memory curator in a background thread after every job completes."""
    import threading

    def _run():
        try:
            from orchestrator.curator import run_curator
            run_curator(job_id)
        except Exception as e:
            print(f"[curator error] {e!r}")

    threading.Thread(target=_run, daemon=True, name=f"curator-{job_id}").start()


def _maybe_auto_improve(job_id: str) -> None:
    """Fire Ralph if the failure buffer has enough new samples since last improvement.

    Runs in a background thread so the job return isn't blocked on the improver.
    """
    try:
        from orchestrator.ralph import MIN_SAMPLES_TO_TRIGGER, improve, pending_failure_count
    except Exception:
        return
    pending = pending_failure_count()
    if pending < MIN_SAMPLES_TO_TRIGGER:
        return
    import threading
    print(f"[job {job_id}] failure buffer at {pending} — triggering Ralph in background.")

    def _run():
        try:
            r = improve(force=False)
            print(f"[ralph] status={r.get('status')}  guide_sha={r.get('guide_sha')}")
        except Exception as e:
            print(f"[ralph error] {e!r}")

    threading.Thread(target=_run, daemon=True, name=f"ralph-auto-{job_id}").start()


def _dispatch(bd: Breakdown, job_id: str) -> list[str]:
    """Spawn sub-agents respecting MAX_CONCURRENT. Higher priority first (lower number)."""
    spawns_sorted = sorted(bd.spawns, key=lambda s: s.priority)
    max_c = _max_concurrent()
    in_flight: list[str] = []
    all_ids: list[str] = []

    job_meta = _read_meta(job_id)
    goal = job_meta.get("goal", "")
    orch_id = job_meta.get("orchestrator_agent_id")
    model_pin = job_meta.get("model_pin", "")

    for s in spawns_sorted:
        try:
            r = estimate(s.task)
        except Exception:
            r = None
        # Estimator is a floor only: it raises under-drafted budgets so agents
        # aren't hard-killed mid-task, but never lowers the orchestrator's draft.
        if r is not None and r.estimated_tokens > s.budget_tokens:
            budget_tokens = r.estimated_tokens
            budget_source = "estimator"
        else:
            budget_tokens = s.budget_tokens
            budget_source = "orchestrator"
        # Wait for a slot
        while sum(1 for a in in_flight if (get_agent(a) or {}).get("status") in ("starting", "running")) >= max_c:
            time.sleep(1.0)
        try:
            # model_pin overrides routing only when the breakdown has no explicit model
            effective_model = s.model or model_pin or None
            aid = spawn_agent(
                task=s.task,
                runtime_name=s.runtime,
                system_prompt="",
                budget_tokens=budget_tokens,
                model=effective_model,
                parent_job_id=job_id,
                estimated_tokens=r.estimated_tokens if r else 0,
                metered_cap_usd=s.metered_cap_usd,
            )
            print(f"  · spawned {aid}  runtime={s.runtime} model={s.model or '—'} budget={budget_tokens} ({budget_source})")
            m = _read_agent_meta(aid)
            m["budget_source"] = budget_source
            if r is not None:
                m["task_type"] = r.task_type
            _write_agent_meta(aid, m)
            observations.log_dispatch(
                job_id=job_id,
                agent_id=aid,
                goal=goal,
                drafted_task=s.task,
                runtime=s.runtime,
                model=s.model,
                budget_tokens=s.budget_tokens,
                estimated_tokens=r.estimated_tokens if r else 0,
                parent_agent_id=orch_id,
            )
            in_flight.append(aid)
            all_ids.append(aid)
        except Exception as e:
            print(f"  ! failed to spawn (runtime={s.runtime}): {e}")
    return all_ids


def _wait_for_children(child_ids: list[str], job_id: str = "", original_task: str = "") -> list[dict]:
    from orchestrator.watchdog import WatchdogAgent
    watchdog = WatchdogAgent()

    final: list[dict] = []
    for aid in child_ids:
        meta = _wait_for_agent(aid)
        result = get_result(aid) or {}
        meta_with_output = {**meta, "final_text": result.get("final_text", "")}
        final.append(meta)
        print(f"  ← {aid} → {meta.get('status')}  tokens={meta.get('tokens_used', 0)}")
        observations.log_outcome(
            job_id=job_id,
            agent_id=aid,
            status=meta.get("status", "unknown"),
            tokens_used=int(meta.get("tokens_used", 0)),
            cost_usd=float(meta.get("cost_usd", 0.0)),
            budget_overrun=bool(meta.get("budget_overrun", False)),
            final_text=result.get("final_text", "") or "",
        )
        if original_task:
            try:
                watchdog.check_alignment(job_id, original_task, [m for m in final] + [meta_with_output])
            except Exception as e:
                print(f"  [watchdog error] {e!r}")
    return final


def _summarize(final: list[dict]) -> dict:
    return {
        "total": len(final),
        "completed": sum(1 for m in final if m.get("status") == "completed"),
        "failed": sum(1 for m in final if m.get("status") in ("failed", "killed_over_budget")),
        "tokens_used": sum(int(m.get("tokens_used", 0)) for m in final),
        "cost_usd": round(sum(float(m.get("cost_usd", 0.0)) for m in final), 6),
    }


def _render_tasks_md(goal: str, bd: Breakdown, problems: list[str], final: Optional[list[dict]] = None) -> str:
    lines = [f"# Job tasks", f"\n## Goal\n{goal}\n", "## Breakdown"]
    for i, s in enumerate(bd.spawns, 1):
        lines.append(f"\n### {i}. [{s.runtime}/{s.model or 'default'}] (budget {s.budget_tokens}, prio {s.priority})")
        lines.append(s.task)
    if problems:
        lines.append("\n## Validation problems")
        for p in problems:
            lines.append(f"- {p}")
    if final:
        lines.append("\n## Outcomes")
        for m in final:
            lines.append(f"- `{m.get('agent_id')}` → **{m.get('status')}**  ({m.get('tokens_used', 0)} tok, ${m.get('cost_usd', 0):.4f})")
    return "\n".join(lines) + "\n"


def get_job(job_id: str) -> Optional[dict]:
    p = _job_dir(job_id) / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_jobs() -> list[dict]:
    if not JOBS_DIR.exists():
        return []
    out = []
    for d in sorted(JOBS_DIR.iterdir()):
        m = d / "meta.json"
        if m.exists():
            try:
                out.append(json.loads(m.read_text()))
            except Exception:
                pass
    return out
