"""Command CLI: spawn / status / kill / btw."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

# Make repo root importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.lifecycle import (  # noqa: E402
    available_runtimes,
    continue_agent,
    inject_message,
    kill_agent,
    spawn_agent,
)
from agents.registry import (  # noqa: E402
    get_agent,
    get_checkpoint,
    get_log,
    get_result,
    list_agents,
)
from core.estimator import estimate  # noqa: E402
from core.metered_ledger import DEFAULT_DAILY_CAP_USD, get_today_spend  # noqa: E402


def _wait_loop(agent_id: str) -> int:
    print("[waiting] ", end="", flush=True)
    last = 0
    while True:
        time.sleep(2)
        meta = get_agent(agent_id)
        if not meta:
            print("\n[error] state vanished")
            return 1
        if meta["status"] not in ("starting", "running"):
            print(f"\n[done] status={meta['status']}  tokens={meta['tokens_used']:,}  cost=${meta['cost_usd']:.4f}")
            res = get_result(agent_id)
            if res and res.get("final_text"):
                print("\n--- final output ---")
                print(res["final_text"][:2000])
            return 0
        if meta["tokens_used"] != last:
            print(f"·{meta['tokens_used']:,}", end="", flush=True)
            last = meta["tokens_used"]
        else:
            print(".", end="", flush=True)


def cmd_spawn(args) -> int:
    est = None
    if not args.runtime or not args.model or args.budget is None:
        est = estimate(args.task)
        runtime = args.runtime or est.recommended_runtime
        model = args.model or est.recommended_model
        budget = args.budget if args.budget is not None else max(est.estimated_tokens * 2, 2000)
        print(f"[estimate] task_type={est.task_type}  runtime={runtime}  model={model}  est_tokens={est.estimated_tokens}  budget={budget}")
    else:
        runtime = args.runtime
        model = args.model
        budget = args.budget

    if runtime not in available_runtimes():
        print(f"error: runtime '{runtime}' not registered. available: {available_runtimes()}", file=sys.stderr)
        return 2

    sys_prompt = ""
    if args.system_prompt:
        sys_prompt = Path(args.system_prompt).read_text()

    agent_id = spawn_agent(
        task=args.task,
        runtime_name=runtime,
        system_prompt=sys_prompt,
        budget_tokens=budget,
        model=model,
        estimated_tokens=(est.estimated_tokens if est else 0),
    )
    print(f"[spawned] {agent_id}  → state/agents/{agent_id}/")

    if args.detach:
        if _handoff_to_commandd(agent_id):
            print(f"[detached] handed off to commandd — watchdog tracks {agent_id} after this process exits.")
        else:
            print("Warning: no commandd running — budget enforcement dies with this process. Start it with: python commandd.py &")
        return 0

    return _wait_loop(agent_id)


def _handoff_to_commandd(agent_id: str) -> bool:
    """Hand a detached agent to the commandd watchdog. True if the daemon accepted."""
    sock_path = _REPO_ROOT / "state" / "commandd.sock"
    if not sock_path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(sock_path))
            s.sendall(f"HANDOFF:{agent_id}\n".encode("utf-8"))
        return True
    except OSError:
        return False


def cmd_status(args) -> int:
    if args.agent:
        meta = get_agent(args.agent)
        if not meta:
            print(f"agent {args.agent} not found", file=sys.stderr)
            return 1
        print(json.dumps(meta, indent=2))
        cp = get_checkpoint(args.agent)
        if cp:
            print("\n--- checkpoint.md ---\n" + cp)
        if args.logs:
            log = get_log(args.agent, "stdout")
            if log:
                print("\n--- stdout (tail) ---\n" + log)
        res = get_result(args.agent)
        if res:
            print("\n--- result.json ---\n" + json.dumps(res, indent=2))
        return 0

    agents = list_agents()
    if not agents:
        print("(no agents yet — run: python -m cli spawn \"your task\")")
        return 0

    header = f"{'AGENT_ID':14}  {'STATUS':18}  {'RUNTIME':12}  {'MODEL':10}  {'TOKENS':>8} / {'CAP':<8}  TASK"
    print(header)
    print("-" * len(header))
    for a in agents:
        cap = a.get("budget_tokens") or 0
        used = a.get("tokens_used") or 0
        print(f"{a['agent_id']:14}  {a['status']:18}  {(a.get('runtime') or ''):12}  {(a.get('model') or ''):10}  {used:>8,} / {cap:<8,}  {(a.get('task') or '')[:60]}")

    spend = get_today_spend()
    print(f"\n[metered today] ${spend['total_usd']:.4f} / ${DEFAULT_DAILY_CAP_USD:.2f}  ({len(spend['by_agent'])} agents)")
    return 0


def cmd_kill(args) -> int:
    ok = kill_agent(args.agent)
    print(f"{'killed' if ok else 'failed to kill (already done?)'}: {args.agent}")
    return 0 if ok else 1


def cmd_btw(args) -> int:
    ok = inject_message(args.agent, args.message)
    print(f"{'queued' if ok else 'failed'}: /btw {args.agent} \"{args.message}\"")
    if ok and args.then_continue:
        new_id = continue_agent(args.agent)
        if new_id:
            print(f"[continued] {args.agent} → {new_id}")
            return _wait_loop(new_id) if not args.detach else 0
        print(f"[error] could not continue {args.agent}")
        return 1
    return 0 if ok else 1


def cmd_continue(args) -> int:
    new_id = continue_agent(args.agent, additional_message=args.message or "")
    if not new_id:
        print(f"[error] agent {args.agent} not found", file=sys.stderr)
        return 1
    print(f"[continued] {args.agent} → {new_id}")
    return _wait_loop(new_id) if not args.detach else 0


def cmd_orchestrate(args) -> int:
    from orchestrator.job import start_job
    job_id = start_job(
        goal=args.goal,
        total_budget_tokens=args.budget_total,
        total_metered_cap_usd=args.metered_cap,
        orchestrator_model=args.orchestrator_model,
        orchestrator_budget=args.orchestrator_budget,
        dry_run=args.dry_run,
        project_hint=args.project or "",
        model_pin=args.model_pin or "",
        use_prompt_pipeline=args.pipeline,
    )
    print(f"\n[job] {job_id}  → state/jobs/{job_id}/")
    return 0


def cmd_jobs(args) -> int:
    from orchestrator.job import get_job, list_jobs
    if args.job:
        meta = get_job(args.job)
        if not meta:
            print(f"job {args.job} not found", file=sys.stderr)
            return 1
        print(json.dumps(meta, indent=2))
        tasks_path = Path("state/jobs") / args.job / "tasks.md"
        if tasks_path.exists():
            print("\n--- tasks.md ---\n" + tasks_path.read_text())
        return 0
    jobs = list_jobs()
    if not jobs:
        print("(no jobs yet — run: python -m cli orchestrate \"your goal\")")
        return 0
    header = f"{'JOB_ID':14}  {'STATUS':22}  {'CHILDREN':>8}  GOAL"
    print(header)
    print("-" * len(header))
    for j in jobs:
        print(f"{j['job_id']:14}  {j['status']:22}  {len(j.get('child_agent_ids') or []):>8}  {j['goal'][:70]}")
    return 0


def cmd_retry(args) -> int:
    from orchestrator.retry import retry_agent
    new_id = retry_agent(args.agent, rationale=args.reason or "")
    if not new_id:
        print(f"[error] could not retry {args.agent}", file=sys.stderr)
        return 1
    print(f"[retried] {args.agent} → {new_id}")
    return _wait_loop(new_id) if not args.detach else 0


def cmd_thumb(args) -> int:
    from orchestrator import observations
    meta = get_agent(args.agent) or {}
    observations.log_thumb(
        job_id=meta.get("parent_job_id") or "",
        agent_id=args.agent,
        direction=args.direction,
        comment=args.comment or "",
    )
    print(f"[thumb {args.direction}] {args.agent}")
    return 0


def cmd_ralph_improve(args) -> int:
    from orchestrator.ralph import improve
    result = improve(
        model=args.model,
        budget_tokens=args.budget,
        force=args.force,
        min_samples=args.min_samples,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in ("applied", "no_change", "skipped") else 1


def cmd_estimate(args) -> int:
    est = estimate(args.task)
    print(json.dumps({
        "task_type": est.task_type,
        "recommended_runtime": est.recommended_runtime,
        "recommended_model": est.recommended_model,
        "estimated_tokens": est.estimated_tokens,
        "requires_metered": est.requires_metered,
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="cli", description="Command — agent orchestration CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spawn", help="Spawn a single agent (blocks until done by default)")
    s.add_argument("task")
    s.add_argument("--runtime", default=None, help="claude_code (default; more added in step 8)")
    s.add_argument("--model", default=None, help="e.g. sonnet, haiku, opus")
    s.add_argument("--budget", type=int, default=None, help="token budget cap")
    s.add_argument("--system-prompt", default=None, help="path to system prompt .md file")
    s.add_argument("--detach", action="store_true", help="don't wait; spawn and return")
    s.set_defaults(func=cmd_spawn)

    s = sub.add_parser("status", help="Show agent(s) status")
    s.add_argument("--agent", default=None, help="if omitted, lists all agents")
    s.add_argument("--logs", action="store_true", help="include stdout tail")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("kill", help="Kill an agent")
    s.add_argument("agent")
    s.set_defaults(func=cmd_kill)

    s = sub.add_parser("btw", help="Inject a /btw message (queued for next continue)")
    s.add_argument("agent")
    s.add_argument("message")
    s.add_argument("--continue", dest="then_continue", action="store_true",
                   help="immediately spawn a continue-turn that consumes the queued msg")
    s.add_argument("--detach", action="store_true", help="don't block on the continue turn")
    s.set_defaults(func=cmd_btw)

    s = sub.add_parser("continue", help="Spawn a follow-up turn for an agent (consumes queued /btw msgs)")
    s.add_argument("agent")
    s.add_argument("--message", default=None, help="optional message to also queue + consume")
    s.add_argument("--detach", action="store_true", help="don't block on the continue turn")
    s.set_defaults(func=cmd_continue)

    s = sub.add_parser("orchestrate", help="Break a goal into parallel sub-agents")
    s.add_argument("goal")
    s.add_argument("--project", default=None, help="project label for historical analytics grouping")
    s.add_argument("--budget-total", type=int, default=100_000, dest="budget_total")
    s.add_argument("--metered-cap", type=float, default=0.50, dest="metered_cap")
    s.add_argument("--orchestrator-model", default="sonnet", dest="orchestrator_model")
    s.add_argument("--orchestrator-budget", type=int, default=8000, dest="orchestrator_budget")
    s.add_argument("--dry-run", action="store_true", help="parse breakdown but don't dispatch sub-agents")
    s.add_argument("--model-pin", default=None, dest="model_pin",
                   help="pin all child agents to this model (e.g. haiku, sonnet, opus)")
    s.add_argument("--pipeline", action="store_true",
                   help="run Architect/Council prompt pipeline on goal before orchestrating")
    s.set_defaults(func=cmd_orchestrate)

    s = sub.add_parser("jobs", help="List jobs or show one job's details")
    s.add_argument("--job", default=None)
    s.set_defaults(func=cmd_jobs)

    s = sub.add_parser("retry", help="Re-spawn a failed agent (emits 'retried' failure signal)")
    s.add_argument("agent")
    s.add_argument("--reason", default=None, help="why this is being retried (for the observation log)")
    s.add_argument("--detach", action="store_true")
    s.set_defaults(func=cmd_retry)

    s = sub.add_parser("thumb", help="👍/👎 an agent's output (feeds Ralph buffer)")
    s.add_argument("agent")
    s.add_argument("direction", choices=["up", "down"])
    s.add_argument("--comment", default=None)
    s.set_defaults(func=cmd_thumb)

    s = sub.add_parser("ralph-improve", help="Run Ralph once: read failure buffer → rewrite orchestrator guide")
    s.add_argument("--model", default="sonnet")
    s.add_argument("--budget", type=int, default=15000)
    s.add_argument("--force", action="store_true", help="run even if below --min-samples")
    s.add_argument("--min-samples", type=int, default=3, dest="min_samples")
    s.set_defaults(func=cmd_ralph_improve)

    s = sub.add_parser("estimate", help="Show estimator decision for a task without spawning")
    s.add_argument("task")
    s.set_defaults(func=cmd_estimate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
