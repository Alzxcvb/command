"""Streamlit dashboard. Run from repo root: streamlit run dashboard/app.py"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.lifecycle import (  # noqa: E402
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
from core.metered_ledger import DEFAULT_DAILY_CAP_USD, get_today_spend  # noqa: E402
from orchestrator import observations  # noqa: E402
from orchestrator.job import list_jobs, start_job  # noqa: E402
from orchestrator.ralph import improve as ralph_improve, pending_failure_count  # noqa: E402
from orchestrator.retry import retry_agent  # noqa: E402

st.set_page_config(page_title="Command — Agent Center", layout="wide", page_icon="🛰️")

# Top bar: title + metered spend + refresh controls
top_left, top_mid, top_right = st.columns([4, 3, 2])
with top_left:
    st.title("Command — Agent Center")
    st.caption("Local agent orchestration · live state from `state/agents/`")
with top_mid:
    spend = get_today_spend()
    used = spend["total_usd"]
    cap = DEFAULT_DAILY_CAP_USD
    pct = min(1.0, used / cap) if cap else 0.0
    st.metric(
        label=f"Metered today (cap ${cap:.2f})",
        value=f"${used:.4f}",
        delta=f"{len(spend['by_agent'])} agent(s) charged",
    )
    st.progress(pct)
    anth_keyed = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if anth_keyed:
        st.caption("⚠️ ANTHROPIC_API_KEY set — Claude Code runs on **metered API** (real $).")
    else:
        st.caption("✓ No ANTHROPIC_API_KEY — Claude Code runs on your **Pro subscription** (quota, not $). "
                   "Per-agent `cost $X` is informational (would-have-cost on metered API).")
with top_right:
    if st.button("↻ Refresh now", use_container_width=True):
        st.rerun()
    auto = st.checkbox("Auto-refresh (2s)", value=True)


def _start_job_async(goal: str, total_budget: int, total_metered_cap: float,
                     orch_model: str, project_hint: str = "") -> None:
    """Run start_job in a thread so the Streamlit request returns immediately."""
    t = threading.Thread(
        target=start_job,
        kwargs={
            "goal": goal,
            "total_budget_tokens": total_budget,
            "total_metered_cap_usd": total_metered_cap,
            "orchestrator_model": orch_model,
            "project_hint": project_hint,
        },
        daemon=True,
        name="job-starter",
    )
    t.start()


with st.sidebar:
    st.header("Ralph — guide improver")
    pending = pending_failure_count()
    st.caption(f"Failure samples since last improvement: **{pending}**  ·  threshold: 3")
    st.caption(f"Current guide sha: `{observations.current_guide_sha() or '—'}`")
    col_r1, col_r2 = st.columns(2)
    if col_r1.button("Improve now (force)", use_container_width=True, key="ralph-force"):
        with st.spinner("Ralph reading the failure buffer…"):
            r = ralph_improve(force=True)
        st.session_state["last_ralph_result"] = r
    if col_r2.button("Improve if ≥3", use_container_width=True, key="ralph-auto"):
        with st.spinner("Ralph checking threshold…"):
            r = ralph_improve(force=False)
        st.session_state["last_ralph_result"] = r
    if "last_ralph_result" in st.session_state:
        st.json(st.session_state["last_ralph_result"])
        if st.button("Dismiss Ralph result", key="ralph-dismiss"):
            del st.session_state["last_ralph_result"]
            st.rerun()

    st.divider()
    st.header("Orchestrate (goal → fan-out)")
    with st.form("orchestrate_form", clear_on_submit=False):
        goal = st.text_area(
            "Goal",
            placeholder="e.g. Draft a 1500-word polycrisis explainer blog post with 3 citations.",
            height=110,
        )
        project_hint = st.text_input("Project label (for analytics)", placeholder="e.g. polycrisis")
        col_a, col_b = st.columns(2)
        total_budget = col_a.number_input("Total tokens", min_value=5_000, max_value=500_000,
                                          value=50_000, step=5_000)
        total_cap = col_b.number_input("Metered cap $", min_value=0.0, max_value=10.0,
                                       value=0.50, step=0.10)
        orch_model = st.selectbox("Orchestrator model", ["sonnet", "opus", "haiku"], index=0)
        if st.form_submit_button("Orchestrate", use_container_width=True) and goal.strip():
            try:
                _start_job_async(goal.strip(), int(total_budget), float(total_cap), orch_model,
                                 project_hint.strip())
                st.success("Job started — orchestrator spinning up.")
            except Exception as e:
                st.error(f"Failed to orchestrate: {e}")
            st.rerun()

    st.divider()
    st.header("Spawn single agent")
    with st.form("spawn_form", clear_on_submit=False):
        task = st.text_area(
            "Task",
            placeholder="Summarize the first 5 lines of README.md",
            height=90,
        )
        provider = st.radio("Provider", ["Claude", "Codex"], horizontal=True, index=0)
        if provider == "Claude":
            runtime = "claude_code"
            model = st.selectbox("Model", ["sonnet", "haiku", "opus"], index=0)
        else:
            runtime = "codex"
            model = st.selectbox("Reasoning effort", ["medium", "low", "high"], index=0)
        budget = st.number_input("Token budget", min_value=500, max_value=200_000,
                                 value=8000, step=500)
        submitted = st.form_submit_button("Spawn", use_container_width=True)
        if submitted and task.strip():
            try:
                new_id = spawn_agent(
                    task=task.strip(),
                    runtime_name=runtime,
                    system_prompt="",
                    budget_tokens=int(budget),
                    model=model,
                )
                st.success(f"Spawned {new_id} ({provider} · {model})")
            except Exception as e:
                st.error(f"Failed to spawn: {e}")
            st.rerun()



def _render_agent_card(a: dict, active: bool, *, key_prefix: str = "") -> None:
    kp = key_prefix or a["agent_id"]
    with st.container(border=True):
        head = st.columns([3, 2, 2, 2, 1, 1, 1])
        head[0].markdown(f"**`{a['agent_id']}`**")
        head[0].caption((a.get("task") or "")[:160])
        head[1].markdown(f"runtime: `{a['runtime']}`")
        head[1].caption(f"model: `{a.get('model') or '—'}`")
        used = a.get("tokens_used") or 0
        cap_t = a.get("budget_tokens") or 0
        est = a.get("estimated_tokens") or 0
        head[2].metric("Tokens used", f"{used:,}")
        if cap_t:
            head[2].progress(min(1.0, used / cap_t))
            head[2].caption(f"cap {cap_t:,}" + (f"  · est {est:,}" if est else ""))
        status_label = a["status"]
        if a.get("budget_overrun"):
            status_label += " ⚠️"
        head[3].markdown(f"**{status_label}**")
        head[3].caption(f"updated {(a.get('updated_at') or '')[:19].replace('T', ' ')}")
        if a.get("cost_usd"):
            head[3].caption(f"cost ${a['cost_usd']:.4f}")
        if active:
            if head[4].button("Kill", key=f"kill-{kp}"):
                kill_agent(a["agent_id"])
                st.rerun()
        else:
            retried_key = f"retried-{a['agent_id']}"
            if st.session_state.get(retried_key):
                head[4].caption(f"↻ retried → `{st.session_state[retried_key]}`")
            else:
                if head[4].button("↻ Retry", key=f"retry-{kp}",
                                  help="Re-spawn this task. Emits a 'retried' failure signal for Ralph."):
                    new_id = retry_agent(a["agent_id"], rationale="manual retry from dashboard")
                    if new_id:
                        st.session_state[retried_key] = new_id
                    else:
                        st.session_state[retried_key] = "failed"
                    st.rerun()
        thumb_key = f"thumbed-{a['agent_id']}"
        if st.session_state.get(thumb_key):
            head[5].caption("👎 ✓")
        else:
            if head[5].button("👎", key=f"thumb-{kp}",
                              help="Mark this output as bad. Feeds Ralph."):
                observations.log_thumb(
                    job_id=a.get("parent_job_id") or "",
                    agent_id=a["agent_id"],
                    direction="down",
                    comment="dashboard 👎",
                )
                st.session_state[thumb_key] = True
                st.rerun()

        with st.expander("Checkpoint · /btw inject · stdout tail"):
            cp = get_checkpoint(a["agent_id"])
            if cp:
                st.markdown(cp)
            msg = st.text_input("/btw message", key=f"btw-input-{kp}",
                                placeholder="follow-up to queue (or send + continue)")
            cols = st.columns(2)
            if cols[0].button("Queue (apply on next continue)", key=f"btw-q-{kp}") and msg:
                inject_message(a["agent_id"], msg)
                st.success("queued")
                st.rerun()
            if cols[1].button("Send + Continue (spawn follow-up turn)", key=f"btw-c-{kp}"):
                if msg:
                    inject_message(a["agent_id"], msg)
                new_id = continue_agent(a["agent_id"])
                if new_id:
                    st.success(f"continued as {new_id}")
                else:
                    st.error("could not continue this agent")
                st.rerun()
            log = get_log(a["agent_id"], "stdout", tail_chars=3000)
            if log:
                st.code(log, language="text")
            if not active:
                res = get_result(a["agent_id"])
                if res:
                    st.json(res)


def _render_agents_tab() -> None:
    # --- Active jobs (orchestrator fan-outs) ---
    jobs = list_jobs()
    active_jobs = [j for j in jobs if j.get("status") in ("orchestrating", "running")]
    done_jobs = [j for j in jobs if j.get("status") not in ("orchestrating", "running")]

    st.markdown(f"### Active jobs ({len(active_jobs)})")
    if not active_jobs:
        st.caption("No fan-out jobs running. Use **Orchestrate** in the sidebar to dispatch a goal.")
    else:
        for j in active_jobs:
            with st.container(border=True):
                top = st.columns([4, 2, 2])
                top[0].markdown(f"**`{j['job_id']}`** · {j.get('status')}")
                top[0].caption(j.get("goal", "")[:200])
                top[1].metric("Children", len(j.get("child_agent_ids") or []))
                top[2].metric("Budget tokens", f"{j.get('total_budget_tokens', 0):,}")

                child_ids = j.get("child_agent_ids") or []
                orch_id = j.get("orchestrator_agent_id")
                all_ids = ([orch_id] if orch_id else []) + child_ids
                if all_ids:
                    st.caption("Side-by-side view of orchestrator + children (running in parallel):")
                    cols = st.columns(max(1, min(len(all_ids), 3)))
                    for idx, aid in enumerate(all_ids):
                        a = get_agent(aid)
                        if not a:
                            continue
                        with cols[idx % len(cols)]:
                            role = "orchestrator" if aid == orch_id else f"child {idx}"
                            st.markdown(f"**{role}** · `{aid}`")
                            st.caption(f"{a.get('runtime')} / {a.get('model') or '—'}")
                            st.caption(f"status: **{a.get('status')}**")
                            used_c = a.get("tokens_used") or 0
                            cap_c = a.get("budget_tokens") or 0
                            if cap_c:
                                st.progress(min(1.0, used_c / cap_c))
                            st.caption(f"{used_c:,} / {cap_c:,} tok")
                            log = get_log(aid, "stdout", tail_chars=800)
                            if log:
                                st.code(log[-500:], language="text")

    agents = list_agents()
    active = [a for a in agents if a["status"] in ("starting", "running")]
    done = sorted(
        [a for a in agents if a["status"] not in ("starting", "running")],
        key=lambda a: a.get("updated_at") or a.get("created_at") or "",
        reverse=True,
    )

    st.markdown(f"### Active agents ({len(active)})")
    if not active:
        st.info('No agents currently running. Spawn one with: `python -m cli spawn "your task"`')
    else:
        for a in active:
            _render_agent_card(a, active=True)

    st.markdown(f"### Completed ({len(done)})  — most recent first")
    if done:
        for a in done[:10]:
            _render_agent_card(a, active=False)

    if done_jobs:
        with st.expander(f"Completed jobs ({len(done_jobs)})"):
            for j in list(reversed(done_jobs))[:10]:
                st.markdown(f"- `{j['job_id']}` · **{j.get('status')}** · {j.get('goal', '')[:120]}")


def _render_analytics() -> None:
    """Historical analytics: one jsonl line per completed job + task types from agent metas."""
    import pandas as pd

    path = _REPO_ROOT / "state" / "analytics.jsonl"
    rows = []
    if path.exists():
        for ln in path.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue

    if not rows:
        st.info("No completed jobs logged yet — analytics accrue when an orchestrated job finishes.")
    else:
        df = pd.DataFrame(rows)
        m1, m2, m3 = st.columns(3)
        m1.metric("Jobs logged", len(df))
        m2.metric("Total tokens", f"{int(df['total_tokens'].sum()):,}")
        m3.metric("Total cost", f"${df['cost_usd'].sum():.4f}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Cost per project ($)**")
            st.bar_chart(df.groupby("project_hint")["cost_usd"].sum())
        with col_r:
            st.markdown("**Tokens per job over time**")
            trend = df[["ts", "total_tokens"]].copy()
            trend["ts"] = trend["ts"].str[:16]
            st.line_chart(trend.set_index("ts"))

    typed = [a for a in list_agents() if a.get("task_type")]
    if typed:
        st.markdown("**Tokens per task type** (recorded at dispatch by the estimator)")
        tdf = pd.DataFrame(
            [{"task_type": a["task_type"], "tokens": a.get("tokens_used") or 0} for a in typed]
        )
        st.bar_chart(tdf.groupby("task_type")["tokens"].sum())
    else:
        st.caption("Tokens per task type appears once jobs dispatch through the estimator.")


tab_agents, tab_analytics = st.tabs(["Agents", "Analytics"])
with tab_agents:
    _render_agents_tab()
with tab_analytics:
    _render_analytics()

if auto:
    time.sleep(2)
    st.rerun()
