# Command — Unified Agent Platform

## What This Is

Command is a unified platform that combines model routing (picking the cheapest AI model per task) with agent orchestration (spawning, monitoring, budgeting, and resuming multiple AI agents). It runs locally on Alex's Mac as a developer tool — no web hosting.

## Inherited Rules

This project inherits behavioral rules from `ClaudeProjects/CLAUDE.md` (parent directory). That file covers: plan mode, task management, self-improvement loop, verification standards, elegance checks, autonomous bug fixing, debugging & fault isolation policy, and code hygiene. Those rules apply here automatically — do not duplicate them.

## Quick Start for New Sessions

1. Read `docs/PROJECT_HISTORY.md` for how we got here
2. Read `docs/IMPLEMENTATION_PLAN.md` for the full build plan
3. Check the phase checklist below for where we left off

## Current Status: CORE VERTICAL SLICE SHIPPED (2026-04-20)

Orchestrator fan-out + Ralph self-improvement loop + Streamlit dashboard all running locally. Claude Code runtime hits Pro subscription quota (no metered API). Multi-provider router (Anthropic/OpenAI/Ollama/OpenRouter) in place.

**Next to continue:** Re-run 4-vignette fan-out test to validate orchestrator decomposition; if it still collapses, thumb-down → Ralph force-improve → verify guide rewrite closes the loop. Then commit uncommitted router work and scaffolding. See `~/.claude/projects/-Users-alexandercoffman-ClaudeProjects/memory/project-command.md` for full status.

## Implementation Phases

### Phase 1: Foundation — DONE
- [x] `agents/runtimes/claude_code.py` Claude Code CLI subprocess runtime (streaming JSON)
- [x] `agents/runtimes/codex.py`, `opencode.py`, `ollama.py` additional runtimes
- [x] `agents/lifecycle.py` spawn/kill/continue with soft/hard budget policy
- [x] `agents/registry.py` in-process registry with `get_agent()`, `get_result()`
- [x] `router/` multi-provider dispatcher (Anthropic/OpenAI/Ollama/OpenRouter fallback)
- [x] `config/budget_limits.yaml` max_concurrent_agents cap

### Phase 2: Local Dashboard — DONE
- [x] Streamlit dashboard at `dashboard/app.py` (port 8521)
- [x] Active jobs grid with side-by-side orchestrator + children view
- [x] Progress bars, live stdout tail, session-state click-once buttons
- [x] Kill/Retry/Thumb-down controls per agent
- [x] Completed agents sorted most-recent-first
- [x] Metered-spend banner distinguishing subscription quota from API $

### Phase 3: Orchestration + Ralph Loop — DONE
- [x] `orchestrator/job.py` `start_job()` fan-out orchestrator
- [x] `orchestrator/breakdown.py` parser for `<spawn .../>` tags
- [x] `orchestrator/prompts/orchestrator.md` the guide Ralph iterates
- [x] `orchestrator/retry.py` `retry_agent(failed_agent_id)`
- [x] `orchestrator/observations.py` dispatch/outcome/retry/thumb log + failure-only filter
- [x] `orchestrator/ralph.py` improver: spawn Sonnet → extract revised guide → git-commit
- [x] Auto-trigger Ralph in background thread when failure buffer ≥3 new samples
- [x] Content-hash guide versioning (SHA-256 12-char)

### Phase 4: CLI — DONE
- [x] `cli/__main__.py` — spawn, status, kill, btw, continue, orchestrate, jobs, estimate
- [x] New in 2026-04-20: retry, thumb up|down, ralph-improve --force

### Phase 5: 2026-06-12 session — DONE
- [x] `config/routing_rules.yaml` + `config/budget_limits.yaml` created (estimator crashed without them)
- [x] Estimator integration into `start_job()` — per-task budget floor, `budget_source` in agent meta
- [x] CLI daemon for detached spawns — `commandd.py` PID watchdog (`docs/DAEMON.md`)
- [x] Historical analytics — `state/analytics.jsonl` + dashboard Analytics tab (cost per project, tokens per task type)
- [x] Phase 3a static mid-turn routing — `core/turn_router.py` rule table + turn ledger + savings widget (no live caller yet; see `docs/BLOCKERS.md`)

### Phase 6: What remains
- [ ] Re-run 4-vignette fan-out test (verify visible parallel decomposition) — needs live agents, run manually
- [ ] If orchestrator still under-decomposes: Ralph force-improve → verify loop closes
- [ ] Live detached-spawn test through commandd (see `docs/BLOCKERS.md`)
- [ ] Wire `route_turn()` into a harness loop once Command owns per-turn API calls
- [ ] Close the commandd budget-enforcement gap (`tokens_used` freezes after CLI exit — `docs/BLOCKERS.md`)

## Architecture

```
command/
  core/           # Router engine + budget manager + classifier
  agents/         # Lifecycle, registry, task queue
  providers/      # Anthropic, OpenAI, Google, OpenRouter, local (Ollama)
  dashboard/      # Local web UI
  integrations/   # Git, GitHub, Claude Code subagent bridge
  reference/      # Study material: claude-code, clawd-code, hive
  docs/           # History, plan, architecture decisions
```

## Model Routing Strategy

| Task Type | Model | Cost |
|-----------|-------|------|
| Research / web search | Haiku 4.5 | 1x |
| Planning / outlining | Haiku 4.5 | 1x |
| Code implementation | Opus 4.6 | 60x |
| Code review | Sonnet 4.6 | 15x |
| Git commit messages | Local (Phi-3 mini) | ~0x |
| Bulk text processing | Gemini Flash | ~0.5x |

## Hardware

MacBook Pro 14" (Nov 2023), Apple M3, 8GB RAM, 494GB SSD. Can run 3-4B local models (Phi-3 mini). 7B quantized is tight with ~2GB headroom.

## Key Lesson That Created This Project

On 2026-04-16, three parallel agents ran on Opus simultaneously, burned the full API rate limit in ~2 minutes, and delivered zero output. With Command's routing, ~70% of that work would have run on Haiku (60x cheaper), and budget management would have paused lower-priority agents before hitting the limit.

## Reference Material (in `reference/`)

- `reference/claude-code/` — instructkr's Claude Code reimplementation (study harness patterns)
- `reference/clawd-code/` — variant of above
- `reference/hive/` — nwyin's multi-agent orchestrator (study coordination patterns)
- Whitepaper: `~/Downloads/The Harness Problem - White Paper.docx` (Richard Davidson, March 2026)
