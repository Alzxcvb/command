# Unified Agent Platform: Merging AI Harness + AI Agent Orchestration

**Date:** 2026-04-17
**Status:** Plan — awaiting Alex's approval before implementation

---

## The Problem We Just Experienced

Three parallel agents launched to work on World3, MDAC, and Polycrisis projects. All three hit the API rate limit and produced almost nothing (~2,200 tokens across 63 tool calls, zero deliverables shipped). Root causes:

1. **No token budgeting** — three agents competed for the same quota pool with no awareness of remaining capacity
2. **No model routing** — all three ran on Opus for everything including research/planning that Haiku handles fine at 1/60th the cost
3. **No progress visibility** — couldn't see what agents were doing until they finished (or failed)
4. **No retry/resume** — when agents hit limits, their partial work was lost to context; no checkpoint to resume from
5. **Overly ambitious prompts** — each agent tried to research + implement + update memory in one shot

These are exactly the problems AI Harness (model routing) and AI Agent Orchestration (agent lifecycle) were designed to solve — separately. Combined, they solve ALL of them.

---

## Current State of Both Projects

### AI Harness (repo: `Alzxcvb/ai-model-router`)
| Aspect | Status |
|--------|--------|
| **What it does** | Routes prompts to the best model per task type |
| **Core capability** | Rules engine + LLM classifier (Gemini Flash), multi-model support |
| **Models supported** | Claude (Opus/Sonnet/Haiku), GPT-4o, Gemini, DeepSeek, Llama, Qwen via OpenRouter |
| **Tests** | 37 passing (pytest) |
| **Architecture** | Bockeler Framework: Context Engineering, Architectural Constraints, Garbage Collection |
| **Gap** | No agent lifecycle, no dashboard, no task persistence |

### AI Agent Orchestration System (no repo yet)
| Aspect | Status |
|--------|--------|
| **What it does** | Manages agent lifecycles: spawn, monitor, checkpoint, resume |
| **Core capability** | Planned — evaluating GasTown and Tau as starting points |
| **Gap** | Nothing built. No model routing, no cost optimization |

### Key Insight: They're Two Halves of One System

| Capability | AI Harness | Orchestration | Combined |
|-----------|-----------|---------------|----------|
| Model routing (Haiku vs Opus) | Yes | No | Yes |
| Cost optimization | Yes (per-call) | No | Yes (per-call + budget) |
| Agent spawn/monitor/kill | No | Yes | Yes |
| Progress tracking | No | Yes | Yes |
| Checkpoint/resume | No | Yes | Yes |
| Task persistence | No | Yes | Yes |
| Dashboard UI | No | Planned | Yes |
| Context engineering | Yes | No | Yes |
| Garbage collection | Yes | No | Yes |
| Multi-provider (OpenAI, Google, Anthropic) | Yes (via OpenRouter) | No | Yes |

**Recommendation: Merge into a single project — "Command" (working name).**

---

## Proposed Architecture: Command

```
command/
  core/
    router.py          # Model routing engine (from AI Harness)
    budget.py           # Token budget manager — per-agent and global caps
    classifier.py       # Task classifier (research/planning/implementation/review)
  agents/
    lifecycle.py        # Spawn, monitor, checkpoint, resume, kill
    registry.py         # Track all active/completed agents + their state
    queue.py            # Task queue with priority + dependency ordering
  providers/
    anthropic.py        # Claude Opus/Sonnet/Haiku (direct API)
    openai.py           # Codex, GPT-4o (direct API)
    google.py           # Gemini Pro/Flash (direct API)
    openrouter.py       # Fallback for everything else (DeepSeek, Llama, Qwen)
    local.py            # On-board LLM (Ollama/llama.cpp) for low-stakes tasks
  dashboard/
    app.py              # Web UI — agent control center
    api.py              # REST API for dashboard + CLI
    templates/          # Dashboard HTML/CSS
  integrations/
    git.py              # Git operations (commit, push, PR) — route to cheap model
    github.py           # GitHub API (issues, PRs, project boards)
    claude_code.py      # Spawn Claude Code subagents programmatically
  config/
    routing_rules.yaml  # Which task types go to which models
    budget_limits.yaml  # Token caps per agent, per session, per day
```

### Model Routing Strategy

| Task Type | Default Model | Why | Estimated Cost Ratio |
|-----------|--------------|-----|---------------------|
| Research / web search | Haiku 4.5 | Fast, cheap, good enough for fetching + summarizing | 1x |
| Planning / outlining | Haiku 4.5 | Structured output, doesn't need deep reasoning | 1x |
| Code implementation | Opus 4.6 | Complex reasoning, architecture decisions | 60x |
| Code review | Sonnet 4.6 | Good balance of quality and cost | 15x |
| Git operations (commit messages, PR descriptions) | Haiku 4.5 or local LLM | Formulaic, low stakes | 1x or ~0x |
| Fact-checking / citation lookup | Gemini Pro | Long context window, good at retrieval | ~5x |
| Bulk text processing | Gemini Flash | Cheapest for high-volume, low-complexity | ~0.5x |
| Codex tasks (code generation) | OpenAI Codex | Specialized for code, good for boilerplate | ~10x |

**Savings estimate:** If the three agents had used this routing, research phases (~70% of work) would have run on Haiku instead of Opus. That's roughly 60x cheaper for those calls — the same budget could have completed all three tasks.

### Dashboard Features (Control Center)

The dashboard is the UI that answers: "What are my agents doing right now?"

**Real-time view:**
- Active agents: name, task, model being used, tokens consumed, % of budget
- Live log stream per agent (collapsible)
- Kill button per agent
- Global token budget meter (used / remaining / rate limit reset time)

**Historical view:**
- Completed agents: what they did, tokens spent, files changed, commits made
- Cost breakdown by model (pie chart)
- Task completion rate and average token efficiency

**Controls:**
- Deploy new agent: pick task, set budget cap, choose model routing profile
- Pause/resume agents
- Priority reordering
- Manual model override ("switch this agent to Opus now")

**Integration:**
- GitHub webhook: show PR status, CI results alongside agent that created them
- Portfolio page link: agents can update project status badges automatically

### On-Board LLM Strategy

For tasks that don't need cloud API calls at all:
- **Git commit messages** — local Llama 3 or Qwen 2.5 via Ollama
- **File renaming / simple refactors** — pattern-based, barely needs an LLM
- **Log parsing / grep assistance** — local model or regex
- **Cost: $0.** Only limited by Alex's local hardware (Mac)

Caveat: Alex's machine specs matter. If running a 7B param model locally is smooth, this saves real money on high-volume low-stakes calls.

---

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
- [ ] Create `Alzxcvb/command` repo (or rename `ai-model-router`)
- [ ] Port existing router + classifier from ai-model-router
- [ ] Add `budget.py` — token tracking per agent with configurable caps
- [ ] Add `lifecycle.py` — spawn/monitor/kill agents (wrapping Claude Code subagent calls)
- [ ] Add provider adapters: Anthropic (direct), OpenAI, Google, OpenRouter
- [ ] Tests for routing decisions + budget enforcement

### Phase 2: Dashboard MVP (Week 3)
- [ ] Streamlit or FastAPI + HTMX dashboard (keep it simple, ship fast)
- [ ] Real-time agent status view
- [ ] Token budget visualization
- [ ] Deploy button (spawn agent with task + budget + routing profile)
- [ ] Run locally only (no deployment — Alex uses it on his Mac during dev sessions)

### Phase 3: Smart Routing (Week 4)
- [ ] Task classifier trained on our actual usage patterns
- [ ] Auto-downgrade: if agent is doing research, switch to Haiku mid-task
- [ ] Auto-escalate: if agent hits a complex problem, offer to switch to Opus
- [ ] Local LLM integration via Ollama for git ops

### Phase 4: Integrations (Week 5-6)
- [ ] GitHub integration: agents create issues, PRs, update project boards
- [ ] Codex integration for bulk code generation tasks
- [ ] Checkpoint/resume: serialize agent state to disk, resume after rate limits
- [ ] Cross-agent communication (Agent A's output feeds Agent B)

### Phase 5: Polish (Week 7-8)
- [ ] Historical analytics: cost per project, tokens per task type
- [ ] Agent templates: "Research Agent", "Implementation Agent", "Review Agent"
- [ ] CLI interface (`command deploy "add SW scenario to World3" --budget 5000 --model haiku`)
- [ ] Documentation + README for portfolio

---

## How This Would Have Changed Today's Outcome

**What happened:**
- 3 agents launched in parallel on Opus → burned quota in ~2 minutes → zero deliverables

**What would happen with Command:**
1. Router classifies all three tasks: research phase → Haiku, implementation → Opus
2. Budget manager sees remaining quota, allocates proportionally (or sequences agents)
3. Dashboard shows: "Agent 1: World3 research — 400 tokens used, Haiku" in real-time
4. When Haiku research phase completes, router escalates to Opus for code changes only
5. If rate limit approaches, budget manager pauses lowest-priority agent, lets highest finish
6. Checkpoint saves partial work — resume after limit resets without re-deriving context

**Estimated savings:** ~80% fewer Opus tokens for same output.

---

## Risks

- **Scope creep** — this is ambitious. Phase 1-2 is the MVP; don't gold-plate.
- **Maintenance burden** — another project to maintain. Mitigate: dogfood it on real projects immediately.
- **Provider API changes** — OpenRouter abstracts most of this, but direct API adapters need version pinning.
- **Local LLM quality** — 7B models may produce garbage for anything beyond templates. Test before relying.

---

## Decisions — LOCKED (2026-04-17)

1. **Merge:** Done. Repo renamed `ai-model-router` → `Alzxcvb/command` on GitHub. Orchestration absorbed.
2. **Dashboard:** Local-only. Runs on Alex's Mac during dev sessions. No web hosting needed.
3. **Priority:** Finish World3/MDAC/Polycrisis manually first, then build Command.
4. **Local LLM:** MacBook Pro M3, 8GB RAM, 233GB free. Can run 3-4B models (Phi-3 mini recommended). 7B quantized works but tight (~2GB headroom). Not viable for background inference during heavy multitasking. Plan: use Phi-3 mini (3.8B) for git ops + formulaic tasks; cloud APIs for everything else.

## Hardware Profile

- MacBook Pro 14" (Nov 2023)
- Apple M3 chip
- 8 GB unified memory
- 494 GB SSD (233 GB free)
- macOS Sequoia 15.6
- 14" Liquid Retina XDR (3024x1964)
