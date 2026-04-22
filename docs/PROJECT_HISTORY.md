# Command — Project History

How this project came to exist, and what it absorbed along the way.

---

## Timeline

### Phase 1: AI Model Router (March 2026)
The original project. A Python routing engine that takes an AI prompt, classifies the task type (code, writing, reasoning, etc.), and sends it to the cheapest model that can handle it well. Uses OpenRouter for access to Claude, GPT-4o, Gemini, DeepSeek, Llama, and Qwen through a single API key.

- **Repo:** Started as `Alzxcvb/ai-model-router`, now renamed to `Alzxcvb/command`
- **What shipped:** Rules engine + Gemini Flash classifier, 37 pytest tests, TypeScript/Express web demo with routing visualization
- **Key finding:** Balanced routing saves ~91% vs always using Claude Opus, with minimal quality loss

### Phase 2: The Harness Problem Whitepaper (March 2026)
Richard Davidson published "The Harness Problem" — a research paper arguing that the wrapper around an AI model (the "harness") matters as much as the model itself. Same model can swing from 6.7% to 68.3% success rate just by changing how you format instructions and manage context.

This introduced the **Bockeler Framework** — three pillars for building good harnesses:
1. **Context Engineering** — curate what the model sees (system prompts, tools, code context)
2. **Architectural Constraints** — deterministic scaffolding (edit formats, sandboxes, linters)
3. **Garbage Collection** — clean up AI output (automated refactoring, quality enforcement)

The Router evolved into "AI Harness" — same repo, broader ambition.

### Phase 3: Studying Existing Harnesses (March-April 2026)
Two open-source reimplementations of Claude Code's architecture were downloaded for study:
- `claude-code/` — by @instructkr (Sigrid Jin), Python port of Claude Code internals
- `clawd-code/` — variant/alternate track of the same porting effort

These were never meant to be used directly — they're blueprints to learn from. Key lessons: how sub-agents work as "context firewalls," how edit formats affect success rates, how MCP (Model Context Protocol) connects tools.

### Phase 4: Evaluating Orchestration Tools (April 2026)
Two external tools were evaluated for managing multiple AI agents working in parallel:

**Hive** (by nwyin) — Downloaded to `hive/`. A multi-agent coordinator that breaks big tasks into pieces, assigns each to a separate AI worker running in its own git worktree, then merges their work through a validation pipeline. Supports three backends: Claude (WebSocket), OpenAI Codex, and Tau.

**Tau** — A lightweight agent process that speaks JSON-RPC. Used as a backend inside Hive. Not downloaded separately.

**GasTown** — Another orchestration tool evaluated as a starting point. Never downloaded — only referenced in planning docs.

### Phase 5: The Rate Limit Incident (April 16, 2026)
Three parallel agents were launched to work on World3 Dashboard, MDAC Better, and Polycrisis Research. All three ran on Opus (the most expensive model) simultaneously, burned through the entire API quota in ~2 minutes, and delivered zero output.

This proved that model routing (Harness) and agent lifecycle management (Orchestration) are the same problem — you can't route models intelligently without knowing what agents are running, and you can't manage agents without controlling which models they use.

### Phase 6: Command (April 17, 2026)
Decision made to merge AI Harness + AI Agent Orchestration into a single platform called **Command**. GitHub repo renamed from `ai-model-router` to `command`. Local folder renamed to match.

Command = Model Router + Agent Lifecycle + Local Dashboard + Multi-Provider Support + Token Budgeting.

---

## What Was Absorbed

| Project | Status | Where It Went |
|---------|--------|---------------|
| AI Model Router | Core of Command | Router engine lives in this repo |
| AI Harness (concept) | Absorbed | Bockeler Framework guides architecture |
| AI Agent Orchestration | Absorbed | Lifecycle management layer of Command |
| GasTown | Never adopted | Concept absorbed into Command's design |

## Reference Material (not absorbed, kept for study)

| Folder | What It Is | Action |
|--------|-----------|--------|
| `claude-code/` | instructkr's Claude Code reimplementation | Move to `command/reference/` |
| `clawd-code/` | Variant of above | Move to `command/reference/` |
| `hive/` | nwyin's multi-agent orchestrator | Move to `command/reference/` |

## Key People / References

- **Richard Davidson** — "The Harness Problem" whitepaper (March 2026). File: `~/Downloads/The Harness Problem - White Paper.docx`
- **Sigrid Jin (@instructkr)** — Claude Code reimplementation author
- **nwyin** — Hive orchestrator author
