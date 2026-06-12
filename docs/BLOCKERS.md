# Blockers

Known gaps that need either a design decision or Alex's hands to resolve. Each entry says what is blocked, why, and what would unblock it.

- Budget enforcement in commandd watchdog is incomplete: `tokens_used` is written by `lifecycle._monitor()` which dies with the CLI process. The watchdog can detect process exit and mark completion, but cannot enforce the token budget without a second mechanism to update tokens_used. (2026-06-12)
- Live end-to-end detached-spawn run not performed: spawning a real Claude Code agent and handing it to commandd was skipped on this machine (8GB M3 — nested agents starve it). To verify manually: `python commandd.py &`, then `source .venv/bin/activate && python -m cli spawn "say hi" --detach`, confirm the handoff line prints, exit the shell, and watch `state/agents/<id>/meta.json` flip to `completed` after the agent process exits. (2026-06-12)
- Mid-turn routing (Phase 3a) has no live caller: `core/turn_router.py` rule table, `record_turn()` ledger, and the dashboard widget are built and tested, but Command's runtimes spawn the claude CLI as a subprocess and do not own per-turn API calls, so `route_turn()` cannot intercept turns yet. Unblocks when the harness loop moves to the Claude Agent SDK (the spec's premise in `tasks/mid-turn-model-routing-and-prompt-flywheel.md`). (2026-06-12)
