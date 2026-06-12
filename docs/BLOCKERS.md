# Blockers

Known gaps that need either a design decision or Alex's hands to resolve. Each entry says what is blocked, why, and what would unblock it.

- Budget enforcement in commandd watchdog is incomplete: `tokens_used` is written by `lifecycle._monitor()` which dies with the CLI process. The watchdog can detect process exit and mark completion, but cannot enforce the token budget without a second mechanism to update tokens_used. (2026-06-12)
- Live end-to-end detached-spawn run not performed: spawning a real Claude Code agent and handing it to commandd was skipped on this machine (8GB M3 — nested agents starve it). To verify manually: `python commandd.py &`, then `source .venv/bin/activate && python -m cli spawn "say hi" --detach`, confirm the handoff line prints, exit the shell, and watch `state/agents/<id>/meta.json` flip to `completed` after the agent process exits. (2026-06-12)
