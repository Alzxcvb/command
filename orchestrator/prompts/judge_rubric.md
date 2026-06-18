# Judge Rubric — Shared Criteria

Used by all reviewers: the prompt pipeline council and the job curator.
Judges must evaluate against every criterion below and flag any that fail.

## Criteria

1. **Goal coverage** — every clause of the stated goal is addressed. Nothing is dropped silently.
2. **Scope precision** — the output does not expand beyond what was asked. No unprompted additions.
3. **Constraint preservation** — any explicit constraints (format, length, audience, language) are intact.
4. **No self-contradiction** — the output does not contradict itself or earlier context in the prompt.
5. **Decomposition integrity** — for multi-task plans: subtasks are collectively exhaustive and non-overlapping. No orphaned or redundant tasks.
6. **Budget realism** — token and cost allocations are consistent with task complexity. No obvious under- or over-budgeting.
7. **No injected assumptions** — the output does not add requirements or constraints not present in the original ask.
8. **Actionability** — the output is directly executable. A downstream agent could start without seeking clarification.
