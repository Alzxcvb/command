"""Parse orchestrator output into spawn instructions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_SPAWN_RE = re.compile(r"<spawn\b(.*?)/>", re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
_DONE_RE = re.compile(r"<done\s*/>")


def _unescape(value: str) -> str:
    return value.encode("utf-8").decode("unicode_escape", errors="replace") if "\\" in value else value


@dataclass
class SpawnInstruction:
    task: str
    runtime: str
    model: Optional[str] = None
    budget_tokens: int = 5000
    priority: int = 2
    metered_cap_usd: float = 0.0
    raw: str = ""


@dataclass
class Breakdown:
    spawns: list[SpawnInstruction] = field(default_factory=list)
    done: bool = False
    parse_errors: list[str] = field(default_factory=list)

    @property
    def total_budget(self) -> int:
        return sum(s.budget_tokens for s in self.spawns)

    @property
    def total_metered_cap(self) -> float:
        return round(sum(s.metered_cap_usd for s in self.spawns), 6)


def parse_breakdown(text: str) -> Breakdown:
    bd = Breakdown(done=bool(_DONE_RE.search(text)))
    for m in _SPAWN_RE.finditer(text):
        attrs = {k: _unescape(v) for k, v in _ATTR_RE.findall(m.group(1))}
        if "task" not in attrs or "runtime" not in attrs:
            bd.parse_errors.append(f"missing task/runtime in: {m.group(0)[:120]}")
            continue
        try:
            bd.spawns.append(SpawnInstruction(
                task=attrs["task"],
                runtime=attrs["runtime"],
                model=attrs.get("model") or None,
                budget_tokens=int(attrs.get("budget_tokens", "5000")),
                priority=int(attrs.get("priority", "2")),
                metered_cap_usd=float(attrs.get("metered_cap_usd", "0") or 0),
                raw=m.group(0),
            ))
        except ValueError as e:
            bd.parse_errors.append(f"bad attr in {m.group(0)[:120]}: {e}")
    return bd


def validate(bd: Breakdown, *, total_budget_tokens: int, total_metered_cap_usd: float) -> list[str]:
    """Return list of human-readable validation problems (empty = OK)."""
    problems: list[str] = []
    if not bd.spawns:
        problems.append("no <spawn> instructions parsed")
    if bd.total_budget > total_budget_tokens:
        problems.append(
            f"sum(budget_tokens)={bd.total_budget} exceeds job total {total_budget_tokens}"
        )
    if bd.total_metered_cap > total_metered_cap_usd:
        problems.append(
            f"sum(metered_cap_usd)=${bd.total_metered_cap} exceeds job cap ${total_metered_cap_usd}"
        )
    return problems
