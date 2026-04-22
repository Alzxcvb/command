"""Budget primitives. Lifecycle enforces caps directly; these are convenience accessors."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentBudget:
    agent_id: str
    cap_tokens: int
    used_tokens: int = 0

    def remaining(self) -> int:
        return max(0, self.cap_tokens - self.used_tokens)

    def should_stop(self) -> bool:
        return self.cap_tokens > 0 and self.used_tokens >= self.cap_tokens

    def record(self, n: int) -> None:
        self.used_tokens += n


@dataclass
class JobBudget:
    job_id: str
    cap_tokens: int = 0
    cap_metered_usd: float = 0.0
    used_tokens: int = 0
    used_metered_usd: float = 0.0

    def remaining_tokens(self) -> int:
        return max(0, self.cap_tokens - self.used_tokens) if self.cap_tokens else -1

    def remaining_metered_usd(self) -> float:
        return max(0.0, self.cap_metered_usd - self.used_metered_usd)
