"""Runtime abstraction. A Runtime spawns and manages a single CLI subprocess."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from typing import Iterator, Literal, Optional, Union


@dataclass
class TextChunk:
    text: str


@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    model: Optional[str] = None
    cost_usd: float = 0.0


@dataclass
class CheckpointMarker:
    note: str = ""


@dataclass
class Done:
    status: Literal["completed", "failed"] = "completed"
    final_text: str = ""
    error: Optional[str] = None


@dataclass
class Error:
    message: str


RuntimeEvent = Union[TextChunk, ToolCall, TokenUsage, CheckpointMarker, Done, Error]


class Runtime(ABC):
    name: str = ""
    metered: bool = False

    @abstractmethod
    def spawn(self, task: str, system_prompt: str, agent_id: str, work_dir: Path) -> Popen:
        ...

    @abstractmethod
    def stream_events(self, process: Popen) -> Iterator[RuntimeEvent]:
        ...

    def kill(self, process: Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()

    def inject(self, process: Popen, message: str) -> bool:
        return False
