"""Claude Code subprocess runtime — uses Claude Pro included quota."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from subprocess import Popen
from typing import Iterator

from .base import (
    CheckpointMarker,
    Done,
    Error,
    Runtime,
    RuntimeEvent,
    TextChunk,
    TokenUsage,
    ToolCall,
)


class ClaudeCodeRuntime(Runtime):
    name = "claude_code"
    metered = False

    def __init__(self, model: str | None = None, binary: str | None = None):
        self.model = model or "sonnet"
        self.binary = binary or os.environ.get("COMMAND_CLAUDE_BIN") or "claude"

    def spawn(self, task: str, system_prompt: str, agent_id: str, work_dir: Path) -> Popen:
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.binary,
            "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            task,
        ]
        if system_prompt:
            sys_path = work_dir / "system_prompt.md"
            sys_path.write_text(system_prompt)
            cmd[-1:-1] = ["--append-system-prompt", system_prompt]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_dir),
            text=True,
            bufsize=1,
        )

    def stream_events(self, process: Popen) -> Iterator[RuntimeEvent]:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                yield TextChunk(text=line)
                continue
            yield from self._parse(evt)
        rc = process.wait()
        if rc != 0:
            stderr = process.stderr.read() if process.stderr else ""
            yield Error(message=f"claude exit {rc}: {stderr[:500]}")

    def _parse(self, evt: dict) -> Iterator[RuntimeEvent]:
        etype = evt.get("type")
        if etype == "assistant":
            msg = evt.get("message", {})
            for block in msg.get("content", []):
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    yield TextChunk(text=block["text"])
                elif btype == "tool_use":
                    yield ToolCall(name=block.get("name", ""), input=block.get("input") or {})
            usage = msg.get("usage")
            if usage:
                yield TokenUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    model=msg.get("model"),
                    cost_usd=0.0,
                )
        elif etype == "result":
            usage = evt.get("usage", {})
            if usage:
                yield TokenUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cost_usd=evt.get("total_cost_usd", 0.0) or 0.0,
                )
            status = "failed" if evt.get("is_error") else "completed"
            yield Done(
                status=status,
                final_text=evt.get("result", "") or "",
                error=evt.get("error") if evt.get("is_error") else None,
            )
