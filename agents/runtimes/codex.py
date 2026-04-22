"""Codex CLI subprocess runtime — uses Codex Pro / ChatGPT account included quota."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from subprocess import Popen
from typing import Iterator

from .base import Done, Error, Runtime, RuntimeEvent, TextChunk, TokenUsage


class CodexRuntime(Runtime):
    name = "codex"
    metered = False

    def __init__(self, model: str | None = None, binary: str | None = None):
        # `None` / "default" → let codex pick (account default model).
        # ChatGPT-account users typically don't get gpt-5-codex.
        self.model = None if not model or model == "default" else model
        self.binary = binary or os.environ.get("COMMAND_CODEX_BIN") or "codex"

    def spawn(self, task: str, system_prompt: str, agent_id: str, work_dir: Path) -> Popen:
        work_dir.mkdir(parents=True, exist_ok=True)
        prompt = task if not system_prompt else f"{system_prompt}\n\n---\n\n{task}"
        cmd = [self.binary, "exec", "--json", "--skip-git-repo-check"]
        if self.model:
            cmd += ["-m", self.model]
        cmd.append(prompt)
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
        final_chunks: list[str] = []
        failed_msg: str | None = None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                yield TextChunk(text=line)
                continue
            etype = evt.get("type", "")
            if etype == "item.completed":
                item = evt.get("item") or {}
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        yield TextChunk(text=text)
                        final_chunks.append(text)
            elif etype == "turn.completed":
                usage = evt.get("usage") or {}
                yield TokenUsage(
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    model=self.model or "codex-default",
                )
            elif etype == "turn.failed":
                err = evt.get("error") or {}
                failed_msg = err.get("message", "turn.failed")
                yield Error(message=f"codex: {failed_msg[:500]}")
            elif etype == "error":
                failed_msg = evt.get("message", "error")
                yield Error(message=f"codex: {failed_msg[:500]}")
        rc = process.wait()
        if failed_msg:
            yield Done(status="failed", error=failed_msg[:500])
        elif rc != 0:
            stderr = process.stderr.read() if process.stderr else ""
            yield Error(message=f"codex exit {rc}: {stderr[:500]}")
            yield Done(status="failed", error=stderr[:500])
        else:
            yield Done(status="completed", final_text="".join(final_chunks))
