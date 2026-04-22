"""OpenCode CLI runtime — wraps OpenRouter; metered against $1/day cap."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from subprocess import Popen
from typing import Iterator

from .base import Done, Error, Runtime, RuntimeEvent, TextChunk, TokenUsage


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    try:
        from router.models import MODELS
        info = MODELS.get(model_id)
        if not info:
            return 0.0
        return round(
            (input_tokens / 1_000_000) * info.cost_per_million_input
            + (output_tokens / 1_000_000) * info.cost_per_million_output,
            6,
        )
    except Exception:
        return 0.0


class OpenCodeRuntime(Runtime):
    name = "opencode"
    metered = True

    def __init__(self, model: str | None = None, binary: str | None = None):
        self.model = model or "deepseek/deepseek-chat-v3"
        self.binary = binary or os.environ.get("COMMAND_OPENCODE_BIN") or "opencode"

    def spawn(self, task: str, system_prompt: str, agent_id: str, work_dir: Path) -> Popen:
        work_dir.mkdir(parents=True, exist_ok=True)
        prompt = task if not system_prompt else f"{system_prompt}\n\n---\n\n{task}"
        # opencode picks model from its own config (~/.opencode/...). The `model`
        # field on this runtime is used solely for cost-attribution lookup.
        cmd = [self.binary, "-q", "-f", "json", "-p", prompt]
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
        saw_usage = False
        for line in process.stdout:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                yield TextChunk(text=line)
                final_chunks.append(line)
                continue
            text = evt.get("text") or evt.get("content") or evt.get("message") or ""
            usage = evt.get("usage") or {}
            if usage:
                inp = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
                out = int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
                yield TokenUsage(
                    input_tokens=inp,
                    output_tokens=out,
                    model=self.model,
                    cost_usd=_cost_usd(self.model, inp, out),
                )
                saw_usage = True
            if text:
                yield TextChunk(text=text)
                final_chunks.append(text)
        rc = process.wait()
        if not saw_usage and final_chunks:
            out_t = _approx_tokens("".join(final_chunks))
            in_t = _approx_tokens("input")
            yield TokenUsage(
                input_tokens=in_t,
                output_tokens=out_t,
                model=self.model,
                cost_usd=_cost_usd(self.model, in_t, out_t),
            )
        if rc != 0:
            stderr = process.stderr.read() if process.stderr else ""
            yield Error(message=f"opencode exit {rc}: {stderr[:500]}")
            yield Done(status="failed", error=stderr[:500])
        else:
            yield Done(status="completed", final_text="".join(final_chunks))
