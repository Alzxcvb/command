"""Ollama local runtime — streams from local HTTP API via inline Python helper."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from subprocess import Popen
from typing import Iterator

from .base import Done, Error, Runtime, RuntimeEvent, TextChunk, TokenUsage

_HELPER = r"""
import json, os, sys, urllib.request

host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
model = os.environ["COMMAND_OLLAMA_MODEL"]
prompt = sys.stdin.read()
body = json.dumps({"model": model, "prompt": prompt, "stream": True}).encode()
req = urllib.request.Request(f"{host}/api/generate", data=body,
                             headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req)
except Exception as e:
    print(json.dumps({"error": str(e)}), flush=True)
    sys.exit(2)
for line in resp:
    line = line.decode("utf-8", "replace").strip()
    if not line:
        continue
    try:
        evt = json.loads(line)
    except json.JSONDecodeError:
        continue
    out = {}
    if evt.get("response"):
        out["text"] = evt["response"]
    if evt.get("done"):
        out["done"] = True
        out["input_tokens"] = evt.get("prompt_eval_count", 0)
        out["output_tokens"] = evt.get("eval_count", 0)
    if out:
        print(json.dumps(out), flush=True)
"""


class OllamaRuntime(Runtime):
    name = "ollama"
    metered = False

    def __init__(self, model: str | None = None, binary: str | None = None):
        self.model = model or os.environ.get("COMMAND_LOCAL_MODEL") or "qwen2.5-coder:3b"

    def spawn(self, task: str, system_prompt: str, agent_id: str, work_dir: Path) -> Popen:
        work_dir.mkdir(parents=True, exist_ok=True)
        prompt = task if not system_prompt else f"{system_prompt}\n\n---\n\n{task}"
        env = dict(os.environ)
        env["COMMAND_OLLAMA_MODEL"] = self.model
        proc = subprocess.Popen(
            [sys.executable, "-c", _HELPER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_dir),
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdin is not None
        proc.stdin.write(prompt)
        proc.stdin.close()
        return proc

    def stream_events(self, process: Popen) -> Iterator[RuntimeEvent]:
        assert process.stdout is not None
        final_chunks: list[str] = []
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("error"):
                yield Error(message=f"ollama: {evt['error']}")
                yield Done(status="failed", error=evt["error"])
                return
            if evt.get("text"):
                yield TextChunk(text=evt["text"])
                final_chunks.append(evt["text"])
            if evt.get("done"):
                yield TokenUsage(
                    input_tokens=int(evt.get("input_tokens", 0)),
                    output_tokens=int(evt.get("output_tokens", 0)),
                    model=self.model,
                )
        rc = process.wait()
        if rc != 0:
            stderr = process.stderr.read() if process.stderr else ""
            yield Error(message=f"ollama helper exit {rc}: {stderr[:500]}")
            yield Done(status="failed", error=stderr[:500])
        else:
            yield Done(status="completed", final_text="".join(final_chunks))
