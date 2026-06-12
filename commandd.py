"""commandd — PID-watchdog daemon for detached agent spawns.

`cli spawn --detach` returns immediately, but the monitor thread that enforces
timeouts and budget caps (agents/lifecycle.py:_monitor) is a daemon thread in
the CLI process: when the CLI exits, enforcement dies with it. The live `proc`
handle that _monitor needs cannot be serialized over a socket, so commandd
implements a PID watchdog instead: the CLI hands off an agent_id over a Unix
socket, and commandd polls the agent's OS pid until it exits (marking the agent
completed) or its meta.json shows a hard budget overrun (killing it).

KNOWN GAP — budget enforcement here is incomplete. `tokens_used` is written
only by lifecycle._monitor(), which dies with the CLI process. After handoff
the field freezes at its last value (0 for a freshly detached spawn), so the
killed_over_budget branch below cannot fire in practice. A second mechanism to
update tokens_used post-handoff is needed. See docs/BLOCKERS.md.

Start:  python commandd.py &
Stop:   kill $(cat state/commandd.pid)
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STATE_ROOT = REPO_ROOT / "state"
AGENTS_DIR = STATE_ROOT / "agents"
SOCK_PATH = STATE_ROOT / "commandd.sock"
PID_PATH = STATE_ROOT / "commandd.pid"

POLL_INTERVAL_S = 2
BUDGET_KILL_MULTIPLIER = 3
SIGTERM_GRACE_S = 3


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _log(msg: str) -> None:
    print(f"[commandd] {msg}", flush=True)


def _meta_path(agent_id: str) -> Path:
    return AGENTS_DIR / agent_id / "meta.json"


def _read_meta(agent_id: str) -> dict | None:
    p = _meta_path(agent_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_meta(agent_id: str, meta: dict) -> None:
    meta["updated_at"] = _now_iso()
    _meta_path(agent_id).write_text(json.dumps(meta, indent=2))


def _watchdog(agent_id: str, pid: int, budget_tokens: int) -> None:
    _log(f"watchdog started: {agent_id} pid={pid} budget={budget_tokens}")
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # Process gone: normal exit. Record final tokens and close out.
            meta = _read_meta(agent_id) or {"agent_id": agent_id}
            tokens = int(meta.get("tokens_used", 0) or 0)
            meta["status"] = "completed"
            meta["ended_at"] = _now_iso()
            _write_meta(agent_id, meta)
            _log(f"{agent_id}: pid {pid} exited — marked completed (tokens_used={tokens})")
            return

        meta = _read_meta(agent_id) or {}
        tokens = int(meta.get("tokens_used", 0) or 0)
        # Dormant in practice — see KNOWN GAP in module docstring: tokens_used
        # stops updating once the spawning CLI process has exited.
        if budget_tokens > 0 and tokens > budget_tokens * BUDGET_KILL_MULTIPLIER:
            _log(f"{agent_id}: tokens_used={tokens:,} > {BUDGET_KILL_MULTIPLIER}x budget {budget_tokens:,} — killing pid {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(SIGTERM_GRACE_S)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            meta = _read_meta(agent_id) or {"agent_id": agent_id}
            meta["status"] = "killed_over_budget"
            meta["ended_at"] = _now_iso()
            _write_meta(agent_id, meta)
            return

        time.sleep(POLL_INTERVAL_S)


def _handle_handoff(agent_id: str) -> None:
    meta = _read_meta(agent_id)
    if meta is None:
        _log(f"warning: handoff for {agent_id} but no readable meta.json — ignoring")
        return
    pid = meta.get("pid")
    if not pid:
        _log(f"warning: handoff for {agent_id} has no pid in meta.json — ignoring")
        return
    budget = int(meta.get("budget_tokens", 0) or 0)
    threading.Thread(
        target=_watchdog,
        args=(agent_id, int(pid), budget),
        daemon=True,
        name=f"watchdog-{agent_id}",
    ).start()


def _cleanup() -> None:
    PID_PATH.unlink(missing_ok=True)
    SOCK_PATH.unlink(missing_ok=True)


def _on_sigterm(signum, frame):
    raise SystemExit(0)


def main() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    SOCK_PATH.unlink(missing_ok=True)  # stale socket from an unclean exit blocks bind()
    PID_PATH.write_text(str(os.getpid()))
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, _on_sigterm)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(SOCK_PATH))
        server.listen(8)
        _log(f"listening on {SOCK_PATH} (pid {os.getpid()})")
        while True:
            conn, _ = server.accept()
            with conn:
                conn.settimeout(5)
                data = b""
                try:
                    while b"\n" not in data:
                        chunk = conn.recv(1024)
                        if not chunk:
                            break
                        data += chunk
                except socket.timeout:
                    _log("warning: client connection timed out mid-message")
                for line in data.decode("utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line.startswith("HANDOFF:"):
                        _handle_handoff(line[len("HANDOFF:"):].strip())
                    elif line:
                        _log(f"warning: unknown message {line!r}")
    finally:
        server.close()
        _cleanup()


if __name__ == "__main__":
    main()
