import json, os, shutil, socket as _socket, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "state"

def test_daemon_handoff_and_watchdog():
    # Create stub agent meta — PID 99999 does not exist, so watchdog marks completed immediately
    agent_dir = STATE / "agents" / "test-agent-001"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "meta.json").write_text(json.dumps({
        "agent_id": "test-agent-001",
        "pid": 99999,
        "budget_tokens": 5000,
        "status": "running",
        "tokens_used": 0,
    }))

    proc = subprocess.Popen(["python", str(REPO / "commandd.py")])
    try:
        sock_path = STATE / "commandd.sock"

        # Wait for socket to appear (up to 5s)
        deadline = time.time() + 5
        while not sock_path.exists() and time.time() < deadline:
            time.sleep(0.1)
        assert sock_path.exists(), "daemon did not create socket within 5s"

        # Send handoff
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
            s.connect(str(sock_path))
            s.sendall(b"HANDOFF:test-agent-001\n")

        # Daemon must still be alive immediately (crash-on-receipt check)
        assert proc.poll() is None, "daemon crashed on handoff message"

        # Wait one poll interval (2s) then verify watchdog flipped status to completed
        time.sleep(2.5)
        meta = json.loads((agent_dir / "meta.json").read_text())
        assert meta["status"] == "completed", f"watchdog did not mark completed; got: {meta['status']}"

        # Shut down and verify PID file cleaned up
        proc.terminate()
        proc.wait(timeout=5)
        assert not (STATE / "commandd.pid").exists(), "PID file not removed on exit"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        # Stub agent must not linger: the dashboard renders everything in state/agents/
        shutil.rmtree(agent_dir, ignore_errors=True)
