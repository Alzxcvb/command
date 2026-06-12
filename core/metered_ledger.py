"""Daily USD cap on metered API spend, file-locked for concurrent writes.

Also holds the per-turn routing ledger (state/turn_ledger.jsonl): one line per
routed turn so the dashboard can show downgrade rate and savings vs all-Opus.
"""
from __future__ import annotations

import fcntl
import json
from datetime import date, datetime, timezone
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent.parent / "state" / "metered_spend.json"
TURN_LEDGER_PATH = Path(__file__).resolve().parent.parent / "state" / "turn_ledger.jsonl"
DEFAULT_DAILY_CAP_USD = 1.00


def _today() -> str:
    return date.today().isoformat()


def _empty_day() -> dict:
    return {"date": _today(), "total_usd": 0.0, "by_agent": {}}


def _load(fh) -> dict:
    fh.seek(0)
    raw = fh.read()
    if not raw.strip():
        return _empty_day()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _empty_day()
    if data.get("date") != _today():
        return _empty_day()
    return data


def _save(fh, data: dict) -> None:
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps(data, indent=2))


def record_metered_spend(agent_id: str, usd: float) -> dict:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text("")
    with LEDGER_PATH.open("r+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            data = _load(fh)
            data["total_usd"] = round(data["total_usd"] + usd, 6)
            data["by_agent"][agent_id] = round(data["by_agent"].get(agent_id, 0.0) + usd, 6)
            _save(fh, data)
            return data
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def get_today_spend() -> dict:
    if not LEDGER_PATH.exists():
        return _empty_day()
    with LEDGER_PATH.open("r") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        try:
            return _load(fh)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def can_spawn_metered(estimated_usd: float, daily_cap_usd: float = DEFAULT_DAILY_CAP_USD) -> bool:
    today = get_today_spend()
    return (today["total_usd"] + estimated_usd) <= daily_cap_usd


def record_turn(turn_id: str, model: str, tokens: int, estimated_cost_usd: float = 0.0) -> None:
    """Append one routed turn to the turn ledger."""
    TURN_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "turn_id": turn_id,
        "model": model,
        "tokens": int(tokens),
        "estimated_cost_usd": round(float(estimated_cost_usd), 6),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with TURN_LEDGER_PATH.open("a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(line) + "\n")
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_turns() -> list[dict]:
    if not TURN_LEDGER_PATH.exists():
        return []
    out = []
    for ln in TURN_LEDGER_PATH.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
