"""v1 static turn-routing rules + savings math + turn ledger."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import core.metered_ledger as ledger
from core.turn_router import (
    HAIKU,
    OPUS,
    SONNET,
    RouterConfig,
    Turn,
    estimate_turn_cost_usd,
    route_turn,
    savings_vs_all_opus,
)


class TestRuleTable:
    def test_destructive_tools_route_to_opus(self):
        assert route_turn(Turn("t1", tools=["Bash"]), []) == OPUS
        assert route_turn(Turn("t2", tools=["Edit"]), []) == OPUS
        assert route_turn(Turn("t3", tools=["Write"]), []) == OPUS

    def test_planning_text_routes_to_opus(self):
        assert route_turn(Turn("t1", text="Design the architecture for the new module"), []) == OPUS

    def test_web_tools_route_to_sonnet(self):
        assert route_turn(Turn("t1", tools=["WebSearch"]), []) == SONNET
        assert route_turn(Turn("t2", tools=["WebFetch"]), []) == SONNET

    def test_lookup_tools_route_to_haiku(self):
        assert route_turn(Turn("t1", tools=["Read"]), []) == HAIKU
        assert route_turn(Turn("t2", tools=["Grep", "Glob"]), []) == HAIKU

    def test_destructive_beats_lookup_and_web(self):
        turn = Turn("t1", tools=["Read", "WebSearch", "Bash"])
        assert route_turn(turn, []) == OPUS

    def test_default_inherits_previous_model(self):
        history = [Turn("t0", model=HAIKU)]
        assert route_turn(Turn("t1"), history) == HAIKU

    def test_default_without_history_uses_config(self):
        assert route_turn(Turn("t1"), []) == SONNET  # RouterConfig default

    def test_pin_model_overrides_everything(self):
        cfg = RouterConfig(pin_model=OPUS)
        assert route_turn(Turn("t1", tools=["Read"]), [], cfg) == OPUS


class TestSavings:
    def test_opus_costs_more_than_haiku(self):
        assert estimate_turn_cost_usd(OPUS, 1_000_000) > estimate_turn_cost_usd(HAIKU, 1_000_000)

    def test_savings_vs_all_opus(self):
        turns = [
            {"model": OPUS, "tokens": 1_000_000},
            {"model": HAIKU, "tokens": 1_000_000},
        ]
        s = savings_vs_all_opus(turns)
        assert s["turns"] == 2
        assert s["downgraded_pct"] == 50.0
        assert s["all_opus_usd"] > s["actual_usd"]
        assert s["saved_usd"] == round(s["all_opus_usd"] - s["actual_usd"], 6)

    def test_empty_turns(self):
        s = savings_vs_all_opus([])
        assert s == {"turns": 0, "downgraded_pct": 0.0, "actual_usd": 0.0,
                     "all_opus_usd": 0.0, "saved_usd": 0.0}


class TestTurnLedger:
    def test_record_and_read_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ledger, "TURN_LEDGER_PATH", tmp_path / "turn_ledger.jsonl")
        ledger.record_turn("turn-1", SONNET, 4200, estimated_cost_usd=0.0231)
        ledger.record_turn("turn-2", HAIKU, 800)

        turns = ledger.read_turns()
        assert len(turns) == 2
        assert turns[0]["turn_id"] == "turn-1"
        assert turns[0]["model"] == SONNET
        assert turns[0]["tokens"] == 4200
        assert turns[0]["estimated_cost_usd"] == 0.0231
        assert turns[0]["ts"]
        assert turns[1]["tokens"] == 800

    def test_read_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ledger, "TURN_LEDGER_PATH", tmp_path / "nope.jsonl")
        assert ledger.read_turns() == []
