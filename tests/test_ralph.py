"""Tests for orchestrator/ralph.py — judge council, verdict parsing, and improve() flow.

All tests are fully isolated: no real API calls, no git operations, no file
mutations to the actual guide or state directories.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestrator import ralph


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

def _make_sample(agent_id: str = "agent-001", fail_reason: str = "failed") -> dict:
    return {
        "agent_id": agent_id,
        "dispatch": {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "runtime": "claude_code",
            "model": "sonnet",
            "budget_tokens": 10000,
            "drafted_task": "Do something useful",
        },
        "outcome": {
            "status": "failed",
            "tokens_used": 0,
            "final_text": "",
        },
        "retries": [],
        "thumbs": [],
        "fail_reasons": [fail_reason],
    }


APPROVE_RESPONSE = """\
Criterion 1: 4/5 - The change directly targets the observed timeout failures.
Criterion 2: 4/5 - Core orchestration intent is preserved.
Criterion 3: 3/5 - Instructions are specific enough.
Criterion 4: 4/5 - Likely to reduce over-budget runs.

Verdict: APPROVE. All criteria meet the threshold. Weakest criterion is 3 (instruction clarity).
"""

REJECT_RESPONSE = """\
Criterion 1: 2/5 - The change does not address the specific timeout failures.
Criterion 2: 4/5 - Core intent is preserved.
Criterion 3: 3/5 - Instructions are reasonably specific.
Criterion 4: 3/5 - Unclear if budget overruns will reduce.

Verdict: REJECT. Weakest criterion is 1 (addresses failures) with a score of 2/5.
"""


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def test_approve_on_verdict_line(self):
        verdict, reason = ralph._parse_verdict(APPROVE_RESPONSE)
        assert verdict == "APPROVE"
        assert len(reason) > 0

    def test_reject_on_verdict_line(self):
        verdict, reason = ralph._parse_verdict(REJECT_RESPONSE)
        assert verdict == "REJECT"
        assert len(reason) > 0

    def test_empty_text_defaults_to_reject(self):
        verdict, _ = ralph._parse_verdict("")
        assert verdict == "REJECT"

    def test_no_verdict_line_uses_last_keyword(self):
        text = "This seems fine. APPROVE is likely here."
        verdict, _ = ralph._parse_verdict(text)
        assert verdict == "APPROVE"

    def test_last_keyword_wins_when_both_present(self):
        # REJECT appears last, so it wins
        text = "Could APPROVE but ultimately REJECT due to ambiguity."
        verdict, _ = ralph._parse_verdict(text)
        assert verdict == "REJECT"

    def test_approve_wins_when_last(self):
        text = "Initial read: REJECT. On reflection: APPROVE."
        verdict, _ = ralph._parse_verdict(text)
        assert verdict == "APPROVE"

    def test_reason_equals_stripped_input(self):
        _, reason = ralph._parse_verdict("  Verdict: APPROVE. Good.  ")
        assert reason == "Verdict: APPROVE. Good."

    def test_verdict_line_case_insensitive(self):
        text = "verdict: approve this change."
        verdict, _ = ralph._parse_verdict(text)
        assert verdict == "APPROVE"


# ---------------------------------------------------------------------------
# _call_judge
# ---------------------------------------------------------------------------

class TestCallJudge:
    def test_judge_model_ids_are_registered(self):
        """Both judge models must be present in the router model registry."""
        from router.models import MODELS
        assert ralph.HAIKU_JUDGE_MODEL in MODELS, (
            f"Haiku judge model {ralph.HAIKU_JUDGE_MODEL!r} not in router registry"
        )
        assert ralph.OPUS_JUDGE_MODEL in MODELS, (
            f"Opus judge model {ralph.OPUS_JUDGE_MODEL!r} not in router registry"
        )

    def test_raises_runtime_error_when_no_keys(self):
        """If all dispatch paths raise ValueError, _call_judge raises RuntimeError."""
        # Build mock providers that all fail with ValueError
        failing_provider = MagicMock()
        failing_provider.call_raw.side_effect = ValueError("No key")

        mock_model = MagicMock()

        # Inject mocks for the lazy-imported modules inside _call_judge
        mock_models_mod = MagicMock()
        mock_models_mod.get_model.return_value = mock_model

        mock_providers_mod = MagicMock()
        mock_providers_mod.get_provider_for.return_value = failing_provider
        mock_providers_mod.AnthropicProvider.return_value = failing_provider
        mock_providers_mod.OpenRouterProvider.return_value = failing_provider

        with patch.dict(sys.modules, {
            "router.models": mock_models_mod,
            "router.providers": mock_providers_mod,
        }):
            with pytest.raises(RuntimeError, match="No API key available"):
                ralph._call_judge(ralph.HAIKU_JUDGE_MODEL, "test prompt")

    def test_fallback_chain_order(self):
        """Router dispatch is tried first, then direct Anthropic, then OpenRouter."""
        call_order: list[str] = []

        def make_failing(name: str):
            m = MagicMock()
            def fail(*a, **kw):
                call_order.append(name)
                raise ValueError(f"No key for {name}")
            m.call_raw.side_effect = fail
            return m

        router_provider = make_failing("router")
        anthropic_provider = make_failing("direct_anthropic")
        openrouter_provider = make_failing("openrouter")

        mock_model = MagicMock()

        mock_models_mod = MagicMock()
        mock_models_mod.get_model.return_value = mock_model

        mock_providers_mod = MagicMock()
        mock_providers_mod.get_provider_for.return_value = router_provider
        mock_providers_mod.AnthropicProvider.return_value = anthropic_provider
        mock_providers_mod.OpenRouterProvider.return_value = openrouter_provider

        with patch.dict(sys.modules, {
            "router.models": mock_models_mod,
            "router.providers": mock_providers_mod,
        }):
            with pytest.raises(RuntimeError):
                ralph._call_judge(ralph.HAIKU_JUDGE_MODEL, "prompt")

        # Router must be tried before direct Anthropic, and direct Anthropic before OpenRouter
        assert call_order == ["router", "direct_anthropic", "openrouter"]


# ---------------------------------------------------------------------------
# judge_guide_revision
# ---------------------------------------------------------------------------

class TestJudgeGuideRevision:
    def test_both_approve_returns_council_approved_true(self):
        with patch.object(ralph, "_call_judge", return_value=APPROVE_RESPONSE):
            result = ralph.judge_guide_revision("original", "proposed", [_make_sample()])
        assert result["council_approved"] is True
        assert result["haiku_verdict"] == "APPROVE"
        assert result["opus_verdict"] == "APPROVE"

    def test_haiku_rejects_returns_council_approved_false(self):
        def side_effect(model_id, prompt):
            return REJECT_RESPONSE if "haiku" in model_id else APPROVE_RESPONSE

        with patch.object(ralph, "_call_judge", side_effect=side_effect):
            result = ralph.judge_guide_revision("original", "proposed", [_make_sample()])
        assert result["council_approved"] is False
        assert result["haiku_verdict"] == "REJECT"
        assert result["opus_verdict"] == "APPROVE"

    def test_opus_rejects_returns_council_approved_false(self):
        def side_effect(model_id, prompt):
            return REJECT_RESPONSE if "opus" in model_id else APPROVE_RESPONSE

        with patch.object(ralph, "_call_judge", side_effect=side_effect):
            result = ralph.judge_guide_revision("original", "proposed", [_make_sample()])
        assert result["council_approved"] is False
        assert result["haiku_verdict"] == "APPROVE"
        assert result["opus_verdict"] == "REJECT"

    def test_both_reject_returns_council_approved_false(self):
        with patch.object(ralph, "_call_judge", return_value=REJECT_RESPONSE):
            result = ralph.judge_guide_revision("original", "proposed", [_make_sample()])
        assert result["council_approved"] is False
        assert result["haiku_verdict"] == "REJECT"
        assert result["opus_verdict"] == "REJECT"

    def test_result_has_required_keys(self):
        with patch.object(ralph, "_call_judge", return_value=APPROVE_RESPONSE):
            result = ralph.judge_guide_revision("original", "proposed", [])
        assert set(result.keys()) == {
            "haiku_verdict", "opus_verdict",
            "haiku_reason", "opus_reason",
            "council_approved",
        }

    def test_reasons_are_non_empty_strings(self):
        with patch.object(ralph, "_call_judge", return_value=APPROVE_RESPONSE):
            result = ralph.judge_guide_revision("orig", "prop", [])
        assert isinstance(result["haiku_reason"], str) and result["haiku_reason"]
        assert isinstance(result["opus_reason"], str) and result["opus_reason"]

    def test_haiku_called_before_opus(self):
        """Haiku (fast, cheap) must be dispatched first."""
        called_models = []

        def side_effect(model_id, prompt):
            called_models.append(model_id)
            return APPROVE_RESPONSE

        with patch.object(ralph, "_call_judge", side_effect=side_effect):
            ralph.judge_guide_revision("orig", "prop", [])

        assert len(called_models) == 2
        assert called_models[0] == ralph.HAIKU_JUDGE_MODEL
        assert called_models[1] == ralph.OPUS_JUDGE_MODEL

    def test_empty_samples_does_not_crash(self):
        with patch.object(ralph, "_call_judge", return_value=APPROVE_RESPONSE):
            result = ralph.judge_guide_revision("orig", "prop", [])
        assert "council_approved" in result


# ---------------------------------------------------------------------------
# improve() integration with council
# ---------------------------------------------------------------------------

def _make_improver_output(revised: str) -> str:
    """Wrap a revised guide string in the expected markers.

    Note: _extract_revised strips leading and trailing newlines from the
    content between markers, so the roundtripped guide will not have a
    trailing newline even if `revised` does.
    """
    return (
        f"Addressing failures by clarifying budget rules.\n\n"
        f"{ralph.REVISED_START}\n{revised}\n{ralph.REVISED_END}"
    )


def _extract_expected(revised: str) -> str:
    """Mimic what _extract_revised does so test assertions match reality."""
    return revised.strip("\n")


class TestImproveCouncilIntegration:
    """Test improve() end-to-end with all external calls mocked."""

    def _base_patches(self, tmp_path, guide_content, improver_output, council_result):
        guide = tmp_path / "orchestrator.md"
        guide.write_text(guide_content)
        rejections = tmp_path / "council_rejections.jsonl"

        return {
            "guide_path": guide,
            "rejections_path": rejections,
            "patches": [
                patch.object(ralph, "GUIDE_PATH", guide),
                patch.object(ralph, "COUNCIL_REJECTIONS_PATH", rejections),
                patch.object(ralph.observations, "failure_samples",
                             return_value=[_make_sample("a"), _make_sample("b"), _make_sample("c")]),
                patch.object(ralph.observations, "last_improvement_ts", return_value=None),
                patch.object(ralph.observations, "current_guide_sha", return_value="deadbeef1234"),
                patch.object(ralph.observations, "record_improvement"),
                patch.object(ralph, "spawn_agent", return_value="agent-ralph-001"),
                patch.object(ralph, "get_agent", return_value={"status": "completed"}),
                patch.object(ralph, "get_result",
                             return_value={"final_text": improver_output}),
                patch.object(ralph, "_git_commit_guide", return_value=True),
                patch.object(ralph, "judge_guide_revision", return_value=council_result),
            ],
        }

    def test_council_approved_writes_guide_and_commits(self, tmp_path):
        original = "# Original\n"
        revised = "# Revised\nBetter instructions.\n"

        cfg = self._base_patches(
            tmp_path,
            original,
            _make_improver_output(revised),
            {
                "haiku_verdict": "APPROVE", "opus_verdict": "APPROVE",
                "haiku_reason": "Looks good", "opus_reason": "Agreed",
                "council_approved": True,
            },
        )

        with _enter_patches(cfg["patches"]) as mocks:
            result = ralph.improve(wait=True)

        assert result["status"] == "applied"
        # _extract_revised strips leading/trailing newlines from the marker content
        assert cfg["guide_path"].read_text() == _extract_expected(revised)
        assert not cfg["rejections_path"].exists()
        mocks["_git_commit_guide"].assert_called_once()
        mocks["record_improvement"].assert_called_once()

    def test_council_rejected_does_not_write_guide(self, tmp_path):
        original = "# Original\n"
        revised = "# Revised bad\n"

        cfg = self._base_patches(
            tmp_path,
            original,
            _make_improver_output(revised),
            {
                "haiku_verdict": "REJECT", "opus_verdict": "APPROVE",
                "haiku_reason": "Fails criterion 1", "opus_reason": "Approved",
                "council_approved": False,
            },
        )

        with _enter_patches(cfg["patches"]) as mocks:
            result = ralph.improve(wait=True)

        assert result["status"] == "council_rejected"
        # Guide must remain unchanged
        assert cfg["guide_path"].read_text() == original
        # No commit
        mocks["_git_commit_guide"].assert_not_called()
        # No record_improvement (processed_through_ts must not advance)
        mocks["record_improvement"].assert_not_called()

    def test_council_rejected_logs_to_rejections_jsonl(self, tmp_path):
        original = "# Original\n"
        revised = "# Rejected revision\n"

        cfg = self._base_patches(
            tmp_path,
            original,
            _make_improver_output(revised),
            {
                "haiku_verdict": "REJECT", "opus_verdict": "REJECT",
                "haiku_reason": "Criterion 1 score 2", "opus_reason": "Too broad",
                "council_approved": False,
            },
        )

        with _enter_patches(cfg["patches"]):
            ralph.improve(wait=True)

        assert cfg["rejections_path"].exists()
        lines = [
            l for l in cfg["rejections_path"].read_text().splitlines() if l.strip()
        ]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "timestamp" in entry
        assert "proposed_sha" in entry
        assert entry["haiku_verdict"] == "REJECT"
        assert entry["opus_verdict"] == "REJECT"
        assert entry["haiku_reason"] == "Criterion 1 score 2"
        assert entry["opus_reason"] == "Too broad"

    def test_council_rejected_result_has_verdict_keys(self, tmp_path):
        original = "# Original\n"
        revised = "# Rejected\n"

        cfg = self._base_patches(
            tmp_path,
            original,
            _make_improver_output(revised),
            {
                "haiku_verdict": "REJECT", "opus_verdict": "APPROVE",
                "haiku_reason": "Bad", "opus_reason": "OK",
                "council_approved": False,
            },
        )

        with _enter_patches(cfg["patches"]):
            result = ralph.improve(wait=True)

        assert result["haiku_verdict"] == "REJECT"
        assert result["opus_verdict"] == "APPROVE"
        assert result["haiku_reason"] == "Bad"
        assert result["opus_reason"] == "OK"

    def test_force_flag_bypasses_min_samples_but_council_still_gates(self, tmp_path):
        original = "# Original\n"
        revised = "# Revised\n"

        guide = tmp_path / "orchestrator.md"
        guide.write_text(original)
        rejections = tmp_path / "council_rejections.jsonl"

        # Only 1 sample — below the default threshold of 3
        council_result = {
            "haiku_verdict": "REJECT", "opus_verdict": "REJECT",
            "haiku_reason": "No", "opus_reason": "No",
            "council_approved": False,
        }
        improver_output = _make_improver_output(revised)

        patches = [
            patch.object(ralph, "GUIDE_PATH", guide),
            patch.object(ralph, "COUNCIL_REJECTIONS_PATH", rejections),
            patch.object(ralph.observations, "failure_samples",
                         return_value=[_make_sample("only-one")]),
            patch.object(ralph.observations, "last_improvement_ts", return_value=None),
            patch.object(ralph.observations, "current_guide_sha", return_value="abc123"),
            patch.object(ralph.observations, "record_improvement"),
            patch.object(ralph, "spawn_agent", return_value="agent-ralph-001"),
            patch.object(ralph, "get_agent", return_value={"status": "completed"}),
            patch.object(ralph, "get_result", return_value={"final_text": improver_output}),
            patch.object(ralph, "_git_commit_guide", return_value=True),
            patch.object(ralph, "judge_guide_revision", return_value=council_result),
        ]

        with _enter_patches(patches) as mocks:
            result = ralph.improve(force=True, wait=True)

        assert result["status"] == "council_rejected"
        assert guide.read_text() == original
        mocks["_git_commit_guide"].assert_not_called()

    def test_improve_skips_without_force_below_min_samples(self, tmp_path):
        guide = tmp_path / "orchestrator.md"
        guide.write_text("# Guide\n")

        patches = [
            patch.object(ralph, "GUIDE_PATH", guide),
            patch.object(ralph.observations, "failure_samples",
                         return_value=[_make_sample("only-one")]),
            patch.object(ralph.observations, "last_improvement_ts", return_value=None),
        ]

        with _enter_patches(patches):
            result = ralph.improve(force=False, min_samples=3, wait=True)

        assert result["status"] == "skipped"

    def test_improve_no_diff_when_markers_absent(self, tmp_path):
        guide = tmp_path / "orchestrator.md"
        guide.write_text("# Guide\n")

        patches = [
            patch.object(ralph, "GUIDE_PATH", guide),
            patch.object(ralph.observations, "failure_samples",
                         return_value=[_make_sample("a"), _make_sample("b"), _make_sample("c")]),
            patch.object(ralph.observations, "last_improvement_ts", return_value=None),
            patch.object(ralph.observations, "current_guide_sha", return_value="abc"),
            patch.object(ralph.observations, "record_improvement"),
            patch.object(ralph, "spawn_agent", return_value="agent-ralph-001"),
            patch.object(ralph, "get_agent", return_value={"status": "completed"}),
            patch.object(ralph, "get_result",
                         return_value={"final_text": "No markers here."}),
        ]

        with _enter_patches(patches):
            result = ralph.improve(wait=True)

        assert result["status"] == "no_diff"


# ---------------------------------------------------------------------------
# Multiple rejections accumulate in the log
# ---------------------------------------------------------------------------

class TestRejectionLogAccumulation:
    def test_two_rejections_append_two_lines(self, tmp_path):
        """Each rejected improvement appends exactly one JSONL entry."""
        rejections_path = tmp_path / "council_rejections.jsonl"
        council_result = {
            "haiku_verdict": "REJECT", "opus_verdict": "REJECT",
            "haiku_reason": "reason A", "opus_reason": "reason B",
            "council_approved": False,
        }

        original_path = ralph.COUNCIL_REJECTIONS_PATH
        try:
            ralph.COUNCIL_REJECTIONS_PATH = rejections_path
            ralph._log_council_rejection(council_result, "revision one")
            ralph._log_council_rejection(council_result, "revision two")
        finally:
            ralph.COUNCIL_REJECTIONS_PATH = original_path

        lines = [l for l in rejections_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)
            assert entry["haiku_verdict"] == "REJECT"
            assert "timestamp" in entry
            assert "proposed_sha" in entry

    def test_proposed_sha_is_12_hex_chars(self, tmp_path):
        rejections_path = tmp_path / "rejections.jsonl"
        original_path = ralph.COUNCIL_REJECTIONS_PATH
        try:
            ralph.COUNCIL_REJECTIONS_PATH = rejections_path
            ralph._log_council_rejection(
                {
                    "haiku_verdict": "REJECT", "opus_verdict": "APPROVE",
                    "haiku_reason": "x", "opus_reason": "y",
                    "council_approved": False,
                },
                "some proposed guide content",
            )
        finally:
            ralph.COUNCIL_REJECTIONS_PATH = original_path

        lines = rejections_path.read_text().splitlines()
        entry = json.loads(lines[0])
        sha = entry["proposed_sha"]
        assert len(sha) == 12
        assert all(c in "0123456789abcdef" for c in sha)


# ---------------------------------------------------------------------------
# Helper: enter a flat list of patch objects as context managers
# ---------------------------------------------------------------------------

class _PatchContext:
    """Enter a list of patch objects and expose a dict of attribute -> mock."""

    def __init__(self, patches):
        self._patches = patches
        self._mocks: dict = {}
        self._entered: list = []

    def __enter__(self) -> dict:
        for p in self._patches:
            mock = p.start()
            attr = getattr(p, "attribute", None)
            if attr:
                self._mocks[attr] = mock
            self._entered.append(p)
        return self._mocks

    def __exit__(self, *args):
        for p in reversed(self._entered):
            p.stop()


def _enter_patches(patches):
    return _PatchContext(patches)
