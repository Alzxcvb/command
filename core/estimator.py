"""Estimate task type, recommended runtime/model, and token budget for a task."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Make repo root importable so `router` package resolves
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from router.classifier import classify  # noqa: E402

CONFIG_PATH = _REPO_ROOT / "config" / "routing_rules.yaml"


@dataclass
class EstimatorResult:
    task_type: str
    recommended_runtime: str
    recommended_model: str
    estimated_tokens: int
    requires_metered: bool


def _load_rules() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def estimate(task: str) -> EstimatorResult:
    rules = _load_rules()

    try:
        cls_result = classify(task, method="rules")
        tt = cls_result.task_type
        task_type_str = tt.value if hasattr(tt, "value") else str(tt).lower()
    except Exception:
        task_type_str = "code"

    rule = rules["task_types"].get(task_type_str) or rules["task_types"]["default"]

    return EstimatorResult(
        task_type=task_type_str,
        recommended_runtime=rule["runtime"],
        recommended_model=rule.get("model", ""),
        estimated_tokens=int(rule.get("estimated_tokens", 5000)),
        requires_metered=bool(rule.get("metered", False)),
    )
