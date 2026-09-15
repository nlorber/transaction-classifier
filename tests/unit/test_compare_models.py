"""Run selection for scripts/compare_models.py, which the scaling benchmark drives per model."""

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location(
        "compare_models", ROOT / "scripts" / "compare_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_runs_keeps_the_given_order(compare):
    assert compare.parse_runs("xgb-balanced, lr") == ["xgb-balanced", "lr"]


def test_parse_runs_rejects_unknown_runs(compare):
    with pytest.raises(argparse.ArgumentTypeError, match="svm"):
        compare.parse_runs("lr,svm")


@pytest.mark.parametrize(("best_round", "stopped"), [(460, False), (461, True), (500, True)])
def test_round_cap_ends_training_when_the_best_round_is_within_patience(
    compare, best_round, stopped
):
    rounds = compare._rounds(best_round, round_cap=500, patience=40)
    assert rounds == {"best_round": best_round, "round_cap": 500, "stopped_by_round_cap": stopped}


def test_default_runs_are_the_committed_comparison(compare):
    assert compare.parse_runs(compare.DEFAULT_RUNS) == [
        "lr",
        "lr-balanced",
        "xgb",
        "xgb-balanced",
        "lgbm",
    ]
