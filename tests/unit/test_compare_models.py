"""Run selection for scripts/compare_models.py, which the scaling benchmark drives per model."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

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
    assert rounds == {
        "best_round": best_round,
        "round_cap": 500,
        "stopped_by_round_cap": stopped,
        "converged": not stopped,
    }


def _early_stop(compare, losses, ran, max_iter):
    losses, ran, snapshots = iter(losses), iter(ran), []
    record = compare.early_stopped_iterations(
        step=lambda asked: min(asked, next(ran)),
        val_loss=lambda: next(losses),
        snapshot=lambda: snapshots.append(True),
        max_iter=max_iter,
        chunk=100,
        patience=3,
    )
    return record, len(snapshots)


def test_logistic_stops_when_validation_loss_stops_improving(compare):
    record, snapshots = _early_stop(
        compare, losses=[3.0, 2.0, 2.5, 2.6, 2.7], ran=[100] * 5, max_iter=5_000
    )
    assert record == {
        "iterations": 500,
        "best_iteration": 200,
        "iteration_cap": 5_000,
        "stop_reason": "validation",
        "converged": True,
    }
    assert snapshots == 2


def test_logistic_stops_when_the_solver_meets_its_tolerance(compare):
    record, _ = _early_stop(compare, losses=[3.0, 2.0], ran=[100, 40], max_iter=5_000)
    assert (record["iterations"], record["stop_reason"], record["converged"]) == (
        140,
        "tolerance",
        True,
    )


def test_logistic_stopped_by_the_cap_has_not_converged(compare):
    record, snapshots = _early_stop(compare, losses=[3.0, 2.0, 1.0], ran=[100] * 3, max_iter=250)
    assert record == {
        "iterations": 250,
        "best_iteration": 250,
        "iteration_cap": 250,
        "stop_reason": "cap",
        "converged": False,
    }
    assert snapshots == 3


def test_train_logistic_returns_test_probabilities_and_a_stop_record(compare):
    rng = np.random.default_rng(0)
    features = sparse.csr_matrix(rng.normal(size=(300, 5)))
    y = np.arange(300) % 3
    proba, _, record = compare._train_logistic(
        features[:200], y[:200], features[200:250], y[200:250], features[250:], True, 300
    )
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert 0 < record["best_iteration"] <= record["iterations"] <= 300
    assert record["stop_reason"] in {"validation", "tolerance", "cap"}


def test_default_runs_are_the_committed_comparison(compare):
    assert compare.parse_runs(compare.DEFAULT_RUNS) == [
        "lr",
        "lr-balanced",
        "xgb",
        "xgb-balanced",
        "lgbm",
    ]
