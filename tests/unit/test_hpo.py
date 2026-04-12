"""Tests for HPO search space and integration."""

import inspect
from unittest.mock import patch

import optuna

from transaction_classifier.core.config import Settings
from transaction_classifier.core.data.splitter import split_by_date
from transaction_classifier.training.hpo.cli import load_and_prepare_data
from transaction_classifier.training.hpo.objective import build_objective_fn
from transaction_classifier.training.hpo.search_space import BASELINE_PARAMS, draw_hyperparams


def test_draw_hyperparams_returns_all_expected_keys():
    study = optuna.create_study()
    trial = study.ask()
    params = draw_hyperparams(trial)
    expected_keys = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "reg_alpha",
        "reg_lambda",
        "min_child_weight",
        "gamma",
        "max_delta_step",
        "max_bin",
    }
    assert set(params.keys()) == expected_keys


def test_draw_hyperparams_values_in_range():
    study = optuna.create_study()
    trial = study.ask()
    params = draw_hyperparams(trial)
    assert 200 <= params["n_estimators"] <= 1500
    assert 3 <= params["max_depth"] <= 10
    assert 0.005 <= params["learning_rate"] <= 0.3
    assert 0.5 <= params["subsample"] <= 1.0
    assert 0.4 <= params["colsample_bytree"] <= 1.0
    assert 0.4 <= params["colsample_bylevel"] <= 1.0
    assert 0.01 <= params["reg_alpha"] <= 10.0
    assert 0.1 <= params["reg_lambda"] <= 20.0
    assert 1 <= params["min_child_weight"] <= 50
    assert 0.0 <= params["gamma"] <= 5.0
    assert 0 <= params["max_delta_step"] <= 5
    assert 128 <= params["max_bin"] <= 512


def test_baseline_params_keys_match_draw_hyperparams():
    study = optuna.create_study()
    trial = study.ask()
    params = draw_hyperparams(trial)
    assert set(BASELINE_PARAMS.keys()) == set(params.keys())


def test_load_and_prepare_data_uses_temporal_split(sample_csv_path):
    """Verify HPO data loading uses temporal split, not random stratified split."""
    settings = Settings(data_path=str(sample_csv_path), min_class_samples=1)
    with patch(
        "transaction_classifier.training.hpo.cli.split_by_date",
        wraps=split_by_date,
    ) as mock_split:
        load_and_prepare_data(settings)
        mock_split.assert_called_once()


def test_objective_uses_explicit_trial_pruned_exception():
    """Verify objective.py uses explicit optuna.TrialPruned, not string-based detection."""
    source = inspect.getsource(build_objective_fn)
    assert '"pruned" in str(' not in source, (
        "objective.py still uses fragile string-based TrialPruned detection"
    )
    assert "optuna.TrialPruned" in source, (
        "objective.py should catch optuna.TrialPruned explicitly"
    )
