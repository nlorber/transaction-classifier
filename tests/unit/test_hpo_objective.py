"""Tests for the HPO objective function factory."""

from unittest.mock import MagicMock, patch

import numpy as np
import optuna

from transaction_classifier.training.hpo.objective import build_objective_fn


@patch("transaction_classifier.training.hpo.objective.xgb")
@patch("transaction_classifier.training.hpo.objective.draw_hyperparams")
def test_objective_returns_accuracy(mock_draw_hyperparams, mock_xgb):
    """Test that the objective function returns accuracy score."""
    mock_draw_hyperparams.return_value = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
    }

    # Mock the DMatrix constructor
    mock_xgb.DMatrix.return_value = MagicMock()

    # Mock xgb.train to return a booster that predicts perfectly
    mock_booster = MagicMock()
    mock_booster.best_iteration = 50
    # 3 classes, 5 samples — predict class 0 for all
    preds = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
        ]
    )
    mock_booster.predict.return_value = preds
    mock_xgb.train.return_value = mock_booster

    X_train = np.random.rand(10, 5)
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    X_val = np.random.rand(5, 5)
    y_val = np.array([0, 1, 2, 0, 1])

    objective = build_objective_fn(X_train, y_train, X_val, y_val, n_classes=3)

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    result = objective(trial)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    mock_xgb.train.assert_called_once()


@patch("transaction_classifier.training.hpo.objective.xgb")
@patch("transaction_classifier.training.hpo.objective.draw_hyperparams")
def test_objective_sets_user_attrs(mock_draw_hyperparams, mock_xgb):
    """Test that user attributes (training_time, balanced_accuracy, etc.) are set."""
    mock_draw_hyperparams.return_value = {
        "n_estimators": 100,
        "max_depth": 6,
    }
    mock_xgb.DMatrix.return_value = MagicMock()

    mock_booster = MagicMock()
    mock_booster.best_iteration = 42
    preds = np.eye(3)  # perfect predictions for 3 samples
    mock_booster.predict.return_value = preds
    mock_xgb.train.return_value = mock_booster

    X_train = np.random.rand(10, 5)
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    X_val = np.random.rand(3, 5)
    y_val = np.array([0, 1, 2])

    objective = build_objective_fn(X_train, y_train, X_val, y_val, n_classes=3)

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    objective(trial)

    assert "training_time" in trial.user_attrs
    assert "balanced_accuracy" in trial.user_attrs
    assert "best_iteration" in trial.user_attrs
    assert "n_estimators_requested" in trial.user_attrs
    assert trial.user_attrs["best_iteration"] == 42


@patch("transaction_classifier.training.hpo.objective.xgb")
@patch("transaction_classifier.training.hpo.objective.draw_hyperparams")
def test_objective_prunes_on_training_error(mock_draw_hyperparams, mock_xgb):
    """Test that generic training exceptions raise TrialPruned instead of returning a score."""
    mock_draw_hyperparams.return_value = {"n_estimators": 100, "max_depth": 6}
    mock_xgb.DMatrix.return_value = MagicMock()
    mock_xgb.train.side_effect = RuntimeError("CUDA out of memory")

    X_train = np.random.rand(10, 5)
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    X_val = np.random.rand(3, 5)
    y_val = np.array([0, 1, 2])

    objective = build_objective_fn(X_train, y_train, X_val, y_val, n_classes=3)

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    import pytest

    with pytest.raises(optuna.TrialPruned, match="CUDA out of memory"):
        objective(trial)


@patch("transaction_classifier.training.hpo.objective.xgb")
@patch("transaction_classifier.training.hpo.objective.draw_hyperparams")
def test_objective_propagates_trial_pruned(mock_draw_hyperparams, mock_xgb):
    """Test that TrialPruned is re-raised so Optuna can handle pruning."""
    mock_draw_hyperparams.return_value = {"n_estimators": 100, "max_depth": 6}
    mock_xgb.DMatrix.return_value = MagicMock()
    mock_xgb.train.side_effect = optuna.TrialPruned()

    X_train = np.random.rand(10, 5)
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    X_val = np.random.rand(3, 5)
    y_val = np.array([0, 1, 2])

    objective = build_objective_fn(X_train, y_train, X_val, y_val, n_classes=3)

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    import pytest

    with pytest.raises(optuna.TrialPruned):
        objective(trial)


@patch("transaction_classifier.training.hpo.objective.xgb")
@patch("transaction_classifier.training.hpo.objective.draw_hyperparams")
def test_objective_passes_device_param(mock_draw_hyperparams, mock_xgb):
    """Test that the device parameter is forwarded to xgb.train params."""
    mock_draw_hyperparams.return_value = {"n_estimators": 100, "max_depth": 6}
    mock_xgb.DMatrix.return_value = MagicMock()

    mock_booster = MagicMock()
    mock_booster.best_iteration = 10
    mock_booster.predict.return_value = np.eye(3)
    mock_xgb.train.return_value = mock_booster

    X_train = np.random.rand(10, 5)
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    X_val = np.random.rand(3, 5)
    y_val = np.array([0, 1, 2])

    objective = build_objective_fn(X_train, y_train, X_val, y_val, n_classes=3, device="cuda")

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    objective(trial)

    call_args = mock_xgb.train.call_args
    xgb_params = call_args[0][0] if call_args[0] else call_args[1].get("params", {})
    assert xgb_params["device"] == "cuda"
