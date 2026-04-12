"""XGBoost hyperparameter search-space definition."""

from typing import Any

import optuna

from ...core.models.xgboost_model import XGBoostModel

# Derived from the model's default constructor to stay in sync automatically.
_EXCLUDED_KEYS = frozenset({"patience", "random_state"})
BASELINE_PARAMS: dict[str, Any] = {
    k: v for k, v in XGBoostModel().hyperparameters().items() if k not in _EXCLUDED_KEYS
}


def draw_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    """Sample a full XGBoost parameter set for a single Optuna trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
    }
