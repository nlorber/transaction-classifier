"""Optuna objective function for XGBoost hyperparameter search."""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import xgboost as xgb
from optuna_integration import XGBoostPruningCallback
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .search_space import draw_hyperparams

logger = logging.getLogger(__name__)


def build_objective_fn(
    X_train: np.ndarray[Any, np.dtype[Any]],  # noqa: N803
    y_train: np.ndarray[Any, np.dtype[Any]],
    X_val: np.ndarray[Any, np.dtype[Any]],  # noqa: N803
    y_val: np.ndarray[Any, np.dtype[Any]],
    n_classes: int,
    device: str = "cpu",
) -> Callable[[optuna.Trial], float]:
    """Return a callable that Optuna will invoke on each trial."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def _objective(trial: optuna.Trial) -> float:
        hp = draw_hyperparams(trial)
        rounds = hp.pop("n_estimators")

        xgb_cfg = {
            **hp,
            "objective": "multi:softprob",
            "num_class": n_classes,
            "tree_method": "hist",
            "device": device,
            "seed": 42,
            "verbosity": 0,
        }

        pruner_cb = XGBoostPruningCallback(trial, "validation-mlogloss")

        t0 = time.time()
        try:
            booster = xgb.train(
                xgb_cfg,
                dtrain,
                num_boost_round=rounds,
                evals=[(dval, "validation")],
                early_stopping_rounds=40,
                callbacks=[pruner_cb],
                verbose_eval=False,
            )
        except optuna.TrialPruned:
            raise
        except (ValueError, MemoryError) as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc)
            raise optuna.TrialPruned(f"Training failed: {exc}") from exc
        except Exception as exc:
            logger.warning("Trial %d raised unexpected error: %s", trial.number, exc)
            raise optuna.TrialPruned(f"Training failed: {exc}") from exc

        elapsed = time.time() - t0

        proba = booster.predict(dval)
        y_hat = np.argmax(proba, axis=1)

        acc = accuracy_score(y_val, y_hat)
        bal_acc = balanced_accuracy_score(y_val, y_hat)

        trial.set_user_attr("training_time", elapsed)
        trial.set_user_attr("balanced_accuracy", bal_acc)
        trial.set_user_attr("best_iteration", booster.best_iteration)
        trial.set_user_attr("n_estimators_requested", rounds)

        logger.info(
            "Trial %d: acc=%.4f bal_acc=%.4f time=%.1fs (iter=%d/%d)",
            trial.number,
            acc,
            bal_acc,
            elapsed,
            booster.best_iteration,
            rounds,
        )
        return float(acc)

    return _objective
