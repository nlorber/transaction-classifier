"""XGBoost-based multi-class classifier for account-code prediction."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from scipy.sparse import spmatrix
from xgboost import XGBClassifier, callback

from .base import ClassifierBase

logger = logging.getLogger(__name__)


class CheckpointCallback(callback.TrainingCallback):
    """Periodically persist intermediate boosters during training."""

    def __init__(self, output_dir: str | Path, interval: int = 25, retain: int = 3):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.retain = retain
        self._saved: list[Path] = []

    def before_training(self, model: Any) -> Any:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return model

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        step = epoch + 1
        if step % self.interval != 0:
            return False

        ckpt = self.output_dir / f"step_{step:05d}.json"
        model.save_model(ckpt)
        self._saved.append(ckpt)

        # Write metadata
        meta: dict[str, Any] = {"iteration": step, "file": ckpt.name}
        if evals_log:
            meta["metrics"] = {
                ds: {m: list(vals) for m, vals in metrics.items()}
                for ds, metrics in evals_log.items()
            }
        (self.output_dir / "latest.json").write_text(json.dumps(meta, indent=2))

        # Prune old checkpoints
        while self.retain > 0 and len(self._saved) > self.retain:
            self._saved.pop(0).unlink(missing_ok=True)

        return False


class ProgressCallback(callback.TrainingCallback):
    """Emit training progress at regular intervals."""

    def __init__(self, every: int = 10, include_metrics: bool = True):
        self.every = every
        self.include_metrics = include_metrics
        self._t0: float | None = None

    def before_training(self, model: Any) -> Any:
        self._t0 = time.time()
        return model

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        step = epoch + 1
        if step % self.every != 0 and epoch != 0:
            return False

        if self._t0 is None:
            raise ValueError("before_training was not called")
        elapsed = time.time() - self._t0
        elapsed_fmt = f"{elapsed / 60:.1f}min" if elapsed > 60 else f"{elapsed:.0f}s"
        parts = [f"  [step {step}] {elapsed_fmt}"]

        if self.include_metrics and evals_log:
            for _ds, metrics in evals_log.items():
                for mname, vals in metrics.items():
                    if vals:
                        parts.append(f"{mname}: {vals[-1]:.4f}")

        logger.info(" | ".join(parts))
        return False


class XGBoostModel(ClassifierBase):
    """XGBoost multi-class wrapper with checkpointing and early-stopping support."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.7,
        colsample_bytree: float = 0.7,
        colsample_bylevel: float = 0.7,
        reg_alpha: float = 1.0,
        reg_lambda: float = 5.0,
        min_child_weight: int = 10,
        gamma: float = 0.5,
        max_delta_step: int = 1,
        patience: int | None = 40,
        max_bin: int = 256,
        random_state: int = 42,
        device: str = "cpu",
        n_jobs: int = -1,
        verbosity: int = 1,
        log_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 25,
        **extra: Any,
    ):
        super().__init__(random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.patience = patience
        self.max_bin = max_bin
        self.device = device
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.log_every = log_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self._extra = extra

        self.model: XGBClassifier | None = None
        self.n_classes_: int | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: spmatrix | np.ndarray[Any, np.dtype[Any]],  # noqa: N803
        y_train: np.ndarray[Any, np.dtype[Any]],
        X_val: spmatrix | np.ndarray[Any, np.dtype[Any]] | None = None,  # noqa: N803
        y_val: np.ndarray[Any, np.dtype[Any]] | None = None,
        sample_weight: np.ndarray[Any, np.dtype[Any]] | None = None,
    ) -> "XGBoostModel":
        self.n_classes_ = len(np.unique(y_train))

        cbs: list[callback.TrainingCallback] = []
        if self.log_every is not None and self.log_every > 0:
            cbs.append(ProgressCallback(every=self.log_every))
        if self.checkpoint_dir is not None:
            cbs.append(CheckpointCallback(self.checkpoint_dir, interval=self.checkpoint_interval))

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            max_delta_step=self.max_delta_step,
            tree_method="hist",
            max_bin=self.max_bin,
            device=self.device,
            objective="multi:softprob",
            num_class=self.n_classes_,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            early_stopping_rounds=self.patience,
            callbacks=cbs or None,
            **self._extra,
        )

        fit_kw: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None and self.patience:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["verbose"] = False

        self.model.fit(X_train, y_train, **fit_kw)
        self._ready = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X: spmatrix | np.ndarray[Any, np.dtype[Any]],  # noqa: N803
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if not self._ready:
            raise ValueError("Model has not been fitted")
        return np.argmax(self.predict_proba(X), axis=1)  # type: ignore[no-any-return]

    def predict_proba(
        self,
        X: spmatrix | np.ndarray[Any, np.dtype[Any]],  # noqa: N803
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if not self._ready or self.model is None:
            raise ValueError("Model has not been fitted")
        booster = self.model.get_booster()
        dm = xgb.DMatrix(X)
        preds = booster.predict(dm)
        if preds.ndim == 1 and self.n_classes_ and self.n_classes_ > 2:
            preds = preds.reshape(-1, self.n_classes_)
        return preds

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "max_delta_step": self.max_delta_step,
            "patience": self.patience,
            "max_bin": self.max_bin,
            "random_state": self.random_state,
        }

    def persist(self, path: str | Path) -> None:
        """Write the trained booster and metadata to disk."""
        if not self._ready or self.model is None:
            raise ValueError("Cannot persist an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.get_booster().save_model(str(path))

        meta = {"n_classes": self.n_classes_, "params": self.hyperparameters()}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    def restore(self, path: str | Path) -> "XGBoostModel":
        """Load a previously persisted model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.n_classes_ = meta.get("n_classes")

        self.model = XGBClassifier()
        self.model.load_model(str(path))
        self._ready = True
        return self
