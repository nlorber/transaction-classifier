"""Compare logistic regression, XGBoost, and LightGBM on the sample dataset.

Every model sees the same feature matrix and the same temporal split as
training: the boosters early-stop on the validation block, and all models are
scored on the held-out test block. XGBoost runs through the production wrapper
with the production hyperparameters (``Settings``). Logistic regression and
XGBoost are each trained without and with balanced class weights, the
production default. Reports top-1/3/5 accuracy, balanced accuracy, weighted F1,
and training time.

Usage:
    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --data data/generated/x.csv --runs xgb-balanced --output /tmp/r.json

Each result records the process's peak resident memory so far; run one model per process
(as scripts/scaling_benchmark.py does) for a per-model figure. Booster results also record
the best round and whether the round cap, rather than early stopping, ended training.
"""

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _train_xgboost(
    settings: Any,
    X_train: spmatrix,
    y_train: np.ndarray,
    X_val: spmatrix,
    y_val: np.ndarray,
    X_test: spmatrix,
    balanced: bool,
) -> tuple[np.ndarray, float, dict[str, object]]:
    from transaction_classifier.core.models.xgboost_model import XGBoostModel

    t0 = time.time()
    model = XGBoostModel(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        learning_rate=settings.learning_rate,
        patience=settings.patience,
        max_bin=settings.max_bin,
        random_state=settings.random_state,
        device=settings.device,
        verbosity=0,
    )
    weights = compute_sample_weight("balanced", y_train) if balanced else None
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=weights)
    elapsed = time.time() - t0
    # best_iteration counts from 0 and is absent when early stopping never ran.
    best = model.model.get_booster().attr("best_iteration")
    best_round = int(best) + 1 if best is not None else settings.n_estimators
    rounds = _rounds(best_round, settings.n_estimators, settings.patience or 0)
    return model.predict_proba(X_test), elapsed, rounds


LIGHTGBM_ROUNDS = 500
LIGHTGBM_PATIENCE = 40


def _rounds(best_round: int, round_cap: int, patience: int) -> dict[str, object]:
    """Where a booster settled. Early stopping needs *patience* rounds past the best one,
    so a best round within patience of the cap means the cap ended training."""
    return {
        "best_round": best_round,
        "round_cap": round_cap,
        "stopped_by_round_cap": best_round + patience > round_cap,
    }


def _train_lightgbm(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_val: spmatrix,
    y_val: np.ndarray,
    X_test: spmatrix,
    n_classes: int,
    balanced: bool,
) -> tuple[np.ndarray, float, dict[str, object]]:
    from lightgbm import LGBMClassifier, early_stopping

    t0 = time.time()
    model = LGBMClassifier(
        n_estimators=LIGHTGBM_ROUNDS,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        min_child_weight=10,
        num_class=n_classes,
        objective="multiclass",
        class_weight="balanced" if balanced else None,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=LIGHTGBM_PATIENCE, verbose=False)],
    )
    elapsed = time.time() - t0
    # best_iteration_ counts from 1.
    rounds = _rounds(model.best_iteration_, LIGHTGBM_ROUNDS, LIGHTGBM_PATIENCE)
    return model.predict_proba(X_test), elapsed, rounds


def _train_logistic(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_test: spmatrix,
    balanced: bool,
) -> tuple[np.ndarray, float, dict[str, object]]:
    t0 = time.time()
    # The matrix mixes TF-IDF weights in [0, 1] with raw amounts in the thousands.
    # saga's step size shrinks with the largest row norm, so unscaled it makes
    # almost no progress; MaxAbsScaler maps each column to [-1, 1] and keeps the
    # matrix sparse.
    model = make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="saga",
            class_weight="balanced" if balanced else None,
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    return model.predict_proba(X_test), elapsed, {}


def _peak_rss_mb() -> float:
    """Peak resident memory of this process so far (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1)


def _score(
    model: str,
    class_weights: str,
    proba: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
    elapsed: float,
) -> dict[str, object]:
    pred = proba.argmax(axis=1)
    return {
        "model": model,
        "class_weights": class_weights,
        "top1_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "top3_accuracy": round(float(top_k_accuracy_score(y_test, proba, k=3, labels=labels)), 4),
        "top5_accuracy": round(float(top_k_accuracy_score(y_test, proba, k=5, labels=labels)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, pred, average="weighted")), 4),
        "train_seconds": round(elapsed, 1),
        "peak_rss_mb": _peak_rss_mb(),
    }


RUN_LABELS: dict[str, tuple[str, str]] = {
    "lr": ("Logistic Regression", "none"),
    "lr-balanced": ("Logistic Regression", "balanced"),
    "xgb": ("XGBoost", "none"),
    "xgb-balanced": ("XGBoost", "balanced"),
    "lgbm": ("LightGBM", "none"),
    "lgbm-balanced": ("LightGBM", "balanced"),
}
DEFAULT_RUNS = "lr,lr-balanced,xgb,xgb-balanced,lgbm"


def parse_runs(value: str) -> list[str]:
    runs = [run.strip() for run in value.split(",") if run.strip()]
    unknown = [run for run in runs if run not in RUN_LABELS]
    if unknown or not runs:
        raise argparse.ArgumentTypeError(
            f"unknown runs {unknown}; choose from {', '.join(RUN_LABELS)}"
        )
    return runs


class Matrices(NamedTuple):
    X_train: spmatrix
    y_train: np.ndarray
    X_val: spmatrix
    y_val: np.ndarray
    X_test: spmatrix
    y_test: np.ndarray
    labels: np.ndarray


def build_matrices(settings: Any, data_path: str) -> Matrices:
    """Feature matrices on the production temporal split and class filter."""
    from transaction_classifier.core.data.loader import read_csv_data
    from transaction_classifier.core.data.splitter import split_by_date
    from transaction_classifier.core.features.engine import DomainFeatureEngine
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    logger.info("Loading data from %s ...", data_path)
    df = read_csv_data(
        data_path,
        target_length=settings.target_length,
        min_class_samples=settings.min_class_samples,
    )
    logger.info("Loaded %d rows, %d classes", len(df), df["target"].nunique())

    train_df, val_df, test_df = split_by_date(
        df, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio
    )
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["target"])
    val_df = val_df[val_df["target"].isin(le.classes_)]
    test_df = test_df[test_df["target"].isin(le.classes_)]
    labels = np.arange(len(le.classes_))
    logger.info(
        "Train: %d | Val: %d | Test: %d | Classes: %d\n",
        len(train_df),
        len(val_df),
        len(test_df),
        len(labels),
    )

    logger.info("Building feature matrix ...")
    extractor = TfidfFeatureExtractor(
        label_vocab_size=settings.tfidf_max_label,
        detail_vocab_size=settings.tfidf_max_detail,
        char_vocab_size=settings.tfidf_max_char,
    )
    engine = DomainFeatureEngine(settings.feature_profile)
    X_train = assemble_feature_matrix(train_df, extractor, engine, fit=True)
    X_val = assemble_feature_matrix(val_df, extractor, engine, fit=False)
    X_test = assemble_feature_matrix(test_df, extractor, engine, fit=False)
    logger.info("Feature shape: %s\n", X_train.shape)
    return Matrices(
        X_train,
        y_train,
        X_val,
        le.transform(val_df["target"]),
        X_test,
        le.transform(test_df["target"]),
        labels,
    )


def main(argv: list[str] | None = None) -> None:
    from transaction_classifier.core.config import Settings

    settings = Settings()
    parser = argparse.ArgumentParser(description="Compare models on one dataset.")
    parser.add_argument("--data", default=settings.data_path, help="CSV to train and score on")
    parser.add_argument("--runs", type=parse_runs, default=parse_runs(DEFAULT_RUNS))
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison.json"))
    args = parser.parse_args(argv)

    m = build_matrices(settings, args.data)
    trainers = {
        "lr": lambda: _train_logistic(m.X_train, m.y_train, m.X_test, False),
        "lr-balanced": lambda: _train_logistic(m.X_train, m.y_train, m.X_test, True),
        "xgb": lambda: _train_xgboost(
            settings, m.X_train, m.y_train, m.X_val, m.y_val, m.X_test, False
        ),
        "xgb-balanced": lambda: _train_xgboost(
            settings, m.X_train, m.y_train, m.X_val, m.y_val, m.X_test, True
        ),
        "lgbm": lambda: _train_lightgbm(
            m.X_train, m.y_train, m.X_val, m.y_val, m.X_test, len(m.labels), False
        ),
        "lgbm-balanced": lambda: _train_lightgbm(
            m.X_train, m.y_train, m.X_val, m.y_val, m.X_test, len(m.labels), True
        ),
    }

    results: list[dict[str, object]] = []
    for run in args.runs:
        model_name, class_weights = RUN_LABELS[run]
        logger.info("Training: %s (class weights: %s) ...", model_name, class_weights)
        proba, elapsed, rounds = trainers[run]()
        score = _score(model_name, class_weights, proba, m.y_test, m.labels, elapsed)
        results.append({**score, **rounds})
        logger.info("  done (%.1fs)\n", elapsed)

    logger.info("=" * 92)
    logger.info("MODEL COMPARISON (held-out test block, synthetic data)")
    logger.info("=" * 92)
    logger.info(
        "%-22s %-9s %8s %8s %8s %10s %9s %9s",
        "Model",
        "Weights",
        "Top-1",
        "Top-3",
        "Top-5",
        "Bal. Acc.",
        "F1 (wtd)",
        "Time (s)",
    )
    logger.info("-" * 92)
    for r in results:
        logger.info(
            "%-22s %-9s %8.4f %8.4f %8.4f %10.4f %9.4f %9.1f",
            r["model"],
            r["class_weights"],
            r["top1_accuracy"],
            r["top3_accuracy"],
            r["top5_accuracy"],
            r["balanced_accuracy"],
            r["f1_weighted"],
            r["train_seconds"],
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("\nResults saved to %s", args.output)


if __name__ == "__main__":
    main()
