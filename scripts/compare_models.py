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
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

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
) -> tuple[np.ndarray, float]:
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
    return model.predict_proba(X_test), elapsed


def _train_lightgbm(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_val: spmatrix,
    y_val: np.ndarray,
    X_test: spmatrix,
    n_classes: int,
) -> tuple[np.ndarray, float]:
    from lightgbm import LGBMClassifier, early_stopping

    t0 = time.time()
    model = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        min_child_weight=10,
        num_class=n_classes,
        objective="multiclass",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=40, verbose=False)],
    )
    elapsed = time.time() - t0
    return model.predict_proba(X_test), elapsed


def _train_logistic(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_test: spmatrix,
    balanced: bool,
) -> tuple[np.ndarray, float]:
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
    return model.predict_proba(X_test), elapsed


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
    }


def main() -> None:
    from transaction_classifier.core.config import Settings
    from transaction_classifier.core.data.loader import read_csv_data
    from transaction_classifier.core.data.splitter import split_by_date
    from transaction_classifier.core.features.engine import DomainFeatureEngine
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    settings = Settings()
    data_path = Path(settings.data_path)
    logger.info("Loading data from %s ...", data_path)
    df = read_csv_data(
        str(data_path),
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
    y_val = le.transform(val_df["target"])
    y_test = le.transform(test_df["target"])
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

    runs = [
        ("Logistic Regression", "none", lambda: _train_logistic(X_train, y_train, X_test, False)),
        (
            "Logistic Regression",
            "balanced",
            lambda: _train_logistic(X_train, y_train, X_test, True),
        ),
        (
            "XGBoost",
            "none",
            lambda: _train_xgboost(settings, X_train, y_train, X_val, y_val, X_test, False),
        ),
        (
            "XGBoost",
            "balanced",
            lambda: _train_xgboost(settings, X_train, y_train, X_val, y_val, X_test, True),
        ),
        (
            "LightGBM",
            "none",
            lambda: _train_lightgbm(X_train, y_train, X_val, y_val, X_test, len(labels)),
        ),
    ]

    results: list[dict[str, object]] = []
    for model_name, class_weights, train in runs:
        logger.info("Training: %s (class weights: %s) ...", model_name, class_weights)
        proba, elapsed = train()
        results.append(_score(model_name, class_weights, proba, y_test, labels, elapsed))
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

    out_path = Path("reports/model_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    main()
