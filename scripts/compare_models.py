"""Compare XGBoost, LightGBM, and logistic regression on the sample dataset.

Trains each model on the same temporal split as training with the same feature
matrix: the boosters early-stop on the validation block, and every model is
scored on the held-out test block. Reports balanced accuracy, weighted F1, and
training time.

Usage:
    uv run python scripts/compare_models.py
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _train_xgboost(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_val: spmatrix,
    y_val: np.ndarray,
    X_test: spmatrix,
    n_classes: int,
) -> tuple[np.ndarray, float]:
    from xgboost import XGBClassifier

    t0 = time.time()
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        min_child_weight=10,
        gamma=0.5,
        tree_method="hist",
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=40,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    return model.predict(X_test), elapsed


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
    return model.predict(X_test), elapsed


def _train_logistic(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_test: spmatrix,
) -> tuple[np.ndarray, float]:
    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        random_state=42,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    return model.predict(X_test), elapsed


def main() -> None:
    from transaction_classifier.core.config import Settings
    from transaction_classifier.core.data.loader import read_csv_data
    from transaction_classifier.core.data.splitter import split_by_date
    from transaction_classifier.core.features.engine import DomainFeatureEngine
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    settings = Settings()
    data_path = Path("data/sample.csv")
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
    n_classes = len(le.classes_)
    logger.info(
        "Train: %d | Val: %d | Test: %d | Classes: %d\n",
        len(train_df),
        len(val_df),
        len(test_df),
        n_classes,
    )

    logger.info("Building feature matrix ...")
    extractor = TfidfFeatureExtractor()
    engine = DomainFeatureEngine(settings.feature_profile)
    X_train = assemble_feature_matrix(train_df, extractor, engine, fit=True)
    X_val = assemble_feature_matrix(val_df, extractor, engine, fit=False)
    X_test = assemble_feature_matrix(test_df, extractor, engine, fit=False)
    logger.info("Feature shape: %s\n", X_train.shape)

    results: list[dict[str, object]] = []

    # --- Logistic Regression (baseline) ---
    logger.info("Training: Logistic Regression ...")
    y_pred_lr, t_lr = _train_logistic(X_train, y_train, X_test)
    results.append(
        {
            "model": "Logistic Regression",
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred_lr)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_lr, average="weighted")), 4),
            "train_seconds": round(t_lr, 1),
        }
    )
    logger.info("  done (%.1fs)\n", t_lr)

    # --- XGBoost ---
    logger.info("Training: XGBoost ...")
    y_pred_xgb, t_xgb = _train_xgboost(X_train, y_train, X_val, y_val, X_test, n_classes)
    results.append(
        {
            "model": "XGBoost",
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred_xgb)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_xgb, average="weighted")), 4),
            "train_seconds": round(t_xgb, 1),
        }
    )
    logger.info("  done (%.1fs)\n", t_xgb)

    # --- LightGBM ---
    logger.info("Training: LightGBM ...")
    y_pred_lgb, t_lgb = _train_lightgbm(X_train, y_train, X_val, y_val, X_test, n_classes)
    results.append(
        {
            "model": "LightGBM",
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred_lgb)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred_lgb, average="weighted")), 4),
            "train_seconds": round(t_lgb, 1),
        }
    )
    logger.info("  done (%.1fs)\n", t_lgb)

    # --- Report ---
    results.sort(key=lambda r: float(str(r["balanced_accuracy"])), reverse=True)

    logger.info("=" * 65)
    logger.info("MODEL COMPARISON (held-out test block, synthetic data)")
    logger.info("=" * 65)
    logger.info(
        "%-25s %12s %12s %10s",
        "Model",
        "Bal. Acc.",
        "F1 (wtd)",
        "Time (s)",
    )
    logger.info("-" * 65)
    for r in results:
        logger.info(
            "%-25s %12.4f %12.4f %10.1f",
            r["model"],
            r["balanced_accuracy"],
            r["f1_weighted"],
            r["train_seconds"],
        )

    out_path = Path("reports/model_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    main()
