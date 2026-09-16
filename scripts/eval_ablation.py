"""Cumulative feature ablation on the sample dataset.

Trains the same XGBoost configuration on a growing feature matrix — TF-IDF only,
then adding numeric, date, and domain features — on the same temporal split as
training (early stopping on the validation block), and reports accuracy and
balanced accuracy on the held-out test block per feature set.

Usage:
    uv run python scripts/eval_ablation.py
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, spmatrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def _train_xgboost(
    X_train: spmatrix | np.ndarray,
    y_train: np.ndarray,
    X_val: spmatrix | np.ndarray,
    y_val: np.ndarray,
    X_test: spmatrix | np.ndarray,
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
        min_child_weight=1,
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


def main() -> None:
    from transaction_classifier.core.config import Settings
    from transaction_classifier.core.data.loader import read_csv_data
    from transaction_classifier.core.data.splitter import split_by_date
    from transaction_classifier.core.features.engine import DomainFeatureEngine
    from transaction_classifier.core.features.standard import (
        build_date_features,
        build_numeric_features,
    )
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
    frames = {"train": train_df, "val": val_df, "test": test_df}

    # Build each dense feature family once per split, mirroring
    # assemble_feature_matrix (which derives a combined amount column first).
    dense: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    engine = DomainFeatureEngine(settings.feature_profile)
    for name, frame in frames.items():
        frame = frame.reset_index(drop=True).copy()
        frame["amount"] = frame["debit"].fillna(0).astype(float) + frame["credit"].fillna(
            0
        ).astype(float)
        numeric = build_numeric_features(frame).values
        date = build_date_features(frame).values
        domain = engine.build(
            frame,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        ).values
        dense[name] = (numeric, date, domain)

    extractor = TfidfFeatureExtractor()
    tfidf = {"train": extractor.fit_transform(train_df.reset_index(drop=True))}
    for name in ("val", "test"):
        tfidf[name] = extractor.transform(frames[name].reset_index(drop=True))

    # Each row adds the next dense family: 0 = TF-IDF only, 3 = all features.
    feature_sets = [
        ("TF-IDF only", 0),
        ("+ numeric", 1),
        ("+ date", 2),
        ("+ domain (all features)", 3),
    ]

    results: list[dict[str, object]] = []
    for label, n_dense in feature_sets:
        X = {
            split: hstack([tfidf[split], *dense[split][:n_dense]]) if n_dense else tfidf[split]
            for split in SPLITS
        }
        logger.info("Training: %s (features: %d) ...", label, X["train"].shape[1])
        y_pred, elapsed = _train_xgboost(
            X["train"], y_train, X["val"], y_val, X["test"], n_classes
        )
        results.append(
            {
                "feature_set": label,
                "n_features": int(X["train"].shape[1]),
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
                "train_seconds": round(elapsed, 1),
            }
        )
        logger.info("  done (%.1fs)\n", elapsed)

    logger.info("=" * 70)
    logger.info("FEATURE ABLATION (cumulative, held-out test block, synthetic data)")
    logger.info("=" * 70)
    logger.info("%-25s %10s %12s %12s", "Feature set", "Features", "Accuracy", "Bal. Acc.")
    logger.info("-" * 70)
    for r in results:
        logger.info(
            "%-25s %10d %12.4f %12.4f",
            r["feature_set"],
            r["n_features"],
            r["accuracy"],
            r["balanced_accuracy"],
        )

    out_path = Path("reports/feature_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
