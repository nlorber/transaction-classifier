"""Measure the effect of balanced class weights on the held-out test block.

Trains the production XGBoost configuration twice on the same temporal split and
feature matrix as training — unweighted, then with balanced sample weights —
early-stopping on the validation block both times. Reports top-K accuracy,
balanced accuracy, and mean recall over the rarest quarter of training classes.
This run backs the default of ``TXCLS_BALANCED_CLASS_WEIGHTS``.

Usage:
    uv run python scripts/eval_class_weights.py
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from transaction_classifier.core.config import Settings
    from transaction_classifier.core.data.loader import read_csv_data
    from transaction_classifier.core.data.splitter import split_by_date
    from transaction_classifier.core.features.engine import DomainFeatureEngine
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.core.features.text import TfidfFeatureExtractor
    from transaction_classifier.core.models.xgboost_model import XGBoostModel
    from transaction_classifier.core.utils.reproducibility import set_seed

    settings = Settings()
    set_seed(settings.random_state)
    logger.info("Loading data from %s ...", settings.data_path)
    df = read_csv_data(
        settings.data_path,
        target_length=settings.target_length,
        min_class_samples=settings.min_class_samples,
    )

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

    extractor = TfidfFeatureExtractor(
        label_vocab_size=settings.tfidf_max_label,
        detail_vocab_size=settings.tfidf_max_detail,
        char_vocab_size=settings.tfidf_max_char,
    )
    engine = DomainFeatureEngine(settings.feature_profile)
    X_train = assemble_feature_matrix(train_df, extractor, engine, fit=True)
    X_val = assemble_feature_matrix(val_df, extractor, engine, fit=False)
    X_test = assemble_feature_matrix(test_df, extractor, engine, fit=False)

    # "Rare" = the bottom quartile of training class counts, among classes the
    # test block actually contains (recall is undefined for the others).
    counts = np.bincount(y_train, minlength=len(labels))
    rare = np.isin(labels, labels[counts <= np.quantile(counts, 0.25)]) & np.isin(labels, y_test)

    results: list[dict[str, object]] = []
    for name, weights in (
        ("unweighted", None),
        ("balanced", compute_sample_weight("balanced", y_train)),
    ):
        logger.info("Training: %s ...", name)
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
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=weights)
        elapsed = time.time() - t0

        proba = model.predict_proba(X_test)
        pred = proba.argmax(axis=1)
        recall = recall_score(y_test, pred, labels=labels, average=None, zero_division=0)
        results.append(
            {
                "weighting": name,
                "top1_accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "top3_accuracy": round(
                    float(top_k_accuracy_score(y_test, proba, k=3, labels=labels)), 4
                ),
                "top5_accuracy": round(
                    float(top_k_accuracy_score(y_test, proba, k=5, labels=labels)), 4
                ),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
                "rare_class_mean_recall": round(float(recall[rare].mean()), 4),
                "rare_classes": int(rare.sum()),
                "best_iteration": int(model.model.best_iteration),
                "max_rounds": settings.n_estimators,
                "train_seconds": round(elapsed, 1),
            }
        )
        logger.info("  %s\n", results[-1])

    out_path = Path("reports/class_weighting.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
