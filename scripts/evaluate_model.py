"""Evaluate the current model and regenerate report artifacts.

Loads the promoted model, runs predictions on the temporal validation split,
and generates confusion matrix, top-K accuracy curve, calibration diagram,
and per-class performance charts.

Usage:
    uv run python scripts/evaluate_model.py
    uv run python scripts/evaluate_model.py --output-dir reports
"""

import logging
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--data-path", default="data/sample.csv", help="Path to CSV data")
@click.option("--model-dir", default="models", help="Path to model store")
@click.option("--output-dir", default="reports", help="Output directory for reports")
@click.option("--train-ratio", default=0.80, type=float, help="Temporal split ratio")
def evaluate(data_path: str, model_dir: str, output_dir: str, train_ratio: float) -> None:
    """Evaluate the promoted model and generate report artifacts."""
    from transaction_classifier.core.artifacts.store import ModelStore
    from transaction_classifier.core.data.loader import read_csv_data
    from transaction_classifier.core.data.splitter import split_by_date
    from transaction_classifier.core.features.engine import DomainFeatureEngine
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.evaluation.visualize import generate_all_reports

    logger.info("Loading model from %s", model_dir)
    store = ModelStore(model_dir)
    bundle = store.load_active()

    logger.info("Loading data from %s", data_path)
    df = read_csv_data(data_path, target_length=6, min_class_samples=1)

    _, val_df = split_by_date(df, train_ratio=train_ratio)
    logger.info("Validation set: %d rows", len(val_df))

    le = bundle.label_encoder
    val_mask = val_df["target"].isin(le.classes_)
    val_df = val_df[val_mask].copy()
    y_true = le.transform(val_df["target"])

    engine = DomainFeatureEngine("config/profiles/french_treasury.yaml")
    X_val = assemble_feature_matrix(val_df, bundle.text_extractor, engine, fit=False)

    proba = bundle.model.predict_proba(X_val)
    y_pred = proba.argmax(axis=1)

    class_names = list(le.classes_)
    metrics = generate_all_reports(y_true, y_pred, proba, class_names, Path(output_dir))

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Balanced accuracy: %.4f", metrics["balanced_accuracy"])
    logger.info("Top-K accuracy: %s", metrics["top_k_accuracy"])
    logger.info("Reports saved to %s", output_dir)


if __name__ == "__main__":
    evaluate()
