"""Smoke test: the real pipeline, trained on real sample data, must learn something."""

import pandas as pd
import pytest

from transaction_classifier.core.config import Settings
from transaction_classifier.core.data.source import CsvDataSource
from transaction_classifier.training.pipeline import TrainingPipeline


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pipeline_beats_majority_class_on_sample_data_slice(tmp_path):
    """Catch a model that trains and serialises fine but has stopped learning.

    Shape and round-trip tests pass for a label mix-up, empty features, or a
    scoring bug. This one trains on the ten most frequent account codes of
    data/sample.csv, exercising the production features, split, class weights
    and early stopping in seconds.
    """
    df = pd.read_csv("data/sample.csv", dtype={"account_code": str})
    top_codes = df["account_code"].value_counts().index[:10]
    slice_path = tmp_path / "sample_top10_codes.csv"
    df[df["account_code"].isin(top_codes)].to_csv(slice_path, index=False)

    settings = Settings(
        data_path=str(slice_path),
        artifact_dir=str(tmp_path / "models"),
        n_estimators=80,
        patience=10,
        tfidf_max_label=500,
        tfidf_max_detail=500,
        tfidf_max_char=300,
    )
    runner = TrainingPipeline(settings, CsvDataSource(settings.data_path))
    manifest, majority_share, n_classes = runner.execute()

    assert n_classes == 10
    # A model that ignores its inputs scores the majority share on accuracy and
    # 1/n_classes on balanced accuracy; a working one clears both by a wide margin.
    assert manifest.metrics["accuracy"] > 2 * majority_share
    assert manifest.metrics["balanced_accuracy"] > 3 / n_classes
