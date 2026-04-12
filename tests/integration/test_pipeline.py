"""Integration tests for the full training pipeline."""

import tempfile

import numpy as np
import pytest

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.config import Settings
from transaction_classifier.core.data.source import CsvDataSource
from transaction_classifier.training.pipeline import TrainingPipeline


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


def _quick_settings(csv_path: str, tmpdir: str) -> Settings:
    return Settings(
        data_path=csv_path,
        artifact_dir=tmpdir,
        min_class_samples=1,
        n_estimators=10,
        max_depth=2,
        patience=None,
        tfidf_max_label=20,
        tfidf_max_detail=20,
        tfidf_max_char=20,
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_full_pipeline_on_fixture(sample_csv_path, domain_engine):
    """Train on the 10-row fixture, verify manifest, promotion, and prediction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _quick_settings(str(sample_csv_path), tmpdir)
        provider = CsvDataSource(settings.data_path)
        runner = TrainingPipeline(settings, provider)

        # 1. Run training
        manifest, baseline_accuracy, n_classes = runner.execute()

        # 2. Verify manifest
        assert manifest.version.startswith("v-")
        assert manifest.num_categories > 0
        assert manifest.metrics["accuracy"] > 0
        assert manifest.n_features > 0

        # 3. Promote and reload
        store = ModelStore(tmpdir)
        store.promote(manifest.version)

        bundle = store.load_active()

        # 4. Verify loaded model can predict
        from transaction_classifier.core.features.pipeline import assemble_feature_matrix

        val_df = provider.fetch(min_class_samples=1, target_length=settings.target_length).head(2)
        X = assemble_feature_matrix(val_df, bundle.text_extractor, domain_engine, fit=False)
        proba = bundle.model.predict_proba(X)

        assert proba.shape[0] == 2
        assert proba.shape[1] == manifest.num_categories


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_baseline_accuracy_is_majority_class_frequency(sample_csv_path):
    """Baseline accuracy should equal the most frequent class proportion in training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _quick_settings(str(sample_csv_path), tmpdir)
        provider = CsvDataSource(settings.data_path)
        runner = TrainingPipeline(settings, provider)

        _, baseline_accuracy, n_classes = runner.execute()

        # Baseline is the majority-class proportion — must be in (0, 1)
        assert 0 < baseline_accuracy < 1
        # With 10 rows across 5 classes, majority class is at most 3/8 = 0.375
        assert baseline_accuracy <= 0.5
        assert n_classes >= 2


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_feature_dimensions_match_between_train_and_reload(sample_csv_path, domain_engine):
    """Feature matrix width from a reloaded model must match the training manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _quick_settings(str(sample_csv_path), tmpdir)
        provider = CsvDataSource(settings.data_path)
        runner = TrainingPipeline(settings, provider)

        manifest, _, _ = runner.execute()

        store = ModelStore(tmpdir)
        store.promote(manifest.version)
        bundle = store.load_active()

        from transaction_classifier.core.features.pipeline import assemble_feature_matrix

        df = provider.fetch(min_class_samples=1, target_length=settings.target_length)
        X = assemble_feature_matrix(df, bundle.text_extractor, domain_engine, fit=False)

        assert X.shape[1] == manifest.n_features


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_proba_rows_sum_to_one(sample_csv_path, domain_engine):
    """Each row of predict_proba output should sum to ~1.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = _quick_settings(str(sample_csv_path), tmpdir)
        provider = CsvDataSource(settings.data_path)
        runner = TrainingPipeline(settings, provider)

        manifest, _, _ = runner.execute()

        store = ModelStore(tmpdir)
        store.promote(manifest.version)
        bundle = store.load_active()

        from transaction_classifier.core.features.pipeline import assemble_feature_matrix

        df = provider.fetch(min_class_samples=1, target_length=settings.target_length).head(3)
        X = assemble_feature_matrix(df, bundle.text_extractor, domain_engine, fit=False)
        proba = bundle.model.predict_proba(X)

        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
