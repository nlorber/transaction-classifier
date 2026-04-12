"""Integration test: full training pipeline on 10-row fixture."""

import tempfile

import pytest

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.config import Settings
from transaction_classifier.core.data.source import CsvDataSource
from transaction_classifier.training.pipeline import TrainingPipeline


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_full_pipeline_on_fixture(sample_csv_path, domain_engine):
    """Train on the 10-row fixture, verify manifest, promotion, and prediction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            data_path=str(sample_csv_path),
            artifact_dir=tmpdir,
            min_class_samples=1,
            n_estimators=10,
            max_depth=2,
            patience=None,
            tfidf_max_label=20,
            tfidf_max_detail=20,
            tfidf_max_char=20,
        )
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
