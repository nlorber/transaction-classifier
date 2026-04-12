"""Unit tests for model store and bundle schema."""

import tempfile

import pytest
from sklearn.preprocessing import LabelEncoder

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.features.text import TfidfFeatureExtractor
from transaction_classifier.core.models.xgboost_model import XGBoostModel


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


@pytest.fixture
def tiny_model(sample_df, domain_engine):
    """Train a tiny XGBoostModel on the 10-row fixture."""
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix

    tf = TfidfFeatureExtractor(
        label_vocab_size=20, detail_vocab_size=20, char_vocab_size=20, min_df=1
    )
    X = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)

    le = LabelEncoder()
    y = le.fit_transform(sample_df["target"])

    model = XGBoostModel(n_estimators=5, max_depth=2, verbosity=0, patience=None)
    model.fit(X, y)

    return model, tf, le


def test_save_and_load_roundtrip(tiny_model, sample_df, domain_engine):
    model, tf, le = tiny_model

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ModelStore(tmpdir)
        manifest = store.save(
            model=model,
            text_extractor=tf,
            label_encoder=le,
            metrics={"accuracy": 0.5},
            config={"test": True},
            n_features=100,
        )

        assert manifest.version.startswith("v-")
        assert manifest.metrics["accuracy"] == 0.5
        assert manifest.num_categories == len(le.classes_)

        # Promote and load back
        store.promote(manifest.version)
        bundle = store.load_active()

        assert bundle.manifest.version == manifest.version
        assert bundle.model.ready
        assert bundle.text_extractor._fitted
        assert len(bundle.label_encoder.classes_) == len(le.classes_)

        # Predictions should work
        from transaction_classifier.core.features.pipeline import assemble_feature_matrix

        X = assemble_feature_matrix(sample_df, bundle.text_extractor, domain_engine, fit=False)
        preds = bundle.model.predict(X)
        assert len(preds) == len(sample_df)


def test_list_versions(tiny_model):
    model, tf, le = tiny_model

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ModelStore(tmpdir)
        store.save(model, tf, le, {}, {})
        import time

        time.sleep(1)  # Ensure different timestamp
        store.save(model, tf, le, {}, {})

        versions = store.available_versions()
        assert len(versions) >= 2


def test_check_for_update(tiny_model):
    model, tf, le = tiny_model

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ModelStore(tmpdir)
        m1 = store.save(model, tf, le, {}, {})
        store.promote(m1.version)

        store.load_active()

        # No update yet
        assert not store.has_update()

        # Save and promote a new version
        import time

        time.sleep(1)  # Ensure different timestamp
        m2 = store.save(model, tf, le, {}, {})
        store.promote(m2.version)

        # Now should detect update
        assert store.has_update()


def test_corrupted_artifact_raises(tiny_model):
    model, tf, le = tiny_model

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ModelStore(tmpdir)
        manifest = store.save(model, tf, le, {}, {})
        store.promote(manifest.version)

        # Corrupt a file
        model_path = store.root / manifest.version / "classifier.json"
        model_path.write_bytes(b"corrupted data")

        store2 = ModelStore(tmpdir)
        with pytest.raises(RuntimeError, match="Hash mismatch"):
            store2.load_active()
