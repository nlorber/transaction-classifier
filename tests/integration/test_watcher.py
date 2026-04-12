"""Test the background model watcher detects symlink changes."""

import tempfile
import time

import pytest
from sklearn.preprocessing import LabelEncoder

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.features.pipeline import assemble_feature_matrix
from transaction_classifier.core.features.text import TfidfFeatureExtractor
from transaction_classifier.core.models.xgboost_model import XGBoostModel


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


def _train_tiny_model(sample_df, store_path, domain_engine):
    """Train and save a minimal model, return (store, manifest)."""
    tf = TfidfFeatureExtractor(
        label_vocab_size=20,
        detail_vocab_size=20,
        char_vocab_size=20,
        min_df=1,
    )
    X = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)
    le = LabelEncoder()
    y = le.fit_transform(sample_df["target"])
    model = XGBoostModel(n_estimators=5, max_depth=2, verbosity=0, patience=None)
    model.fit(X, y)

    store = ModelStore(store_path)
    manifest = store.save(model, tf, le, {"accuracy": 0.5}, {})
    return store, manifest


def test_watcher_detects_symlink_change(sample_df, domain_engine):
    """ModelStore.has_update() returns True after symlink swap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train two model versions
        store, manifest_v1 = _train_tiny_model(sample_df, tmpdir, domain_engine)
        store.promote(manifest_v1.version)

        store.load_active()

        # No update yet
        assert store.has_update() is False

        # Train a second version and promote it
        time.sleep(1.1)  # ensure different version timestamp
        _, manifest_v2 = _train_tiny_model(sample_df, tmpdir, domain_engine)
        assert manifest_v1.version != manifest_v2.version

        store.promote(manifest_v2.version)

        # Now the watcher should detect the change
        assert store.has_update() is True

        # Reload and verify it's the new version
        bundle_v2 = store.load_active()
        assert bundle_v2.manifest.version == manifest_v2.version
        assert store.has_update() is False
