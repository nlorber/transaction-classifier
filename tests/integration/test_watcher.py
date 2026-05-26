"""Test the background model watcher detects symlink changes."""

import tempfile
import time
from types import SimpleNamespace

import pytest
from sklearn.preprocessing import LabelEncoder

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.config import Settings
from transaction_classifier.core.features.pipeline import assemble_feature_matrix
from transaction_classifier.core.features.text import TfidfFeatureExtractor
from transaction_classifier.core.models.xgboost_model import XGBoostModel
from transaction_classifier.inference.app import _ModelReloadHandler


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


def test_reload_handler_detects_symlink_change(sample_df, domain_engine):
    """_ModelReloadHandler._try_reload() updates app.state.predictor on symlink swap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train v1 and promote it
        store, manifest_v1 = _train_tiny_model(sample_df, tmpdir, domain_engine)
        store.promote(manifest_v1.version)

        # Load the initial predictor
        from transaction_classifier.inference.predictor import reload_predictor

        predictor_v1 = reload_predictor(store, 3, domain_engine)

        # Build a mock app with state
        mock_settings = Settings(artifact_dir=tmpdir, sandbox_mode=False)
        mock_app = SimpleNamespace(
            state=SimpleNamespace(
                predictor=predictor_v1,
                settings=mock_settings,
            )
        )

        handler = _ModelReloadHandler(mock_app, store, domain_engine, debounce_secs=0.0)  # type: ignore[arg-type]

        # Train v2 and promote it
        time.sleep(1.1)
        _, manifest_v2 = _train_tiny_model(sample_df, tmpdir, domain_engine)
        assert manifest_v1.version != manifest_v2.version
        store.promote(manifest_v2.version)

        # Directly invoke reload logic (simulates a filesystem event)
        handler._try_reload()

        # Verify the predictor was updated to v2
        assert mock_app.state.predictor.bundle.manifest.version == manifest_v2.version

        handler.shutdown()


class TestMaxDelayForceFire:
    """Unit-level tests for the max_delay hard cap on the debounce handler."""

    def _make_handler(
        self,
        sample_df: object,
        tmpdir: str,
        domain_engine: object,
        debounce_secs: float = 60.0,
        max_delay_secs: float = 0.05,
    ) -> "_ModelReloadHandler":
        from types import SimpleNamespace

        from transaction_classifier.core.config import Settings
        from transaction_classifier.inference.predictor import reload_predictor

        store, manifest_v1 = _train_tiny_model(sample_df, tmpdir, domain_engine)  # type: ignore[arg-type]
        store.promote(manifest_v1.version)
        predictor_v1 = reload_predictor(store, 3, domain_engine)  # type: ignore[arg-type]
        mock_settings = Settings(artifact_dir=tmpdir, sandbox_mode=False)
        mock_app = SimpleNamespace(
            state=SimpleNamespace(predictor=predictor_v1, settings=mock_settings)
        )
        return _ModelReloadHandler(mock_app, store, domain_engine, debounce_secs, max_delay_secs)  # type: ignore[arg-type]

    def test_rolling_debounce_resets_timer(self, sample_df, domain_engine):
        """Consecutive events within debounce window keep rescheduling the timer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(
                sample_df, tmpdir, domain_engine, debounce_secs=10.0, max_delay_secs=30.0
            )
            from watchdog.events import FileCreatedEvent

            ev = FileCreatedEvent("x")
            handler.on_any_event(ev)
            timer1 = handler._timer
            handler.on_any_event(ev)
            timer2 = handler._timer
            # Each call should have created a new (different) timer object
            assert timer1 is not timer2
            handler.shutdown()

    def test_force_fire_when_max_delay_exceeded(self, sample_df, domain_engine):
        """When elapsed >= max_delay the handler fires immediately without waiting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(
                sample_df, tmpdir, domain_engine, debounce_secs=60.0, max_delay_secs=0.05
            )
            from watchdog.events import FileCreatedEvent

            ev = FileCreatedEvent("x")
            # First event starts the burst clock.
            handler.on_any_event(ev)
            # Wait for max_delay to expire.
            time.sleep(0.15)
            # Second event should detect elapsed >= max_delay and fire immediately.
            handler.on_any_event(ev)
            # After the force-fire path the first_event_time is reset to None.
            time.sleep(0.05)  # let the daemon thread complete
            assert handler._first_event_time is None
            handler.shutdown()
