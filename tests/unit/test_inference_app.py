"""Tests for the FastAPI application factory and lifespan."""

import tempfile

import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.config import Settings
from transaction_classifier.core.features.text import TfidfFeatureExtractor
from transaction_classifier.core.models.xgboost_model import XGBoostModel
from transaction_classifier.inference.app import create_app, get_app


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


class TestCreateApp:
    def test_creates_fastapi_app(self):
        settings = Settings(sandbox_mode=True)
        app = create_app(settings)
        assert app.title == "Transaction Classifier API"

    def test_default_settings_when_none(self):
        app = create_app(None)
        assert app.state.settings is not None

    def test_middleware_is_added(self):
        settings = Settings(sandbox_mode=True)
        app = create_app(settings)
        # The middleware stack should contain our LatencyMiddleware
        middleware_names = [m.cls.__name__ for m in app.user_middleware]
        assert "LatencyMiddleware" in middleware_names


class TestGetApp:
    def test_returns_app(self):
        app = get_app()
        assert app.title == "Transaction Classifier API"


class TestLifespanSandboxMode:
    def test_sandbox_mode_skips_model_loading(self):
        settings = Settings(sandbox_mode=True)
        app = create_app(settings)
        with TestClient(app):
            assert app.state.predictor is None
            assert app.state.store is None


class TestLifespanApiKeyWarning:
    def _capture_app_warnings(self, app: object) -> list[str]:
        """Run the app inside TestClient and return warning messages from the app logger."""
        import logging as _logging

        records: list[str] = []

        class _CapHandler(_logging.Handler):
            def emit(self, record: _logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _CapHandler()
        handler.setLevel(_logging.WARNING)
        log = _logging.getLogger("transaction_classifier.inference.app")
        log.addHandler(handler)
        try:
            with TestClient(app):
                pass
        finally:
            log.removeHandler(handler)
        return records

    def test_warns_when_no_api_keys_in_non_sandbox_mode(self):
        """App warns at startup when running without authentication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(sandbox_mode=False, api_keys=[], artifact_dir=tmpdir)
            app = create_app(settings)
            messages = self._capture_app_warnings(app)
        assert any("TXCLS_API_KEYS" in m for m in messages)

    def test_no_warning_when_api_keys_present(self):
        """App does not warn when api_keys are configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(sandbox_mode=False, api_keys=["secret-key"], artifact_dir=tmpdir)
            app = create_app(settings)
            messages = self._capture_app_warnings(app)
        assert not any("TXCLS_API_KEYS" in m for m in messages)

    def test_no_warning_in_sandbox_mode(self):
        """No api_keys warning is emitted in sandbox mode."""
        settings = Settings(sandbox_mode=True, api_keys=[])
        app = create_app(settings)
        messages = self._capture_app_warnings(app)
        assert not any("TXCLS_API_KEYS" in m for m in messages)


class TestLifespanWithModel:
    def test_loads_model_on_startup(self, sample_df, domain_engine):
        """Full lifespan test: create a model in a tmpdir and start the app."""
        from transaction_classifier.core.features.pipeline import assemble_feature_matrix

        tf = TfidfFeatureExtractor(
            label_vocab_size=20, detail_vocab_size=20, char_vocab_size=20, min_df=1
        )
        X = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)
        le = LabelEncoder()
        y = le.fit_transform(sample_df["target"])
        model = XGBoostModel(n_estimators=5, max_depth=2, verbosity=0, patience=None)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(tmpdir)
            manifest = store.save(model, tf, le, {"accuracy": 0.5}, {})
            store.promote(manifest.version)

            settings = Settings(sandbox_mode=False, artifact_dir=tmpdir)
            app = create_app(settings)

            with TestClient(app):
                assert app.state.predictor is not None
                assert app.state.store is not None

    def test_missing_model_starts_without_crashing(self):
        """When no model exists, the app starts with predictor=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(sandbox_mode=False, artifact_dir=tmpdir)
            app = create_app(settings)

            with TestClient(app):
                assert app.state.predictor is None
                assert app.state.store is not None
