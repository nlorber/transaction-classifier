"""Unit tests for admin/ops endpoints (refresh, confidence-histogram)."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transaction_classifier.core.config import Settings
from transaction_classifier.inference.routes import ops


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


@pytest.fixture
def ops_app(sample_df, domain_engine):
    """Minimal app with ops router and a mock engine."""
    from sklearn.preprocessing import LabelEncoder

    from transaction_classifier.core.artifacts.schema import Manifest, ModelBundle
    from transaction_classifier.core.artifacts.store import ModelStore
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.core.features.text import TfidfFeatureExtractor
    from transaction_classifier.core.models.xgboost_model import XGBoostModel
    from transaction_classifier.inference.predictor import Predictor

    tf = TfidfFeatureExtractor(
        label_vocab_size=20, detail_vocab_size=20, char_vocab_size=20, min_df=1
    )
    X = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)
    le = LabelEncoder()
    y = le.fit_transform(sample_df["target"])
    model = XGBoostModel(n_estimators=5, max_depth=2, verbosity=0, patience=None)
    model.fit(X, y)

    manifest = Manifest(version="v-test", metrics={"accuracy": 0.5})
    bundle = ModelBundle(model=model, text_extractor=tf, label_encoder=le, manifest=manifest)
    engine = Predictor(bundle, default_top_k=3, domain_engine=domain_engine)

    loader = MagicMock(spec=ModelStore)

    app = FastAPI()
    app.state.predictor = engine
    app.state.store = loader
    app.state.settings = Settings(sandbox_mode=False)
    app.state.start_time = time.time()
    app.include_router(ops.router)

    return TestClient(app), loader, engine


class TestRefresh:
    """POST /ops/refresh endpoint."""

    def test_refresh(self, ops_app):
        client, loader, engine = ops_app
        with patch(
            "transaction_classifier.inference.routes.ops.reload_predictor", return_value=engine
        ):
            resp = client.post("/ops/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reloaded"
        assert data["model_version"] == "v-test"

    def test_refresh_file_not_found_returns_404(self, ops_app):
        client, _, _ = ops_app
        with patch(
            "transaction_classifier.inference.routes.ops.reload_predictor",
            side_effect=FileNotFoundError("no artifacts"),
        ):
            resp = client.post("/ops/refresh")
        assert resp.status_code == 404

    def test_refresh_generic_error_returns_500(self, ops_app):
        client, _, _ = ops_app
        with patch(
            "transaction_classifier.inference.routes.ops.reload_predictor",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post("/ops/refresh")
        assert resp.status_code == 500

    def test_refresh_sandbox_mode_is_noop(self, ops_app):
        client, _, _ = ops_app
        client.app.state.settings.sandbox_mode = True
        resp = client.post("/ops/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "sandbox"
        client.app.state.settings.sandbox_mode = False


class TestConfidenceHistogram:
    """POST /ops/confidence-histogram endpoint."""

    def test_histogram_returns_valid_structure(self, ops_app):
        client, _, _ = ops_app
        resp = client.post(
            "/ops/confidence-histogram",
            json={
                "transactions": [
                    {"description": "URSSAF COTISATIONS", "debit": 1234.56},
                    {"description": "EDF FACTURE", "debit": 89.0},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_samples"] == 2
        assert 0 <= data["mean_confidence"] <= 1
        assert 0 <= data["median_confidence"] <= 1
        assert len(data["histogram"]["bin_edges"]) == 11  # 10 bins + 1
        assert len(data["histogram"]["counts"]) == 10
        assert sum(data["histogram"]["counts"]) == 2

    def test_histogram_custom_bins(self, ops_app):
        client, _, _ = ops_app
        resp = client.post(
            "/ops/confidence-histogram",
            params={"n_bins": 5},
            json={"transactions": [{"description": "TEST", "debit": 100}]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["histogram"]["bin_edges"]) == 6  # 5 bins + 1

    def test_histogram_no_model_returns_503(self, ops_app):
        client, _, _ = ops_app
        client.app.state.predictor = None
        resp = client.post(
            "/ops/confidence-histogram",
            json={"transactions": [{"description": "TEST"}]},
        )
        assert resp.status_code == 503

    def test_histogram_sandbox_mode_returns_400(self, ops_app):
        client, _, _ = ops_app
        client.app.state.settings.sandbox_mode = True
        resp = client.post(
            "/ops/confidence-histogram",
            json={"transactions": [{"description": "TEST"}]},
        )
        assert resp.status_code == 400
        assert "sandbox" in resp.json()["detail"].lower()
        client.app.state.settings.sandbox_mode = False
