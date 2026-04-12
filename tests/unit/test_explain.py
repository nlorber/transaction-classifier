"""Unit tests for the /explain endpoint."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transaction_classifier.core.config import Settings
from transaction_classifier.inference.routes import explain


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


@pytest.fixture
def explain_app(sample_df, domain_engine):
    """Minimal app with explain router and a trained engine."""
    from sklearn.preprocessing import LabelEncoder

    from transaction_classifier.core.artifacts.schema import Manifest, ModelBundle
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

    app = FastAPI()
    app.state.engines = {"demo": engine}
    app.state.loaders = {"demo": MagicMock()}
    app.state.settings = Settings(sandbox_mode=False)
    app.state.start_time = time.time()
    app.include_router(explain.router)

    return TestClient(app)


class TestExplainEndpoint:
    """POST /explain/{client_id}."""

    def test_returns_contributions(self, explain_app):
        pytest.importorskip("shap")
        resp = explain_app.post(
            "/explain/demo",
            json={"transactions": [{"description": "URSSAF COTISATIONS", "debit": 1234.56}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert "predicted_code" in data["results"][0]
        assert "contributions" in data["results"][0]
        assert len(data["results"][0]["contributions"]) > 0
        assert "model_version" in data

    def test_max_features_param(self, explain_app):
        pytest.importorskip("shap")
        resp = explain_app.post(
            "/explain/demo",
            params={"max_features": 3},
            json={"transactions": [{"description": "EDF FACTURE", "debit": 89.0}]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"][0]["contributions"]) <= 3

    def test_unknown_client_returns_404(self, explain_app):
        resp = explain_app.post(
            "/explain/unknown",
            json={"transactions": [{"description": "TEST"}]},
        )
        assert resp.status_code == 404

    def test_shap_not_installed_returns_501(self, explain_app):
        with patch(
            "transaction_classifier.inference.predictor.Predictor.explain",
            side_effect=ImportError("No module named 'shap'"),
        ):
            resp = explain_app.post(
                "/explain/demo",
                json={"transactions": [{"description": "TEST"}]},
            )
            assert resp.status_code == 501

    def test_sandbox_mode_returns_fixed_response(self, explain_app):
        explain_app.app.state.settings.sandbox_mode = True
        resp = explain_app.post(
            "/explain/demo",
            json={"transactions": [{"description": "TEST"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] == "sandbox"
        assert data["results"][0]["predicted_code"] == "100000"
        explain_app.app.state.settings.sandbox_mode = False

    def test_invalid_target_class_returns_422(self, explain_app):
        pytest.importorskip("shap")
        resp = explain_app.post(
            "/explain/demo",
            params={"target_class": "999999"},
            json={"transactions": [{"description": "TEST"}]},
        )
        assert resp.status_code == 422
        assert "999999" in resp.json()["detail"]
