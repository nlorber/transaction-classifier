"""Integration tests for the FastAPI application."""

import tempfile
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.config import Settings
from transaction_classifier.core.features.text import TfidfFeatureExtractor
from transaction_classifier.core.models.xgboost_model import XGBoostModel
from transaction_classifier.inference.predictor import Predictor
from transaction_classifier.inference.routes import classify, health, ops


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


@pytest.fixture
def client_with_model(sample_df, domain_engine):
    """Create a TestClient with a tiny trained model, no lifespan."""
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

        bundle = store.load_active()

        # Build a minimal app without lifespan (avoids async background task issues)
        app = FastAPI()
        engine = Predictor(bundle, default_top_k=3, domain_engine=domain_engine)
        app.state.predictor = engine
        app.state.store = store
        app.state.settings = Settings(artifact_dir=tmpdir)
        app.state.start_time = time.time()

        app.include_router(classify.router)
        app.include_router(health.router)
        app.include_router(ops.router)

        yield TestClient(app)


def test_health_endpoint(client_with_model):
    resp = client_with_model.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single(client_with_model):
    resp = client_with_model.post(
        "/classify",
        json={
            "transactions": [
                {
                    "description": "URSSAF COTISATIONS",
                    "remarks": "PRLV SEPA CPY:FR123 NBE:URSSAF",
                    "debit": 1234.56,
                    "credit": 0,
                    "posting_date": "2025-01-15",
                }
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert "predictions" in data["results"][0]
    assert len(data["results"][0]["predictions"]) > 0
    assert "code" in data["results"][0]["predictions"][0]
    assert "confidence" in data["results"][0]["predictions"][0]
    assert "model_version" in data


def test_predict_batch(client_with_model):
    resp = client_with_model.post(
        "/classify",
        json={
            "transactions": [
                {"description": "URSSAF", "posting_date": "2025-01-15", "debit": 100},
                {"description": "EDF", "posting_date": "2025-02-10", "debit": 50},
            ],
            "top_k": 2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert len(r["predictions"]) == 2


def test_predict_missing_description(client_with_model):
    resp = client_with_model.post("/classify", json={"transactions": [{"debit": 100}]})
    assert resp.status_code == 422


def test_predict_empty_batch_returns_422(client_with_model):
    resp = client_with_model.post("/classify", json={"transactions": []})
    assert resp.status_code == 422


def test_predict_batch_exceeds_max_size(client_with_model):
    settings = client_with_model.app.state.settings
    settings.batch_limit = 5
    resp = client_with_model.post(
        "/classify",
        json={
            "transactions": [{"description": f"TX {i}"} for i in range(6)],
        },
    )
    assert resp.status_code == 422
    assert "limit is 5" in resp.json()["detail"]
    settings.batch_limit = 100  # reset


def test_predict_batch_at_max_size(client_with_model):
    settings = client_with_model.app.state.settings
    settings.batch_limit = 3
    resp = client_with_model.post(
        "/classify",
        json={
            "transactions": [{"description": f"TX {i}"} for i in range(3)],
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 3
    settings.batch_limit = 100  # reset


def test_ready_endpoint(client_with_model):
    resp = client_with_model.get("/ready")
    assert resp.status_code == 200


@pytest.fixture
def client_with_auth(client_with_model):
    """Reconfigure client's app to require API keys."""
    settings = client_with_model.app.state.settings
    settings.api_keys = ["test-predict-key"]
    settings.admin_api_keys = ["test-admin-key"]
    yield client_with_model
    # Reset
    settings.api_keys = []
    settings.admin_api_keys = []


class TestAuthEnforcement:
    """Verify auth is enforced end-to-end on the real app."""

    def test_predict_requires_key(self, client_with_auth):
        resp = client_with_auth.post(
            "/classify",
            json={
                "transactions": [{"description": "TEST"}],
            },
        )
        assert resp.status_code == 403

    def test_predict_with_valid_key(self, client_with_auth):
        resp = client_with_auth.post(
            "/classify",
            json={"transactions": [{"description": "URSSAF COTISATIONS", "debit": 100}]},
            headers={"X-API-Key": "test-predict-key"},
        )
        assert resp.status_code == 200

    def test_admin_requires_key(self, client_with_auth):
        resp = client_with_auth.post("/ops/refresh")
        assert resp.status_code == 403

    def test_admin_with_valid_key(self, client_with_auth):
        resp = client_with_auth.post(
            "/ops/refresh",
            headers={"X-API-Key": "test-admin-key"},
        )
        assert resp.status_code == 200

    def test_health_remains_open(self, client_with_auth):
        resp = client_with_auth.get("/health")
        assert resp.status_code == 200

    def test_ready_remains_open(self, client_with_auth):
        resp = client_with_auth.get("/ready")
        assert resp.status_code == 200
