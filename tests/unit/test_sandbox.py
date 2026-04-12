"""Unit tests for sandbox mode."""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transaction_classifier.core.config import Settings
from transaction_classifier.inference.routes import classify, health, ops


@pytest.fixture
def sandbox_client():
    """Create a TestClient for a sandbox-mode app."""
    app = FastAPI()
    settings = Settings(sandbox_mode=True)
    app.state.settings = settings
    app.state.engines = {}
    app.state.loaders = {}
    app.state.start_time = time.time()

    app.include_router(classify.router)
    app.include_router(health.router)
    app.include_router(ops.router)

    return TestClient(app)


class TestSandboxPredict:
    """Sandbox mode returns fixed predictions."""

    def test_predict_returns_fixed_response(self, sandbox_client):
        resp = sandbox_client.post(
            "/classify/sandbox",
            json={
                "transactions": [{"description": "TEST TRANSACTION"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] == "sandbox"
        assert len(data["results"]) == 1
        preds = data["results"][0]["predictions"]
        assert len(preds) == 3
        assert preds[0] == {"code": "100000", "confidence": 0.90}
        assert preds[1] == {"code": "100001", "confidence": 0.06}
        assert preds[2] == {"code": "100002", "confidence": 0.04}

    def test_predict_batch_returns_one_result_per_transaction(self, sandbox_client):
        resp = sandbox_client.post(
            "/classify/sandbox",
            json={
                "transactions": [
                    {"description": "TX 1"},
                    {"description": "TX 2"},
                    {"description": "TX 3"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3

    def test_predict_respects_top_k(self, sandbox_client):
        resp = sandbox_client.post(
            "/classify/sandbox",
            json={
                "transactions": [{"description": "TEST"}],
                "top_k": 1,
            },
        )
        assert resp.status_code == 200
        preds = resp.json()["results"][0]["predictions"]
        assert len(preds) == 1
        assert preds[0] == {"code": "100000", "confidence": 0.90}

    def test_predict_top_k_above_default_returns_requested_count(self, sandbox_client):
        resp = sandbox_client.post(
            "/classify/sandbox",
            json={
                "transactions": [{"description": "TEST"}],
                "top_k": 10,
            },
        )
        assert resp.status_code == 200
        preds = resp.json()["results"][0]["predictions"]
        assert len(preds) == 10


class TestSandboxHealth:
    """Sandbox mode returns sandbox status."""

    def test_health_returns_sandbox_status(self, sandbox_client):
        resp = sandbox_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "sandbox"
        assert data["model_loaded"] is False

    def test_ready_returns_200(self, sandbox_client):
        resp = sandbox_client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True


class TestSandboxAdmin:
    """Sandbox mode admin is a no-op."""

    def test_reload_is_noop(self, sandbox_client):
        resp = sandbox_client.post("/ops/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "sandbox"
        assert data["clients"] == []
