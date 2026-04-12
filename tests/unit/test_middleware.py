"""Tests for the LatencyMiddleware."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from transaction_classifier.inference.middleware import LatencyMiddleware


@pytest.fixture
def app_with_middleware():
    """Minimal FastAPI app with the request timing middleware."""
    app = FastAPI()
    app.add_middleware(LatencyMiddleware)

    @app.get("/classify/test")
    async def predict_endpoint():
        return PlainTextResponse("ok")

    @app.get("/health")
    async def health_endpoint():
        return PlainTextResponse("healthy")

    return TestClient(app)


def test_response_has_request_id_header(app_with_middleware):
    resp = app_with_middleware.get("/health")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) == 8


def test_predict_path_logs_at_info(app_with_middleware, caplog):
    with caplog.at_level(logging.INFO, logger="transaction_classifier.inference.middleware"):
        resp = app_with_middleware.get("/classify/test")
    assert resp.status_code == 200
    assert any("rid=" in rec.message for rec in caplog.records)


def test_non_predict_path_does_not_log_info(app_with_middleware, caplog):
    with caplog.at_level(logging.INFO, logger="transaction_classifier.inference.middleware"):
        resp = app_with_middleware.get("/health")
    assert resp.status_code == 200
    assert not any("rid=" in rec.message for rec in caplog.records)
