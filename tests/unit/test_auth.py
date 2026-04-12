"""Unit tests for API key authentication."""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from transaction_classifier.core.config import Settings
from transaction_classifier.inference.auth import AuthContext, require_admin_key, require_api_key


def _make_app(dependency, api_keys=None, admin_api_keys=None):
    """Create a minimal app with a single protected endpoint and settings on app.state."""
    app = FastAPI()
    app.state.settings = Settings(
        api_keys=api_keys or [],
        admin_api_keys=admin_api_keys or [],
    )

    @app.get("/protected", dependencies=[Depends(dependency)])
    def protected():
        return {"ok": True}

    return app


class TestRequireApiKey:
    """Tests for the predict-tier API key dependency."""

    def test_valid_key_passes(self):
        app = _make_app(require_api_key, api_keys=["test-key-1"])
        resp = TestClient(app).get("/protected", headers={"X-API-Key": "test-key-1"})
        assert resp.status_code == 200

    def test_invalid_key_returns_403(self):
        app = _make_app(require_api_key, api_keys=["test-key-1"])
        resp = TestClient(app).get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    def test_missing_header_returns_403(self):
        app = _make_app(require_api_key, api_keys=["test-key-1"])
        resp = TestClient(app).get("/protected")
        assert resp.status_code == 403

    def test_second_key_in_list_passes(self):
        app = _make_app(require_api_key, api_keys=["key-a", "key-b"])
        resp = TestClient(app).get("/protected", headers={"X-API-Key": "key-b"})
        assert resp.status_code == 200

    def test_no_keys_configured_allows_all(self):
        """Dev mode: if api_keys is empty, auth is bypassed."""
        app = _make_app(require_api_key)
        resp = TestClient(app).get("/protected")
        assert resp.status_code == 200


class TestRequireAdminKey:
    """Tests for the admin-tier API key dependency."""

    def test_valid_admin_key_passes(self):
        app = _make_app(require_admin_key, admin_api_keys=["admin-secret"])
        resp = TestClient(app).get("/protected", headers={"X-API-Key": "admin-secret"})
        assert resp.status_code == 200

    def test_predict_key_rejected_for_admin(self):
        app = _make_app(
            require_admin_key,
            api_keys=["predict-key"],
            admin_api_keys=["admin-secret"],
        )
        resp = TestClient(app).get("/protected", headers={"X-API-Key": "predict-key"})
        assert resp.status_code == 403

    def test_no_admin_keys_configured_allows_all(self):
        """Dev mode: if admin_api_keys is empty, auth is bypassed."""
        app = _make_app(require_admin_key)
        resp = TestClient(app).get("/protected")
        assert resp.status_code == 200


_predict_dep = Depends(require_api_key)
_admin_dep = Depends(require_admin_key)


class TestAuthContext:
    """Tests that auth dependencies return typed AuthContext."""

    def test_predict_tier_returns_context(self):
        app = FastAPI()
        app.state.settings = Settings(api_keys=["test-key"])

        @app.get("/ctx")
        async def get_ctx(auth: AuthContext = _predict_dep):  # noqa: B008
            return {"tier": auth.tier}

        resp = TestClient(app).get("/ctx", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        assert resp.json()["tier"] == "predict"

    def test_admin_tier_returns_context(self):
        app = FastAPI()
        app.state.settings = Settings(admin_api_keys=["admin-key"])

        @app.get("/ctx")
        async def get_ctx(auth: AuthContext = _admin_dep):  # noqa: B008
            return {"tier": auth.tier}

        resp = TestClient(app).get("/ctx", headers={"X-API-Key": "admin-key"})
        assert resp.status_code == 200
        assert resp.json()["tier"] == "admin"

    def test_dev_mode_returns_predict_context(self):
        app = FastAPI()
        app.state.settings = Settings()

        @app.get("/ctx")
        async def get_ctx(auth: AuthContext = _predict_dep):  # noqa: B008
            return {"tier": auth.tier}

        resp = TestClient(app).get("/ctx")
        assert resp.status_code == 200
        assert resp.json()["tier"] == "predict"
