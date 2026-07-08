"""API-key authentication dependencies for FastAPI."""

import hmac
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class AuthContext:
    """Typed authentication result indicating the caller's access tier."""

    tier: Literal["predict", "admin"]


def _matches_any(candidate: str, allowed: list[str]) -> bool:
    """Compare *candidate* against every key without short-circuit to prevent timing leaks."""
    encoded = candidate.encode()
    results = [hmac.compare_digest(encoded, k.encode()) for k in allowed]
    return any(results)


def _auth_bypassed(request: Request) -> bool:
    """Auth is skipped only on explicit opt-out: TXCLS_AUTH_DISABLED or sandbox mode."""
    settings = request.app.state.settings
    return bool(settings.auth_disabled or settings.sandbox_mode)


async def require_api_key(
    request: Request,
    api_key: str | None = Security(_header),
) -> AuthContext:
    """Enforce API-key auth on prediction endpoints.

    Fails closed: with no keys configured, every request is rejected unless auth
    is explicitly disabled (``TXCLS_AUTH_DISABLED``) or the app runs in sandbox mode.
    """
    if _auth_bypassed(request):
        return AuthContext(tier="predict")
    keys = request.app.state.settings.api_keys
    if not api_key or not _matches_any(api_key, keys):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return AuthContext(tier="predict")


async def require_admin_key(
    request: Request,
    api_key: str | None = Security(_header),
) -> AuthContext:
    """Enforce API-key auth on admin endpoints.

    Fails closed: with no admin keys configured, every request is rejected unless
    auth is explicitly disabled (``TXCLS_AUTH_DISABLED``) or the app runs in sandbox mode.
    """
    if _auth_bypassed(request):
        return AuthContext(tier="admin")
    keys = request.app.state.settings.admin_api_keys
    if not api_key or not _matches_any(api_key, keys):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return AuthContext(tier="admin")
