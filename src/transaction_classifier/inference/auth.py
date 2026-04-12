"""API-key authentication dependencies for FastAPI."""

import hmac

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _matches_any(candidate: str, allowed: list[str]) -> bool:
    """Constant-time comparison of *candidate* against every key in *allowed*."""
    encoded = candidate.encode()
    return any(hmac.compare_digest(encoded, k.encode()) for k in allowed)


async def require_api_key(
    request: Request,
    api_key: str | None = Security(_header),
) -> None:
    """Enforce API-key auth on prediction endpoints.

    Auth is bypassed when the key list is empty (development mode).
    """
    keys = request.app.state.settings.api_keys
    if not keys:
        return
    if not api_key or not _matches_any(api_key, keys):
        raise HTTPException(status_code=403, detail="Invalid API key")


async def require_admin_key(
    request: Request,
    api_key: str | None = Security(_header),
) -> None:
    """Enforce API-key auth on admin endpoints.

    Auth is bypassed when the admin key list is empty (development mode).
    """
    keys = request.app.state.settings.admin_api_keys
    if not keys:
        return
    if not api_key or not _matches_any(api_key, keys):
        raise HTTPException(status_code=403, detail="Invalid API key")
