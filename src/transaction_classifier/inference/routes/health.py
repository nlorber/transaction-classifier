"""Health and readiness probes."""

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..schemas import StatusResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=StatusResponse)
def health(request: Request) -> StatusResponse:
    """Liveness check — returns 200 if the process is alive."""
    settings = request.app.state.settings
    boot = request.app.state.start_time

    if settings.sandbox_mode:
        return StatusResponse(
            status="sandbox",
            model_loaded=False,
            uptime_seconds=time.time() - boot,
        )

    loaded = request.app.state.predictor is not None

    return StatusResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        uptime_seconds=time.time() - boot,
    )


@router.get("/ready", response_model=None)
def ready(request: Request) -> dict[str, bool] | JSONResponse:
    """Readiness check — 200 only when the model is serving."""
    if request.app.state.settings.sandbox_mode:
        return {"ready": True}

    if request.app.state.predictor is None:
        return JSONResponse({"ready": False}, status_code=503)
    return {"ready": True}
