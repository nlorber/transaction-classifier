"""Admin routes for model management and monitoring."""

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_admin_key
from ..predictor import reload_predictor
from ..schemas import ClassifyRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ops", tags=["ops"], dependencies=[Depends(require_admin_key)])


@router.post("/refresh")
def refresh_model(request: Request, client_id: str | None = None) -> dict[str, Any]:
    """Force-reload model(s) from disk.

    Pass ``?client_id=<id>`` to reload a single client; omit to reload all.
    """
    settings = request.app.state.settings
    if settings.sandbox_mode:
        return {"status": "sandbox", "clients": []}

    loaders = request.app.state.loaders
    engines = request.app.state.engines

    targets = [client_id] if client_id else list(loaders.keys())
    reloaded = []
    for cid in targets:
        ldr = loaders.get(cid)
        if ldr is None:
            raise HTTPException(status_code=404, detail=f"Unknown client: '{cid}'")
        try:
            from ...core.features.engine import DomainFeatureEngine

            domain_engine = DomainFeatureEngine(settings.feature_profile)
            engine = reload_predictor(ldr, settings.default_top_k, domain_engine)
            engines[cid] = engine
            logger.info("Reloaded %s → %s", cid, engine.bundle.manifest.version)
            reloaded.append({"client_id": cid, "model_version": engine.bundle.manifest.version})
        except FileNotFoundError as err:
            raise HTTPException(status_code=404, detail=f"No model artifacts for '{cid}'") from err
        except Exception as err:
            logger.exception("Reload failed for %s", cid)
            raise HTTPException(status_code=500, detail=f"Reload failed for '{cid}'") from err

    return {"status": "reloaded", "clients": reloaded}


@router.post("/confidence-histogram/{client_id}")
def confidence_histogram(
    client_id: str,
    body: ClassifyRequest,
    request: Request,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return a confidence histogram for a batch of transactions.

    Useful for monitoring prediction confidence drift over time.
    Compare histograms across time windows to detect distribution shifts.
    """
    settings = request.app.state.settings
    if settings.sandbox_mode:
        raise HTTPException(status_code=400, detail="Not available in sandbox mode")

    engines = request.app.state.engines
    if client_id not in engines:
        raise HTTPException(status_code=503, detail=f"No model loaded for '{client_id}'")

    engine = engines[client_id]
    frame = engine.build_frame(body.transactions)
    from ...core.features.pipeline import assemble_feature_matrix

    if engine.domain_engine is None:
        raise HTTPException(status_code=500, detail="domain_engine not configured")
    X = assemble_feature_matrix(
        frame, engine.bundle.text_extractor, engine.domain_engine, fit=False
    )
    proba = engine.bundle.model.predict_proba(X)

    max_conf = np.max(proba, axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(max_conf, bins=edges)

    return {
        "client_id": client_id,
        "model_version": engine.bundle.manifest.version,
        "n_samples": len(body.transactions),
        "mean_confidence": round(float(max_conf.mean()), 4),
        "median_confidence": round(float(np.median(max_conf)), 4),
        "histogram": {
            "bin_edges": [round(float(e), 2) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }
