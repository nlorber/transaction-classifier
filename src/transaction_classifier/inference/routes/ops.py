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
def refresh_model(request: Request) -> dict[str, Any]:
    """Force-reload model from disk."""
    settings = request.app.state.settings
    if settings.sandbox_mode:
        return {"status": "sandbox"}

    store = request.app.state.store
    try:
        from ...core.features.engine import DomainFeatureEngine

        domain_engine = DomainFeatureEngine(settings.feature_profile)
        predictor = reload_predictor(store, settings.default_top_k, domain_engine)
        request.app.state.predictor = predictor
        logger.info("Reloaded model → %s", predictor.bundle.manifest.version)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="No model artifacts found") from err
    except Exception as err:
        logger.exception("Reload failed")
        raise HTTPException(status_code=500, detail="Reload failed") from err

    return {"status": "reloaded", "model_version": predictor.bundle.manifest.version}


@router.post("/confidence-histogram")
def confidence_histogram(
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

    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    frame = predictor.build_frame(body.transactions)
    from ...core.features.pipeline import assemble_feature_matrix

    if predictor.domain_engine is None:
        raise HTTPException(status_code=500, detail="domain_engine not configured")
    X = assemble_feature_matrix(
        frame, predictor.bundle.text_extractor, predictor.domain_engine, fit=False
    )
    proba = predictor.bundle.model.predict_proba(X)

    max_conf = np.max(proba, axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(max_conf, bins=edges)

    return {
        "model_version": predictor.bundle.manifest.version,
        "n_samples": len(body.transactions),
        "mean_confidence": round(float(max_conf.mean()), 4),
        "median_confidence": round(float(np.median(max_conf)), 4),
        "histogram": {
            "bin_edges": [round(float(e), 2) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }
