"""Explainability endpoint — per-transaction SHAP feature contributions."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..auth import require_api_key
from ..schemas import (
    ClassifyRequest,
    ExplainItemResult,
    ExplainResponse,
    FeatureContribution,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["explain"], dependencies=[Depends(require_api_key)])

_SANDBOX_CONTRIBUTIONS = [
    FeatureContribution(feature="ent_social_contributions", value=1.0, shap_value=0.32),
    FeatureContribution(feature="desc_urssaf", value=0.85, shap_value=0.21),
    FeatureContribution(feature="amt_medium", value=1.0, shap_value=0.08),
]


@router.post("/explain/{client_id}", response_model=ExplainResponse)
def explain(
    client_id: str,
    body: ClassifyRequest,
    request: Request,
    max_features: int = Query(default=10, ge=1, le=50),
    target_class: str | None = Query(default=None),
) -> ExplainResponse:
    """Return SHAP feature contributions for each transaction's top prediction.

    Requires the ``explain`` extra (``shap`` package). Returns 501 if not installed.
    """
    settings = request.app.state.settings

    if len(body.transactions) > settings.batch_limit:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Batch too large: {len(body.transactions)} items, limit is {settings.batch_limit}"
            ),
        )

    if settings.sandbox_mode:
        items = [
            ExplainItemResult(
                predicted_code="100000",
                confidence=0.90,
                contributions=_SANDBOX_CONTRIBUTIONS[:max_features],
            )
            for _ in body.transactions
        ]
        return ExplainResponse(results=items, model_version="sandbox")

    engines = request.app.state.engines
    if client_id not in engines:
        if client_id not in request.app.state.loaders:
            raise HTTPException(status_code=404, detail=f"Unknown client: '{client_id}'")
        raise HTTPException(status_code=503, detail=f"No model loaded for client '{client_id}'")

    engine = engines[client_id]

    try:
        items = engine.explain(
            body.transactions,
            max_features=max_features,
            target_class=target_class,
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="shap not installed — install with: uv sync --extra explain",
        ) from None
    except (IndexError, KeyError):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown target_class: '{target_class}' is not in the model's label set",
        ) from None

    return ExplainResponse(
        results=items,
        model_version=engine.bundle.manifest.version,
    )
