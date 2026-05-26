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


@router.post("/explain", response_model=ExplainResponse)
def explain(
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

    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    if target_class is not None:
        valid_classes: list[str] = predictor.bundle.label_encoder.classes_.tolist()
        if target_class not in valid_classes:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Unknown target_class: '{target_class}' is not in the model's label set",
                    "valid_classes": valid_classes,
                },
            )

    try:
        items = predictor.explain(
            body.transactions,
            max_features=max_features,
            target_class=target_class,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="shap not installed — install with: uv sync --extra explain",
        ) from exc

    return ExplainResponse(
        results=items,
        model_version=predictor.bundle.manifest.version,
    )
