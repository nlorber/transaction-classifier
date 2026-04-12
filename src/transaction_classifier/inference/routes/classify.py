"""Classification endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import require_api_key
from ..schemas import (
    ClassifyItemResult,
    ClassifyRequest,
    ClassifyResponse,
    ScoredCode,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["classify"], dependencies=[Depends(require_api_key)])

_SANDBOX_SCORES = [0.90, 0.06, 0.04] + [0.0] * 7
_SANDBOX_ITEMS = [
    ScoredCode(code=f"{100000 + i}", confidence=_SANDBOX_SCORES[i]) for i in range(10)
]


@router.post("/classify/{client_id}", response_model=ClassifyResponse)
def classify(client_id: str, body: ClassifyRequest, request: Request) -> ClassifyResponse:
    """Predict accounting codes for one or more transactions."""
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
            ClassifyItemResult(predictions=_SANDBOX_ITEMS[: body.top_k]) for _ in body.transactions
        ]
        return ClassifyResponse(results=items, model_version="sandbox")

    engines = request.app.state.engines
    if client_id not in engines:
        if client_id not in request.app.state.loaders:
            raise HTTPException(status_code=404, detail=f"Unknown client: '{client_id}'")
        raise HTTPException(status_code=503, detail=f"No model loaded for client '{client_id}'")

    engine = engines[client_id]
    items = engine.classify(body.transactions, top_k=body.top_k)
    return ClassifyResponse(
        results=items,
        model_version=engine.bundle.manifest.version,
    )
