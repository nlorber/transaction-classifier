"""Pydantic request / response models for the classification API."""

from pydantic import BaseModel, Field


class TransactionPayload(BaseModel):
    """A single transaction to be classified."""

    description: str = Field(..., max_length=1000)
    remarks: str = Field(default="", max_length=5000)
    debit: float = 0.0
    credit: float = 0.0
    posting_date: str = ""
    reference: str = Field(default="", max_length=200)


class ClassifyRequest(BaseModel):
    """Batch classification request."""

    transactions: list[TransactionPayload] = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class ScoredCode(BaseModel):
    """One predicted account code with its confidence score."""

    code: str
    confidence: float


class ClassifyItemResult(BaseModel):
    """Predictions for a single transaction."""

    predictions: list[ScoredCode]


class ClassifyResponse(BaseModel):
    """Top-level response wrapping all item results."""

    results: list[ClassifyItemResult]
    model_version: str


class StatusResponse(BaseModel):
    """Health / liveness probe response."""

    status: str
    model_loaded: bool
    uptime_seconds: float = 0.0


class FeatureContribution(BaseModel):
    """A single feature's contribution to a prediction."""

    feature: str
    value: float
    shap_value: float


class ExplainItemResult(BaseModel):
    """SHAP explanation for a single transaction."""

    predicted_code: str
    confidence: float
    contributions: list[FeatureContribution]


class ExplainResponse(BaseModel):
    """Top-level response for the /explain endpoint."""

    results: list[ExplainItemResult]
    model_version: str
