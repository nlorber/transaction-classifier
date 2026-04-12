"""Artifact schema definitions for model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder

    from ..features.text import TfidfFeatureExtractor
    from ..models.xgboost_model import XGBoostModel


class Manifest(BaseModel):
    """Metadata persisted alongside each model bundle."""

    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    num_categories: int = 0
    n_features: int = 0
    checksums: dict[str, str] = Field(default_factory=dict)
    status: str = "candidate"


@dataclass
class ModelBundle:
    """All artefacts required for inference on a single model version."""

    model: XGBoostModel
    text_extractor: TfidfFeatureExtractor
    label_encoder: LabelEncoder
    manifest: Manifest
