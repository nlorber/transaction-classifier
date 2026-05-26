"""Predictor — turns raw transactions into scored predictions."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..core.artifacts.schema import ModelBundle
from ..core.artifacts.store import ModelStore
from ..core.features.engine import DomainFeatureEngine
from ..core.features.pipeline import assemble_feature_matrix
from .schemas import ClassifyItemResult, ScoredCode, TransactionPayload

if TYPE_CHECKING:
    from .schemas import ExplainItemResult

logger = logging.getLogger(__name__)


def reload_predictor(
    store: ModelStore,
    default_top_k: int,
    domain_engine: DomainFeatureEngine,
) -> "Predictor":
    """Load the active bundle and return a fresh predictor."""
    bundle = store.load_active()
    return Predictor(bundle, default_top_k=default_top_k, domain_engine=domain_engine)


class Predictor:
    """Holds a loaded model bundle and serves classification requests."""

    def __init__(
        self,
        bundle: ModelBundle,
        default_top_k: int = 3,
        domain_engine: DomainFeatureEngine | None = None,
    ):
        self.bundle = bundle
        self.default_top_k = default_top_k
        self.domain_engine = domain_engine

    def classify(
        self, transactions: list[TransactionPayload], top_k: int | None = None
    ) -> list[ClassifyItemResult]:
        """Return ranked predictions for each transaction."""
        k = top_k or self.default_top_k
        frame = self.build_frame(transactions)
        if self.domain_engine is None:
            raise ValueError("domain_engine must be set")
        X = assemble_feature_matrix(
            frame, self.bundle.text_extractor, self.domain_engine, fit=False
        )
        proba = self.bundle.model.predict_proba(X)

        items: list[ClassifyItemResult] = []
        for row_idx in range(len(frame)):
            row_p = proba[row_idx]
            best_idx = np.argsort(row_p)[-k:][::-1]
            codes = self.bundle.label_encoder.inverse_transform(best_idx)
            scores = row_p[best_idx]

            items.append(
                ClassifyItemResult(
                    predictions=[
                        ScoredCode(code=str(c), confidence=round(float(s), 4))
                        for c, s in zip(codes, scores, strict=True)
                    ]
                )
            )
        return items

    def explain(
        self,
        transactions: list[TransactionPayload],
        max_features: int = 10,
        target_class: str | None = None,
    ) -> list["ExplainItemResult"]:
        """Return per-transaction SHAP feature contributions."""
        import shap

        from ..core.features.pipeline import collect_feature_names
        from .schemas import ExplainItemResult, FeatureContribution

        frame = self.build_frame(transactions)
        if self.domain_engine is None:
            raise ValueError("domain_engine must be set")
        X = assemble_feature_matrix(
            frame, self.bundle.text_extractor, self.domain_engine, fit=False
        )
        feature_names = collect_feature_names(
            frame, self.bundle.text_extractor, self.domain_engine
        )

        if self.bundle.model.model is None:
            raise ValueError("Model has not been fitted")
        explainer = shap.TreeExplainer(self.bundle.model.model.get_booster())

        proba = self.bundle.model.predict_proba(X)
        classes = self.bundle.label_encoder.classes_

        # Ensure the matrix supports row-slicing (COO does not; CSR does).
        X_csr = X.tocsr() if hasattr(X, "tocsr") else X

        results: list[ExplainItemResult] = []
        for row_idx in range(len(frame)):
            # Materialise one row at a time to avoid a full-batch dense allocation.
            row = X_csr[row_idx]
            row_dense = row.toarray() if hasattr(row, "toarray") else np.asarray(row)
            row_dense = row_dense.reshape(1, -1)

            row_shap_values = explainer.shap_values(row_dense)
            # row_shap_values: list of (1, n_features) or (1, n_features, n_classes)
            row_shap_arr = (
                np.stack(row_shap_values, axis=-1)
                if isinstance(row_shap_values, list)
                else row_shap_values
            )
            if row_shap_arr.ndim == 2:
                row_shap_arr = row_shap_arr[:, :, np.newaxis]
            # Shape is now (1, n_features, n_classes)

            if target_class is not None:
                class_idx = int(np.where(classes == target_class)[0][0])
            else:
                class_idx = int(np.argmax(proba[row_idx]))

            code = str(classes[class_idx])
            confidence = round(float(proba[row_idx, class_idx]), 4)
            row_shap = row_shap_arr[0, :, class_idx]
            row_vals = row_dense[0]

            top_idx = np.argsort(np.abs(row_shap))[-max_features:][::-1]
            contributions = [
                FeatureContribution(
                    feature=feature_names[i] if i < len(feature_names) else f"f_{i}",
                    value=round(float(row_vals[i]), 4),
                    shap_value=round(float(row_shap[i]), 4),
                )
                for i in top_idx
            ]
            results.append(
                ExplainItemResult(
                    predicted_code=code,
                    confidence=confidence,
                    contributions=contributions,
                )
            )
        return results

    @staticmethod
    def build_frame(transactions: list[TransactionPayload]) -> pd.DataFrame:
        """Convert API payloads into a DataFrame matching the training schema."""
        rows = [
            {
                "description": t.description,
                "remarks": t.remarks,
                "debit": t.debit,
                "credit": t.credit,
                "posting_date": t.posting_date,
                "reference": t.reference,
            }
            for t in transactions
        ]
        df = pd.DataFrame(rows)
        df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0)
        df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0)
        df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
        return df
