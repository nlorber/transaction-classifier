"""Tests for explain endpoint Pydantic schemas."""

from transaction_classifier.inference.schemas import (
    ExplainItemResult,
    ExplainResponse,
    FeatureContribution,
)


def test_feature_contribution_fields():
    fc = FeatureContribution(feature="ent_social_contributions", value=1.0, shap_value=0.32)
    assert fc.feature == "ent_social_contributions"
    assert fc.value == 1.0
    assert fc.shap_value == 0.32


def test_explain_item_result():
    item = ExplainItemResult(
        predicted_code="431000",
        confidence=0.72,
        contributions=[
            FeatureContribution(feature="f1", value=1.0, shap_value=0.5),
        ],
    )
    assert item.predicted_code == "431000"
    assert len(item.contributions) == 1


def test_explain_response():
    resp = ExplainResponse(
        results=[
            ExplainItemResult(
                predicted_code="431000",
                confidence=0.72,
                contributions=[],
            )
        ],
        model_version="v_test",
    )
    assert len(resp.results) == 1
    assert resp.model_version == "v_test"
