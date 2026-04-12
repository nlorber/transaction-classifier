"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from transaction_classifier.inference.schemas import (
    ClassifyItemResult,
    ClassifyRequest,
    ScoredCode,
    StatusResponse,
    TransactionPayload,
)


def test_transaction_payload_minimal():
    t = TransactionPayload(description="Test")
    assert t.description == "Test"
    assert t.debit == 0.0
    assert t.credit == 0.0
    assert t.remarks == ""


def test_transaction_payload_full():
    t = TransactionPayload(
        description="URSSAF",
        remarks="PRLV SEPA",
        debit=1234.56,
        credit=0,
        posting_date="2025-01-15",
        reference="REF123",
    )
    assert t.debit == 1234.56


def test_transaction_payload_requires_description():
    with pytest.raises(ValidationError):
        TransactionPayload()


def test_classify_item_result():
    result = ClassifyItemResult(
        predictions=[
            ScoredCode(code="401000", confidence=0.85),
            ScoredCode(code="411000", confidence=0.10),
        ],
    )
    assert len(result.predictions) == 2
    assert result.predictions[0].code == "401000"


def test_classify_request_top_k_bounds():
    t = TransactionPayload(description="test")
    req = ClassifyRequest(transactions=[t], top_k=5)
    assert req.top_k == 5

    with pytest.raises(ValidationError):
        ClassifyRequest(transactions=[t], top_k=0)

    with pytest.raises(ValidationError):
        ClassifyRequest(transactions=[t], top_k=21)


def test_status_response():
    h = StatusResponse(status="healthy", model_loaded=True)
    assert h.status == "healthy"
