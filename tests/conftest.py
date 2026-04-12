"""Shared test fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv_path():
    """Path to the 10-row sample CSV."""
    return FIXTURES_DIR / "sample_10rows.csv"


@pytest.fixture
def sample_df(sample_csv_path):
    """Loaded DataFrame from the sample CSV (with target column)."""
    from transaction_classifier.core.data.loader import read_csv_data

    return read_csv_data(sample_csv_path, target_length=6, min_class_samples=1)


@pytest.fixture
def sample_transactions():
    """List of TransactionPayload for API testing."""
    from transaction_classifier.inference.schemas import TransactionPayload

    return [
        TransactionPayload(
            description="URSSAF COTISATIONS",
            remarks="PRLV SEPA CPY:FR123 NBE:URSSAF",
            debit=1234.56,
            credit=0,
            posting_date="2025-01-15",
        ),
        TransactionPayload(
            description="VIREMENT CLIENT DUPONT",
            remarks="VIR SEPA REF:VIR2025001 NPY:SAS DUPONT",
            debit=0,
            credit=5000.0,
            posting_date="2025-02-10",
        ),
    ]
