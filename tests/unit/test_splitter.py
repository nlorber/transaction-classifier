"""Unit tests for chronological splitting, including the defensive error paths."""

import logging

import pandas as pd
import pytest

from transaction_classifier.core.data.splitter import split_by_date


def test_missing_date_column_raises():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="posting_date"):
        split_by_date(df)


def test_unparseable_dates_are_discarded(caplog):
    df = pd.DataFrame(
        {
            "posting_date": pd.to_datetime(
                ["2025-01-01", None, "2025-02-01", "2025-03-01"], errors="coerce"
            ),
            "target": ["a", "b", "c", "d"],
        }
    )
    with caplog.at_level(logging.WARNING):
        train, val = split_by_date(df, train_ratio=0.5)

    # The single NaT row is dropped before splitting.
    assert len(train) + len(val) == 3
    assert "Discarding" in caplog.text


def test_chronological_order_is_respected():
    df = pd.DataFrame(
        {
            "posting_date": pd.to_datetime(["2025-03-01", "2025-01-01", "2025-02-01"]),
            "target": ["late", "early", "mid"],
        }
    )
    train, val = split_by_date(df, train_ratio=0.7)  # int(3 * 0.7) -> 2 train rows
    assert list(train["target"]) == ["early", "mid"]
    assert list(val["target"]) == ["late"]
