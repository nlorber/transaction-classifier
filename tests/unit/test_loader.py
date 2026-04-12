"""Tests for prepare_dataframe()."""

import datetime

import pandas as pd
import pytest

from transaction_classifier.core.data.loader import prepare_dataframe


def _make_df(**overrides):
    """Build a minimal valid transactions DataFrame."""
    base = {
        "account_code": ["401ABC", "401ABC", "627XYZ", "627XYZ"],
        "description": ["A", "B", "C", "D"],
        "reference": ["REF1", None, "REF3", "REF4"],
        "remarks": ["Com A", None, "Com C", "Com D"],
        "credit": ["100.0", "0", "0", "50.0"],
        "debit": ["0", "200.0", "300.0", "0"],
        "posting_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_normalize_creates_target_column():
    result = prepare_dataframe(_make_df(), target_length=3, min_class_samples=1)
    assert "target" in result.columns
    assert list(result["target"]) == ["401", "401", "627", "627"]


def test_normalize_fills_null_text_fields():
    df = _make_df()
    df["description"] = pd.array(["A", None, "C", "D"], dtype=object)
    result = prepare_dataframe(df, target_length=6, min_class_samples=1)
    assert result["reference"].iloc[1] == ""
    assert result["remarks"].iloc[1] == ""
    assert result["description"].iloc[1] == ""


def test_normalize_converts_numeric_columns():
    result = prepare_dataframe(_make_df(), target_length=6, min_class_samples=1)
    assert result["debit"].dtype == float
    assert result["credit"].dtype == float


def test_normalize_parses_date_column():
    result = prepare_dataframe(_make_df(), target_length=6, min_class_samples=1)
    assert pd.api.types.is_datetime64_any_dtype(result["posting_date"])


def test_normalize_accepts_native_python_types():
    """psycopg2 returns Python floats and dates, not strings -- both must work."""
    df = pd.DataFrame(
        {
            "account_code": ["401ABC", "401ABC"],
            "description": ["A", "B"],
            "reference": ["REF1", "REF2"],
            "remarks": ["Com A", "Com B"],
            "credit": [100.0, 200.0],
            "debit": [0.0, 0.0],
            "posting_date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        }
    )
    result = prepare_dataframe(df, target_length=3, min_class_samples=1)
    assert pd.api.types.is_datetime64_any_dtype(result["posting_date"])
    assert result["credit"].dtype == float


def test_normalize_min_samples_1_keeps_all_rows():
    """min_class_samples=1 must not filter anything -- it is the 'no filter' sentinel."""
    df = pd.DataFrame(
        {
            "account_code": ["401ABC", "627XYZ", "999ZZZ"],  # three singletons
            "description": ["A", "B", "C"],
            "reference": ["", "", ""],
            "remarks": ["", "", ""],
            "credit": ["100", "0", "0"],
            "debit": ["0", "200", "300"],
            "posting_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    result = prepare_dataframe(df, target_length=3, min_class_samples=1)
    assert len(result) == 3


def test_normalize_filters_rare_classes():
    df = pd.DataFrame(
        {
            "account_code": ["401ABC", "401ABC", "401ABC", "627XYZ"],
            "description": ["A", "B", "C", "D"],
            "reference": ["", "", "", ""],
            "remarks": ["", "", "", ""],
            "credit": ["100", "200", "300", "400"],
            "debit": ["0", "0", "0", "0"],
            "posting_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )
    result = prepare_dataframe(df, target_length=3, min_class_samples=2)
    assert len(result) == 3
    assert "627" not in result["target"].values


def test_normalize_raises_if_empty_after_filtering():
    with pytest.raises(ValueError, match="No rows remain"):
        prepare_dataframe(_make_df(), target_length=3, min_class_samples=99)


def test_normalize_raises_on_missing_columns():
    df = pd.DataFrame({"account_code": ["401ABC"], "description": ["A"]})
    with pytest.raises(ValueError, match="Missing columns"):
        prepare_dataframe(df)


def test_normalize_raises_if_all_amounts_zero():
    df = _make_df(credit=["0", "0", "0", "0"], debit=["0", "0", "0", "0"])
    with pytest.raises(ValueError, match="zero"):
        prepare_dataframe(df, target_length=3, min_class_samples=1)
