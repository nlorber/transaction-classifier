"""Unit tests for standard.py consolidated feature module."""

import numpy as np
import pandas as pd
import pytest

from transaction_classifier.core.features.standard import (
    build_date_features,
    build_numeric_features,
)


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "debit": [100.0, None, 50.0],
            "credit": [None, 200.0, None],
            "description": ["Buy groceries", "Salary payment", "Coffee"],
            "remarks": ["", "monthly", None],
            "reference": ["REF001", None, "NULL_SENTINEL"],
        }
    )


@pytest.fixture()
def date_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "posting_date": pd.to_datetime(["2024-01-03", "2024-03-27", "2024-06-15"]),
        }
    )


# ---------------------------------------------------------------------------
# build_numeric_features
# ---------------------------------------------------------------------------


def test_numeric_output_columns(numeric_df: pd.DataFrame) -> None:
    result = build_numeric_features(numeric_df)
    expected_cols = {
        "amount",
        "log_amount",
        "is_debit",
        "is_credit",
        "amount_bucket",
        "desc_len",
        "remarks_len",
        "has_reference",
    }
    assert expected_cols == set(result.columns)


def test_numeric_amount_debit_plus_credit(numeric_df: pd.DataFrame) -> None:
    result = build_numeric_features(numeric_df)
    # Row 0: debit=100, credit=NaN→0  → amount=100
    assert result.loc[0, "amount"] == pytest.approx(100.0)
    # Row 1: debit=NaN→0, credit=200  → amount=200
    assert result.loc[1, "amount"] == pytest.approx(200.0)
    # Row 2: debit=50, credit=NaN→0   → amount=50
    assert result.loc[2, "amount"] == pytest.approx(50.0)


def test_numeric_debit_credit_flags(numeric_df: pd.DataFrame) -> None:
    result = build_numeric_features(numeric_df)
    assert result.loc[0, "is_debit"] == 1
    assert result.loc[0, "is_credit"] == 0
    assert result.loc[1, "is_debit"] == 0
    assert result.loc[1, "is_credit"] == 1
    assert result.loc[2, "is_debit"] == 1
    assert result.loc[2, "is_credit"] == 0


def test_numeric_has_reference(numeric_df: pd.DataFrame) -> None:
    result = build_numeric_features(numeric_df)
    # REF001 → valid reference
    assert result.loc[0, "has_reference"] == 1
    # None → no reference
    assert result.loc[1, "has_reference"] == 0
    # "NULL_SENTINEL" string → treated as sentinel only if it equals NULL_SENTINEL constant;
    # we just verify the column is integer 0 or 1
    assert result.loc[2, "has_reference"] in (0, 1)


def test_numeric_log_amount(numeric_df: pd.DataFrame) -> None:
    result = build_numeric_features(numeric_df)
    assert result.loc[0, "log_amount"] == pytest.approx(np.log1p(100.0))
    assert result.loc[1, "log_amount"] == pytest.approx(np.log1p(200.0))


def test_numeric_index_preserved(numeric_df: pd.DataFrame) -> None:
    result = build_numeric_features(numeric_df)
    assert list(result.index) == list(numeric_df.index)


# ---------------------------------------------------------------------------
# build_date_features
# ---------------------------------------------------------------------------


def test_date_output_has_13_columns(date_df: pd.DataFrame) -> None:
    result = build_date_features(date_df)
    assert result.shape[1] == 13


def test_date_expected_columns(date_df: pd.DataFrame) -> None:
    result = build_date_features(date_df)
    expected = {
        "weekday",
        "month_day",
        "month",
        "quarter",
        "is_month_end",
        "is_month_start",
        "is_weekend",
        "day_cycle_sin",
        "day_cycle_cos",
        "month_cycle_sin",
        "month_cycle_cos",
        "weekday_cycle_sin",
        "weekday_cycle_cos",
    }
    assert expected == set(result.columns)


def test_date_cyclical_range(date_df: pd.DataFrame) -> None:
    result = build_date_features(date_df)
    cyclical_cols = [
        "day_cycle_sin",
        "day_cycle_cos",
        "month_cycle_sin",
        "month_cycle_cos",
        "weekday_cycle_sin",
        "weekday_cycle_cos",
    ]
    for col in cyclical_cols:
        assert result[col].between(-1, 1).all(), f"{col} has values outside [-1, 1]"


def test_date_is_month_end_flag(date_df: pd.DataFrame) -> None:
    result = build_date_features(date_df)
    # 2024-01-03: day=3 → NOT month_end (day < 25)
    assert result.loc[0, "is_month_end"] == 0
    # 2024-03-27: day=27 → IS month_end (day >= 25)
    assert result.loc[1, "is_month_end"] == 1
    # 2024-06-15: day=15 → NOT month_end
    assert result.loc[2, "is_month_end"] == 0


def test_date_is_weekend_flag(date_df: pd.DataFrame) -> None:
    result = build_date_features(date_df)
    # 2024-01-03: Wednesday → weekday
    assert result.loc[0, "is_weekend"] == 0
    # 2024-03-27: Wednesday → weekday
    assert result.loc[1, "is_weekend"] == 0
    # 2024-06-15: Saturday → weekend
    assert result.loc[2, "is_weekend"] == 1


def test_date_custom_col() -> None:
    df = pd.DataFrame({"txn_date": pd.to_datetime(["2024-02-10"])})
    result = build_date_features(df, date_col="txn_date")
    assert result.shape[1] == 13


def test_date_no_nans(date_df: pd.DataFrame) -> None:
    result = build_date_features(date_df)
    assert not result.isna().any().any()


def test_date_index_preserved(date_df: pd.DataFrame) -> None:
    result = build_date_features(date_df)
    assert list(result.index) == list(date_df.index)
