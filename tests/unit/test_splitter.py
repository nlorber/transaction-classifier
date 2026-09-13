"""Unit tests for chronological splitting, including the defensive error paths."""

import logging

import pandas as pd
import pytest

from transaction_classifier.core.data.splitter import split_by_date, temporal_partition_stats


def _dated(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "posting_date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "target": [f"c{i % 3}" for i in range(n)],
        }
    )


def test_missing_date_column_raises():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="posting_date"):
        split_by_date(df)


@pytest.mark.parametrize(
    ("train_ratio", "val_ratio"),
    [(0.0, 0.15), (0.70, 0.0), (0.85, 0.15), (0.90, 0.20)],
)
def test_ratios_that_leave_a_block_empty_raise(train_ratio, val_ratio):
    with pytest.raises(ValueError, match="test block"):
        split_by_date(_dated(20), train_ratio=train_ratio, val_ratio=val_ratio)


def test_unparseable_dates_are_discarded(caplog):
    df = pd.DataFrame(
        {
            "posting_date": pd.to_datetime(
                ["2025-01-01", None, "2025-02-01", "2025-03-01", "2025-04-01"], errors="coerce"
            ),
            "target": ["a", "b", "c", "d", "e"],
        }
    )
    with caplog.at_level(logging.WARNING):
        train, val, test = split_by_date(df, train_ratio=0.5, val_ratio=0.25)

    # The single NaT row is dropped before splitting.
    assert len(train) + len(val) + len(test) == 4
    assert "Discarding" in caplog.text


def test_default_ratios_give_70_15_15_blocks():
    train, val, test = split_by_date(_dated(100))
    assert (len(train), len(val), len(test)) == (70, 15, 15)


def test_blocks_are_chronological_and_disjoint():
    shuffled = _dated(20).sample(frac=1.0, random_state=0)
    train, val, test = split_by_date(shuffled, train_ratio=0.5, val_ratio=0.25)

    assert train["posting_date"].max() < val["posting_date"].min()
    assert val["posting_date"].max() < test["posting_date"].min()
    assert len(train) + len(val) + len(test) == 20


def test_partition_stats_cover_all_three_blocks():
    df = _dated(20)
    train, val, test = split_by_date(df, train_ratio=0.5, val_ratio=0.25)
    test = pd.concat(
        [test, pd.DataFrame({"posting_date": [pd.Timestamp("2026-01-01")], "target": ["new"]})]
    )

    stats = temporal_partition_stats(train, val, test)

    assert (stats["train_rows"], stats["val_rows"], stats["test_rows"]) == (10, 5, 6)
    assert stats["total_rows"] == 21
    assert stats["test_exclusive_classes"] == 1
    assert stats["test_coverage"] == pytest.approx(5 / 6)
    assert stats["val_coverage"] == 1.0
