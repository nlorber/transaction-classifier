"""Chronological data splitting for temporal validation."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def split_by_date(
    df: pd.DataFrame,
    date_col: str = "posting_date",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition a DataFrame chronologically into train, validation, and test blocks.

    The earliest *train_ratio* fraction (by date order) trains, the next
    *val_ratio* fraction validates (early stopping, tuning), and the most recent
    remainder is the held-out test block. Keeping test untouched by every
    fitting decision is what makes the metrics reported on it unbiased.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' is not present in the DataFrame")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            f"Need train_ratio > 0, val_ratio > 0 and a non-empty test block; "
            f"got train_ratio={train_ratio}, val_ratio={val_ratio}"
        )

    # Exclude rows where the date could not be parsed
    missing_dates = df[date_col].isna()
    if missing_dates.any():
        logger.warning("Discarding %d rows whose dates could not be parsed", missing_dates.sum())
        df = df[~missing_dates]

    ordered = df.sort_values(date_col).reset_index(drop=True)
    train_end = int(len(ordered) * train_ratio)
    val_end = train_end + int(len(ordered) * val_ratio)

    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:val_end].copy(),
        ordered.iloc[val_end:].copy(),
    )


def partition_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target",
) -> dict[str, Any]:
    """Basic size and class-count statistics for a train/val/test split."""
    parts = {"train": train_df, "val": val_df, "test": test_df}
    total = sum(len(part) for part in parts.values())
    stats: dict[str, Any] = {"total_rows": total}
    for name, part in parts.items():
        stats[f"{name}_rows"] = len(part)
        stats[f"{name}_fraction"] = len(part) / total
        stats[f"{name}_n_classes"] = part[target_col].nunique()
    return stats


def temporal_partition_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "posting_date",
    target_col: str = "target",
) -> dict[str, Any]:
    """Extended split statistics including date ranges and class-overlap analysis.

    Overlap is measured against the training classes, since rows of a class the
    model never saw cannot be scored.
    """
    stats = partition_stats(train_df, val_df, test_df, target_col)
    train_labels = set(train_df[target_col].unique())

    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        stats[f"{name}_dates"] = {
            "first": str(part[date_col].min()),
            "last": str(part[date_col].max()),
        }
    for name, part in (("val", val_df), ("test", test_df)):
        labels = set(part[target_col].unique())
        stats[f"{name}_shared_classes"] = len(labels & train_labels)
        stats[f"{name}_exclusive_classes"] = len(labels - train_labels)
        stats[f"{name}_coverage"] = (
            part[target_col].isin(train_labels).sum() / len(part) if len(part) > 0 else 0.0
        )

    return stats
