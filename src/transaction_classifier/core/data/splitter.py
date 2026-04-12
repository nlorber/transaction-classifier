"""Chronological data splitting for temporal validation."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def split_by_date(
    df: pd.DataFrame,
    date_col: str = "posting_date",
    train_ratio: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a DataFrame chronologically into training and validation sets.

    The first *train_ratio* fraction (by date order) becomes the training set;
    the remainder is used for validation.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' is not present in the DataFrame")

    # Exclude rows where the date could not be parsed
    missing_dates = df[date_col].isna()
    if missing_dates.any():
        logger.warning("Discarding %d rows whose dates could not be parsed", missing_dates.sum())
        df = df[~missing_dates]

    ordered = df.sort_values(date_col).reset_index(drop=True)
    boundary = int(len(ordered) * train_ratio)

    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def partition_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "target",
) -> dict[str, Any]:
    """Basic size and class-count statistics for a train/val split."""
    combined = len(train_df) + len(val_df)
    return {
        "total_rows": combined,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_fraction": len(train_df) / combined,
        "val_fraction": len(val_df) / combined,
        "train_n_classes": train_df[target_col].nunique(),
        "val_n_classes": val_df[target_col].nunique(),
    }


def temporal_partition_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    date_col: str = "posting_date",
    target_col: str = "target",
) -> dict[str, Any]:
    """Extended split statistics including date ranges and class-overlap analysis."""
    base = partition_stats(train_df, val_df, target_col)

    train_labels = set(train_df[target_col].unique())
    val_labels = set(val_df[target_col].unique())

    shared = train_labels & val_labels
    only_in_train = train_labels - val_labels
    unseen_in_val = val_labels - train_labels

    base.update(
        {
            "train_dates": {
                "first": str(train_df[date_col].min()),
                "last": str(train_df[date_col].max()),
            },
            "val_dates": {
                "first": str(val_df[date_col].min()),
                "last": str(val_df[date_col].max()),
            },
            "shared_classes": len(shared),
            "train_exclusive_classes": len(only_in_train),
            "val_exclusive_classes": len(unseen_in_val),
            "val_coverage": (
                val_df[target_col].isin(train_labels).sum() / len(val_df)
                if len(val_df) > 0
                else 0.0
            ),
        }
    )

    return base
