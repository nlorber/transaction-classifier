"""CSV ingestion and DataFrame normalisation."""

from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = frozenset(
    {
        "debit",
        "credit",
        "posting_date",
        "account_code",
        "description",
        "remarks",
        "reference",
    }
)


def _check_required_columns(df: pd.DataFrame) -> None:
    """Raise if any expected column is absent."""
    absent = _REQUIRED_COLUMNS - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}. Present: {sorted(df.columns)}")


def prepare_dataframe(
    df: pd.DataFrame,
    target_length: int = 6,
    min_class_samples: int | None = None,
) -> pd.DataFrame:
    """Validate column presence, coerce types, derive *target*, and filter rare classes."""
    _check_required_columns(df)
    df = df.copy()

    # Coerce amounts
    df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0)
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0)

    if (df["debit"] == 0).all() and (df["credit"] == 0).all():
        raise ValueError(
            "Both debit and credit are entirely zero after conversion — "
            "verify that the source data contains valid numeric amounts."
        )

    # Parse dates
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")

    # Ensure text columns are strings
    for col in ("description", "remarks", "reference"):
        df[col] = df[col].fillna("")

    # Derive classification target from the account code prefix
    df["target"] = df["account_code"].str[:target_length]

    # Drop classes with too few examples
    if min_class_samples is not None and min_class_samples > 1:
        freq = df["target"].value_counts()
        keep = freq[freq >= min_class_samples].index
        df = df[df["target"].isin(keep)].copy()

    if df.empty:
        raise ValueError(
            "No rows remain after class-frequency filtering. "
            "Lower --min-samples or check the source data."
        )

    return df


def read_csv_data(
    csv_path: str | Path,
    target_length: int = 6,
    min_class_samples: int | None = None,
) -> pd.DataFrame:
    """Read a CSV file and return a normalised DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    raw = pd.read_csv(csv_path, dtype=str)
    return prepare_dataframe(raw, target_length=target_length, min_class_samples=min_class_samples)
