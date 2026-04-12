"""Consolidated standard (non-domain) numeric and date features."""

import numpy as np
import pandas as pd

from ..data.preprocessor import NULL_SENTINEL

# Fine-grained ordinal bucketing (log-scale bins → single integer feature).
# DomainFeatureEngine uses a separate coarse scheme (binary one-hot columns)
# to let the model learn non-linear amount effects independently.

_TWO_PI = 2 * np.pi


def _log_amount_edges(n_bins: int = 8) -> list[float]:
    """Generate amount-bucket edges on a log10 scale from 1 to 100,000."""
    return [
        -np.inf,
        0,
        *np.logspace(0, 5, n_bins - 1).round(0).tolist(),
        np.inf,
    ]


_AMOUNT_EDGES: list[float] = _log_amount_edges()


def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive numeric columns from raw transaction data.

    Produces: amount, log_amount, is_debit, is_credit, amount_bucket,
    desc_len, remarks_len, has_reference.
    """
    out = pd.DataFrame(index=df.index)

    debit = df["debit"].fillna(0).astype(float)
    credit = df["credit"].fillna(0).astype(float)

    out["amount"] = debit + credit
    out["log_amount"] = np.log1p(out["amount"])
    out["is_debit"] = (debit > 0).astype(int)
    out["is_credit"] = (credit > 0).astype(int)

    bucket = pd.cut(
        out["amount"],
        bins=_AMOUNT_EDGES,
        labels=range(len(_AMOUNT_EDGES) - 1),
        include_lowest=True,
    )
    out["amount_bucket"] = bucket.cat.codes.replace(-1, 0).astype(int)

    out["desc_len"] = df["description"].fillna("").str.len()
    out["remarks_len"] = df["remarks"].fillna("").str.len()

    out["has_reference"] = (
        df["reference"].notna() & (df["reference"] != "") & (df["reference"] != NULL_SENTINEL)
    ).astype(int)

    return out


def build_date_features(df: pd.DataFrame, date_col: str = "posting_date") -> pd.DataFrame:
    """Return a DataFrame of temporal features aligned to *df*'s index.

    Produces: weekday, month_day, month, quarter, is_month_end, is_month_start,
    is_weekend, day_cycle_sin, day_cycle_cos, month_cycle_sin, month_cycle_cos,
    weekday_cycle_sin, weekday_cycle_cos.
    """
    raw = df[date_col]
    dates = (
        raw if pd.api.types.is_datetime64_any_dtype(raw) else pd.to_datetime(raw, errors="coerce")
    )

    out = pd.DataFrame(index=df.index)

    # Categorical components
    out["weekday"] = dates.dt.dayofweek
    out["month_day"] = dates.dt.day
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter

    # Binary flags
    out["is_month_end"] = (dates.dt.day >= 25).astype(int)
    out["is_month_start"] = (dates.dt.day <= 5).astype(int)
    out["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encodings
    out["day_cycle_sin"] = np.sin(_TWO_PI * dates.dt.day / 31)
    out["day_cycle_cos"] = np.cos(_TWO_PI * dates.dt.day / 31)
    out["month_cycle_sin"] = np.sin(_TWO_PI * dates.dt.month / 12)
    out["month_cycle_cos"] = np.cos(_TWO_PI * dates.dt.month / 12)
    out["weekday_cycle_sin"] = np.sin(_TWO_PI * dates.dt.dayofweek / 7)
    out["weekday_cycle_cos"] = np.cos(_TWO_PI * dates.dt.dayofweek / 7)

    # Fill NaN for rows with unparseable dates
    for col in out.columns:
        if out[col].isna().any():
            if col in ("weekday", "month_day", "month", "quarter"):
                mode = out[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else 0
                out[col] = out[col].fillna(fill)
            else:
                out[col] = out[col].fillna(0)

    return out
