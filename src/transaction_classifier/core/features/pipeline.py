"""Feature-matrix assembly — combines text, numeric, date, and domain features."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, spmatrix

from .engine import DomainFeatureEngine
from .standard import build_date_features, build_numeric_features
from .text import TfidfFeatureExtractor


def collect_feature_names(
    df: pd.DataFrame,
    text_extractor: TfidfFeatureExtractor,
    domain_engine: DomainFeatureEngine,
) -> list[str]:
    """Return ordered feature names matching the columns of ``assemble_feature_matrix``.

    Uses a single-row sample from *df* to derive dense column names.
    Order: [text (desc_, rem_, chr_)] + [numeric] + [date] + [domain].
    """
    names: list[str] = list(text_extractor.feature_names)

    sample = df.iloc[:1]
    names.extend(build_numeric_features(sample).columns)
    names.extend(build_date_features(sample).columns)
    names.extend(domain_engine.feature_names)

    return names


def assemble_feature_matrix(
    df: pd.DataFrame,
    text_extractor: TfidfFeatureExtractor | None,
    domain_engine: DomainFeatureEngine,
    fit: bool = True,
    skip_text: bool = False,
) -> spmatrix | np.ndarray[Any, np.dtype[Any]]:
    """Build the complete feature matrix for a set of transactions.

    Args:
        df: Raw transaction DataFrame.
        text_extractor: A ``TfidfFeatureExtractor`` instance (may be ``None``
            when *skip_text* is ``True``).
        domain_engine: Config-driven domain feature engine.
        fit: Whether to fit the text extractor (``True`` during training).
        skip_text: If ``True``, omit TF-IDF features and return a dense array.

    Returns:
        A sparse matrix (when TF-IDF is included) or a dense ndarray.
    """
    df = df.reset_index(drop=True)

    # Resolve amount column
    debit = df["debit"].fillna(0).astype(float)
    credit = df["credit"].fillna(0).astype(float)
    df = df.copy()
    df["amount"] = debit + credit

    numeric_feats = build_numeric_features(df)
    date_feats = build_date_features(df)
    domain_feats = domain_engine.build(
        df,
        text_cols=["remarks", "description"],
        amount_col="amount",
        date_col="posting_date",
        comment_col="remarks",
    )

    dense_block = pd.concat(
        [
            numeric_feats.reset_index(drop=True),
            date_feats.reset_index(drop=True),
            domain_feats.reset_index(drop=True),
        ],
        axis=1,
    ).values

    if skip_text:
        return dense_block

    if text_extractor is None:
        raise ValueError("text_extractor must be set before calling transform")
    sparse_text = text_extractor.fit_transform(df) if fit else text_extractor.transform(df)

    return hstack([sparse_text, dense_block])
