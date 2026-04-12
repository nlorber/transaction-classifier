"""Unit tests for feature engineering."""

import pandas as pd
import pytest


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


def test_build_numeric_features(sample_df):
    from transaction_classifier.core.features.standard import build_numeric_features

    features = build_numeric_features(sample_df)
    assert len(features) == len(sample_df)
    assert "amount" in features.columns
    assert "log_amount" in features.columns
    assert "is_debit" in features.columns
    assert "is_credit" in features.columns
    assert features["amount"].min() >= 0


def test_create_date_features(sample_df):
    from transaction_classifier.core.features.standard import build_date_features

    features = build_date_features(sample_df)
    assert len(features) == len(sample_df)
    assert "weekday" in features.columns
    assert "month_cycle_sin" in features.columns
    assert not features.isna().any().any()


def test_build_domain_features(sample_df, domain_engine):
    debit = sample_df["debit"].fillna(0).astype(float)
    credit = sample_df["credit"].fillna(0).astype(float)
    df = sample_df.copy()
    df["amount"] = debit + credit
    features = domain_engine.build(
        df,
        text_cols=["remarks", "description"],
        amount_col="amount",
        date_col="posting_date",
        comment_col="remarks",
    )
    assert len(features) == len(sample_df)
    # Should detect known entities
    assert "ent_social_contributions" in features.columns
    assert "ent_energy_provider" in features.columns


def test_assemble_feature_matrix_pipeline(sample_df, domain_engine):
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    tf = TfidfFeatureExtractor(
        label_vocab_size=50,
        detail_vocab_size=50,
        char_vocab_size=50,
        min_df=1,
    )
    X = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)
    assert X.shape[0] == len(sample_df)
    assert X.shape[1] > 0


def test_text_featurizer_fit_transform(sample_df):
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    tf = TfidfFeatureExtractor(
        label_vocab_size=50,
        detail_vocab_size=50,
        char_vocab_size=50,
        min_df=1,
    )
    X = tf.fit_transform(sample_df)
    assert X.shape[0] == len(sample_df)

    # Transform again should produce same shape
    X2 = tf.transform(sample_df)
    assert X2.shape == X.shape


def test_domain_features_include_fiscal_periods(sample_df, domain_engine):
    """Fiscal period features must be generated when posting_date is present."""
    assert "posting_date" in sample_df.columns
    debit = sample_df["debit"].fillna(0).astype(float)
    credit = sample_df["credit"].fillna(0).astype(float)
    df = sample_df.copy()
    df["amount"] = debit + credit
    features = domain_engine.build(
        df,
        text_cols=["remarks", "description"],
        amount_col="amount",
        date_col="posting_date",
        comment_col="remarks",
    )
    fiscal_cols = [
        c
        for c in features.columns
        if c.startswith("fiscal_quarter") or c.startswith("fiscal_year")
    ]
    assert len(fiscal_cols) >= 4, f"Expected fiscal period columns, got: {fiscal_cols}"
    # At least some rows should have non-zero fiscal indicators
    assert features[fiscal_cols].sum().sum() > 0


def test_domain_amount_features_nonzero_for_credits(domain_engine):
    """Credit-only rows (debit=0) must produce non-zero domain amount features."""
    df = pd.DataFrame(
        {
            "description": ["VIREMENT CLIENT"],
            "remarks": ["VIR SEPA"],
            "debit": [0.0],
            "credit": [1500.0],
            "posting_date": pd.to_datetime(["2025-06-15"]),
            "amount": [1500.0],
        }
    )
    features = domain_engine.build(
        df,
        text_cols=["remarks", "description"],
        amount_col="amount",
        date_col="posting_date",
        comment_col="remarks",
    )
    amount_cols = [
        c
        for c in features.columns
        if c.startswith("amt_") or c.startswith("minimum_wage") or c.startswith("typical_salary")
    ]
    assert features[amount_cols].sum().sum() > 0, (
        "Credit-only row should have non-zero amount features"
    )


def test_feature_dimensions_match_between_fit_and_transform(sample_df, domain_engine):
    """Feature count at training (fit=True) must match inference (fit=False)."""
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    tf = TfidfFeatureExtractor(
        label_vocab_size=50,
        detail_vocab_size=50,
        char_vocab_size=50,
        min_df=1,
    )

    X_train = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)

    # Inference batch: single row, different transaction type, credit-only
    inference_df = pd.DataFrame(
        {
            "description": ["PAIEMENT CB AMAZON"],
            "remarks": ["PAIEMENT CB 1234"],
            "debit": [0.0],
            "credit": [42.99],
            "posting_date": ["2025-12-31"],
            "reference": [""],
        }
    )
    X_infer = assemble_feature_matrix(inference_df, tf, domain_engine, fit=False)

    assert X_train.shape[1] == X_infer.shape[1], (
        f"Train has {X_train.shape[1]} features but inference has {X_infer.shape[1]}"
    )


def test_text_featurizer_not_fitted_raises():
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    tf = TfidfFeatureExtractor()
    with pytest.raises(ValueError, match="Call fit"):
        tf.transform(pd.DataFrame({"description": ["test"], "remarks": ["test"]}))


def test_collect_feature_names_matches_matrix_columns(sample_df, domain_engine):
    """Feature name list must have exactly as many entries as the feature matrix has columns."""
    from transaction_classifier.core.features.pipeline import (
        assemble_feature_matrix,
        collect_feature_names,
    )
    from transaction_classifier.core.features.text import TfidfFeatureExtractor

    tf = TfidfFeatureExtractor(
        label_vocab_size=20, detail_vocab_size=20, char_vocab_size=20, min_df=1
    )
    X = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)
    names = collect_feature_names(sample_df, tf, domain_engine)

    assert len(names) == X.shape[1]
    assert all(isinstance(n, str) for n in names)
    # Spot check: TF-IDF names are prefixed
    assert any(n.startswith("desc_") for n in names)
    assert any(n.startswith("rem_") for n in names)
    assert any(n.startswith("chr_") for n in names)
    # Spot check: dense feature names present
    assert "amount" in names
    assert "weekday" in names
