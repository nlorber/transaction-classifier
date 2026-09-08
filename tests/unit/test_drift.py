"""Unit tests for PSI drift detection."""

import math

import numpy as np
import pandas as pd
import pytest

from transaction_classifier.core.evaluation.drift import (
    PSI_MODERATE_MAX,
    PSI_STABLE_MAX,
    build_baseline,
    compute_psi,
    evaluate_drift,
    psi_verdict,
)

CLASSES = np.array(["401000", "411000", "512000", "606300"])


def _proba(n_rows, n_classes=None, seed=42):
    """Row-stochastic probability matrix."""
    raw = np.random.default_rng(seed).random((n_rows, n_classes or len(CLASSES)))
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.fixture
def baseline(sample_df):
    """Baseline built from the 10-row sample with synthetic validation proba."""
    return build_baseline(sample_df, _proba(len(sample_df)), CLASSES)


class TestComputePsi:
    """compute_psi formula behaviour."""

    def test_identical_distributions_score_zero(self):
        proportions = [0.25, 0.25, 0.30, 0.20]
        assert compute_psi(proportions, proportions) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_scores_significant(self):
        assert compute_psi([0.7, 0.2, 0.1], [0.1, 0.2, 0.7]) > PSI_MODERATE_MAX

    def test_empty_bin_stays_finite(self):
        # Without the epsilon clip the log-ratio would diverge here.
        psi = compute_psi([1.0, 0.0], [0.5, 0.5])
        assert math.isfinite(psi)

    def test_psi_is_symmetric(self):
        a, b = [0.6, 0.4], [0.3, 0.7]
        assert compute_psi(a, b) == pytest.approx(compute_psi(b, a))


class TestPsiVerdict:
    """Threshold boundaries, pinned deliberately."""

    def test_below_stable_threshold(self):
        assert psi_verdict(0.0) == "stable"
        assert psi_verdict(PSI_STABLE_MAX - 0.001) == "stable"

    def test_stable_threshold_is_exclusive(self):
        assert psi_verdict(PSI_STABLE_MAX) == "moderate"

    def test_moderate_threshold_is_exclusive(self):
        assert psi_verdict(PSI_MODERATE_MAX - 0.001) == "moderate"
        assert psi_verdict(PSI_MODERATE_MAX) == "significant"


class TestBuildBaseline:
    """build_baseline output shape and content."""

    def test_structure(self, baseline, sample_df):
        assert baseline["schema_version"] == 1
        assert baseline["reference_size"] == len(sample_df)
        assert set(baseline["input_features"]) == {
            "amount",
            "desc_len",
            "is_debit",
            "has_reference",
            "amount_bucket",
            "weekday",
        }

    def test_categorical_bin_counts(self, baseline):
        features = baseline["input_features"]
        assert len(features["amount_bucket"]["ref_proportions"]) == 9
        assert len(features["weekday"]["ref_proportions"]) == 7
        assert len(features["is_debit"]["ref_proportions"]) == 2

    def test_continuous_edges_produce_ten_bins(self, baseline):
        amount = baseline["input_features"]["amount"]
        assert amount["kind"] == "continuous"
        assert len(amount["bin_edges"]) == 9  # interior edges
        assert len(amount["ref_proportions"]) == 10

    def test_proportions_sum_to_one(self, baseline):
        for spec in baseline["input_features"].values():
            assert sum(spec["ref_proportions"]) == pytest.approx(1.0)
        for spec in baseline["output"].values():
            assert sum(spec["ref_proportions"]) == pytest.approx(1.0)

    def test_output_reference_uses_model_classes(self, baseline):
        """Reference is the model's own predictions, not the label distribution."""
        assert baseline["output"]["predicted_class_distribution"]["categories"] == list(CLASSES)

    def test_reference_sizes_track_their_own_split(self, baseline, sample_df):
        """Input refs come from the train frame, output refs from val_proba."""
        assert baseline["reference_size"] == len(sample_df)
        assert baseline["output_reference_size"] == len(sample_df)

        val_only = build_baseline(sample_df, _proba(4), CLASSES)
        assert val_only["reference_size"] == len(sample_df)
        assert val_only["output_reference_size"] == 4

    def test_tiny_sample_with_duplicate_quantile_edges(self, sample_df):
        """10 rows over 10 deciles yields duplicate edges — must not raise."""
        tiny = sample_df.head(3)
        result = build_baseline(tiny, _proba(3), CLASSES)
        assert sum(result["input_features"]["amount"]["ref_proportions"]) == pytest.approx(1.0)


class TestEvaluateDrift:
    """evaluate_drift end-to-end against a baseline."""

    def test_returns_valid_structure(self, baseline, sample_df):
        proba = _proba(len(sample_df))

        report = evaluate_drift(sample_df, proba, CLASSES, baseline)

        assert set(report["input_drift"]) == set(baseline["input_features"])
        assert set(report["output_drift"]) == {
            "predicted_class_distribution",
            "confidence_distribution",
        }
        for entry in (*report["input_drift"].values(), *report["output_drift"].values()):
            assert math.isfinite(entry["psi"])
            assert entry["verdict"] in {"stable", "moderate", "significant"}
        assert report["overall_verdict"] in {"stable", "moderate", "significant"}

    def test_reference_data_shows_no_drift(self, baseline, sample_df):
        """Replaying the exact reference rows and proba must score ~0 everywhere."""
        proba = _proba(len(sample_df))  # same seed as the baseline fixture

        report = evaluate_drift(sample_df, proba, CLASSES, baseline)

        for name, entry in (*report["input_drift"].items(), *report["output_drift"].items()):
            assert entry["psi"] == pytest.approx(0.0, abs=1e-3), f"{name} drifted vs itself"
        assert report["overall_verdict"] == "stable"

    def test_shifted_batch_is_detected(self, baseline, sample_df):
        """Forcing every row to a credit must move the is_debit PSI."""
        shifted = sample_df.copy()
        shifted["debit"] = 0.0
        shifted["credit"] = 250.0
        proba = _proba(len(shifted))

        report = evaluate_drift(shifted, proba, CLASSES, baseline)

        assert report["input_drift"]["is_debit"]["psi"] > PSI_MODERATE_MAX
        assert report["overall_verdict"] == "significant"

    def test_undated_rows_are_excluded_not_imputed(self, baseline, sample_df):
        """Masking undated rows must score identically to dropping them."""
        partly = sample_df.copy()
        partly.loc[partly.index[::2], "posting_date"] = pd.NaT
        dated_only = partly[partly["posting_date"].notna()]

        masked = evaluate_drift(partly, _proba(len(partly)), CLASSES, baseline)
        filtered = evaluate_drift(dated_only, _proba(len(dated_only)), CLASSES, baseline)

        assert masked["input_drift"]["weekday"] == filtered["input_drift"]["weekday"]

    def test_fully_undated_batch_omits_weekday(self, baseline, sample_df):
        """No parseable date anywhere means nothing to compare, not maximal PSI."""
        undated = sample_df.copy()
        undated["posting_date"] = pd.NaT

        report = evaluate_drift(undated, _proba(len(undated)), CLASSES, baseline)

        assert "weekday" not in report["input_drift"]
        assert set(report["input_drift"]) < set(baseline["input_features"])
        # The remaining features are unaffected by the missing dates.
        assert report["input_drift"]["amount"]["psi"] == pytest.approx(0.0, abs=1e-3)

    def test_verdict_matches_reported_psi(self, baseline, sample_df):
        """Rounding happens before judging, so the two can never disagree."""
        proba = _proba(len(sample_df))

        report = evaluate_drift(sample_df, proba, CLASSES, baseline)

        for entry in report["input_drift"].values():
            assert entry["verdict"] == psi_verdict(entry["psi"])
