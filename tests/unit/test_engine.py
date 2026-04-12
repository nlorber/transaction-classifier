"""Tests for the config-driven domain feature engine."""

import re
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from transaction_classifier.core.features.engine import (
    DerivedFeatureRule,
    DomainFeatureEngine,
    FeatureProfile,
    VatSignalConfig,
)

PROFILE_PATH = Path("config/profiles/french_treasury.yaml")


class TestFeatureProfileLoading:
    """Profile loading and validation from YAML."""

    def test_load_french_treasury_profile(self) -> None:
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        assert len(profile.entities) == 13
        assert profile.amount_signals is not None
        assert len(profile.amount_signals.buckets) == 5
        assert len(profile.fiscal_indicators) == 10
        assert len(profile.text_signals) == 13
        assert profile.structured_fields is not None
        assert len(profile.structured_fields.field_patterns) == 12
        assert len(profile.structured_fields.transaction_modes) == 6

    def test_entity_patterns_are_valid_regex(self) -> None:
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        for _name, entity in profile.entities.items():
            for pattern in entity.patterns:
                re.compile(pattern)  # should not raise

    def test_amount_buckets_are_ordered(self) -> None:
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        assert profile.amount_signals is not None
        bounds = list(profile.amount_signals.buckets.values())
        for i in range(1, len(bounds)):
            prev_upper = bounds[i - 1][1] if bounds[i - 1][1] is not None else float("inf")
            curr_lower = bounds[i][0] if bounds[i][0] is not None else float("-inf")
            assert curr_lower >= prev_upper

    def test_invalid_regex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureProfile.model_validate({"text_signals": {"bad": "[unclosed"}})

    def test_invalid_day_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureProfile.model_validate(
                {"fiscal_indicators": {"bad": {"min_day": 25, "max_day": 5}}}
            )

    def test_invalid_named_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureProfile.model_validate(
                {
                    "amount_signals": {
                        "buckets": {},
                        "named_ranges": {"bad": [5000, 1000]},
                    }
                }
            )

    def test_empty_profile_valid(self) -> None:
        profile = FeatureProfile.model_validate({})
        assert len(profile.entities) == 0
        assert profile.amount_signals is None


@pytest.fixture()
def engine() -> DomainFeatureEngine:
    return DomainFeatureEngine(PROFILE_PATH)


@pytest.fixture()
def sample_engine_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": ["DGFIP IMPOTS", "Salaire Janvier", "Loyer Janvier"],
            "remarks": ["PRLV SEPA urssaf cotisations", "virement salaire", ""],
            "debit": [500.0, 0.0, 49.99],
            "credit": [0.0, 2500.0, 0.0],
            "posting_date": pd.to_datetime(["2026-01-15", "2026-03-31", "2026-07-10"]),
            "amount": [500.0, 2500.0, 49.99],
        }
    )


class TestDomainFeatureEngineEntities:
    """Entity detection applied to remarks (text_cols[0]) and description (text_cols[1])."""

    def test_urssaf_detected_in_remarks(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 0 remarks = "PRLV SEPA urssaf cotisations" → social_contributions matches
        assert result["ent_social_contributions"].iloc[0] == 1
        assert result["ent_social_contributions"].iloc[1] == 0

    def test_dgfip_detected_in_description(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 0 description = "DGFIP IMPOTS" → desc_ent_tax_authority matches
        assert result["desc_ent_tax_authority"].iloc[0] == 1
        assert result["desc_ent_tax_authority"].iloc[1] == 0

    def test_payroll_detected_in_description(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 1 description = "Salaire Janvier" → desc_ent_payroll matches
        assert result["desc_ent_payroll"].iloc[1] == 1
        assert result["desc_ent_payroll"].iloc[0] == 0

    def test_rental_detected_in_description(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 2 description = "Loyer Janvier" → desc_ent_rental matches
        assert result["desc_ent_rental"].iloc[2] == 1
        assert result["desc_ent_rental"].iloc[0] == 0

    def test_no_false_positives_for_empty_remarks(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 2 remarks = "" → no entity columns should fire
        entity_cols = [c for c in result.columns if c.startswith("ent_")]
        assert result.loc[2, entity_cols].sum() == 0


class TestDomainFeatureEngineAmounts:
    """Amount bucketing and named salary band detection."""

    def test_medium_bucket_for_500(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # 500.0 → small bucket [50, 500) — upper bound exclusive — actually boundary: 500 >= 50
        # and 500 < 500 is False, so it falls in medium [500, 5000)
        assert result["amt_medium"].iloc[0] == 1
        assert result["amt_small"].iloc[0] == 0

    def test_micro_bucket_for_49_99(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # 49.99 → micro bucket [0, 50)
        assert result["amt_micro"].iloc[2] == 1
        assert result["amt_small"].iloc[2] == 0

    def test_typical_salary_band_for_2500(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # 2500.0 → typical_salary range [2200, 5500]
        assert result["typical_salary_band"].iloc[1] == 1
        assert result["minimum_wage_band"].iloc[1] == 0

    def test_log_scale_for_500(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        import numpy as np

        # floor(log10(500 + 1)) = floor(2.699) = 2
        expected = float(np.floor(np.log10(501)))
        assert result["amt_log_scale"].iloc[0] == expected

    def test_divisible_by_100_for_500(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["divisible_by_100"].iloc[0] == 1
        assert result["divisible_by_100"].iloc[2] == 0


class TestDomainFeatureEngineFiscal:
    """Fiscal indicator derivation from posting_date."""

    def test_quarter_end_march_31(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # 2026-03-31 → quarter_end (months=[3,6,9,12]) → True
        assert result["fiscal_quarter_end"].iloc[1] == 1
        assert result["fiscal_quarter_end"].iloc[2] == 0

    def test_summer_july(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # 2026-07-10 → summer (months=[7,8]) → True
        assert result["fiscal_summer"].iloc[2] == 1
        assert result["fiscal_summer"].iloc[0] == 0

    def test_vat_window_for_jan_dates(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # vat_window: months=[1,4,7,10], min_day=17, max_day=24
        # 2026-01-15 → month=1 ✓, day=15 < 17 → False
        assert result["fiscal_vat_window"].iloc[0] == 0
        # 2026-03-31 → month=3 not in [1,4,7,10] → False
        assert result["fiscal_vat_window"].iloc[1] == 0

    def test_month_close_for_march_31(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # month_close: min_day=25 (no month constraint)
        # 2026-03-31 day=31 ≥ 25 → True
        assert result["fiscal_month_close"].iloc[1] == 1
        # 2026-07-10 day=10 < 25 → False
        assert result["fiscal_month_close"].iloc[2] == 0

    def test_year_start_january(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # year_start: months=[1]
        # 2026-01-15 → True; others False
        assert result["fiscal_year_start"].iloc[0] == 1
        assert result["fiscal_year_start"].iloc[1] == 0


class TestDomainFeatureEngineTextSignals:
    """Text signal pattern detection on remarks column."""

    def test_channel_direct_debit_detected(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 0 remarks = "PRLV SEPA urssaf cotisations"
        # channel_direct_debit pattern: \bprel[eè]vement\b|\bprlv\b
        # "PRLV" matches \bprlv\b (case-insensitive)
        assert result["channel_direct_debit"].iloc[0] == 1
        assert result["channel_direct_debit"].iloc[2] == 0

    def test_channel_transfer_detected(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 1 remarks = "virement salaire"
        # channel_transfer pattern: \bvir(?:ement)?\b
        assert result["channel_transfer"].iloc[1] == 1
        assert result["channel_transfer"].iloc[2] == 0

    def test_no_signals_for_empty_remarks(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        text_signal_cols = list(FeatureProfile.from_yaml(PROFILE_PATH).text_signals.keys())
        assert result.loc[2, text_signal_cols].sum() == 0


class TestDomainFeatureEngineStructuredFields:
    """SEPA field extraction and transaction mode classification."""

    def test_is_sepa_true_for_sepa_in_text(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 0 remarks = "PRLV SEPA urssaf cotisations" → "sepa" in lowered text
        assert result["is_sepa"].iloc[0] == 1

    def test_is_sepa_false_for_empty(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 2 remarks = "" → no field patterns, no sepa keyword
        assert result["is_sepa"].iloc[2] == 0

    def test_txn_type_direct_debit_for_prlv_sepa(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 0 remarks = "PRLV SEPA urssaf cotisations"
        # triggers ["prlv sepa", "prelevement"] → "prlv sepa" matches
        assert result["txn_type_direct_debit"].iloc[0] == 1
        assert result["txn_type_other"].iloc[0] == 0

    def test_txn_type_transfer_for_virement(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 1 remarks = "virement salaire"
        # triggers ["vir sepa", "virement"] → "virement" matches
        assert result["txn_type_transfer"].iloc[1] == 1
        assert result["txn_type_other"].iloc[1] == 0

    def test_txn_type_default_for_empty(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # Row 2 remarks = "" → no mode trigger matches → default "other"
        assert result["txn_type_other"].iloc[2] == 1

    def test_has_field_columns_present(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        # All 10 field patterns should produce has_* columns
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        assert profile.structured_fields is not None
        for field_name in profile.structured_fields.field_patterns:
            assert f"has_{field_name}" in result.columns

    def test_sepa_field_extraction_from_structured_comment(
        self, engine: DomainFeatureEngine
    ) -> None:
        df = pd.DataFrame(
            {
                "description": ["Payment"],
                "remarks": ["PRLV SEPA CPY :ACME01 RUM :FR-2026-001 NPY :John Doe"],
                "debit": [100.0],
                "credit": [0.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
                "amount": [100.0],
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["has_cpy"].iloc[0] == 1
        assert result["has_rum"].iloc[0] == 1
        assert result["has_npy"].iloc[0] == 1
        assert result["has_ibe"].iloc[0] == 0


class TestDomainFeatureEngineFeatureNames:
    """feature_names is deterministic and matches build() output columns."""

    def test_feature_names_matches_build_columns(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        result = engine.build(
            sample_engine_df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert engine.feature_names == list(result.columns)

    def test_feature_names_is_deterministic(self, engine: DomainFeatureEngine) -> None:
        names_a = engine.feature_names
        names_b = engine.feature_names
        assert names_a == names_b

    def test_feature_names_contains_all_families(self, engine: DomainFeatureEngine) -> None:
        names = engine.feature_names
        profile = FeatureProfile.from_yaml(PROFILE_PATH)

        # Entity columns for text_cols[0]
        for entity_name in profile.entities:
            assert f"ent_{entity_name}" in names

        # Text signal columns
        for signal_name in profile.text_signals:
            assert signal_name in names

        # Entity columns for text_cols[1]
        for entity_name in profile.entities:
            assert f"desc_ent_{entity_name}" in names

        # Amount signal columns
        assert "amt_log_scale" in names
        assert profile.amount_signals is not None
        for divisor in profile.amount_signals.round_divisors:
            assert f"divisible_by_{divisor}" in names
        for bucket_name in profile.amount_signals.buckets:
            assert f"amt_{bucket_name}" in names
        for range_name in profile.amount_signals.named_ranges:
            assert f"{range_name}_band" in names

        # Fiscal indicator columns
        for indicator_name in profile.fiscal_indicators:
            assert f"fiscal_{indicator_name}" in names

        # Structured field columns
        assert profile.structured_fields is not None
        for field_name in profile.structured_fields.field_patterns:
            assert f"has_{field_name}" in names
        assert "is_sepa" in names
        for mode in profile.structured_fields.transaction_modes:
            assert f"txn_type_{mode.name}" in names
        assert f"txn_type_{profile.structured_fields.default}" in names

    def test_feature_names_no_duplicates(self, engine: DomainFeatureEngine) -> None:
        names = engine.feature_names
        assert len(names) == len(set(names))

    def test_feature_names_contains_vat_signals(self, engine: DomainFeatureEngine) -> None:
        names = engine.feature_names
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        assert profile.vat_signals is not None
        for rate in profile.vat_signals.rates:
            label = f"{rate:.3f}".replace(".", "")
            assert f"vat_compatible_{label}" in names

    def test_feature_names_contains_derived_features(self, engine: DomainFeatureEngine) -> None:
        names = engine.feature_names
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        assert profile.structured_fields is not None
        for derived_name in profile.structured_fields.derived_features:
            assert derived_name in names

    def test_result_index_matches_input_index(
        self, engine: DomainFeatureEngine, sample_engine_df: pd.DataFrame
    ) -> None:
        # Slice to non-default index
        df = sample_engine_df.iloc[1:].copy()
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert list(result.index) == list(df.index)


class TestVatSignals:
    """VAT round-amount detection."""

    def test_vat_compatible_020_for_round_ht(self, engine: DomainFeatureEngine) -> None:
        """120.00 / 1.20 = 100.0 (exact integer) -> compatible."""
        df = pd.DataFrame(
            {
                "description": ["Test"],
                "remarks": [""],
                "amount": [120.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["vat_compatible_0200"].iloc[0] == 1

    def test_vat_compatible_010_for_round_ht(self, engine: DomainFeatureEngine) -> None:
        """110.00 / 1.10 = 100.0 (exact integer) -> compatible."""
        df = pd.DataFrame(
            {
                "description": ["Test"],
                "remarks": [""],
                "amount": [110.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["vat_compatible_0100"].iloc[0] == 1

    def test_vat_not_compatible_for_non_round_ht(self, engine: DomainFeatureEngine) -> None:
        """123.45 / 1.20 = 102.875 (not integer) -> not compatible at 20%."""
        df = pd.DataFrame(
            {
                "description": ["Test"],
                "remarks": [""],
                "amount": [123.45],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["vat_compatible_0200"].iloc[0] == 0

    def test_vat_zero_amount_not_compatible(self, engine: DomainFeatureEngine) -> None:
        """Zero-amount rows should NOT be flagged as VAT-compatible."""
        df = pd.DataFrame(
            {
                "description": ["Test"],
                "remarks": [""],
                "amount": [0.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        for col in result.columns:
            if col.startswith("vat_compatible_"):
                assert result[col].iloc[0] == 0, f"{col} should be 0 for zero amount"

    def test_vat_signal_column_names(self, engine: DomainFeatureEngine) -> None:
        expected = [
            "vat_compatible_0200",
            "vat_compatible_0100",
            "vat_compatible_0055",
            "vat_compatible_0021",
        ]
        names = engine.feature_names
        for col in expected:
            assert col in names


class TestVatSignalConfigValidation:
    """VatSignalConfig Pydantic validation."""

    def test_valid_rates(self) -> None:
        cfg = VatSignalConfig(rates=[0.20, 0.10])
        assert len(cfg.rates) == 2

    def test_rate_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VatSignalConfig(rates=[0.0])

    def test_rate_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VatSignalConfig(rates=[1.0])

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VatSignalConfig(rates=[-0.1])

    def test_default_tolerance(self) -> None:
        cfg = VatSignalConfig(rates=[0.20])
        assert cfg.tolerance == 0.005


class TestDerivedFeatureRule:
    """DerivedFeatureRule Pydantic model."""

    def test_valid_rule(self) -> None:
        rule = DerivedFeatureRule(source_field="ibe", rule="starts_with", value="FR")
        assert rule.source_field == "ibe"

    def test_invalid_rule_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DerivedFeatureRule(source_field="ibe", rule="contains", value="FR")


class TestDerivedFeatures:
    """Derived feature extraction (is_domestic_iban)."""

    def test_domestic_iban_detected(self, engine: DomainFeatureEngine) -> None:
        df = pd.DataFrame(
            {
                "description": ["Payment"],
                "remarks": ["PRLV SEPA IBE :FR7612345678901 NPY :Vendor"],
                "amount": [100.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["is_domestic_iban"].iloc[0] == 1

    def test_foreign_iban_not_domestic(self, engine: DomainFeatureEngine) -> None:
        df = pd.DataFrame(
            {
                "description": ["Payment"],
                "remarks": ["PRLV SEPA IBE :DE89370400440532013000 NPY :Vendor"],
                "amount": [100.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["is_domestic_iban"].iloc[0] == 0

    def test_no_iban_not_domestic(self, engine: DomainFeatureEngine) -> None:
        df = pd.DataFrame(
            {
                "description": ["Payment"],
                "remarks": ["PRLV SEPA NPY :Vendor"],
                "amount": [100.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["is_domestic_iban"].iloc[0] == 0


class TestPaymentProcessorEntity:
    """Entity detection for consolidated payment_processor entity."""

    def test_paypal_detected(self, engine: DomainFeatureEngine) -> None:
        df = pd.DataFrame(
            {
                "description": ["PAYPAL PAYMENT"],
                "remarks": ["paypal transaction ref123"],
                "amount": [50.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["ent_payment_processor"].iloc[0] == 1

    def test_stripe_detected(self, engine: DomainFeatureEngine) -> None:
        df = pd.DataFrame(
            {
                "description": ["Stripe payout"],
                "remarks": ["stripe settlement"],
                "amount": [200.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["ent_payment_processor"].iloc[0] == 1

    def test_sumup_detected(self, engine: DomainFeatureEngine) -> None:
        df = pd.DataFrame(
            {
                "description": ["SumUp collection"],
                "remarks": ["sumup daily batch"],
                "amount": [150.0],
                "posting_date": pd.to_datetime(["2026-01-15"]),
            }
        )
        result = engine.build(
            df,
            text_cols=["remarks", "description"],
            amount_col="amount",
            date_col="posting_date",
            comment_col="remarks",
        )
        assert result["ent_payment_processor"].iloc[0] == 1

    def test_no_paypal_entity_column(self, engine: DomainFeatureEngine) -> None:
        """The old paypal entity should no longer exist as a separate column."""
        names = engine.feature_names
        assert "ent_paypal" not in names
        assert "desc_ent_paypal" not in names


class TestSourcesCitation:
    """Profile sources block loading."""

    def test_sources_loaded(self) -> None:
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        assert len(profile.sources) > 0
        assert "fiscal_calendar" in profile.sources
        assert "tva_rates" in profile.sources

    def test_sources_are_urls(self) -> None:
        profile = FeatureProfile.from_yaml(PROFILE_PATH)
        for _key, url in profile.sources.items():
            assert url.startswith("https://")

    def test_empty_sources_valid(self) -> None:
        profile = FeatureProfile.model_validate({})
        assert profile.sources == {}
