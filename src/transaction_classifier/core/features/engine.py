"""Config-driven domain feature engine.

Loads domain-specific feature rules from a YAML profile and applies them
generically. The profile defines entity patterns, amount signals, fiscal
indicators, text signals, and structured field extraction rules.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ..data.preprocessor import clean_html_for_extraction


class EntityConfig(BaseModel):
    patterns: list[str]


class DerivedFeatureRule(BaseModel):
    source_field: str
    rule: Literal["starts_with"]
    value: str


class VatSignalConfig(BaseModel):
    rates: list[float]
    tolerance: float = 0.005

    @field_validator("rates")
    @classmethod
    def _rates_positive(cls, v: list[float]) -> list[float]:
        for r in v:
            if r <= 0 or r >= 1:
                raise ValueError(f"VAT rate must be in (0, 1), got {r}")
        return v


class AmountSignalConfig(BaseModel):
    buckets: dict[str, list[float | None]]
    named_ranges: dict[str, list[float]] = Field(default_factory=dict)
    round_divisors: list[int] = Field(default_factory=list)
    log_clip_max: int = 7

    @field_validator("buckets")
    @classmethod
    def _buckets_valid(cls, v: dict[str, list[float | None]]) -> dict[str, list[float | None]]:
        prev_upper: float = 0
        for name, bounds in v.items():
            if len(bounds) != 2:
                raise ValueError(f"Bucket '{name}' must have exactly [lower, upper]")
            lo = bounds[0] if bounds[0] is not None else float("-inf")
            hi = bounds[1] if bounds[1] is not None else float("inf")
            if lo < prev_upper:
                raise ValueError(f"Bucket '{name}' overlaps with previous bucket")
            prev_upper = hi
        return v

    @field_validator("named_ranges")
    @classmethod
    def _ranges_valid(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        for name, bounds in v.items():
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"Range '{name}' must be [lower, upper] with lower < upper")
        return v


class FiscalIndicatorConfig(BaseModel):
    months: list[int] | None = None
    min_day: int | None = None
    max_day: int | None = None

    @model_validator(mode="after")
    def _day_range_valid(self) -> FiscalIndicatorConfig:
        if self.min_day is not None and self.max_day is not None and self.min_day > self.max_day:
            raise ValueError(f"min_day ({self.min_day}) > max_day ({self.max_day})")
        return self


class TransactionModeConfig(BaseModel):
    name: str
    triggers: list[str]


class StructuredFieldConfig(BaseModel):
    field_patterns: dict[str, str]
    transaction_modes: list[TransactionModeConfig] = Field(default_factory=list)
    derived_features: dict[str, DerivedFeatureRule] = Field(default_factory=dict)
    default: str = "other"


class FeatureProfile(BaseModel):
    """Validated domain feature configuration loaded from YAML."""

    sources: dict[str, str] = Field(default_factory=dict)
    entities: dict[str, EntityConfig] = Field(default_factory=dict)
    amount_signals: AmountSignalConfig | None = None
    fiscal_indicators: dict[str, FiscalIndicatorConfig] = Field(default_factory=dict)
    text_signals: dict[str, str] = Field(default_factory=dict)
    vat_signals: VatSignalConfig | None = None
    structured_fields: StructuredFieldConfig | None = None

    @field_validator("text_signals")
    @classmethod
    def _compile_text_patterns(cls, v: dict[str, str]) -> dict[str, str]:
        for name, pattern in v.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex for text signal '{name}': {e}") from e
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> FeatureProfile:
        """Load and validate a profile from a YAML file."""
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        return cls.model_validate(data)


class DomainFeatureEngine:
    """Config-driven feature engineering engine.

    Loads a ``FeatureProfile`` from YAML and applies all configured feature
    families (entity detection, text signals, amount signals, fiscal indicators,
    structured field extraction) to a DataFrame via :meth:`build`.
    """

    def __init__(self, profile_path: str | Path) -> None:
        self._profile = FeatureProfile.from_yaml(profile_path)

        # Pre-compile entity patterns: join alternatives with |, case-insensitive
        self._entity_patterns: dict[str, re.Pattern[str]] = {
            name: re.compile("|".join(cfg.patterns), re.IGNORECASE)
            for name, cfg in self._profile.entities.items()
        }

        # Pre-compile text signal patterns
        self._text_signal_patterns: dict[str, re.Pattern[str]] = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self._profile.text_signals.items()
        }

        # Pre-compile structured field patterns
        self._field_patterns: dict[str, re.Pattern[str]] = {}
        if self._profile.structured_fields is not None:
            self._field_patterns = {
                name: re.compile(pattern, re.IGNORECASE)
                for name, pattern in self._profile.structured_fields.field_patterns.items()
            }

    @property
    def feature_names(self) -> list[str]:
        """Deterministic ordered list of all output column names."""
        names: list[str] = []

        # 1. Entity columns for text_cols[0] (remarks)
        for entity_name in self._profile.entities:
            names.append(f"ent_{entity_name}")

        # 2. Text signal columns
        for signal_name in self._profile.text_signals:
            names.append(signal_name)

        # 3. Entity columns for text_cols[1] (description)
        for entity_name in self._profile.entities:
            names.append(f"desc_ent_{entity_name}")

        # 4. Amount signal columns
        if self._profile.amount_signals is not None:
            cfg = self._profile.amount_signals
            names.append("amt_log_scale")
            for divisor in cfg.round_divisors:
                names.append(f"divisible_by_{divisor}")
            for bucket_name in cfg.buckets:
                names.append(f"amt_{bucket_name}")
            for range_name in cfg.named_ranges:
                names.append(f"{range_name}_band")

        # 5. Fiscal indicator columns
        for indicator_name in self._profile.fiscal_indicators:
            names.append(f"fiscal_{indicator_name}")

        # 6. Structured field columns
        if self._profile.structured_fields is not None:
            sf = self._profile.structured_fields
            for field_name in sf.field_patterns:
                names.append(f"has_{field_name}")
            names.append("is_sepa")
            for mode in sf.transaction_modes:
                names.append(f"txn_type_{mode.name}")
            names.append(f"txn_type_{sf.default}")

            # 7. Derived feature columns
            for derived_name in sf.derived_features:
                names.append(derived_name)

        # 8. VAT signal columns
        if self._profile.vat_signals is not None:
            for rate in self._profile.vat_signals.rates:
                label = f"{rate:.3f}".replace(".", "")
                names.append(f"vat_compatible_{label}")

        return names

    def build(
        self,
        df: pd.DataFrame,
        text_cols: list[str],
        amount_col: str,
        date_col: str,
        comment_col: str,
    ) -> pd.DataFrame:
        """Build all feature families and return a single concatenated DataFrame.

        The result has the same index as ``df`` and columns in the order defined
        by :attr:`feature_names`.

        Parameters
        ----------
        df:
            Source DataFrame.
        text_cols:
            Two-element list: [remarks_col, description_col].  Entity detection
            runs on both; text signals run only on text_cols[0].
        amount_col:
            Numeric column for amount signals.
        date_col:
            Datetime column for fiscal indicators.
        comment_col:
            Text column for structured field extraction (typically the remarks).
        """
        parts: list[pd.DataFrame] = []

        # 1. Entity detection on text_cols[0] → prefix ent_
        parts.append(self._detect_entities(df[text_cols[0]], prefix="ent_"))

        # 2. Text signal detection on text_cols[0]
        parts.append(self._detect_text_signals(df[text_cols[0]]))

        # 3. Entity detection on text_cols[1] → prefix desc_ent_
        parts.append(self._detect_entities(df[text_cols[1]], prefix="desc_ent_"))

        # 4. Amount signals
        parts.append(self._compute_amount_signals(df[amount_col]))

        # 5. Fiscal indicators
        parts.append(self._derive_fiscal_indicators(df[date_col]))

        # 6. Structured field extraction (includes derived features)
        parts.append(self._extract_structured_fields(df[comment_col]))

        # 7. VAT signals
        parts.append(self._compute_vat_signals(df[amount_col]))

        result = pd.concat(
            [p.reset_index(drop=True) for p in parts],
            axis=1,
        )
        result.index = df.index
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_entities(self, text_series: pd.Series[Any], prefix: str) -> pd.DataFrame:
        """Binary entity detection: 1 if any pattern in the entity matches."""
        data: dict[str, Any] = {}
        lowered = text_series.fillna("").str.lower()
        for name, pattern in self._entity_patterns.items():
            data[f"{prefix}{name}"] = lowered.str.contains(pattern, regex=True).astype(int)
        return pd.DataFrame(data)

    def _detect_text_signals(self, text_series: pd.Series[Any]) -> pd.DataFrame:
        """Binary text signal detection: 1 if the signal pattern matches."""
        data: dict[str, Any] = {}
        lowered = text_series.fillna("").str.lower()
        for name, pattern in self._text_signal_patterns.items():
            data[name] = lowered.str.contains(pattern, regex=True).astype(int)
        return pd.DataFrame(data)

    def _compute_amount_signals(self, amount_series: pd.Series[Any]) -> pd.DataFrame:
        """Numeric amount features: log scale, divisibility, buckets, named ranges."""
        if self._profile.amount_signals is None:
            return pd.DataFrame()

        cfg = self._profile.amount_signals
        amt = amount_series.abs().fillna(0.0)
        data: dict[str, Any] = {}

        # Log scale
        data["amt_log_scale"] = np.floor(np.log10(amt + 1)).clip(0, cfg.log_clip_max)

        # Round divisors
        for divisor in cfg.round_divisors:
            data[f"divisible_by_{divisor}"] = ((amt % divisor) == 0).astype(int)

        # Buckets: half-open intervals [lower, upper)
        for bucket_name, bounds in cfg.buckets.items():
            lo = bounds[0] if bounds[0] is not None else float("-inf")
            hi = bounds[1] if bounds[1] is not None else float("inf")
            data[f"amt_{bucket_name}"] = ((amt >= lo) & (amt < hi)).astype(int)

        # Named ranges: inclusive [lower, upper]
        for range_name, range_bounds in cfg.named_ranges.items():
            rng_lo: float = range_bounds[0]
            rng_hi: float = range_bounds[1]
            data[f"{range_name}_band"] = ((amt >= rng_lo) & (amt <= rng_hi)).astype(int)

        return pd.DataFrame(data)

    def _compute_vat_signals(self, amount_series: pd.Series[Any]) -> pd.DataFrame:
        """Check if amount / (1 + rate) yields a round integer for each VAT rate."""
        if self._profile.vat_signals is None:
            return pd.DataFrame()

        cfg = self._profile.vat_signals
        amt = amount_series.abs().fillna(0.0)
        data: dict[str, Any] = {}

        for rate in cfg.rates:
            ht = amt / (1 + rate)
            remainder = (ht - np.round(ht)).abs()
            label = f"{rate:.3f}".replace(".", "")
            data[f"vat_compatible_{label}"] = ((remainder <= cfg.tolerance) & (amt > 0)).astype(
                int
            )

        return pd.DataFrame(data)

    def _derive_fiscal_indicators(self, date_series: pd.Series[Any]) -> pd.DataFrame:
        """Binary fiscal indicators derived from a datetime series."""
        data: dict[str, Any] = {}
        dates = (
            date_series
            if pd.api.types.is_datetime64_any_dtype(date_series)
            else pd.to_datetime(date_series, errors="coerce")
        )
        months = dates.dt.month
        days = dates.dt.day

        for name, indicator in self._profile.fiscal_indicators.items():
            mask = pd.Series([True] * len(date_series), index=date_series.index)

            if indicator.months is not None:
                mask &= months.isin(indicator.months)
            if indicator.min_day is not None:
                mask &= days >= indicator.min_day
            if indicator.max_day is not None:
                mask &= days <= indicator.max_day

            data[f"fiscal_{name}"] = mask.astype(int)

        return pd.DataFrame(data)

    def _extract_structured_fields(self, comment_series: pd.Series[Any]) -> pd.DataFrame:
        """Extract SEPA-style structured fields and classify transaction mode."""
        if self._profile.structured_fields is None:
            return pd.DataFrame()

        sf = self._profile.structured_fields
        n = len(comment_series)

        # Initialise output arrays
        has_fields: dict[str, list[int]] = {name: [0] * n for name in sf.field_patterns}
        is_sepa: list[int] = [0] * n

        # Transaction mode: one-hot; include default column
        all_mode_names = [m.name for m in sf.transaction_modes] + [sf.default]
        txn_modes: dict[str, list[int]] = {f"txn_type_{m}": [0] * n for m in all_mode_names}

        # Derived features
        derived: dict[str, list[int]] = {name: [0] * n for name in sf.derived_features}

        # Track raw extracted values per row for derived feature computation
        field_values: list[dict[str, str]] = [{} for _ in range(n)]

        for i, raw in enumerate(comment_series):
            cleaned = clean_html_for_extraction(str(raw) if not isinstance(raw, str) else raw)
            lowered = cleaned.lower()

            # Field pattern extraction
            any_field_matched = False
            for field_name, pattern in self._field_patterns.items():
                match = pattern.search(cleaned)
                if match:
                    has_fields[field_name][i] = 1
                    any_field_matched = True
                    field_values[i][field_name] = match.group(1)

            # is_sepa: any field matched OR "sepa" in text
            if any_field_matched or "sepa" in lowered:
                is_sepa[i] = 1

            # Transaction mode: first trigger match wins
            matched_mode: str | None = None
            for mode in sf.transaction_modes:
                for trigger in mode.triggers:
                    if trigger in lowered:
                        matched_mode = mode.name
                        break
                if matched_mode is not None:
                    break

            if matched_mode is None:
                matched_mode = sf.default

            txn_modes[f"txn_type_{matched_mode}"][i] = 1

            # Derived feature rules
            for derived_name, rule in sf.derived_features.items():
                extracted = field_values[i].get(rule.source_field, "")
                if rule.rule == "starts_with":
                    if extracted.upper().startswith(rule.value.upper()):
                        derived[derived_name][i] = 1
                else:
                    raise ValueError(f"Unknown derived feature rule: {rule.rule!r}")

        data: dict[str, Any] = {}
        for field_name in sf.field_patterns:
            data[f"has_{field_name}"] = has_fields[field_name]
        data["is_sepa"] = is_sepa
        data.update(txn_modes)
        for derived_name in sf.derived_features:
            data[derived_name] = derived[derived_name]

        return pd.DataFrame(data)
