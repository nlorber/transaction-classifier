"""Consistency checks for scripts/generate_sample_data.py and the data it wrote."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location(
        "generate_sample_data", ROOT / "scripts" / "generate_sample_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(
        ROOT / "data" / "sample.csv",
        dtype={"account_code": str},
        keep_default_na=False,
        parse_dates=["posting_date"],
    )


def test_sampling_distributions_are_normalised(gen):
    """scripts/estimate_ceiling.py reads these distributions as exact probabilities."""
    for code in gen.ACCOUNT_CODES:
        assert sum(gen.date_distribution(code).values()) == pytest.approx(1.0), code
        assert sum(gen.entity_weights(code).values()) == pytest.approx(1.0), code
        for description, *_ in gen.TEMPLATES[code]:
            assert sum(p for _, p in gen.label_variants(description)) == pytest.approx(1.0)


def test_sample_follows_calendar_rules(gen, sample):
    # Label noise can record a row under its sibling, but siblings share a rule.
    for code, (months, first_day, last_day) in gen.CALENDAR_RULES.items():
        dates = sample.loc[sample["account_code"] == code, "posting_date"]
        in_window = dates.dt.day >= first_day
        if last_day is not None:
            in_window &= dates.dt.day <= last_day
        if months is not None:
            in_window &= dates.dt.month.isin(months)
        assert not dates.empty and in_window.all(), code


def test_sample_descriptions_fit_the_bank_label_width(gen, sample):
    assert sample["description"].str.len().max() <= gen.BANK_LABEL_LENGTH
