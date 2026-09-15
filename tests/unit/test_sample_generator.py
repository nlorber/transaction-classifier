"""Consistency checks for scripts/generate_sample_data.py and the data it wrote."""

import importlib.util
from collections import Counter
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


@pytest.fixture(scope="module")
def small(gen):
    return gen.generate(n_classes=120, n_rows=6_000, seed=7)


def test_sampling_distributions_are_normalised(gen):
    """scripts/estimate_ceiling.py reads these distributions as exact probabilities."""
    codes = [*gen.ACCOUNT_CODES, *gen.sub_account_codes(gen.DEFAULT_CLASSES)]
    for code in codes:
        assert sum(gen.date_distribution(code).values()) == pytest.approx(1.0), code
        assert sum(gen.entity_weights(code).values()) == pytest.approx(1.0), code
        for description, *_ in gen.templates_for(code):
            assert sum(p for _, p in gen.label_variants(description)) == pytest.approx(1.0)


def test_every_catalogue_account_has_templates(gen):
    assert set(gen.ACCOUNT_CODES) <= set(gen.TEMPLATES)


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


def test_class_count_is_exact_and_every_class_survives_filtering(small):
    rows = Counter(line.account_code for transaction in small for line in transaction.lines)
    assert len(rows) == 120
    assert min(rows.values()) >= 10


@pytest.mark.parametrize(
    ("n_classes", "n_rows"),
    [(100, 10_000), (300, 10_000), (300, 50_000), (300, 200_000), (100, 50_000), (1_200, 50_000)],
)
def test_row_count_is_within_one_percent_of_the_target(gen, n_classes, n_rows):
    counts = gen.transaction_counts(n_classes, n_rows)
    lines = sum(n * gen.lines_per_transaction(code) for code, n in counts.items())
    assert abs(lines - n_rows) <= 0.01 * n_rows


def test_too_few_classes_is_rejected(gen):
    with pytest.raises(ValueError, match="--classes"):
        gen.transaction_counts(len(gen.general_codes()) + 1, 10_000)


def test_too_few_rows_is_rejected(gen):
    with pytest.raises(ValueError, match="--rows"):
        gen.transaction_counts(1_200, 10_000)


def test_generation_is_deterministic(gen):
    assert gen.generate(100, 4_000, seed=3) == gen.generate(100, 4_000, seed=3)


def test_each_sub_account_books_one_distinct_counterparty(gen):
    codes = gen.sub_account_codes(1_200)
    pools = [gen.entity_pool(code) for code in codes]
    assert all(len(pool) == 1 for pool in pools)
    assert len({pool[0] for pool in pools}) == len(codes)
