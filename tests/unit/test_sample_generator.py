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


@pytest.mark.parametrize(
    ("kind", "cents", "expected"),
    [
        (None, 4_321, [4_321]),
        ("vat", 12_000, [10_000, 2_000]),
        ("vat", 1_001, [834, 167]),
        ("vat_fixed_asset", 60_000, [50_000, 10_000]),
        ("card", 5_000, [5_000, 50]),
        ("loan", 100_000, [80_000, 18_000, 2_000]),
        ("loan", 50_001, [40_001, 9_000, 1_000]),
    ],
)
def test_line_amounts(gen, kind, cents, expected):
    assert gen.line_amounts(kind, cents) == expected


def test_each_split_has_at_most_one_residual_line(gen):
    for specs in gen.SPLITS.values():
        assert sum(line.residual for line in specs) <= 1


def test_vat_purchases_are_catalogue_accounts(gen):
    assert set(gen.ACCOUNT_CODES) >= gen.VAT_DIRECT_PURCHASES


def test_split_lines_follow_their_template(gen, small):
    for transaction in small:
        kind = gen.split_kind(transaction.primary)
        assert [line.cents for line in transaction.lines] == gen.line_amounts(
            kind, transaction.cents
        )
        assert all(line.cents > 0 for line in transaction.lines)
        if kind in ("vat", "vat_fixed_asset", "loan"):
            assert sum(line.cents for line in transaction.lines) == transaction.cents


def test_capitalisation_threshold_applies_to_the_amount_before_vat(gen):
    [(*_, (_, expensed_max))] = gen.AMOUNT_DECIDED_TEMPLATES["606300"]
    [(*_, (capitalised_min, _))] = gen.AMOUNT_DECIDED_TEMPLATES["218000"]
    assert gen.line_amounts("vat", round(expensed_max * 100))[0] < 50_000
    assert gen.line_amounts("vat_fixed_asset", round(capitalised_min * 100))[0] >= 50_000


def test_reform_accounts_switch_on_the_reform_date(gen, small):
    sides = set()
    for transaction in small:
        if transaction.primary in gen.PRE_REFORM_ACCOUNTS:
            before = transaction.posting_date < gen.REFORM_DATE
            expected = (
                gen.PRE_REFORM_ACCOUNTS[transaction.primary] if before else transaction.primary
            )
            assert transaction.lines[0].account_code == expected
            sides.add(before)
    assert sides == {True, False}


def test_general_codes_include_both_sides_of_the_reform(gen):
    for new, old in gen.PRE_REFORM_ACCOUNTS.items():
        assert {new, old} <= gen.general_codes()


def test_reform_accounts_get_a_floor_for_each_side(gen):
    for code in gen.PRE_REFORM_ACCOUNTS:
        assert gen.transaction_floor(code) == 2 * gen.MIN_TRANSACTIONS
