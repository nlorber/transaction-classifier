"""Bayes-optimal accuracy ceiling for the synthetic sample data.

scripts/generate_sample_data.py allocates transactions to primary accounts, then for each
draws a template, a counterparty, a posting date and an amount in cents. It passes the
description through a bank-label channel (prefix variant, an occasional dropped character,
truncation) and writes one row per journal line: a split template's rows share every text
field and the date, each with its own account and amount. A line's account can depend on
the posting date (the 2025 chart-of-accounts reform), and the primary line is occasionally
recorded under a sibling account.

No classifier can beat the Bayes rule under that process: predict the recorded code with the
highest posterior given the row. This script evaluates that posterior exactly for every row of
the held-out test block of the committed default sample, so the result sits directly next to
the model's own test-block metrics.

Observable per row: description, remarks, the line's amount and side, and the posting date.
The reference column and the random tokens inside the remarks (bank references, cheque and
invoice numbers) are drawn independently of the account, so they carry no signal beyond their
format.

Usage:
    uv run python scripts/estimate_ceiling.py
"""

import importlib.util
import json
import logging
import math
import re
from collections import defaultdict
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GENERATOR_PATH = Path(__file__).resolve().parent / "generate_sample_data.py"
TOP_K = (1, 3, 5)

# Remarks placeholders filled from the RNG: value regex and the probability of one value.
RANDOM_TOKENS: dict[str, tuple[str, float]] = {
    "ref8": (r"FR\d{8}", 1 / 90_000_000),
    "ref6": (r"\d{6}", 1 / 900_000),
    "inv": (r"FAC\d{7}", 1 / 2_000),
}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")

# Observed label -> (counterparty or None, month or None) -> P(label | template, that fill).
DescriptionTable = dict[str, dict[tuple[str | None, int | None], float]]


class Observed(NamedTuple):
    description: str
    remarks: str
    cents: int
    is_debit: bool
    posting_date: date


class Emitter(NamedTuple):
    """One journal line of one template of one primary account."""

    primary: str
    account: str
    kind: str | None
    line_index: int
    is_debit: bool
    low_cents: int
    high_cents: int
    high: float
    descriptions: DescriptionTable
    remarks: re.Pattern[str] | None
    remarks_token_p: float
    counterparty_p: dict[str, float]
    date_p: dict[date, float]
    weight: float  # transactions of the primary account × P(template)


def load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("generate_sample_data", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(template: str, token_patterns: dict[str, str]) -> re.Pattern[str]:
    """Regex matching exactly the strings *template* can fill to.

    A placeholder repeated in one template is filled with a single value, so its
    later occurrences are back-references.
    """
    parts: list[str] = []
    seen: set[str] = set()
    end = 0
    for match in _PLACEHOLDER.finditer(template):
        name = match.group(1)
        parts.append(re.escape(template[end : match.start()]))
        parts.append(f"(?P={name})" if name in seen else f"(?P<{name}>{token_patterns[name]})")
        seen.add(name)
        end = match.end()
    parts.append(re.escape(template[end:]))
    return re.compile("".join(parts))


def _token_probability(template: str) -> float:
    names = set(_PLACEHOLDER.findall(template))
    return float(np.prod([RANDOM_TOKENS[name][1] for name in names if name in RANDOM_TOKENS]))


@cache
def _description_table(gen: Any, pattern: str, pool: tuple[str, ...]) -> DescriptionTable:
    """Every bank label a description pattern can produce, with its probability per fill.

    Descriptions only use the counterparty and month placeholders, so each
    rendering can be enumerated: every prefix variant and fill, kept whole or
    with one character dropped, then truncated to the bank's label width.
    """
    names = set(_PLACEHOLDER.findall(pattern))
    if not names <= {"entity", "month"}:
        raise ValueError(f"Cannot enumerate description pattern {pattern!r}")
    entities: tuple[str | None, ...] = pool if "entity" in names else (None,)
    months: tuple[int | None, ...] = tuple(range(1, 13)) if "month" in names else (None,)
    width = gen.BANK_LABEL_LENGTH

    table: defaultdict[str, defaultdict[tuple[str | None, int | None], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for rendered, variant_p in gen.label_variants(pattern):
        for entity in entities:
            for month in months:
                text = rendered
                if entity is not None:
                    text = text.replace("{entity}", entity)
                if month is not None:
                    text = text.replace("{month}", gen.MONTHS_FR[month - 1])
                fill = (entity, month)
                table[text[:width]][fill] += variant_p * (1 - gen.TYPO_RATE)
                dropped_p = variant_p * gen.TYPO_RATE / len(text)
                for position in range(len(text)):
                    table[(text[:position] + text[position + 1 :])[:width]][fill] += dropped_p
    return {label: dict(fills) for label, fills in table.items()}


def compile_emitters(gen: Any, n_classes: int, n_rows: int) -> list[Emitter]:
    counts = gen.transaction_counts(n_classes, n_rows)
    entities = {name for code in counts for name in gen.entity_pool(code)}
    token_patterns = {
        "entity": "|".join(re.escape(name) for name in sorted(entities, key=len, reverse=True)),
        "iban": r"[A-Z]{2}\d+",
        "bic": r"[A-Z0-9]{8}",
        "country": r"[A-Z]{2}",
        "date6": r"\d{6}",
        "month": "|".join(gen.MONTHS_FR),
        "quarter": "|".join(gen.QUARTERS_FR),
        "year": r"\d{4}",
        **{name: regex for name, (regex, _) in RANDOM_TOKENS.items()},
    }
    emitters: list[Emitter] = []
    for primary, count in counts.items():
        templates = gen.templates_for(primary)
        kind = gen.split_kind(primary)
        counterparty_p = gen.entity_weights(primary)
        for description, remarks, is_debit, (low, high) in templates:
            descriptions = _description_table(gen, description, tuple(counterparty_p))
            compiled = (
                None
                if remarks is None
                else _compile(gen.format_comment_html(remarks), token_patterns)
            )
            for index, line in enumerate(gen.split_specs(kind)):
                emitters.append(
                    Emitter(
                        primary=primary,
                        account=line.account or primary,
                        kind=kind,
                        line_index=index,
                        is_debit=is_debit if line.is_debit is None else line.is_debit,
                        low_cents=round(low * 100),
                        high_cents=round(high * 100),
                        high=float(high),
                        descriptions=descriptions,
                        remarks=compiled,
                        remarks_token_p=_token_probability(remarks or ""),
                        counterparty_p=counterparty_p,
                        date_p=gen.date_distribution(primary),
                        weight=count / len(templates),
                    )
                )
    return emitters


@cache
def _round_amount_pmf(
    low_cents: int, high_cents: int, high: float, magnitudes: tuple[int, ...]
) -> dict[int, float]:
    """P(drawn cents) under the round branch of generate_amount_cents, clipping included."""
    pmf: defaultdict[int, float] = defaultdict(float)
    for magnitude in magnitudes:
        k_max = max(1, int(high / magnitude))
        for k in range(1, k_max + 1):
            value = min(high_cents, max(low_cents, magnitude * k * 100))
            pmf[value] += 1 / k_max / len(magnitudes)
    return dict(pmf)


def drawn_probability(cents: int, emitter: Emitter, gen: Any) -> float:
    """P(drawn amount = cents | template): a clipped round multiple, else a uniform cent."""
    in_range = emitter.low_cents <= cents <= emitter.high_cents
    uniform = 1 / (emitter.high_cents - emitter.low_cents + 1) if in_range else 0.0
    rounded = _round_amount_pmf(
        emitter.low_cents, emitter.high_cents, emitter.high, tuple(gen.ROUND_MAGNITUDES)
    ).get(cents, 0.0)
    return (1 - gen.ROUND_AMOUNT_RATE) * uniform + gen.ROUND_AMOUNT_RATE * rounded


def line_probability(cents: int, emitter: Emitter, gen: Any) -> float:
    """P(this line's amount = cents | emitter), summed over the drawn amounts that book it."""
    specs = gen.split_specs(emitter.kind)
    if len(specs) == 1:
        return drawn_probability(cents, emitter, gen)
    share = specs[emitter.line_index].share
    # Every line lies within len(specs) / 2 cents of drawn × share, so the drawn amount
    # lies in this window.
    first = max(emitter.low_cents, math.floor((cents - len(specs)) / share))
    last = min(emitter.high_cents, math.ceil((cents + len(specs)) / share))
    return sum(
        drawn_probability(drawn, emitter, gen)
        for drawn in range(first, last + 1)
        if gen.line_amounts(emitter.kind, drawn)[emitter.line_index] == cents
    )


def _consistent(groups: dict[str, str], row: Observed, gen: Any) -> bool:
    """Tokens derived from the posting date or the counterparty must match them."""
    when = row.posting_date
    derived = {
        "date6": when.strftime("%d%m%y"),
        "month": gen.MONTHS_FR[when.month - 1],
        "quarter": gen.QUARTERS_FR[(when.month - 1) // 3],
        "year": str(when.year),
    }
    if "entity" in groups:
        bank = gen.counterparty_bank(groups["entity"])
        derived.update(zip(("iban", "bic", "country"), bank, strict=True))
    return all(groups[name] == value for name, value in derived.items() if name in groups)


def likelihood(row: Observed, emitter: Emitter, gen: Any) -> float:
    """P(row | emitter), omitting the reference column every emitter shares."""
    if row.is_debit != emitter.is_debit:
        return 0.0
    fills = emitter.descriptions.get(row.description)
    p = emitter.date_p.get(row.posting_date, 0.0)
    if p == 0.0 or not fills:
        return 0.0

    remarks_entity = None
    if not row.remarks:
        p *= 1.0 if emitter.remarks is None else 1 - gen.REMARKS_RATE
    else:
        remarks = emitter.remarks.fullmatch(row.remarks) if emitter.remarks else None
        if remarks is None or not _consistent(remarks.groupdict(), row, gen):
            return 0.0
        p *= gen.REMARKS_RATE * emitter.remarks_token_p
        remarks_entity = remarks.groupdict().get("entity")

    # One drawn counterparty fills both fields, and the description's month is the
    # posting month. An unobserved counterparty contributes no factor.
    fill_p = 0.0
    for (entity, month), description_p in fills.items():
        if month is not None and month != row.posting_date.month:
            continue
        if entity is not None and remarks_entity is not None and entity != remarks_entity:
            continue
        drawn = entity if entity is not None else remarks_entity
        drawn_p = 1.0 if drawn is None else emitter.counterparty_p.get(drawn, 0.0)
        fill_p += description_p * drawn_p
    if fill_p == 0.0:
        return 0.0
    return p * fill_p * line_probability(row.cents, emitter, gen)


def label_noise_matrix(gen: Any, codes: list[str]) -> np.ndarray:
    """M[true, recorded]: the chance a row of the true code is recorded under each code."""
    index = {code: i for i, code in enumerate(codes)}
    matrix = np.eye(len(codes))
    for code, sibling in gen.SIBLING_ACCOUNTS.items():
        matrix[index[code], index[code]] = 1 - gen.LABEL_NOISE_RATE
        matrix[index[code], index[sibling]] = gen.LABEL_NOISE_RATE
    return matrix


def topk_credit(scores: np.ndarray, true_idx: int, k: int) -> float:
    """Chance the true code lands in the top k when exact ties are broken at random."""
    true_score = scores[true_idx]
    higher = int((scores > true_score).sum())
    tied = int((scores == true_score).sum())
    return float(np.clip((k - higher) / tied, 0.0, 1.0))


def compute_ceiling(
    gen: Any,
    df: Any,
    n_classes: int,
    n_rows: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, object]:
    """Score the Bayes classifier on the held-out test block of a generated sample."""
    from transaction_classifier.core.data.splitter import split_by_date

    train_df, _, test_df = split_by_date(df, train_ratio=train_ratio, val_ratio=val_ratio)
    test_df = test_df[test_df["target"].isin(set(train_df["target"]))]

    codes = sorted(gen.general_codes() | set(gen.sub_account_codes(n_classes)))
    index = {code: i for i, code in enumerate(codes)}
    noise = label_noise_matrix(gen, codes)
    by_description: defaultdict[str, list[Emitter]] = defaultdict(list)
    for emitter in compile_emitters(gen, n_classes, n_rows):
        for label in emitter.descriptions:
            by_description[label].append(emitter)

    topk_hits: dict[int, list[float]] = {k: [] for k in TOP_K}
    expected_top1: list[float] = []
    unexplained = 0
    for record in test_df.itertuples(index=False):
        row = Observed(
            description=record.description,
            remarks=record.remarks,
            cents=round((record.debit or record.credit) * 100),
            is_debit=record.debit > 0,
            posting_date=record.posting_date.date(),
        )
        true_scores = np.zeros(len(codes))
        for emitter in by_description.get(row.description, []):
            p = likelihood(row, emitter, gen)
            if p:
                account = gen.booked_account(emitter.account, row.posting_date)
                true_scores[index[account]] += emitter.weight * p
        # Score recorded codes: a row of true code c is recorded as l with noise[c, l].
        scores = true_scores @ noise
        true_idx = index[record.target]
        if scores[true_idx] == 0.0:
            unexplained += 1
            continue
        for k in TOP_K:
            topk_hits[k].append(topk_credit(scores, true_idx, k))
        expected_top1.append(float(scores.max() / scores.sum()))

    # Every real row was generated by some emitter and recorded under this code, so a zero
    # score means this model of the generator has drifted from generate_sample_data.py.
    if unexplained:
        raise ValueError(f"{unexplained} rows have zero likelihood under their recorded code.")

    return {
        "evaluated_on": "held-out temporal test block",
        "rows": len(expected_top1),
        **{f"bayes_top{k}_accuracy": round(float(np.mean(topk_hits[k])), 4) for k in TOP_K},
        "expected_bayes_top1_accuracy": round(float(np.mean(expected_top1)), 4),
    }


def main() -> None:
    from transaction_classifier.core.config import Settings
    from transaction_classifier.core.data.loader import read_csv_data

    settings = Settings()
    if settings.target_length != 6:
        raise SystemExit("The ceiling is defined over the generator's 6-digit account codes.")

    gen = load_generator()
    df = read_csv_data(
        settings.data_path,
        target_length=settings.target_length,
        min_class_samples=settings.min_class_samples,
    )
    counts = gen.transaction_counts(gen.DEFAULT_CLASSES, gen.DEFAULT_ROWS)
    expected_rows = sum(n * gen.lines_per_transaction(code) for code, n in counts.items())
    if df["account_code"].nunique() != gen.DEFAULT_CLASSES or len(df) != expected_rows:
        raise SystemExit(
            "The ceiling is defined for the default sample: run "
            "scripts/generate_sample_data.py with no arguments first."
        )

    logger.info("Scoring the Bayes classifier on the test block of %s ...", settings.data_path)
    try:
        result = compute_ceiling(
            gen,
            df,
            gen.DEFAULT_CLASSES,
            gen.DEFAULT_ROWS,
            settings.train_ratio,
            settings.val_ratio,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info("%s", result)

    out_path = Path("reports/ceiling.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
