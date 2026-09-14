"""Bayes-optimal accuracy ceiling for the synthetic sample data.

scripts/generate_sample_data.py draws each row's true account code first, then a
template, a counterparty, a posting date and an amount. It passes the
description through a bank-label channel (prefix variant, an occasional dropped
character, truncation) and records the code, occasionally as a sibling account.
No classifier can beat the Bayes rule under that process: predict the recorded
code with the highest posterior given the row. This script evaluates that
posterior exactly for every row of the held-out test block (same rows, split
and class filter as training) and scores it there, so the result sits directly
next to the model's own test-block metrics.

Observable per row: description, remarks, amount, debit/credit side and posting
date. The reference column and the random tokens inside the remarks (bank
references, cheque and invoice numbers) are drawn independently of the account
code, so they carry no signal beyond their format.

Usage:
    uv run python scripts/estimate_ceiling.py
"""

import importlib.util
import json
import logging
import re
from collections import defaultdict
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
    amount: float
    is_debit: bool
    posting_date: Any


class CompiledTemplate(NamedTuple):
    is_debit: bool
    low: float
    high: float
    descriptions: DescriptionTable
    remarks: re.Pattern[str] | None
    remarks_token_p: float
    counterparty_p: dict[str, float]
    date_p: dict[Any, float]
    choice_p: float


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


def compile_templates(gen: Any) -> dict[str, list[CompiledTemplate]]:
    entities = {name for code in gen.ACCOUNT_CODES for name in gen.entity_pool(code)}
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
    compiled: dict[str, list[CompiledTemplate]] = {}
    for code in gen.ACCOUNT_CODES:
        variants = gen.TEMPLATES[code]
        counterparty_p = gen.entity_weights(code)
        compiled[code] = [
            CompiledTemplate(
                is_debit=is_debit,
                low=float(low),
                high=float(high),
                descriptions=_description_table(gen, description, tuple(counterparty_p)),
                remarks=(
                    None
                    if remarks is None
                    else _compile(gen.format_comment_html(remarks), token_patterns)
                ),
                remarks_token_p=_token_probability(remarks or ""),
                counterparty_p=counterparty_p,
                date_p=gen.date_distribution(code),
                choice_p=1 / len(variants),
            )
            for description, remarks, is_debit, (low, high) in variants
        ]
    return compiled


@cache
def _round_amount_pmf(low: float, high: float, magnitudes: tuple[int, ...]) -> dict[float, float]:
    """P(amount) under the round branch of generate_amount, clipping included."""
    pmf: dict[float, float] = {}
    for magnitude in magnitudes:
        k_max = max(1, int(high / magnitude))
        values = np.clip(magnitude * np.arange(1, k_max + 1), low, high)
        for value, count in zip(*np.unique(values, return_counts=True), strict=True):
            key = round(float(value), 2)
            pmf[key] = pmf.get(key, 0.0) + count / k_max / len(magnitudes)
    return pmf


def amount_probability(amount: float, low: float, high: float, gen: Any) -> float:
    """P(amount | template): a clipped round multiple, else a uniform draw rounded to cents."""
    cent_overlap = min(high, amount + 0.005) - max(low, amount - 0.005)
    uniform = max(cent_overlap, 0.0) / (high - low)
    rounded = _round_amount_pmf(low, high, tuple(gen.ROUND_MAGNITUDES)).get(round(amount, 2), 0.0)
    return (1 - gen.ROUND_AMOUNT_RATE) * uniform + gen.ROUND_AMOUNT_RATE * rounded


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


def likelihood(row: Observed, template: CompiledTemplate, gen: Any) -> float:
    """P(row | true code, template), omitting the reference column every code shares."""
    if row.is_debit != template.is_debit:
        return 0.0
    p = amount_probability(row.amount, template.low, template.high, gen)
    p *= template.date_p.get(row.posting_date.date(), 0.0)
    fills = template.descriptions.get(row.description)
    if p == 0.0 or not fills:
        return 0.0

    remarks_entity = None
    if not row.remarks:
        p *= 1.0 if template.remarks is None else 1 - gen.REMARKS_RATE
    else:
        remarks = template.remarks.fullmatch(row.remarks) if template.remarks else None
        if remarks is None or not _consistent(remarks.groupdict(), row, gen):
            return 0.0
        p *= gen.REMARKS_RATE * template.remarks_token_p
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
        drawn_p = 1.0 if drawn is None else template.counterparty_p.get(drawn, 0.0)
        fill_p += description_p * drawn_p
    return p * fill_p * template.choice_p


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


def main() -> None:
    from transaction_classifier.core.config import Settings
    from transaction_classifier.core.data.loader import read_csv_data
    from transaction_classifier.core.data.splitter import split_by_date

    settings = Settings()
    if settings.target_length != 6:
        raise SystemExit("The ceiling is defined over the generator's 6-digit account codes.")

    gen = load_generator()
    codes = list(gen.ACCOUNT_CODES)
    total = sum(gen.ACCOUNT_CODES.values())
    prior = np.array([gen.ACCOUNT_CODES[code] / total for code in codes])
    noise = label_noise_matrix(gen, codes)
    templates = compile_templates(gen)

    df = read_csv_data(
        settings.data_path,
        target_length=settings.target_length,
        min_class_samples=settings.min_class_samples,
    )
    train_df, _, test_df = split_by_date(
        df, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio
    )
    test_df = test_df[test_df["target"].isin(set(train_df["target"]))]
    logger.info("Scoring the Bayes classifier on %d test-block rows ...", len(test_df))

    topk_hits: dict[int, list[float]] = {k: [] for k in TOP_K}
    expected_top1: list[float] = []
    unexplained = 0
    for record in test_df.itertuples(index=False):
        row = Observed(
            description=record.description,
            remarks=record.remarks,
            amount=float(record.debit or record.credit),
            is_debit=record.debit > 0,
            posting_date=record.posting_date,
        )
        true_code_scores = prior * np.array(
            [sum(likelihood(row, template, gen) for template in templates[code]) for code in codes]
        )
        # Score recorded codes: a row of true code c is recorded as l with noise[c, l].
        scores = true_code_scores @ noise
        true_idx = codes.index(record.target)
        if scores[true_idx] == 0.0:
            unexplained += 1
            continue
        for k in TOP_K:
            topk_hits[k].append(topk_credit(scores, true_idx, k))
        expected_top1.append(float(scores.max() / scores.sum()))

    # Every real row was generated by some code and recorded under this one, so a zero
    # score means this model of the generator has drifted from generate_sample_data.py.
    if unexplained:
        raise SystemExit(f"{unexplained} rows have zero likelihood under their recorded code.")

    result: dict[str, object] = {
        "evaluated_on": "held-out temporal test block",
        "rows": len(expected_top1),
        **{f"bayes_top{k}_accuracy": round(float(np.mean(topk_hits[k])), 4) for k in TOP_K},
        "expected_bayes_top1_accuracy": round(float(np.mean(expected_top1)), 4),
    }
    logger.info("%s", result)

    out_path = Path("reports/ceiling.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
