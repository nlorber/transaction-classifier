"""Bayes-optimal accuracy ceiling for the synthetic sample data.

scripts/generate_sample_data.py draws each row's account code first, then a
template, a counterparty, a date and an amount. No classifier can beat the
Bayes rule under that process: predict the code with the highest posterior
P(code | row). This script evaluates that posterior exactly for every row of
the held-out test block (same rows, split and class filter as training) and
scores the Bayes classifier on them, so the result sits directly next to the
model's own test-block metrics.

Observable per row: description, remarks, amount, debit/credit side and
posting date. The date, the reference column and the random tokens inside the
text (bank references, cheque and invoice numbers) are drawn independently of
the account code, so they carry no signal beyond their format.

Usage:
    uv run python scripts/estimate_ceiling.py
"""

import importlib.util
import json
import logging
import re
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GENERATOR_PATH = Path(__file__).resolve().parent / "generate_sample_data.py"
TOP_K = (1, 3, 5)

# Placeholders filled from the RNG: value regex and the probability of one value.
RANDOM_TOKENS: dict[str, tuple[str, float]] = {
    "ref8": (r"FR\d{8}", 1 / 90_000_000),
    "ref6": (r"\d{6}", 1 / 900_000),
    "inv": (r"FAC\d{7}", 1 / 2_000),
}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


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
    description: re.Pattern[str]
    description_token_p: float
    remarks: re.Pattern[str] | None
    remarks_token_p: float
    pool: list[str]
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
        compiled[code] = [
            CompiledTemplate(
                is_debit=is_debit,
                low=float(low),
                high=float(high),
                description=_compile(description, token_patterns),
                description_token_p=_token_probability(description),
                remarks=(
                    None
                    if remarks is None
                    else _compile(gen.format_comment_html(remarks), token_patterns)
                ),
                remarks_token_p=_token_probability(remarks or ""),
                pool=gen.entity_pool(code),
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
    """P(row | code, template), omitting factors every code shares (date, reference)."""
    if row.is_debit != template.is_debit:
        return 0.0
    p = amount_probability(row.amount, template.low, template.high, gen)
    if p == 0.0:
        return 0.0

    description = template.description.fullmatch(row.description)
    if description is None or not _consistent(description.groupdict(), row, gen):
        return 0.0
    p *= template.description_token_p
    entities = {description.groupdict().get("entity")}

    if not row.remarks:
        p *= 1.0 if template.remarks is None else 1 - gen.REMARKS_RATE
    else:
        remarks = template.remarks.fullmatch(row.remarks) if template.remarks else None
        if remarks is None or not _consistent(remarks.groupdict(), row, gen):
            return 0.0
        p *= gen.REMARKS_RATE * template.remarks_token_p
        entities.add(remarks.groupdict().get("entity"))

    # Description and remarks are filled with the same drawn counterparty.
    entities.discard(None)
    if len(entities) > 1:
        return 0.0
    if entities:
        (entity,) = entities
        p *= template.pool.count(entity) / len(template.pool)
    return p * template.choice_p


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
        scores = prior * np.array(
            [sum(likelihood(row, template, gen) for template in templates[code]) for code in codes]
        )
        true_idx = codes.index(record.target)
        if scores[true_idx] == 0.0:
            unexplained += 1
            continue
        for k in TOP_K:
            topk_hits[k].append(topk_credit(scores, true_idx, k))
        expected_top1.append(float(scores.max() / scores.sum()))

    # Every real row was generated by its true code, so a zero likelihood means this
    # model of the generator has drifted from scripts/generate_sample_data.py.
    if unexplained:
        raise SystemExit(f"{unexplained} rows have zero likelihood under their true code.")

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
