"""Numbers quoted in the docs must match the committed report files they come from.

Each report under reports/ is written by a script; the README, model card and
design doc copy its values into tables. These tests fail when a rerun updates a
report but not the prose, or when a doc is edited by hand.
"""

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _doc(relative_path: str) -> str:
    """Doc text with bold markers stripped, so emphasised table cells still match."""
    return (ROOT / relative_path).read_text(encoding="utf-8").replace("**", "")


def _report(name: str) -> Any:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def test_readme_key_results_match_reports():
    metrics = _report("metrics.json")
    ceiling = _report("ceiling.json")
    readme = _doc("README.md")

    expected = [
        f"| Top-{k} accuracy | {_pct(metrics['top_k_accuracy'][str(k)])} |" for k in (1, 3, 5, 10)
    ]
    expected += [
        f"| Balanced accuracy | {_pct(metrics['balanced_accuracy'])} |",
        f"| Classes | {metrics['n_classes']} |",
        f"| Evaluation samples | {metrics['n_samples']:,} |",
        f"| Bayes ceiling, top-1 / top-5 | {_pct(ceiling['bayes_top1_accuracy'])} / "
        f"{_pct(ceiling['bayes_top5_accuracy'])} |",
    ]
    missing = [row for row in expected if row not in readme]
    assert missing == []


def test_model_card_metrics_match_report():
    metrics = _report("metrics.json")
    card = _doc("docs/MODEL_CARD.md")

    expected = [
        f"| Top-{k} accuracy | {metrics['top_k_accuracy'][str(k)]:.3f} |" for k in (1, 3, 5, 10)
    ]
    expected += [
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.3f} |",
        f"n = {metrics['n_samples']:,} evaluation samples",
    ]
    missing = [row for row in expected if row not in card]
    assert missing == []


@pytest.mark.parametrize("doc", ["README.md", "docs/DESIGN.md"])
def test_feature_ablation_tables_match_report(doc):
    text = _doc(doc)
    expected = [
        f"| {row['feature_set']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} |"
        for row in _report("feature_ablation.json")
    ]
    missing = [row for row in expected if row not in text]
    assert missing == []


def test_class_weighting_table_matches_report():
    design = _doc("docs/DESIGN.md")
    labels = {"unweighted": "none", "balanced": "balanced (default)"}
    columns = (
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "balanced_accuracy",
        "rare_class_mean_recall",
    )
    expected = [
        "| " + " | ".join([labels[row["weighting"]], *(f"{row[c]:.4f}" for c in columns)]) + " |"
        for row in _report("class_weighting.json")
    ]
    missing = [row for row in expected if row not in design]
    assert missing == []


def test_model_comparison_table_matches_report():
    readme = _doc("README.md")
    columns = ("top1_accuracy", "top5_accuracy", "balanced_accuracy", "f1_weighted")
    expected = [
        "| "
        + " | ".join(
            [
                row["model"],
                row["class_weights"],
                *(f"{row[c]:.4f}" for c in columns),
                f"{row['train_seconds']}s",
            ]
        )
        + " |"
        for row in _report("model_comparison.json")
    ]
    missing = [row for row in expected if row not in readme]
    assert missing == []
