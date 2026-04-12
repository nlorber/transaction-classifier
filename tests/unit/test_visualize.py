"""Tests for evaluation/visualize.py — generate_all_reports."""

import json

import numpy as np
import pytest

from transaction_classifier.evaluation.visualize import generate_all_reports


@pytest.fixture
def dummy_predictions():
    """Minimal classification data for 5 classes, 50 samples."""
    rng = np.random.default_rng(42)
    n_samples, n_classes = 50, 5
    y_true = rng.integers(0, n_classes, size=n_samples)
    proba = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y_pred = proba.argmax(axis=1)
    class_names = [f"C{i:03d}" for i in range(n_classes)]
    return y_true, y_pred, proba, class_names


def test_generate_all_reports_creates_expected_files(dummy_predictions, tmp_path):
    y_true, y_pred, proba, class_names = dummy_predictions

    generate_all_reports(y_true, y_pred, proba, class_names, tmp_path)

    expected_files = [
        "confusion_matrix.png",
        "topk_accuracy.png",
        "per_class_performance.png",
        "class_distribution.png",
        "confidence_calibration.png",
        "metrics.json",
    ]
    for fname in expected_files:
        assert (tmp_path / fname).exists(), f"Missing output: {fname}"


def test_generate_all_reports_returns_valid_metrics(dummy_predictions, tmp_path):
    y_true, y_pred, proba, class_names = dummy_predictions

    metrics = generate_all_reports(y_true, y_pred, proba, class_names, tmp_path)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "top_k_accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["n_samples"] == len(y_true)
    assert metrics["n_classes"] == len(class_names)


def test_generate_all_reports_metrics_json_matches_return(dummy_predictions, tmp_path):
    y_true, y_pred, proba, class_names = dummy_predictions

    metrics = generate_all_reports(y_true, y_pred, proba, class_names, tmp_path)

    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["accuracy"] == pytest.approx(metrics["accuracy"])
    assert saved["balanced_accuracy"] == pytest.approx(metrics["balanced_accuracy"])
