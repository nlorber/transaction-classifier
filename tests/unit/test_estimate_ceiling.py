"""The exact Bayes ceiling must explain every row of a freshly generated sample."""

import importlib.util
from pathlib import Path

import pytest

from transaction_classifier.core.data.loader import read_csv_data

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load("generate_sample_data")


@pytest.fixture(scope="module")
def ceiling():
    return _load("estimate_ceiling")


def test_ceiling_explains_every_test_row(gen, ceiling, tmp_path):
    path = tmp_path / "sample.csv"
    gen.write_csv(gen.generate(100, 4_000, seed=11), path)
    df = read_csv_data(path, target_length=6, min_class_samples=10)

    result = ceiling.compute_ceiling(gen, df, 100, 4_000, train_ratio=0.70, val_ratio=0.15)

    assert result["rows"] > 0
    top1, top3, top5 = (result[f"bayes_top{k}_accuracy"] for k in (1, 3, 5))
    assert 0 < top1 <= top3 <= top5 <= 1
