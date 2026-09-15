"""Fit bookkeeping for scripts/scaling_benchmark.py: results, time-outs and crashes."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location(
        "scaling_benchmark", ROOT / "scripts" / "scaling_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_that_writes_its_result_is_ok(bench, tmp_path):
    output = tmp_path / "result.json"
    code = (
        "import json, pathlib; "
        f"pathlib.Path({str(output)!r}).write_text(json.dumps([{{'top1_accuracy': 0.5}}]))"
    )
    fit = bench.run_fit([sys.executable, "-c", code], budget_seconds=60, output=output)
    assert fit["status"] == "ok"
    assert fit["top1_accuracy"] == 0.5


def test_fit_past_the_budget_is_dnf(bench, tmp_path):
    fit = bench.run_fit(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        budget_seconds=0.5,
        output=tmp_path / "unused.json",
    )
    assert fit["status"] == "dnf"
    assert fit["elapsed_seconds"] < 30


def test_crashing_fit_is_an_error_with_its_last_stderr_line(bench, tmp_path):
    fit = bench.run_fit(
        [sys.executable, "-c", "raise SystemExit('boom')"],
        budget_seconds=60,
        output=tmp_path / "unused.json",
    )
    assert fit["status"] == "error"
    assert fit["error"] == "boom"


def test_grid_matches_the_spec(bench):
    assert [(c.name, c.classes, c.rows) for c in bench.CELLS] == [
        ("default", 100, 10_000),
        ("rows-10k", 300, 10_000),
        ("centre", 300, 50_000),
        ("rows-200k", 300, 200_000),
        ("classes-100", 100, 50_000),
        ("classes-1200", 1_200, 50_000),
    ]
    assert bench.RUNS == ("lr-balanced", "xgb-balanced", "lgbm-balanced")
    assert (bench.MAX_ROUNDS, bench.MAX_ITER) == (2_000, 5_000)
    assert bench.DEFAULT_BUDGET_SECONDS == 10_800
