"""Scaling benchmark: how each model's accuracy, training time and memory move with company size.

Generates one synthetic dataset per grid cell (scripts/generate_sample_data.py), then trains
logistic regression, XGBoost and LightGBM with balanced class weights on each through
scripts/compare_models.py, one model per subprocess. Every model gets room to converge: the
boosters up to MAX_ROUNDS rounds with early stopping on the validation block, logistic
regression up to MAX_ITER solver iterations, and each result records whether it converged.
A fit that runs past the time budget is killed and recorded as DNF. Results accumulate in reports/scaling_benchmark.json, and a rerun
skips fits already recorded there, so an interrupted run resumes where it stopped.

The grid varies one factor at a time around a shared centre: rows at 300 classes (volume)
and classes at 50,000 rows (class count). Along the class axis, rows per class fall as
classes grow.

Usage:
    uv run python scripts/scaling_benchmark.py
    uv run python scripts/scaling_benchmark.py --budget 10800 --cells centre,classes-1200
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any, NamedTuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT / "data" / "generated"
REPORT_PATH = ROOT / "reports" / "scaling_benchmark.json"
SEED = 42
RUNS = ("lr-balanced", "xgb-balanced", "lgbm-balanced")
# The model names compare_models.RUN_LABELS gives these runs, so DNF and error rows
# carry the same label as rows with results.
MODEL_NAMES = {
    "lr-balanced": "Logistic Regression",
    "xgb-balanced": "XGBoost",
    "lgbm-balanced": "LightGBM",
}
RUN_PACKAGES = ("scikit-learn", "xgboost", "lightgbm")
# Caps high enough that convergence, not the cap, ends training where the budget allows;
# the budget fits MAX_ROUNDS XGBoost rounds on the largest cells.
MAX_ROUNDS = 2000
MAX_ITER = 5000
DEFAULT_BUDGET_SECONDS = 10_800


class Cell(NamedTuple):
    name: str
    classes: int
    rows: int


CELLS = (
    Cell("default", 100, 10_000),
    Cell("rows-10k", 300, 10_000),
    Cell("centre", 300, 50_000),
    Cell("rows-200k", 300, 200_000),
    Cell("classes-100", 100, 50_000),
    Cell("classes-1200", 1_200, 50_000),
)


def ensure_dataset(cell: Cell) -> Path:
    path = GENERATED_DIR / f"sample_{cell.classes}classes_{cell.rows}rows_seed{SEED}.csv"
    if not path.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_sample_data.py"),
                "--classes",
                str(cell.classes),
                "--rows",
                str(cell.rows),
                "--seed",
                str(SEED),
                "--output",
                str(path),
            ],
            check=True,
        )
    return path


def run_fit(
    command: list[str], budget_seconds: float, output: Path, cwd: Path | None = None
) -> dict[str, Any]:
    """Run one fit as a subprocess and return its result row with status ok, dnf or error."""
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command, timeout=budget_seconds, capture_output=True, text=True, cwd=cwd
        )
    except subprocess.TimeoutExpired:
        return {"status": "dnf", "elapsed_seconds": round(time.monotonic() - start, 1)}
    elapsed = round(time.monotonic() - start, 1)
    if completed.returncode != 0:
        lines = completed.stderr.strip().splitlines() or ["no output"]
        return {"status": "error", "elapsed_seconds": elapsed, "error": lines[-1]}
    [row] = json.loads(output.read_text())
    return {"status": "ok", "elapsed_seconds": elapsed, **row}


def environment() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        cpu = brand or cpu
    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    versions = {package: metadata.version(package) for package in RUN_PACKAGES}
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "ram_gb": round(ram_bytes / 2**30, 1),
        "python": platform.python_version(),
        **versions,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the scaling benchmark grid.")
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET_SECONDS, help="seconds allowed per fit"
    )
    parser.add_argument("--cells", default=",".join(cell.name for cell in CELLS))
    args = parser.parse_args(argv)

    wanted = set(args.cells.split(","))
    report: dict[str, Any] = (
        json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {"fits": []}
    )
    report.update(
        {
            "seed": SEED,
            "budget_seconds": args.budget,
            "max_rounds": MAX_ROUNDS,
            "max_iter": MAX_ITER,
            "environment": environment(),
        }
    )
    done = {(fit["cell"], fit["run"]) for fit in report["fits"]}

    for cell in (cell for cell in CELLS if cell.name in wanted):
        data = ensure_dataset(cell)
        for run in RUNS:
            if (cell.name, run) in done:
                logger.info("Skipping %s / %s (already recorded)", cell.name, run)
                continue
            logger.info(
                "Fitting %s on %s (%d classes, %d rows) ...",
                run,
                cell.name,
                cell.classes,
                cell.rows,
            )
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "result.json"
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_models.py"),
                    "--data",
                    str(data),
                    "--runs",
                    run,
                    "--output",
                    str(output),
                    "--max-rounds",
                    str(MAX_ROUNDS),
                    "--max-iter",
                    str(MAX_ITER),
                ]
                fit = run_fit(command, args.budget, output, cwd=ROOT)
            logger.info("  %s after %.0fs", fit["status"], fit["elapsed_seconds"])
            report["fits"].append(
                {
                    "cell": cell.name,
                    "classes": cell.classes,
                    "rows": cell.rows,
                    "run": run,
                    "model": MODEL_NAMES[run],
                    **fit,
                }
            )
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
