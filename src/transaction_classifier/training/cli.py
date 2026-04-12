"""CLI entry point for model training."""

import logging
import sys

import click

from ..core.artifacts.store import ModelStore
from ..core.config import Settings
from ..core.data.source import CsvDataSource, DataSource, PostgresDataSource
from ..core.utils.logging import setup_logging
from .pipeline import TrainingPipeline
from .validator import QualityGate


@click.command()
@click.option("--data-path", default=None, help="Path to CSV data (overrides env)")
@click.option("--artifact-dir", default=None, help="Model store root (overrides env)")
@click.option("--min-samples", default=None, type=int, help="Min samples per class")
@click.option("--n-estimators", default=None, type=int, help="Boosting rounds")
@click.option(
    "--auto-promote/--no-auto-promote",
    default=False,
    help="Promote automatically if quality gate passes",
)
@click.option("--hpo", "run_hpo", is_flag=True, help="Run HPO instead of training")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def train(
    data_path: str | None,
    artifact_dir: str | None,
    min_samples: int | None,
    n_estimators: int | None,
    auto_promote: bool,
    run_hpo: bool,
    verbose: bool,
) -> None:
    """Train the account-code prediction model."""
    if run_hpo:
        from .hpo.cli import run as hpo_run

        ctx = click.Context(hpo_run)
        ctx.invoke(hpo_run, verbose=verbose)
        return

    settings = Settings()

    if data_path:
        settings.data_path = data_path
    if artifact_dir:
        settings.artifact_dir = artifact_dir
    if min_samples is not None:
        settings.min_class_samples = min_samples
    if n_estimators is not None:
        settings.n_estimators = n_estimators

    setup_logging(
        level="DEBUG" if verbose else settings.log_level,
        fmt=settings.log_format,
    )
    log = logging.getLogger("transaction_classifier.training")
    log.info("Starting training pipeline")

    provider: DataSource
    if settings.pg_dsn:
        provider = PostgresDataSource(
            settings.pg_dsn, query=settings.pg_query, row_limit=settings.pg_row_limit
        )
    else:
        provider = CsvDataSource(settings.data_path)

    runner = TrainingPipeline(settings, provider)
    try:
        manifest, baseline_accuracy, n_classes = runner.execute()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("Training failed: %s", exc)
        sys.exit(1)
    except Exception:
        log.exception("Training failed")
        sys.exit(1)

    if auto_promote:
        gate = QualityGate(min_lift=settings.min_lift)
        result = gate.check(manifest, baseline_accuracy=baseline_accuracy, n_classes=n_classes)
        if result.passed:
            store = ModelStore(settings.artifact_dir)
            gate.approve_and_promote(store, manifest)
            log.info("Model %s promoted", manifest.version)
        else:
            log.error("Model %s failed quality gate — not promoted", manifest.version)
            sys.exit(1)

    log.info("Training complete: %s", manifest.version)


if __name__ == "__main__":
    train()
