"""CLI entry point for model training."""

import logging
import sys
from pathlib import Path

import click

from ..core.artifacts.store import ModelStore
from ..core.config import Settings
from ..core.data.registry import ClientRegistry
from ..core.data.source import CsvDataSource, DataSource, PostgresDataSource
from ..core.utils.logging import setup_logging
from .pipeline import TrainingPipeline
from .validator import QualityGate


def _train_single_client(
    settings: Settings,
    provider: DataSource,
    vault_path: str,
    auto_promote: bool,
    log: logging.Logger,
) -> bool:
    """Train one client and optionally promote. Returns True on success."""
    local_cfg = settings.model_copy(update={"artifact_dir": vault_path})
    runner = TrainingPipeline(local_cfg, provider)
    try:
        manifest, baseline_accuracy, n_classes = runner.execute()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.error("Training failed: %s", exc)
        return False
    except Exception:
        log.exception("Training failed")
        return False

    if auto_promote:
        gate = QualityGate(min_lift=settings.min_lift)
        result = gate.check(manifest, baseline_accuracy=baseline_accuracy, n_classes=n_classes)
        if result.passed:
            store = ModelStore(vault_path)
            gate.approve_and_promote(store, manifest)
            log.info("Model %s promoted", manifest.version)
        else:
            log.error("Model %s failed quality gate — not promoted", manifest.version)
            return False

    log.info("Training complete: %s", manifest.version)
    return True


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
@click.option(
    "--client",
    "client_id",
    default=None,
    help="Train a single client by id (trains all if omitted)",
)
@click.option("--hpo", "run_hpo", is_flag=True, help="Run HPO instead of training")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def train(
    data_path: str | None,
    artifact_dir: str | None,
    min_samples: int | None,
    n_estimators: int | None,
    auto_promote: bool,
    client_id: str | None,
    run_hpo: bool,
    verbose: bool,
) -> None:
    """Train the account-code prediction model."""
    if run_hpo:
        from .hpo.cli import run as hpo_run

        ctx = click.Context(hpo_run)
        ctx.invoke(hpo_run, client_id=client_id, verbose=verbose)
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

    catalog = ClientRegistry(settings.client_registry_path)  # type: ignore[attr-defined]  # Task 4 refactor
    entries = catalog.clients

    if entries:
        if client_id is not None:
            entry = catalog.get(client_id)
            if entry is None:
                log.error("Unknown client: %s", client_id)
                sys.exit(1)
            entries = [entry]

        had_failure = False
        for entry in entries:
            log.info("Training client: %s", entry.client_id)
            if entry.query is None:
                log.error("Client %s has no query configured in clients.yaml", entry.client_id)
                had_failure = True
                continue
            provider = PostgresDataSource(
                entry.db_url,
                query=entry.query,
                row_limit=settings.pg_row_limit,
            )
            path = str(Path(settings.artifact_dir) / entry.client_id)
            if not _train_single_client(settings, provider, path, auto_promote, log):
                had_failure = True

        if had_failure:
            sys.exit(1)
        return

    # Fallback: local CSV development
    csv_provider = CsvDataSource(settings.data_path)
    if not _train_single_client(settings, csv_provider, settings.artifact_dir, auto_promote, log):
        sys.exit(1)


if __name__ == "__main__":
    train()
