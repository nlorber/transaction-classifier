"""CLI entry points for hyperparameter optimisation."""

import logging
from datetime import date
from pathlib import Path
from typing import Any

import click
from sklearn.preprocessing import LabelEncoder

from ...core.config import Settings
from ...core.data.registry import ClientRegistry
from ...core.data.source import CsvDataSource, DataSource, PostgresDataSource
from ...core.data.splitter import split_by_date
from ...core.features.engine import DomainFeatureEngine
from ...core.features.pipeline import assemble_feature_matrix
from ...core.features.text import TfidfFeatureExtractor
from ...core.utils.logging import setup_logging
from .objective import build_objective_fn
from .search_space import BASELINE_PARAMS

logger = logging.getLogger(__name__)


def load_and_prepare_data(
    settings: Settings,
    client_id: str | None,
) -> tuple[Any, Any, Any, Any, int]:
    """Ingest data, split, encode, and build features for HPO."""
    provider: DataSource
    if client_id:
        catalog = ClientRegistry(settings.client_registry_path)
        entry = catalog.get(client_id)
        if entry is None:
            raise click.ClickException(f"Unknown client: {client_id}")
        if entry.query is None:
            raise click.ClickException(
                f"Client {entry.client_id} has no query configured in clients.yaml"
            )
        provider = PostgresDataSource(
            entry.db_url,
            query=entry.query,
            row_limit=settings.pg_row_limit,
        )
        logger.info("Fetching data for client %s …", entry.client_id)
    else:
        provider = CsvDataSource(settings.data_path)
        logger.info("Fetching data from CSV: %s …", settings.data_path)

    df = provider.fetch(
        min_class_samples=settings.min_class_samples,
        target_length=settings.target_length,
    )
    logger.info("Loaded %d rows, %d classes", len(df), df["target"].nunique())

    train_df, val_df = split_by_date(df, train_ratio=0.85)
    logger.info("Split: train=%d, val=%d", len(train_df), len(val_df))

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["target"])
    val_mask = val_df["target"].isin(le.classes_)
    val_df = val_df[val_mask]
    y_val = le.transform(val_df["target"])
    n_classes = len(le.classes_)
    logger.info("Classes: %d", n_classes)

    logger.info("Building features …")
    extractor = TfidfFeatureExtractor(
        label_vocab_size=settings.tfidf_max_label,
        detail_vocab_size=settings.tfidf_max_detail,
        char_vocab_size=settings.tfidf_max_char,
    )
    domain_engine = DomainFeatureEngine(settings.feature_profile)
    X_train = assemble_feature_matrix(train_df, extractor, domain_engine, fit=True)
    X_val = assemble_feature_matrix(val_df, extractor, domain_engine, fit=False)
    logger.info("Feature shape: %s", X_train.shape)

    return X_train, y_train, X_val, y_val, n_classes


@click.group()
def hpo() -> None:
    """Hyperparameter optimisation for the XGBoost classifier."""
    pass


@hpo.command()
@click.option(
    "--client",
    "client_id",
    default=None,
    help="Client ID (uses Postgres). Omit for CSV.",
)
@click.option("--n-trials", default=200, type=int, help="Max number of trials")
@click.option("--timeout", default=28800, type=int, help="Max seconds (default: 8h)")
@click.option("--storage", default="results/hpo.db", help="SQLite storage path")
@click.option("--study-name", default=None, help="Study name (auto-generated if omitted)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    client_id: str | None,
    n_trials: int,
    timeout: int,
    storage: str,
    study_name: str | None,
    verbose: bool,
) -> None:
    """Launch hyperparameter optimisation."""
    import optuna

    setup_logging(level="DEBUG" if verbose else "INFO", fmt="text")
    if not verbose:
        logging.getLogger("optuna").setLevel(logging.WARNING)

    settings = Settings()
    X_train, y_train, X_val, y_val, n_classes = load_and_prepare_data(settings, client_id)

    study_name = study_name or f"hpo_{client_id or 'csv'}_{date.today()}"
    storage_path = Path(storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=20,
            interval_steps=10,
        ),
    )

    study.enqueue_trial(BASELINE_PARAMS)
    objective = build_objective_fn(
        X_train,
        y_train,
        X_val,
        y_val,
        n_classes,
        device=settings.device,
    )

    logger.info(
        "Starting study=%s  n_trials=%d  timeout=%ds",
        study_name,
        n_trials,
        timeout,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info("Optimisation complete. Trials: %d", len(study.trials))
    logger.info("Best accuracy: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    best = study.best_trial
    if "training_time" in best.user_attrs:
        logger.info("Best training time: %.1fs", best.user_attrs["training_time"])
    if "balanced_accuracy" in best.user_attrs:
        logger.info("Best balanced accuracy: %.4f", best.user_attrs["balanced_accuracy"])


@hpo.command()
@click.option("--storage", default="results/hpo.db", help="SQLite storage path")
@click.option(
    "--study-name",
    default=None,
    help="Study name (lists available if omitted)",
)
@click.option("--top", default=10, type=int, help="Number of top trials to show")
@click.option("--output-dir", default="results", help="Directory to save plots")
@click.option("--no-plots", is_flag=True, help="Skip plot generation")
def report(
    storage: str, study_name: str | None, top: int, output_dir: str, no_plots: bool
) -> None:
    """Generate report from a completed HPO study."""
    import optuna

    from .report import print_best_config, print_top_trials, save_plots

    setup_logging(level="INFO", fmt="text")
    storage_url = f"sqlite:///{storage}"

    if study_name is None:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        if not summaries:
            click.echo(f"No studies found in {storage}")
            return
        if len(summaries) == 1:
            study_name = summaries[0].study_name
            click.echo(f"Auto-selected study: {study_name}")
        else:
            click.echo("Available studies:")
            for s in summaries:
                best_val = s.best_trial.value if s.best_trial else "N/A"
                click.echo(f"  - {s.study_name} ({s.n_trials} trials, best={best_val})")
            return

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print_top_trials(study, n=top)
    print_best_config(study)

    if not no_plots:
        save_plots(study, Path(output_dir))
