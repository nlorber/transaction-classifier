"""HPO study reporting — top trials, parameter importance, plots."""

import logging
from pathlib import Path
from typing import Any

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)

logger = logging.getLogger(__name__)


def print_top_trials(study: optuna.Study, n: int = 10) -> None:
    """Log the best *n* completed trials, sorted by objective value."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: t.value or 0.0, reverse=True)

    logger.info("=" * 80)
    logger.info("TOP %d TRIALS (of %d completed)", min(n, len(done)), len(done))
    logger.info("=" * 80)

    for rank, trial in enumerate(done[:n], 1):
        t_time = trial.user_attrs.get("training_time", -1)
        bal = trial.user_attrs.get("balanced_accuracy", -1)
        best_it = trial.user_attrs.get("best_iteration", -1)
        logger.info(
            "  #%d  Trial %d: acc=%.4f  bal_acc=%.4f  time=%.1fs  best_iter=%s",
            rank,
            trial.number,
            trial.value,
            bal,
            t_time,
            best_it,
        )

    state_counts: dict[str, int] = {}
    for t in study.trials:
        key = t.state.name
        state_counts[key] = state_counts.get(key, 0) + 1
    logger.info("Trial states: %s", state_counts)


def print_best_config(study: optuna.Study) -> None:
    """Log the winning hyperparameter set as a copy-pasteable dict."""
    winner = study.best_trial
    logger.info("=" * 80)
    logger.info("BEST CONFIG (copy-paste into Settings / XGBoostModel)")
    logger.info("=" * 80)
    lines = ["{"]
    for key, val in sorted(winner.params.items()):
        fmt = f"{val:.6f}" if isinstance(val, float) else repr(val)
        lines.append(f'    "{key}": {fmt},')
    lines.append("}")
    logger.info("\n".join(lines))

    if winner.user_attrs:
        logger.info("User attrs: %s", winner.user_attrs)


def save_plots(study: optuna.Study, output_dir: Path) -> None:
    """Render and save Optuna visualisation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[tuple[str, Any]] = [
        ("optimization_history", plot_optimization_history),
        ("param_importances", plot_param_importances),
        ("parallel_coordinate", plot_parallel_coordinate),
    ]
    for name, plot_fn in plots:
        try:
            fig = plot_fn(study)
            fig.write_image(str(output_dir / f"{name}.png"), scale=2)
            logger.info("Saved %s.png", name)
        except (ValueError, ImportError) as exc:
            logger.warning("Could not generate %s: %s", name, exc)
        except OSError as exc:
            logger.warning("Could not write %s: %s", name, exc)
        except Exception as exc:
            logger.warning("Could not save %s: %s", name, exc)
