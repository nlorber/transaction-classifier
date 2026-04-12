"""Tests for HPO report functions: print_top_trials, print_best_config, save_plots."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna

from transaction_classifier.training.hpo.report import (
    print_best_config,
    print_top_trials,
    save_plots,
)


def _create_study_with_trials(n_complete=3, n_pruned=1):
    """Helper to create an in-memory study with completed and pruned trials."""
    study = optuna.create_study(direction="maximize")

    for i in range(n_complete):
        study.add_trial(
            optuna.trial.create_trial(
                params={"learning_rate": 0.01 * (i + 1), "max_depth": 5 + i},
                distributions={
                    "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.3),
                    "max_depth": optuna.distributions.IntDistribution(3, 10),
                },
                values=[0.80 + i * 0.05],
                user_attrs={
                    "training_time": 10.0 + i,
                    "balanced_accuracy": 0.75 + i * 0.03,
                    "best_iteration": 100 + i * 10,
                },
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

    for _ in range(n_pruned):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.PRUNED,
                values=[0.0],
                params={},
                distributions={},
            )
        )

    return study


class TestPrintTopTrials:
    def test_logs_top_trials(self, caplog):
        study = _create_study_with_trials(n_complete=5, n_pruned=1)
        with caplog.at_level(logging.INFO, logger="transaction_classifier.training.hpo.report"):
            print_top_trials(study, n=3)
        messages = " ".join(caplog.messages)
        assert "TOP 3 TRIALS" in messages
        assert "accuracy=" in messages or "acc=" in messages
        assert "Trial states:" in messages

    def test_logs_all_when_fewer_than_n(self, caplog):
        study = _create_study_with_trials(n_complete=2, n_pruned=0)
        with caplog.at_level(logging.INFO, logger="transaction_classifier.training.hpo.report"):
            print_top_trials(study, n=10)
        messages = " ".join(caplog.messages)
        assert "TOP 2 TRIALS (of 2 completed)" in messages

    def test_includes_trial_states_summary(self, caplog):
        study = _create_study_with_trials(n_complete=2, n_pruned=1)
        with caplog.at_level(logging.INFO, logger="transaction_classifier.training.hpo.report"):
            print_top_trials(study, n=10)
        messages = " ".join(caplog.messages)
        assert "COMPLETE" in messages
        assert "PRUNED" in messages


class TestPrintBestConfig:
    def test_logs_best_params_as_dict(self, caplog):
        study = _create_study_with_trials(n_complete=3)
        with caplog.at_level(logging.INFO, logger="transaction_classifier.training.hpo.report"):
            print_best_config(study)
        messages = "\n".join(caplog.messages)
        assert "BEST CONFIG" in messages
        assert "learning_rate" in messages

    def test_logs_user_attributes(self, caplog):
        study = _create_study_with_trials(n_complete=2)
        with caplog.at_level(logging.INFO, logger="transaction_classifier.training.hpo.report"):
            print_best_config(study)
        messages = "\n".join(caplog.messages)
        assert "User attrs" in messages

    def test_formats_float_with_six_decimals(self, caplog):
        study = optuna.create_study(direction="maximize")
        study.add_trial(
            optuna.trial.create_trial(
                params={"lr": 0.123456789},
                distributions={
                    "lr": optuna.distributions.FloatDistribution(0.0, 1.0),
                },
                values=[0.9],
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
        with caplog.at_level(logging.INFO, logger="transaction_classifier.training.hpo.report"):
            print_best_config(study)
        messages = "\n".join(caplog.messages)
        assert "0.123457" in messages

    def test_formats_int_params_without_decimals(self, caplog):
        study = optuna.create_study(direction="maximize")
        study.add_trial(
            optuna.trial.create_trial(
                params={"max_depth": 7},
                distributions={
                    "max_depth": optuna.distributions.IntDistribution(3, 10),
                },
                values=[0.9],
                user_attrs={},
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
        with caplog.at_level(logging.INFO, logger="transaction_classifier.training.hpo.report"):
            print_best_config(study)
        messages = "\n".join(caplog.messages)
        # Int values should appear without .000000
        assert '"max_depth": 7,' in messages


class TestSavePlots:
    def test_creates_output_dir(self, tmp_path):
        study = _create_study_with_trials(n_complete=3)
        output_dir = tmp_path / "plots" / "nested"
        with (
            patch(
                "transaction_classifier.training.hpo.report.plot_optimization_history"
            ) as mock_opt,
            patch("transaction_classifier.training.hpo.report.plot_param_importances") as mock_imp,
            patch(
                "transaction_classifier.training.hpo.report.plot_parallel_coordinate"
            ) as mock_coord,
        ):
            mock_fig = MagicMock()
            mock_opt.return_value = mock_fig
            mock_imp.return_value = mock_fig
            mock_coord.return_value = mock_fig

            save_plots(study, output_dir)

        assert output_dir.exists()
        mock_fig.write_image.assert_called()

    def test_handles_plot_failures_gracefully(self, tmp_path, caplog):
        study = _create_study_with_trials(n_complete=3)
        with (
            patch(
                "transaction_classifier.training.hpo.report.plot_optimization_history",
                side_effect=RuntimeError("plot error"),
            ),
            patch(
                "transaction_classifier.training.hpo.report.plot_param_importances",
                side_effect=RuntimeError("plot error"),
            ),
            patch(
                "transaction_classifier.training.hpo.report.plot_parallel_coordinate",
                side_effect=RuntimeError("plot error"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            save_plots(study, tmp_path)

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) == 3

    def test_saves_all_three_plots(self, tmp_path):
        study = _create_study_with_trials(n_complete=3)
        written_files = []

        def track_write(path, **kwargs):
            written_files.append(Path(path).name)

        with (
            patch(
                "transaction_classifier.training.hpo.report.plot_optimization_history"
            ) as mock_opt,
            patch("transaction_classifier.training.hpo.report.plot_param_importances") as mock_imp,
            patch(
                "transaction_classifier.training.hpo.report.plot_parallel_coordinate"
            ) as mock_coord,
        ):
            for mock_fn in (mock_opt, mock_imp, mock_coord):
                fig = MagicMock()
                fig.write_image = track_write
                mock_fn.return_value = fig

            save_plots(study, tmp_path)

        assert "optimization_history.png" in written_files
        assert "param_importances.png" in written_files
        assert "parallel_coordinate.png" in written_files
