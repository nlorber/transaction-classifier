"""Tests for HPO CLI commands (run and report)."""

from unittest.mock import MagicMock, patch

import optuna
from click.testing import CliRunner

from transaction_classifier.training.hpo.cli import hpo, load_and_prepare_data


class TestReportCommand:
    """Tests for the 'hpo report' subcommand."""

    def test_no_studies_found(self, tmp_path):
        """When no studies exist, prints a message and exits cleanly."""
        db_path = tmp_path / "empty.db"
        # Create an empty storage so get_all_study_summaries returns []
        optuna.create_study(
            study_name="__setup__",
            storage=f"sqlite:///{db_path}",
        )
        optuna.delete_study(
            study_name="__setup__",
            storage=f"sqlite:///{db_path}",
        )

        runner = CliRunner()
        result = runner.invoke(hpo, ["report", "--storage", str(db_path)])
        assert result.exit_code == 0
        assert "No studies found" in result.output

    def test_auto_selects_single_study(self, tmp_path):
        """When exactly one study exists and no --study-name given, auto-selects it."""
        db_path = tmp_path / "single.db"
        storage_url = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="my_study",
            storage=storage_url,
            direction="maximize",
        )
        study.add_trial(
            optuna.trial.create_trial(
                params={"lr": 0.01},
                distributions={"lr": optuna.distributions.FloatDistribution(0.001, 0.1)},
                values=[0.85],
                user_attrs={"training_time": 5.0, "balanced_accuracy": 0.80, "best_iteration": 50},
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

        runner = CliRunner()
        result = runner.invoke(hpo, ["report", "--storage", str(db_path), "--no-plots"])
        assert result.exit_code == 0
        assert "Auto-selected study: my_study" in result.output

    def test_lists_multiple_studies(self, tmp_path):
        """When multiple studies exist and no --study-name given, lists them."""
        db_path = tmp_path / "multi.db"
        storage_url = f"sqlite:///{db_path}"

        for name in ("study_a", "study_b"):
            s = optuna.create_study(study_name=name, storage=storage_url, direction="maximize")
            s.add_trial(
                optuna.trial.create_trial(
                    params={"lr": 0.01},
                    distributions={"lr": optuna.distributions.FloatDistribution(0.001, 0.1)},
                    values=[0.80],
                    state=optuna.trial.TrialState.COMPLETE,
                )
            )

        runner = CliRunner()
        result = runner.invoke(hpo, ["report", "--storage", str(db_path)])
        assert result.exit_code == 0
        assert "Available studies:" in result.output
        assert "study_a" in result.output
        assert "study_b" in result.output

    def test_report_with_explicit_study_name(self, tmp_path):
        """When --study-name is given, loads that study directly."""
        db_path = tmp_path / "explicit.db"
        storage_url = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="my_study", storage=storage_url, direction="maximize"
        )
        study.add_trial(
            optuna.trial.create_trial(
                params={"lr": 0.01},
                distributions={"lr": optuna.distributions.FloatDistribution(0.001, 0.1)},
                values=[0.85],
                user_attrs={"training_time": 5.0, "balanced_accuracy": 0.80, "best_iteration": 50},
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            hpo,
            ["report", "--storage", str(db_path), "--study-name", "my_study", "--no-plots"],
        )
        assert result.exit_code == 0


class TestRunCommand:
    """Tests for the 'hpo run' subcommand."""

    def _make_mock_study(self):
        mock_study = MagicMock()
        mock_study.trials = []
        mock_study.best_value = 0.85
        mock_best_trial = MagicMock()
        mock_best_trial.user_attrs = {"training_time": 5.0, "balanced_accuracy": 0.80}
        mock_best_trial.params = {"learning_rate": 0.01}
        mock_study.best_trial = mock_best_trial
        mock_study.best_params = {"learning_rate": 0.01}
        return mock_study

    @patch("transaction_classifier.training.hpo.cli.load_and_prepare_data")
    @patch("transaction_classifier.training.hpo.cli.Settings")
    def test_run_creates_study_and_optimizes(self, mock_settings_cls, mock_load_data, tmp_path):
        """Test that 'hpo run' creates a study, enqueues defaults, and calls optimize."""
        import numpy as np

        mock_settings_cls.return_value = MagicMock(device="cpu")
        mock_load_data.return_value = (
            np.random.rand(10, 5),
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            np.random.rand(3, 5),
            np.array([0, 1, 2]),
            3,
        )

        mock_study = self._make_mock_study()

        with (
            patch("transaction_classifier.training.hpo.cli.build_objective_fn") as mock_obj,
            patch("optuna.create_study", return_value=mock_study),
        ):
            mock_obj.return_value = lambda trial: 0.85

            runner = CliRunner()
            result = runner.invoke(
                hpo,
                [
                    "run",
                    "--n-trials",
                    "1",
                    "--timeout",
                    "10",
                    "--storage",
                    str(tmp_path / "test.db"),
                    "--study-name",
                    "test_run_study",
                ],
            )

        assert result.exit_code == 0
        mock_study.enqueue_trial.assert_called_once()
        mock_study.optimize.assert_called_once()

    @patch("transaction_classifier.training.hpo.cli.load_and_prepare_data")
    @patch("transaction_classifier.training.hpo.cli.Settings")
    def test_run_with_verbose_flag(self, mock_settings_cls, mock_load_data, tmp_path):
        """Test that --verbose flag changes logging level."""
        import numpy as np

        mock_settings_cls.return_value = MagicMock(device="cpu")
        mock_load_data.return_value = (
            np.random.rand(10, 5),
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            np.random.rand(3, 5),
            np.array([0, 1, 2]),
            3,
        )

        mock_study = self._make_mock_study()

        with (
            patch("transaction_classifier.training.hpo.cli.build_objective_fn") as mock_obj,
            patch("optuna.create_study", return_value=mock_study),
        ):
            mock_obj.return_value = lambda trial: 0.85

            runner = CliRunner()
            result = runner.invoke(
                hpo,
                [
                    "run",
                    "--n-trials",
                    "1",
                    "--timeout",
                    "10",
                    "--storage",
                    str(tmp_path / "verbose.db"),
                    "--study-name",
                    "verbose_study",
                    "-v",
                ],
            )

        assert result.exit_code == 0


class TestLoadAndPrepareData:
    """Tests for the data loading helper function."""

    def test_csv_mode(self, sample_csv_path):
        """Verify CSV mode works end-to-end."""
        from transaction_classifier.core.config import Settings

        settings = Settings(data_path=str(sample_csv_path), min_class_samples=1)
        X_train, y_train, X_val, y_val, n_classes = load_and_prepare_data(settings)
        assert X_train.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert n_classes > 0
