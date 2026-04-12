"""Extended tests for training CLI — covering auto-promote, settings overrides, CSV fallback."""

from unittest.mock import MagicMock, patch

import yaml
from click.testing import CliRunner

from transaction_classifier.training.cli import train

_QUERY = "SELECT * FROM transactions LIMIT %(limit)s"


def _write_registry(path, clients):
    path.write_text(yaml.dump({"clients": clients}))
    return str(path)


class TestAutoPromote:
    """Tests for the --auto-promote flag."""

    def test_auto_promote_succeeds_when_validation_passes(self, tmp_path, monkeypatch):
        registry_path = _write_registry(
            tmp_path / "clients.yaml",
            [{"id": "c1", "database_url": "postgresql://localhost/db", "query": _QUERY}],
        )
        monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", registry_path)
        monkeypatch.setenv("TXCLS_ARTIFACT_DIR", str(tmp_path / "models"))

        def fake_execute(self):
            m = MagicMock()
            m.version = "v-test"
            m.metrics = {"accuracy": 0.90, "balanced_accuracy": 0.85}
            return m, 0.40, 10

        with (
            patch(
                "transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute
            ),
            patch("transaction_classifier.training.cli.QualityGate") as MockGate,
            patch("transaction_classifier.training.cli.ModelStore"),
        ):
            mock_gate = MagicMock()
            mock_gate.check.return_value = MagicMock(passed=True)
            MockGate.return_value = mock_gate

            result = CliRunner().invoke(train, ["--client", "c1", "--auto-promote"])

        assert result.exit_code == 0

    def test_auto_promote_exits_1_when_validation_fails(self, tmp_path, monkeypatch):
        registry_path = _write_registry(
            tmp_path / "clients.yaml",
            [{"id": "c1", "database_url": "postgresql://localhost/db", "query": _QUERY}],
        )
        monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", registry_path)
        monkeypatch.setenv("TXCLS_ARTIFACT_DIR", str(tmp_path / "models"))

        def fake_execute(self):
            m = MagicMock()
            m.version = "v-test"
            m.metrics = {"accuracy": 0.10, "balanced_accuracy": 0.05}
            return m, 0.40, 10

        with (
            patch(
                "transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute
            ),
            patch("transaction_classifier.training.cli.QualityGate") as MockGate,
        ):
            mock_gate = MagicMock()
            mock_gate.check.return_value = MagicMock(passed=False)
            MockGate.return_value = mock_gate

            result = CliRunner().invoke(train, ["--client", "c1", "--auto-promote"])

        assert result.exit_code == 1


class TestSettingsOverrides:
    """Tests for CLI flags that override Settings values."""

    def test_overrides_data_path_and_artifact_dir(self, tmp_path, monkeypatch):
        # No clients configured -> CSV fallback path
        monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", str(tmp_path / "empty.yaml"))
        (tmp_path / "empty.yaml").write_text(yaml.dump({"clients": []}))

        captured_settings = {}

        def fake_execute(self):
            captured_settings["data_path"] = self.settings.data_path
            captured_settings["artifact_dir"] = self.settings.artifact_dir
            m = MagicMock()
            m.version = "v-test"
            return m, 0.40, 10

        with patch(
            "transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute
        ):
            result = CliRunner().invoke(
                train,
                ["--data-path", "/custom/data.csv", "--artifact-dir", "/custom/models"],
            )

        assert result.exit_code == 0
        assert captured_settings["data_path"] == "/custom/data.csv"
        assert captured_settings["artifact_dir"] == "/custom/models"

    def test_overrides_min_samples_and_n_estimators(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", str(tmp_path / "empty.yaml"))
        (tmp_path / "empty.yaml").write_text(yaml.dump({"clients": []}))

        captured_settings = {}

        def fake_execute(self):
            captured_settings["min_samples"] = self.settings.min_class_samples
            captured_settings["n_estimators"] = self.settings.n_estimators
            m = MagicMock()
            m.version = "v-test"
            return m, 0.40, 10

        with patch(
            "transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute
        ):
            result = CliRunner().invoke(
                train,
                ["--min-samples", "42", "--n-estimators", "200"],
            )

        assert result.exit_code == 0
        assert captured_settings["min_samples"] == 42
        assert captured_settings["n_estimators"] == 200


class TestCSVFallback:
    """Tests for the no-clients CSV fallback path."""

    def test_csv_fallback_when_no_clients(self, tmp_path, monkeypatch):
        """When no clients in registry, falls back to CSV data source."""
        monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", str(tmp_path / "empty.yaml"))
        (tmp_path / "empty.yaml").write_text(yaml.dump({"clients": []}))

        def fake_execute(self):
            m = MagicMock()
            m.version = "v-test"
            return m, 0.40, 10

        with patch(
            "transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute
        ):
            result = CliRunner().invoke(train, [])

        assert result.exit_code == 0

    def test_csv_fallback_exits_1_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", str(tmp_path / "empty.yaml"))
        (tmp_path / "empty.yaml").write_text(yaml.dump({"clients": []}))

        def fake_execute(self):
            raise RuntimeError("CSV not found")

        with patch(
            "transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute
        ):
            result = CliRunner().invoke(train, [])

        assert result.exit_code == 1


class TestHPODelegation:
    """Tests for the --hpo flag that delegates to hpo run."""

    def test_hpo_flag_invokes_hpo_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", str(tmp_path / "empty.yaml"))
        (tmp_path / "empty.yaml").write_text(yaml.dump({"clients": []}))

        with patch("transaction_classifier.training.cli.click.Context") as MockCtx:
            mock_ctx = MagicMock()
            MockCtx.return_value = mock_ctx

            CliRunner().invoke(train, ["--hpo"])

        mock_ctx.invoke.assert_called_once()
