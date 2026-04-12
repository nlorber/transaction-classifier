"""Tests for training CLI behavior."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from transaction_classifier.training.cli import train


def test_csv_fallback_when_no_pg_dsn(tmp_path, monkeypatch):
    """When pg_dsn is empty, CLI uses CsvDataSource."""
    monkeypatch.setenv("TXCLS_DATA_PATH", str(tmp_path / "data.csv"))

    captured = {}

    def fake_execute(self):
        captured["provider_type"] = type(self.provider).__name__
        m = MagicMock()
        m.version = "v-test"
        return m, 0.40, 10

    with patch("transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute):
        result = CliRunner().invoke(train, [])

    assert result.exit_code == 0
    assert captured["provider_type"] == "CsvDataSource"


def test_postgres_when_pg_dsn_set(tmp_path, monkeypatch):
    """When pg_dsn is set, CLI uses PostgresDataSource."""
    monkeypatch.setenv("TXCLS_PG_DSN", "postgresql://localhost/db")
    monkeypatch.setenv("TXCLS_PG_QUERY", "SELECT * FROM t LIMIT %(limit)s")

    captured = {}

    def fake_execute(self):
        captured["provider_type"] = type(self.provider).__name__
        m = MagicMock()
        m.version = "v-test"
        return m, 0.40, 10

    with patch("transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute):
        result = CliRunner().invoke(train, [])

    assert result.exit_code == 0
    assert captured["provider_type"] == "PostgresDataSource"


def test_training_failure_exits_1(tmp_path, monkeypatch):
    """Training failure causes exit code 1."""
    monkeypatch.setenv("TXCLS_DATA_PATH", str(tmp_path / "data.csv"))

    def fake_execute(self):
        raise RuntimeError("Training failed")

    with patch("transaction_classifier.training.pipeline.TrainingPipeline.execute", fake_execute):
        result = CliRunner().invoke(train, [])

    assert result.exit_code == 1
