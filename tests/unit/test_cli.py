"""Tests for multi-client training CLI behavior."""

from unittest.mock import patch

import yaml
from click.testing import CliRunner

from transaction_classifier.training.cli import train

_QUERY = "SELECT * FROM transactions LIMIT %(limit)s"


def _write_registry(path, clients):
    path.write_text(yaml.dump({"clients": clients}))
    return str(path)


def test_unknown_client_exits_1(tmp_path, monkeypatch):
    registry_path = _write_registry(
        tmp_path / "clients.yaml",
        [{"id": "c1", "database_url": "postgresql://localhost/db", "query": _QUERY}],
    )
    monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", registry_path)
    result = CliRunner().invoke(train, ["--client", "nonexistent"])
    assert result.exit_code == 1


def test_client_flag_trains_only_specified_client(tmp_path, monkeypatch):
    registry_path = _write_registry(
        tmp_path / "clients.yaml",
        [
            {"id": "c1", "database_url": "postgresql://localhost/db", "query": _QUERY},
            {"id": "c2", "database_url": "postgresql://localhost/db", "query": _QUERY},
        ],
    )
    monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", registry_path)
    monkeypatch.setenv("TXCLS_ARTIFACT_DIR", str(tmp_path / "models"))

    trained_clients = []

    def fake_train(settings, provider, vault_path, auto_promote, log):
        trained_clients.append(vault_path.split("/")[-1])
        return True

    with patch("transaction_classifier.training.cli._train_single_client", fake_train):
        CliRunner().invoke(train, ["--client", "c1"])

    assert "c1" in trained_clients
    assert "c2" not in trained_clients


def test_no_client_flag_trains_all_clients(tmp_path, monkeypatch):
    registry_path = _write_registry(
        tmp_path / "clients.yaml",
        [
            {"id": "c1", "database_url": "postgresql://localhost/db", "query": _QUERY},
            {"id": "c2", "database_url": "postgresql://localhost/db", "query": _QUERY},
        ],
    )
    monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", registry_path)
    monkeypatch.setenv("TXCLS_ARTIFACT_DIR", str(tmp_path / "models"))

    trained_clients = []

    def fake_train(settings, provider, vault_path, auto_promote, log):
        trained_clients.append(vault_path.split("/")[-1])
        return True

    with patch("transaction_classifier.training.cli._train_single_client", fake_train):
        CliRunner().invoke(train, [])

    assert set(trained_clients) == {"c1", "c2"}


def test_failed_client_continues_others_and_exits_1(tmp_path, monkeypatch):
    """If one client's training fails, remaining clients still train; exit code is 1."""
    registry_path = _write_registry(
        tmp_path / "clients.yaml",
        [
            {"id": "c1", "database_url": "postgresql://localhost/db", "query": _QUERY},
            {"id": "c2", "database_url": "postgresql://localhost/db", "query": _QUERY},
        ],
    )
    monkeypatch.setenv("TXCLS_CLIENT_REGISTRY_PATH", registry_path)
    monkeypatch.setenv("TXCLS_ARTIFACT_DIR", str(tmp_path / "models"))

    call_count = [0]
    trained_clients = []

    def fake_train(settings, provider, vault_path, auto_promote, log):
        call_count[0] += 1
        if call_count[0] == 1:
            return False  # c1 fails quality gate / training
        trained_clients.append(vault_path.split("/")[-1])
        return True

    with patch("transaction_classifier.training.cli._train_single_client", fake_train):
        result = CliRunner().invoke(train, [])

    assert result.exit_code == 1
    assert "c2" in trained_clients  # c2 trained despite c1 failing
