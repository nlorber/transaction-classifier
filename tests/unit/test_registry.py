"""Tests for ClientRegistry."""

from transaction_classifier.core.data.registry import ClientConfig, ClientRegistry

_QUERY = "SELECT * FROM transactions LIMIT %(limit)s"


def test_empty_clients_list(tmp_path):
    (tmp_path / "clients.yaml").write_text("clients: []\n")
    assert ClientRegistry(tmp_path / "clients.yaml").clients == []


def test_loads_two_clients(tmp_path):
    (tmp_path / "clients.yaml").write_text(
        "clients:\n"
        f"  - id: acme_corp\n"
        f"    database_url: postgresql://localhost/db\n"
        f"    query: '{_QUERY}'\n"
        f"  - id: globex\n"
        f"    database_url: postgresql://other/db\n"
        f"    query: '{_QUERY}'\n"
    )
    reg = ClientRegistry(tmp_path / "clients.yaml")
    assert len(reg.clients) == 2
    assert reg.clients[0] == ClientConfig(
        client_id="acme_corp",
        db_url="postgresql://localhost/db",
        query=_QUERY,
    )
    assert reg.clients[1].client_id == "globex"


def test_get_returns_client_by_id(tmp_path):
    (tmp_path / "clients.yaml").write_text(
        f"clients:\n  - id: c1\n    database_url: postgresql://localhost/db\n    query: '{_QUERY}'\n"
    )
    reg = ClientRegistry(tmp_path / "clients.yaml")
    assert reg.get("c1") == ClientConfig(
        client_id="c1", db_url="postgresql://localhost/db", query=_QUERY
    )
    assert reg.get("nonexistent") is None


def test_client_without_query_defaults_to_none(tmp_path):
    (tmp_path / "clients.yaml").write_text(
        "clients:\n  - id: c1\n    database_url: postgresql://localhost/db\n"
    )
    reg = ClientRegistry(tmp_path / "clients.yaml")
    assert reg.clients[0].query is None


def test_missing_file_returns_empty_list():
    reg = ClientRegistry("/nonexistent/path/clients.yaml")
    assert reg.clients == []


def test_malformed_entry_raises_value_error(tmp_path):
    (tmp_path / "clients.yaml").write_text(
        "clients:\n  - database_url: postgresql://localhost/db\n"  # missing 'id'
    )
    import pytest

    with pytest.raises(ValueError, match="missing key"):
        ClientRegistry(tmp_path / "clients.yaml")


def test_clients_property_returns_copy(tmp_path):
    (tmp_path / "clients.yaml").write_text(
        f"clients:\n  - id: c1\n    database_url: postgresql://localhost/db\n    query: '{_QUERY}'\n"
    )
    reg = ClientRegistry(tmp_path / "clients.yaml")
    reg.clients.clear()  # mutate the returned copy
    assert len(reg.clients) == 1  # internal state unchanged
