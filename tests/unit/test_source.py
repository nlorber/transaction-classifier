"""Tests for PostgresDataSource."""

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from transaction_classifier.core.data.source import PostgresDataSource

_COLUMNS = [
    "account_code",
    "description",
    "reference",
    "remarks",
    "credit",
    "debit",
    "posting_date",
]

_ROWS = [
    ("401ABC", "Payment A", "REF1", "Com A", 100.0, 0.0, datetime.date(2024, 1, 1)),
    ("401ABC", "Payment B", "REF2", "Com B", 200.0, 0.0, datetime.date(2024, 1, 2)),
    ("627XYZ", "Invoice C", "REF3", "Com C", 0.0, 300.0, datetime.date(2024, 1, 3)),
    ("627XYZ", "Invoice D", "REF4", "Com D", 0.0, 400.0, datetime.date(2024, 1, 4)),
]

# psycopg2 cursor.description entries are 7-tuples; only [0] (name) is used
_DESCRIPTION = [(col, None, None, None, None, None, None) for col in _COLUMNS]

_QUERY = "SELECT * FROM transactions LIMIT %(limit)s"


def _make_mock_conn(rows=_ROWS, description=_DESCRIPTION):
    mock_cursor = MagicMock()
    mock_cursor.description = description
    mock_cursor.fetchall.return_value = rows
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


def test_postgres_source_returns_normalized_dataframe():
    mock_conn, _ = _make_mock_conn()
    with patch("psycopg2.connect", return_value=mock_conn):
        source = PostgresDataSource("postgresql://user:pass@host/db", query=_QUERY)
        result = source.fetch(min_class_samples=1, target_length=3)

    assert "target" in result.columns
    assert set(result["target"]) == {"401", "627"}
    assert pd.api.types.is_datetime64_any_dtype(result["posting_date"])
    assert result["debit"].dtype == float


def test_postgres_source_connects_with_given_url():
    mock_conn, _ = _make_mock_conn()
    with patch("psycopg2.connect", return_value=mock_conn) as mock_connect:
        PostgresDataSource("postgresql://user:pass@host/db", query=_QUERY).fetch(
            min_class_samples=1
        )
        mock_connect.assert_called_once_with("postgresql://user:pass@host/db")


def test_postgres_source_closes_connection_on_success():
    mock_conn, _ = _make_mock_conn()
    with patch("psycopg2.connect", return_value=mock_conn):
        PostgresDataSource("postgresql://user:pass@host/db", query=_QUERY).fetch(
            min_class_samples=1
        )
    mock_conn.close.assert_called_once()


def test_postgres_source_closes_connection_on_error():
    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = RuntimeError("DB error")
    with (
        patch("psycopg2.connect", return_value=mock_conn),
        pytest.raises(RuntimeError, match="DB error"),
    ):
        PostgresDataSource("postgresql://user:pass@host/db", query=_QUERY).fetch()
    mock_conn.close.assert_called_once()


def test_postgres_source_executes_query_with_limit_param():
    mock_conn, mock_cursor = _make_mock_conn()
    with patch("psycopg2.connect", return_value=mock_conn):
        PostgresDataSource("postgresql://user:pass@host/db", query=_QUERY).fetch(
            min_class_samples=1
        )

    call_args = mock_cursor.execute.call_args
    executed_sql = call_args[0][0]
    executed_params = call_args[0][1]
    assert "%(limit)s" in executed_sql
    assert "limit" in executed_params


def test_postgres_source_passes_row_limit_as_param():
    mock_conn, mock_cursor = _make_mock_conn()
    with patch("psycopg2.connect", return_value=mock_conn):
        PostgresDataSource("postgresql://user:pass@host/db", query=_QUERY, row_limit=1000).fetch(
            min_class_samples=1
        )

    executed_params = mock_cursor.execute.call_args[0][1]
    assert executed_params["limit"] == 1000


def test_postgres_source_raises_if_psycopg2_missing():
    with (
        patch.dict("sys.modules", {"psycopg2": None}),
        pytest.raises(ImportError, match="psycopg2-binary is required"),
    ):
        PostgresDataSource("postgresql://user:pass@host/db", query=_QUERY).fetch()
