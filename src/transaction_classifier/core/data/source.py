"""Data-provider abstraction (CSV and PostgreSQL backends)."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from .loader import prepare_dataframe, read_csv_data


class DataSource(ABC):
    """Common interface for all data backends."""

    @abstractmethod
    def fetch(self, min_class_samples: int = 10, target_length: int = 6) -> pd.DataFrame:
        """Return a normalised DataFrame ready for feature engineering."""


class CsvDataSource(DataSource):
    """Reads transaction data from a local CSV file."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def fetch(self, min_class_samples: int = 10, target_length: int = 6) -> pd.DataFrame:
        return read_csv_data(
            self.filepath,
            target_length=target_length,
            min_class_samples=min_class_samples,
        )


class PostgresDataSource(DataSource):
    """Fetches transaction data from a PostgreSQL database.

    Callers supply the SQL query directly — this keeps the provider
    decoupled from any particular database schema.
    """

    def __init__(self, dsn: str, query: str, row_limit: int = 500_000):
        if "%(limit)s" not in query:
            raise ValueError("query must contain a %(limit)s placeholder so row_limit is enforced")
        self.dsn = dsn
        self.row_limit = row_limit
        self._query = query

    @contextmanager
    def _connect(self) -> Generator[Any, None, None]:
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2-binary is required for the Postgres provider. "
                "Install with: uv sync --extra train"
            ) from None

        conn = psycopg2.connect(self.dsn)
        try:
            yield conn
        finally:
            conn.close()

    def fetch(self, min_class_samples: int = 10, target_length: int = 6) -> pd.DataFrame:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(self._query, {"limit": self.row_limit})
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()

        raw_df = pd.DataFrame(rows, columns=columns)
        return prepare_dataframe(
            raw_df,
            target_length=target_length,
            min_class_samples=min_class_samples,
        )
