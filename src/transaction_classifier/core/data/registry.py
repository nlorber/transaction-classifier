"""Multi-client configuration catalogue (loaded from clients.yaml)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClientConfig:
    """Immutable descriptor for a single client's data connection."""

    client_id: str
    db_url: str
    query: str | None = None


class ClientRegistry:
    """Reads a YAML registry and provides lookup by client identifier."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: list[ClientConfig] = self._parse()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self) -> list[ClientConfig]:
        if not self.path.exists():
            return []
        with open(self.path) as fh:
            blob: dict[str, Any] = yaml.safe_load(fh) or {}
        try:
            return [
                ClientConfig(
                    client_id=row["id"],
                    db_url=row["database_url"],
                    query=row.get("query"),
                )
                for row in blob.get("clients", [])
            ]
        except KeyError as exc:
            raise ValueError(f"Malformed entry in {self.path}: missing key {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def clients(self) -> list[ClientConfig]:
        """Return a shallow copy of all registered clients."""
        return list(self._entries)

    def get(self, client_id: str) -> ClientConfig | None:
        """Look up a single client by its identifier."""
        return next((e for e in self._entries if e.client_id == client_id), None)
