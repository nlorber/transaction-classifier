"""CLI entry point for the prediction API server."""

import click
import uvicorn

from ..core.config import Settings


@click.command()
@click.option("--host", default=None, help="Bind address (overrides env)")
@click.option("--port", default=None, type=int, help="Bind port (overrides env)")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def serve(host: str | None, port: int | None, reload: bool, verbose: bool) -> None:
    """Start the prediction API server."""
    settings = Settings()
    uvicorn.run(
        "transaction_classifier.inference.app:get_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level="debug" if verbose else "info",
    )
