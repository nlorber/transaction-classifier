"""FastAPI application factory and lifespan management."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from ..core.artifacts.store import ModelStore
from ..core.config import Settings
from ..core.data.registry import ClientRegistry
from ..core.features.engine import DomainFeatureEngine
from ..core.utils.logging import setup_logging
from .middleware import LatencyMiddleware
from .predictor import Predictor, reload_predictor
from .routes import classify, explain, health, ops

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load client models on startup; watch for updates in the background."""
    settings: Settings = app.state.settings
    app.state.start_time = time.time()

    if settings.sandbox_mode:
        logger.info("Sandbox mode — model loading skipped")
        app.state.engines = {}
        app.state.loaders = {}
        yield
        return

    domain_engine = DomainFeatureEngine(settings.feature_profile)

    catalog = ClientRegistry(settings.client_registry_path)
    loaders: dict[str, ModelStore] = {}
    engines: dict[str, Predictor] = {}

    for entry in catalog.clients:
        vault_path = Path(settings.artifact_dir) / entry.client_id
        store = ModelStore(vault_path)
        loaders[entry.client_id] = store
        try:
            engine = reload_predictor(store, settings.default_top_k, domain_engine)
            engines[entry.client_id] = engine
            logger.info(
                "Loaded model for %s: %s (%d categories)",
                entry.client_id,
                engine.bundle.manifest.version,
                engine.bundle.manifest.num_categories,
            )
        except FileNotFoundError:
            logger.warning("No model for client %s — will return 503", entry.client_id)

    app.state.engines = engines
    app.state.loaders = loaders

    async def _poll_for_updates() -> None:
        interval = settings.reload_poll_secs
        while True:
            await asyncio.sleep(interval)
            for cid, store in list(loaders.items()):
                try:
                    if store.has_update():
                        logger.info("New version detected for %s, reloading …", cid)
                        engine = reload_predictor(store, settings.default_top_k, domain_engine)
                        app.state.engines[cid] = engine
                        logger.info("Reloaded %s → %s", cid, engine.bundle.manifest.version)
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning("Reload failed for %s: %s", cid, exc)
                except Exception:
                    logger.exception("Reload failed for %s", cid)

    watcher = asyncio.create_task(_poll_for_updates())
    yield
    watcher.cancel()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and wire the FastAPI application."""
    if settings is None:
        settings = Settings()

    setup_logging(level=settings.log_level, fmt=settings.log_format)

    app = FastAPI(
        title="Transaction Classifier API",
        description="Account-code prediction for French financial transactions",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(LatencyMiddleware)

    app.include_router(classify.router)
    app.include_router(health.router)
    app.include_router(ops.router)
    app.include_router(explain.router)

    return app


def get_app() -> FastAPI:
    """Lazy factory for uvicorn (avoids side effects at import time)."""
    return create_app()
