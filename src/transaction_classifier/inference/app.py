"""FastAPI application factory and lifespan management."""

import logging
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..core.artifacts.store import ModelStore
from ..core.config import Settings
from ..core.features.engine import DomainFeatureEngine
from ..core.utils.logging import setup_logging
from .middleware import LatencyMiddleware
from .predictor import reload_predictor
from .routes import classify, explain, health, ops

logger = logging.getLogger(__name__)


class _ModelReloadHandler(FileSystemEventHandler):
    """Debounced filesystem handler that reloads the model when the symlink moves."""

    def __init__(
        self,
        app: FastAPI,
        store: ModelStore,
        domain_engine: DomainFeatureEngine,
        debounce_secs: float,
    ) -> None:
        self._app = app
        self._store = store
        self._domain_engine = domain_engine
        self._debounce_secs = debounce_secs
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:  # noqa: ARG002
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_secs, self._try_reload)
            self._timer.start()

    def _try_reload(self) -> None:
        try:
            if not self._store.has_update():
                return
            settings: Settings = self._app.state.settings
            predictor = reload_predictor(self._store, settings.default_top_k, self._domain_engine)
            self._app.state.predictor = predictor
            logger.info(
                "Hot-reloaded model: %s (%d categories)",
                predictor.bundle.manifest.version,
                predictor.bundle.manifest.num_categories,
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            logger.warning("Reload failed: %s", exc)

    def shutdown(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup and watch for symlink changes."""
    settings: Settings = app.state.settings
    app.state.start_time = time.time()

    if settings.sandbox_mode:
        logger.info("Sandbox mode — model loading skipped")
        app.state.predictor = None
        app.state.store = None
        yield
        return

    domain_engine = DomainFeatureEngine(settings.feature_profile)
    store = ModelStore(Path(settings.artifact_dir))
    app.state.store = store

    try:
        predictor = reload_predictor(store, settings.default_top_k, domain_engine)
        app.state.predictor = predictor
        logger.info(
            "Loaded model: %s (%d categories)",
            predictor.bundle.manifest.version,
            predictor.bundle.manifest.num_categories,
        )
    except FileNotFoundError:
        app.state.predictor = None
        logger.warning("No model found — will return 503 until a model is trained")

    handler = _ModelReloadHandler(app, store, domain_engine, settings.model_watch_debounce_secs)
    observer = Observer()
    observer.schedule(handler, str(store.root), recursive=False)
    observer.start()

    try:
        yield
    finally:
        handler.shutdown()
        observer.stop()
        observer.join(timeout=5)


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
