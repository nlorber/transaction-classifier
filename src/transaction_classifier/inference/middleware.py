"""HTTP middleware for request timing and tracing."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LatencyMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and log latency for prediction endpoints."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = uuid.uuid4().hex[:8]
        t0 = time.time()

        response: Response = await call_next(request)

        ms = (time.time() - t0) * 1000
        response.headers["X-Request-ID"] = rid

        if request.url.path.startswith("/classify"):
            logger.info(
                "rid=%s %s %s → %d (%.1fms)",
                rid,
                request.method,
                request.url.path,
                response.status_code,
                ms,
            )

        return response
