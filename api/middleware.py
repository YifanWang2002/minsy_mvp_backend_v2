"""API Middleware."""

import time

from fastapi import FastAPI, Request

from utils.logger import logger


def setup_middleware(app: FastAPI) -> None:
    """Setup custom middleware for the application."""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log requests and add X-Process-Time header to responses."""
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # Log the request with our logger
        status_code = response.status_code
        method = request.method
        path = request.url.path

        # Color code by status
        if status_code >= 500:
            logger.error(f"{method} {path} → {status_code} ({process_time*1000:.1f}ms)")
        elif status_code >= 400:
            logger.warning(f"{method} {path} → {status_code} ({process_time*1000:.1f}ms)")
        else:
            logger.info(f"{method} {path} → {status_code} ({process_time*1000:.1f}ms)")

        return response
