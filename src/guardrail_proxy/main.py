"""FastAPI application factory for the Guardrail Proxy."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from guardrail_proxy.api.logging_middleware import AccessLogMiddleware, configure_access_logger
from guardrail_proxy.api.routes import router

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_app() -> FastAPI:
    """
    Assemble and return the FastAPI application.

    Separating app creation from module-level instantiation lets tests import
    the factory without triggering lifespan hooks.
    """
    configure_access_logger()
    app = FastAPI(
        title="Guardrail Proxy",
        description=(
            "Intercepts and analyses LLM-bound prompts for prompt injection "
            "and PII leaks before forwarding to the downstream model."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(AccessLogMiddleware)
    app.include_router(router)

    if os.path.isdir(_STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_ui() -> FileResponse:
            return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    return app


# Module-level singleton used by uvicorn and tests.
app = create_app()
