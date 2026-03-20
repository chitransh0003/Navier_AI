"""
main.py — FastAPI application entry point for the NAVIER AI microservice.

Usage
-----
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

    # Or directly:
    python -m app.main
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.engine.model import get_detector

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("navier")


# ── Lifespan — warm up model on startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load the ML model singleton so first request is not slow."""
    logger.info("🛢️  NAVIER AI Core starting up...")
    t0 = time.perf_counter()
    get_detector()
    logger.info("✅ Hybrid PINN-LSTM detector ready in %.2f s", time.perf_counter() - t0)
    yield
    logger.info("🛑 NAVIER AI Core shut down.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NAVIER AI Core",
        description=(
            "Hybrid Physics-Informed Neural Network (PINN) + LSTM microservice "
            "for petroleum pipeline leak detection, sensor confidence scoring, "
            "and Remaining Useful Life estimation.\n\n"
            "**Endpoints:**\n"
            "- `POST /analyze` — Full PINN-LSTM pipeline analysis\n"
            "- `POST /simulate_leak` — Synthetic leak scenario simulation\n"
            "- `GET /sensor_status` — Lightweight sensor health check\n"
            "- `GET /health` — Service health probe\n"
            "- `GET /model/info` — Model architecture metadata\n"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS (allow React frontend + Express.js backend) ─────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing header ─────────────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-t0)*1000:.2f}"
        return response

    # ── Global error handler ─────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Unexpected internal error.", "type": type(exc).__name__},
        )

    # ── Mount routes ─────────────────────────────────────────────────────────
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
