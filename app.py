"""
src/app.py — FastAPI application factory
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.routes.ai import router as ai_router

load_dotenv()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("✅ Kortex AI Service started")
    yield
    logger.info("👋 Kortex AI Service shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kortex AI Service",
        version="1.0.0",
        description="Pure AI processing — summarization, categorization, labels, memory.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ai_router)

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "service": "kortex-ai"}

    return app