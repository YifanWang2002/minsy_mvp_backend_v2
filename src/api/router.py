"""Top-level API router registry."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.health import router as health_router
from src.api.routers.sessions import router as sessions_router
from src.api.routers.strategies import router as strategies_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(sessions_router)
api_router.include_router(strategies_router)
