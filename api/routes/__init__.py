"""API Routes module."""

from fastapi import APIRouter

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.strategy import router as strategy_router
from api.routes.backtest import router as backtest_router
from api.routes.portfolio import router as portfolio_router

router = APIRouter(prefix="/api/v1")

# Include all routers
router.include_router(health_router, tags=["health"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(strategy_router, prefix="/strategies", tags=["strategies"])
router.include_router(backtest_router, prefix="/backtest", tags=["backtest"])
router.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])


@router.get("/")
async def api_root():
    """API v1 root endpoint."""
    return {
        "message": "Welcome to Minsy API v1",
        "endpoints": {
            "health": "/api/v1/health",
            "strategies": "/api/v1/strategies",
            "backtest": "/api/v1/backtest",
            "portfolio": "/api/v1/portfolio",
            "chat": "/api/v1/chat",
        },
    }
