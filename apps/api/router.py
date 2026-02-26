"""Top-level API router registry."""

from __future__ import annotations

from fastapi import APIRouter

from apps.api.routes.auth import router as auth_router
from apps.api.routes.backtests import router as backtests_router
from apps.api.routes.broker_accounts import router as broker_accounts_router
from apps.api.routes.chat import router as chat_router
from apps.api.routes.deployments import router as deployments_router
from apps.api.routes.health import router as health_router
from apps.api.routes.market_data import router as market_data_router
from apps.api.routes.notification_preferences import router as notification_preferences_router
from apps.api.routes.portfolio import router as portfolio_router
from apps.api.routes.sessions import router as sessions_router
from apps.api.routes.social_connectors import router as social_connectors_router
from apps.api.routes.strategies import router as strategies_router
from apps.api.routes.trade_approvals import router as trade_approvals_router
from apps.api.routes.trading_preferences import router as trading_preferences_router
from apps.api.routes.trading_stream import router as trading_stream_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(sessions_router)
api_router.include_router(strategies_router)
api_router.include_router(backtests_router)
api_router.include_router(social_connectors_router)
api_router.include_router(notification_preferences_router)
api_router.include_router(trading_preferences_router)
api_router.include_router(trade_approvals_router)
api_router.include_router(broker_accounts_router)
api_router.include_router(deployments_router)
api_router.include_router(market_data_router)
api_router.include_router(portfolio_router)
api_router.include_router(trading_stream_router)
