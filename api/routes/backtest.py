"""Backtest routes."""

from fastapi import APIRouter

from utils.logger import logger

router = APIRouter()


@router.post("/run")
async def run_backtest():
    """Run a backtest for a strategy."""
    logger.info("Starting backtest run")
    return {"id": "backtest_placeholder", "status": "running"}


@router.get("/{backtest_id}")
async def get_backtest(backtest_id: str):
    """Get backtest results."""
    logger.debug(f"Getting backtest: {backtest_id}")
    return {
        "id": backtest_id,
        "status": "completed",
        "metrics": {
            "return_pct": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        },
    }


@router.get("/")
async def list_backtests():
    """List all backtests."""
    logger.info("Listing backtests")
    return {"backtests": []}
