"""Strategy routes."""

from fastapi import APIRouter

from utils.logger import logger

router = APIRouter()


@router.get("/")
async def list_strategies():
    """List all strategies."""
    logger.info("Listing strategies")
    return {"strategies": []}


@router.post("/")
async def create_strategy():
    """Create a new strategy."""
    logger.info("Creating new strategy")
    return {"id": "strategy_placeholder", "name": "New Strategy"}


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get a specific strategy."""
    logger.debug(f"Getting strategy: {strategy_id}")
    return {"id": strategy_id, "name": "Strategy", "versions": []}


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str):
    """Update a strategy."""
    logger.info(f"Updating strategy: {strategy_id}")
    return {"id": strategy_id, "updated": True}


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy."""
    logger.warning(f"Deleting strategy: {strategy_id}")
    return {"deleted": True}
