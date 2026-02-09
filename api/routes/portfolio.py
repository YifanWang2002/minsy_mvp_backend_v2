"""Portfolio routes."""

from fastapi import APIRouter

from utils.logger import logger

router = APIRouter()


@router.get("/")
async def get_portfolio():
    """Get portfolio overview."""
    logger.info("Getting portfolio overview")
    return {
        "deployed_strategies": [],
        "total_value": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
    }


@router.post("/deploy")
async def deploy_strategy():
    """Deploy a strategy to paper/live trading."""
    logger.info("Deploying strategy")
    return {"id": "deployment_placeholder", "status": "running"}


@router.post("/{deployment_id}/pause")
async def pause_deployment(deployment_id: str):
    """Pause a deployed strategy."""
    logger.warning(f"Pausing deployment: {deployment_id}")
    return {"id": deployment_id, "status": "paused"}


@router.post("/{deployment_id}/resume")
async def resume_deployment(deployment_id: str):
    """Resume a paused deployment."""
    logger.info(f"Resuming deployment: {deployment_id}")
    return {"id": deployment_id, "status": "running"}


@router.post("/{deployment_id}/stop")
async def stop_deployment(deployment_id: str):
    """Stop a deployed strategy."""
    logger.warning(f"Stopping deployment: {deployment_id}")
    return {"id": deployment_id, "status": "stopped"}
