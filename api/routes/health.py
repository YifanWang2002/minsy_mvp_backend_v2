"""Health check routes."""

from fastapi import APIRouter

from utils.logger import logger

router = APIRouter()


@router.get("/health")
async def health():
    """Simple health check for load balancers."""
    logger.debug("Health check requested")
    return {"status": "ok"}
