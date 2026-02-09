"""API Dependencies."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from api.config import Settings


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


# Future dependencies:
# async def get_db():
#     """Get database session."""
#     pass
#
# async def get_current_user():
#     """Get current authenticated user."""
#     pass
