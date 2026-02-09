"""Chat routes for AI conversation."""

from fastapi import APIRouter

from utils.logger import logger, log_agent

router = APIRouter()


@router.get("/")
async def list_conversations():
    """List all conversations."""
    logger.info("Listing conversations")
    return {"conversations": []}


@router.post("/")
async def create_conversation():
    """Create a new conversation."""
    log_agent("create_conversation", "New conversation started")
    return {"id": "conv_placeholder", "messages": []}


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    logger.debug(f"Getting conversation: {conversation_id}")
    return {"id": conversation_id, "messages": []}


@router.post("/{conversation_id}/messages")
async def send_message(conversation_id: str):
    """Send a message in a conversation."""
    log_agent("send_message", f"Message sent to {conversation_id}")
    return {"id": "msg_placeholder", "content": "placeholder response"}
