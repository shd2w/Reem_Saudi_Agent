"""
Cancel booking node for handling cancellation requests.
"""
from loguru import logger
from ..booking_state import BookingState


async def cancel_booking_node(state: BookingState) -> BookingState:
    """
    Handle booking cancellation request.
    """
    sender_name = state["arabic_name"]
    
    logger.info(f"❌ [NODE:cancel_booking] User cancelled booking")
    
    # Clear booking state
    state["step"] = "cancelled"
    state["started"] = False
    state["awaiting_confirmation"] = False
    
    # CRITICAL: Use LLM to generate natural cancellation message (NO TEMPLATES!)
    from app.core.response_generator import get_response_generator
    response_gen = get_response_generator()
    
    message_content = await response_gen.handle_cancellation(user_name=sender_name)
    
    state["messages"].append({
        "role": "assistant",
        "content": message_content
    })
    
    return state
