"""
User input processing node.

Extracts entities from user messages based on current step.
Extracted from BookingAgent._extract_and_update_booking_info()
"""
from loguru import logger
from ..booking_state import BookingState


# CONFIRMATION KEYWORDS (from original)
CONFIRMATION_KEYWORDS = ['يلا', 'تمام', 'اوك', 'ok', 'yes', 'نعم', 'ماشي', 'زين', 'طيب', 'لا', 'no']

# SOCIAL AND INTENT KEYWORDS (from original)
SOCIAL_AND_INTENT_KEYWORDS = [
    'حجز', 'احجز', 'أحجز', 'موعد', 'book', 'booking', 'appointment',
    'اختار', 'أختار', 'اخترلي', 'أخترلي', 'انت اختار', 'choose', 'select',
    'أي واحد', 'اي واحد', 'ما يهم', 'مايهم', 'ما يهمني', 'انت شوف',
    'انت اشوف', 'أنت شوف', 'على راحتك', 'عادي', 'any', 'whatever', 'you choose'
]


async def process_user_input_node(
    state: BookingState,
    api_client
) -> BookingState:
    """
    Extract entities from user message and update state.
    
    Maps to: _extract_and_update_booking_info()
    
    This node handles:
    - Service selection by name/number
    - Doctor selection by name/number
    - Date extraction (باكر, اليوم, etc.)
    - Time extraction (صباح, ظهر, etc.)
    - Loop detection
    - Social keyword filtering
    """
    message = state["current_message"]
    message_lower = message.lower()
    is_pure_intent = state.get("is_pure_intent", False)
    current_step = state.get("step", "")
    
    logger.info(f"🔍 [NODE:process_input] Processing: '{message[:30]}...' (step={current_step})")
    
    # SKIP extraction for pure intent keywords
    if is_pure_intent:
        logger.info(f"⏭️ [NODE:process_input] Skipping - pure intent keyword")
        return state
    
    # SKIP extraction for confirmation keywords
    if message_lower.strip() in CONFIRMATION_KEYWORDS:
        logger.info(f"⏭️ [NODE:process_input] Skipping - confirmation keyword")
        return state
    
    # LOOP DETECTION (PRESERVE EXACT LOGIC)
    last_message = state.get("last_user_message", "")
    repeat_count = state.get("message_repeat_count", 0)
    
    if message_lower == last_message.lower():
        repeat_count += 1
        state["message_repeat_count"] = repeat_count
        if repeat_count >= 3:
            logger.warning(f"🔁 [NODE:process_input] Loop detected: '{message}' x{repeat_count}")
            state["step"] = "loop_detected"
            return state
    else:
        state["last_user_message"] = message_lower
        state["message_repeat_count"] = 1
    
    # No extraction needed - routing will handle selection
    # The select_* nodes will process the actual selection
    logger.info(f"✅ [NODE:process_input] Input logged, routing will handle selection")
    
    return state


async def handle_error_node(state: BookingState) -> BookingState:
    """
    Handle error states with recovery.
    """
    last_error = state.get("last_error", {})
    error_msg = last_error.get("message", "Unknown error")
    error_node = last_error.get("node", "unknown")
    sender_name = state["arabic_name"]
    
    logger.error(f"🔴 [NODE:handle_error] Error in {error_node}: {error_msg}")
    
    # Increment critical failures
    critical_failures = state.get("critical_failures", 0) + 1
    state["critical_failures"] = critical_failures
    
    if critical_failures >= 3:
        logger.error(f"🚨 [NODE:handle_error] CRITICAL: {critical_failures} consecutive failures")
        state["step"] = "catastrophic_failure"
        
        # CRITICAL: Use LLM to generate natural catastrophic error message (NO TEMPLATES!)
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        
        message_content = await response_gen.handle_catastrophic_error(
            user_name=sender_name
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": message_content
        })
    else:
        # Try to recover - RESET to start state for fresh attempt
        logger.warning(f"⚠️ [NODE:handle_error] Attempting recovery - resetting to start state")
        
        # CRITICAL FIX (Bug 28): Handle error_no_service_type specially
        # If missing service_type_id, ask for service again instead of generic error
        current_step = state.get("step")
        
        if current_step == "error_no_service_type":
            logger.warning(f"⚠️ [NODE:handle_error] Error was 'no service type' - asking for service selection")
            
            # Ask for service instead of generic error
            state["step"] = "awaiting_service_selection"
            state["last_error"] = None
            state["critical_failures"] = 0
            
            # Generate service request
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            
            response = await response_gen.ask_for_service(
                user_name=sender_name,
                services=None  # Will use defaults
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            return state
        
        # For other errors, reset to start
        state["step"] = "start"
        
        # Clear ALL error-related state
        state["last_error"] = None
        state["critical_failures"] = 0  # Reset failure counter on recovery
        
        # Clear old registration data to allow fresh start
        if "registration" in state:
            state["registration"] = None
        if "national_id" in state:
            del state["national_id"]
        if "gender" in state:
            del state["gender"]
        if "name" in state:
            del state["name"]
        
        # CRITICAL: Use LLM to generate natural error recovery message (NO TEMPLATES!)
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        
        message_content = await response_gen.handle_registration_error_recovery(
            user_name=sender_name
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": message_content
        })
    
    return state


async def handle_loop_node(state: BookingState) -> BookingState:
    """
    Handle loop detection (user stuck).
    """
    sender_name = state["arabic_name"]
    
    logger.warning(f"🔁 [NODE:handle_loop] User stuck in loop")
    
    state["step"] = "loop_help"
    
    # CRITICAL: Use LLM to generate natural help message (NO TEMPLATES!)
    from app.core.response_generator import get_response_generator
    response_gen = get_response_generator()
    
    message_content = await response_gen.handle_user_stuck_in_loop(
        user_name=sender_name
    )
    
    state["messages"].append({
        "role": "assistant",
        "content": message_content
    })
    
    # Reset loop counter
    state["message_repeat_count"] = 0
    state["last_user_message"] = ""
    
    return state


async def await_user_input_node(state: BookingState) -> BookingState:
    """
    Pause and wait for user input.
    
    This node marks the state as waiting for user message.
    LangGraph will interrupt here and resume when new message arrives.
    """
    logger.info(f"⏸️ [NODE:await_user_input] Waiting for user input (step={state.get('step')})")
    
    # Just return state - LangGraph will pause here
    return state
