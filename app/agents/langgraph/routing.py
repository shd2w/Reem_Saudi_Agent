"""
Routing functions for LangGraph booking flow.

Determines next node based on current state.
Maps to implicit routing logic in BookingAgent._handle_new_booking()
"""
from loguru import logger
from .booking_state import BookingState
from langgraph.graph import END
import re


def _is_registration_complete(state: BookingState) -> bool:
    """
    Verify registration is actually complete before allowing service selection.
    
    CRITICAL: Prevents flow bypass with incomplete data.
    
    Args:
        state: Current booking state
        
    Returns:
        True if all required fields present and valid, False otherwise
    """
    # Required fields for complete registration
    required_fields = {
        "name": state.get("name") or state.get("registration", {}).get("name"),
        "national_id": state.get("national_id"),
        "gender": state.get("gender"),
        "phone_number": state.get("phone_number"),
        "patient_id": state.get("patient_id")  # Must have created patient
    }
    
    # Check all required fields present
    for field_name, field_value in required_fields.items():
        if not field_value:
            logger.error(f"❌ Registration incomplete: missing {field_name}")
            return False
    
    # Validate national ID format (10 digits, starts with 1 or 2)
    national_id = required_fields["national_id"]
    if not re.match(r'^[12]\d{9}$', national_id):
        logger.error(f"❌ Registration incomplete: invalid national_id format '{national_id}'")
        return False
    
    # Validate gender is valid value
    gender = required_fields["gender"]
    if gender not in ["male", "female"]:
        logger.error(f"❌ Registration incomplete: invalid gender '{gender}'")
        return False
    
    # All checks passed
    logger.debug(f"✅ Registration complete: name={required_fields['name']}, id={national_id}, gender={gender}, patient_id={required_fields['patient_id']}")
    return True

def route_after_confirmation(state: BookingState) -> str:
    """
    LLM-FIRST: Handles routing after user confirms they want to register.
    Uses LLM-detected intent only, no keyword matching.
    """
    router_intent = state.get("router_intent", "")
    message = state.get("current_message", "")
    
    logger.info(f"🤖 [ROUTER:confirmation] LLM-FIRST - Using router intent: {router_intent}")
    
    # LLM-FIRST: Trust the LLM's intent classification
    # LLM understands: نعم، بينا، يلا، تمام, etc. naturally
    
    if router_intent == "cancel":
        logger.info("🚫 [ROUTER] LLM detected cancellation intent")
        state["step"] = "registration_cancelled"
        return "await_user_input"
    
    # CRITICAL: Treat both "confirmation" and "registration" as confirmation
    # When user says "بينا" or "أبغى أسجل", LLM may classify as either
    elif router_intent in ["confirmation", "registration"]:
        logger.info(f"✅ [ROUTER] LLM detected confirmation (intent={router_intent}), proceeding to start_registration.")
        return "start_registration"
    
    elif router_intent == "negation":
        logger.info(f"🤷 [ROUTER] LLM detected negation, awaiting next input.")
        return "await_user_input"
    
    else:
        # User said something else - wait for clarification
        logger.info(f"🤷 [ROUTER] LLM classified as {router_intent}, awaiting next input.")
        return "await_user_input"


def route_patient_flow(state: BookingState) -> str:
    """
    Route patient verification flow.
    
    Decision tree:
    - patient_verified + service already selected → fetch_resources (skip service selection!)
    - patient_verified + no service → fetch_service_types
    - needs_registration → start_registration
    - error → handle_error
    """
    step = state.get("step", "")
    
    if step == "patient_verified":
        # CRITICAL STATE SYNC FIX: Check if service was already selected by Router!
        # If yes, skip fetch_service_types and go directly to resource selection
        selected_service_id = state.get("selected_service_id")
        selected_service_name = state.get("selected_service_name")
        
        if selected_service_name:
            logger.info(f"🔀 [ROUTER:patient] ✅ Service already selected: {selected_service_name} → SKIPPING to fetch_resources")
            # Mark service as selected so we can skip service selection nodes
            state["step"] = "service_selected"
            state["selected_service"] = selected_service_name  # Ensure it's in expected field
            return "fetch_resources"
        else:
            logger.info("🔀 [ROUTER:patient] step=patient_verified → fetch_service_types (no service yet)")
            return "fetch_service_types"
    elif step == "needs_registration":
        logger.info("🔀 [ROUTER:patient] step=needs_registration → confirm_registration (not found; ask consent)")
        return "confirm_registration"  # Ask for confirmation first (Issue #30)
    elif step == "patient_verification_error":
        logger.info("🔀 [ROUTER:patient] step=patient_verification_error → handle_error")
        return "handle_error"
    else:
        logger.info(f"🔀 [ROUTER:patient] step={step} → await_user_input (default)")
        return "await_user_input"


def route_service_flow(state: BookingState) -> str:
    """
    Route service selection flow.
    
    Decision tree:
    - service_type_selected → fetch_services
    - service_selected → fetch_resources
    - error → handle_error
    - waiting → await_user_input
    """
    step = state.get("step", "")
    
    if step == "awaiting_service_type":
        return "await_user_input"
    elif step == "service_type_selected":
        return "fetch_services"
    elif step == "awaiting_service":
        return "await_user_input"
    elif step == "service_selected":
        return "fetch_resources"
    elif step.endswith("_error"):
        return "handle_error"
    else:
        return "await_user_input"


def route_resource_flow(state: BookingState) -> str:
    """
    Route resource selection flow (doctor/specialist/device).
    
    Decision tree based on resource_type:
    - awaiting_doctor → await_user_input
    - doctor_selected → fetch_time_slots
    - awaiting_specialist → await_user_input
    - specialist_selected → fetch_time_slots
    - awaiting_device → await_user_input
    - device_selected → fetch_time_slots
    """
    step = state.get("step", "")
    resource_type = state.get("resource_type")
    
    if step == "awaiting_doctor":
        return "await_user_input"
    elif step == "doctor_selected":
        return "fetch_time_slots"
    elif step == "awaiting_specialist":
        return "await_user_input"
    elif step == "specialist_selected":
        return "fetch_time_slots"
    elif step == "awaiting_device":
        return "await_user_input"
    elif step == "device_selected":
        return "fetch_time_slots"
    elif step == "doctor_not_found" or step == "specialist_not_found" or step == "device_not_found":
        return "await_user_input"
    elif step.endswith("_error"):
        return "handle_error"
    else:
        return "await_user_input"


def route_time_flow(state: BookingState) -> str:
    """
    Route time slot selection flow.
    
    Decision tree:
    - awaiting_time_slot → await_user_input
    - time_slot_selected → confirm_booking
    - no_slots → await_user_input (let user try different selection)
    """
    step = state.get("step", "")
    
    if step == "awaiting_time_slot":
        return "await_user_input"
    elif step == "time_slot_selected":
        return "confirm_booking"
    elif step == "no_slots_available":
        return "await_user_input"
    elif step.endswith("_error"):
        return "handle_error"
    else:
        return "await_user_input"


def route_confirmation_flow(state: BookingState) -> str:
    """
    Route confirmation flow.
    
    Decision tree:
    - awaiting_confirmation → await_user_input
    - user confirms → create_booking
    - booking_created → send_confirmation
    - completed → END
    """
    step = state.get("step", "")
    router_intent = state.get("router_intent", "")
    
    if step == "awaiting_confirmation":
        # LLM-FIRST: Use router intent instead of keyword matching
        logger.info(f"🤖 [ROUTER:booking_confirmation] LLM-FIRST - Using router intent: {router_intent}")
        
        if router_intent == "confirmation":
            return "create_booking"
        elif router_intent in ["negation", "cancel"]:
            return "cancel_booking"
        else:
            return "await_user_input"
    
    elif step == "booking_created":
        return "send_confirmation"
    elif step == "completed":
        return END
    elif step.endswith("_error"):
        return "handle_error"
    else:
        return "await_user_input"


def route_error_flow(state: BookingState) -> str:
    """
    Route error handling flow.
    
    Can route back to recovery points or give up.
    """
    step = state.get("step", "")
    critical_failures = state.get("critical_failures", 0)
    
    if critical_failures >= 3:
        return END
    elif step == "error_recovery":
        return "await_user_input"
    elif step == "catastrophic_failure":
        return END
    else:
        return "await_user_input"


def route_error_state(state: BookingState) -> str:
    """
    Route from error state to recovery or end.
    
    CRITICAL: Never return to error_recovery (creates loop!)
    Instead, reset to start for fresh attempt.
    """
    step = state.get("step", "")
    critical_failures = state.get("critical_failures", 0)
    
    if critical_failures >= 3:
        logger.error(f"🚨 [ROUTER:error] 3+ failures, ending conversation")
        return END
    elif step == "catastrophic_failure":
        logger.error(f"🚨 [ROUTER:error] Catastrophic failure, ending conversation")
        return END
    elif step == "error_recovery":
        # CRITICAL: Don't stay in error_recovery (creates loop!)
        # Reset to start for fresh attempt
        logger.warning(f"⚠️ [ROUTER:error] error_recovery detected - should have been reset to start")
        return "verify_patient"
    elif step == "start":
        # Error handler reset to start, proceed with verification
        logger.info(f"✅ [ROUTER:error] Recovered to start state")
        return "verify_patient"
    else:
        # Other error states
        logger.warning(f"⚠️ [ROUTER:error] Unknown error state: {step}, awaiting input")
        return "await_user_input"


def route_next_step(state: BookingState) -> str:
    """
    Main router: Determine next node based on current state.
    
    This is the heart of the conversation flow.
    Maps to: implicit routing logic in _handle_new_booking()
    """
    current_step = state.get("step", "start")
    
    logger.info(f"🔀 [ROUTER] Current step: {current_step}")
    
    # Error state routing
    if current_step.startswith("error_") or current_step.endswith("_error"):
        # CRITICAL: Don't route error_recovery back to handle_error (creates loop!)
        if current_step == "error_recovery":
            logger.warning(f"⚠️ [ROUTER] error_recovery detected - resetting to start")
            state["step"] = "start"
            return "verify_patient"
        return "handle_error"
    
    # Loop detection
    if current_step == "loop_detected" or current_step == "loop_help":
        return "handle_loop"
    
    # Patient verification flow
    if current_step == "start":
        return "verify_patient"
    
    if current_step == "patient_verification_error":
        return "handle_error"
    
    # Handle registration confirmation step
    if current_step == "awaiting_registration_confirmation":
        # CRITICAL: Use router's intelligent classification instead of static patterns!
        # Router has already used LLM to understand user's intent (خلا، يالا، نعم, etc.)
        router_intent = state.get("router_intent", "").lower()
        
        # CRITICAL: Treat "registration" intent as confirmation when in confirmation step
        if router_intent in ["confirmation", "registration"]:
            logger.info(f"✅ [ROUTER] User confirmed registration (router_intent={router_intent}), proceeding to start_registration")
            return "start_registration"
        elif router_intent in ["negation", "question", "cancel"]:
            logger.info(f"⚠️ [ROUTER] User declined/questioned (router_intent={router_intent}), staying at await_user_input")
            return "await_user_input"
        else:
            # LLM-FIRST: If router_intent not available, treat as unknown
            logger.warning(f"⚠️ [ROUTER] No router_intent available, defaulting to await_user_input")
            return "await_user_input"
    
    if current_step == "needs_registration":
        return "start_registration"
    
    # Handle name collection step (Issue: awaiting_name → handle_registration_name)
    if current_step == "awaiting_name" or current_step == "registration_name":
        # CRITICAL: Check if user is CONFIRMING a name that was already extracted
        # If router_intent=confirmation AND name exists in state, skip to next step
        router_intent = state.get("router_intent", "").lower()
        has_name = state.get("name")
        current_message = state.get("current_message", "").strip()
        
        # DEBUG: Log what we have
        logger.info(f"🔍 [ROUTER DEBUG] step={current_step}, router_intent={router_intent}, has_name={bool(has_name)}, name_value={has_name}")
        
        # CRITICAL: If name already confirmed AND user sent a 10-digit number, they're giving the national ID!
        if has_name and current_message.isdigit() and len(current_message) == 10:
            logger.info(f"✅ [ROUTER] Name confirmed, user sent national ID ({current_message}), skipping to registration_id")
            return "handle_registration_id"
        
        if router_intent == "confirmation" and has_name:
            logger.info(f"✅ [ROUTER] Name confirmation detected (name={has_name}), skipping to registration_id")
            return "handle_registration_id"
        
        # CRITICAL FIX: If name already exists, skip collection (Issue: Inconsistent step tracking)
        if has_name:
            logger.info(f"✅ [ROUTER] Name already exists ({has_name}), proceeding to registration_id")
            return "handle_registration_id"
        
        # Otherwise, collect the name
        logger.info(f"➡️ [ROUTER] No name yet, collecting name (router_intent={router_intent})")
        return "handle_registration_name"
    
    if current_step == "registration_id":
        # CRITICAL: Check if user is CONFIRMING a national_id that was already extracted
        router_intent = state.get("router_intent", "").lower()
        if router_intent == "confirmation" and state.get("national_id"):
            logger.info(f"✅ [ROUTER] National ID confirmation detected, proceeding to create patient")
            # Still go to handle_registration_id but it will use the confirmed ID
        return "handle_registration_id"
    
    # REMOVED: Gender is now auto-detected by LLM, no need for awaiting_gender step
    # if current_step == "awaiting_gender":
    #     logger.info(f"➡️ [ROUTER] Awaiting gender response, routing to registration_id node")
    #     return "handle_registration_id"
    
    # Issue #4: Handle multiline registration
    if current_step == "registration_complete_multiline":
        return "handle_multiline_registration"
    
    if current_step == "registration_complete" or current_step == "patient_verified":
        # CRITICAL: Verify registration is actually complete before proceeding to booking
        # Issue: System bypassed to service selection with incomplete data
        if not _is_registration_complete(state):
            logger.error(f"⚠️ [ROUTER] Registration marked complete but data incomplete! Blocking service selection.")
            return "verify_patient"  # Go back to verification to fix missing data
        
        # CRITICAL STATE SYNC FIX: Check if service was already selected by Router!
        selected_service_name = state.get("selected_service_name")
        if selected_service_name:
            logger.info(f"✅ [ROUTER] ✅ Service already selected: {selected_service_name} → SKIPPING to fetch_resources")
            state["step"] = "service_selected"
            state["selected_service"] = selected_service_name
            return "fetch_resources"
        
        logger.info(f"✅ [ROUTER] Registration verified complete, proceeding to service selection")
        return "fetch_service_types"
    
    # Service type selection flow
    if current_step == "awaiting_service_selection":
        # CRITICAL STATE SYNC FIX: Check if service was already selected by Router!
        selected_service_name = state.get("selected_service_name")
        if selected_service_name:
            logger.info(f"✅ [ROUTER] ✅ Service already selected: {selected_service_name} → SKIPPING to fetch_resources")
            state["step"] = "service_selected"
            state["selected_service"] = selected_service_name
            return "fetch_resources"
        
        # User hasn't selected a service yet - route to service type selection
        # This happens when patient verification blocks because no service was discussed
        logger.info(f"➡️ [ROUTER] No service selected yet, routing to fetch_service_types")
        return "fetch_service_types"
    
    if current_step == "awaiting_service_type":
        return "select_service_type"
    
    if current_step == "service_type_selected":
        return "fetch_services"
    
    # Service selection flow
    if current_step == "awaiting_service":
        return "select_service"
    
    if current_step == "service_selected":
        return "fetch_resources"
    
    # Resource selection flow
    if current_step == "awaiting_doctor":
        return "select_doctor"
    
    if current_step == "awaiting_specialist":
        return "select_specialist"
    
    if current_step == "awaiting_device":
        return "select_device"
    
    if current_step in ["doctor_selected", "specialist_selected", "device_selected"]:
        return "fetch_time_slots"
    
    # Time slot selection flow
    if current_step == "awaiting_time_slot":
        return "select_time_slot"
    
    if current_step == "time_slot_selected":
        return "confirm_booking"
    
    # Confirmation and completion
    if current_step == "awaiting_confirmation":
        # Check message for confirmation
        message = state.get("current_message", "").lower()
        confirmation_words = ["نعم", "yes", "تأكيد", "أكد", "يلا", "تمام"]
        
        if any(word in message for word in confirmation_words):
            return "create_booking"
        else:
            return "await_user_input"
    
    if current_step == "booking_created":
        return "send_confirmation"
    
    if current_step == "completed":
        return END
    
    # Default: wait for user input
    logger.info(f"🔀 [ROUTER] No specific route for step '{current_step}' - awaiting input")
    return "await_user_input"
