"""
Time slot selection and booking confirmation nodes.

Extracted from BookingAgent and booking_helpers.
"""
from datetime import datetime
from loguru import logger
from ..booking_state import BookingState


async def fetch_time_slots_node(
    state: BookingState,
    api_client
) -> BookingState:
    """
    Fetch available time slots for selected service and resource.
    
    Maps to: show_available_time_slots() in booking_helpers
    """
    service_id = state.get("service_id")
    resource_type = state.get("resource_type", "doctor")
    doctor_id = state.get("doctor_id")
    specialist_id = state.get("specialist_id")
    device_id = state.get("device_id")
    sender_name = state["arabic_name"]
    
    logger.info(f"🔍 [NODE:fetch_time_slots] service={service_id}, resource_type={resource_type}")
    
    try:
        # Build params based on resource type
        params = {
            "service_id": service_id,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        if resource_type == "doctor" and doctor_id:
            params["doctor_id"] = doctor_id
        elif resource_type == "specialist" and specialist_id:
            params["specialist_id"] = specialist_id
        elif resource_type == "device" and device_id:
            params["device_id"] = device_id
        else:
            logger.warning(f"⚠️ [NODE:fetch_time_slots] No resource ID found for type {resource_type}")
        
        slots_result = await api_client.get("/slots", params=params)
        slots = slots_result.get("data") or slots_result.get("slots") or []
        
        logger.info(f"✅ [NODE:fetch_time_slots] Fetched {len(slots)} slots")
        
        if not slots:
            state["step"] = "no_slots_available"
            
            # CRITICAL: Use LLM to generate natural "no slots" message (NO TEMPLATES!)
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            
            message_content = await response_gen.handle_no_slots_available(
                user_name=sender_name
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": message_content
            })
            return state
        
        # Save slots to state
        state["available_slots"] = slots[:10]
        state["step"] = "awaiting_time_slot"
        
        # Format slots with numbers
        slots_list = [
            f"{i+1}. {slot.get('date')} - {slot.get('time')}"
            for i, slot in enumerate(slots[:10])
        ]
        
        # CRITICAL: Use LLM to generate natural slot presentation (NO TEMPLATES!)
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        
        message_content = await response_gen.present_time_slots(
            user_name=sender_name,
            slots=slots_list
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": message_content
        })
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [NODE:fetch_time_slots] Error: {e}", exc_info=True)
        state["step"] = "time_slots_fetch_error"
        state["last_error"] = {"message": str(e), "node": "fetch_time_slots"}
        return state


async def select_time_slot_node(state: BookingState) -> BookingState:
    """
    Process user's time slot selection.
    """
    message = state["current_message"]
    slots = state.get("available_slots", [])
    
    logger.info(f"🔍 [NODE:select_time_slot] Processing selection: {message[:30]}...")
    
    selected_slot = None
    
    # Try number selection
    if message.strip().isdigit():
        index = int(message.strip()) - 1
        if 0 <= index < len(slots):
            selected_slot = slots[index]
            logger.info(f"✅ [NODE:select_time_slot] Selected slot #{index + 1}")
    
    if selected_slot:
        state["preferred_date"] = selected_slot.get("date")
        state["preferred_time"] = selected_slot.get("time")
        state["step"] = "time_slot_selected"
    else:
        logger.warning(f"⚠️ [NODE:select_time_slot] Invalid selection: {message}")
        state["step"] = "time_slot_selection_failed"
    
    return state


async def confirm_booking_node(
    state: BookingState,
    api_client
) -> BookingState:
    """
    Show booking summary and request confirmation.
    
    Maps to: request_booking_confirmation() in booking_helpers
    """
    sender_name = state["arabic_name"]
    service_name = state.get("service_name", "الخدمة")
    doctor_name = state.get("doctor_name", "الدكتور")
    date = state.get("preferred_date", "التاريخ")
    time = state.get("preferred_time", "الوقت")
    
    logger.info(f"📋 [NODE:confirm_booking] Requesting confirmation")
    
    try:
        # Fetch price from service details
        service_id = state.get("service_id")
        price = "حسب الاستشارة"
        
        if service_id:
            try:
                service_details = await api_client.get(f"/services/{service_id}")
                price = service_details.get("price", "حسب الاستشارة")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch service price: {e}")
        
        # Mark as awaiting confirmation
        state["awaiting_confirmation"] = True
        state["step"] = "awaiting_confirmation"
        
        # CRITICAL: Use LLM to generate natural booking confirmation (NO TEMPLATES!)
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        
        booking_details = {
            "service": service_name,
            "doctor": doctor_name,
            "date": date,
            "time": time,
            "price": price
        }
        
        confirmation_text = await response_gen.request_booking_confirmation(
            user_name=sender_name,
            booking_details=booking_details
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": confirmation_text
        })
        
        logger.info(f"✅ [NODE:confirm_booking] Confirmation request sent")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [NODE:confirm_booking] Error: {e}", exc_info=True)
        state["step"] = "confirmation_error"
        state["last_error"] = {"message": str(e), "node": "confirm_booking"}
        return state


async def create_booking_node(
    state: BookingState,
    api_client
) -> BookingState:
    """
    Create booking via API.
    
    Maps to: complete_booking_with_details() in booking_helpers
    """
    sender_name = state["arabic_name"]
    phone_number = state["phone_number"]
    
    logger.info(f"🎯 [NODE:create_booking] Creating booking")
    
    try:
        # CRITICAL VALIDATION
        required_fields = {
            "service_id": state.get("service_id"),
            "service_name": state.get("service_name"),
            "doctor_id": state.get("doctor_id"),
            "preferred_date": state.get("preferred_date"),
            "preferred_time": state.get("preferred_time")
        }
        
        missing_fields = [k for k, v in required_fields.items() if not v]
        
        if missing_fields:
            logger.error(f"🚫 [NODE:create_booking] VALIDATION FAILED: Missing {missing_fields}")
            state["step"] = "validation_failed"
            state["last_error"] = {"message": f"Missing fields: {missing_fields}", "node": "create_booking"}
            return state
        
        logger.info(f"✅ [NODE:create_booking] Validation passed")
        
        # Prepare booking data
        booking_data = {
            "patient_phone": phone_number,
            "service_id": state.get("service_id"),
            "doctor_id": state.get("doctor_id"),
            "appointment_date": state.get("preferred_date"),
            "appointment_time": state.get("preferred_time"),
            "notes": f"Booked via WhatsApp by {sender_name}"
        }
        
        # Create booking
        result = await api_client.post("/appointments", data=booking_data)
        
        booking_id = result.get("id") or result.get("booking_id") or "N/A"
        confirmation_code = result.get("confirmation_code") or f"WJ{booking_id}"
        
        logger.info(f"✅ [NODE:create_booking] Booking created - ID: {booking_id}")
        
        state["booking_id"] = booking_id
        state["step"] = "booking_created"
        
        service_name = state.get("service_name", "الخدمة")
        doctor_name = state.get("doctor_name", "الدكتور")
        date = state.get("preferred_date")
        time = state.get("preferred_time")
        
        # CRITICAL: Use LLM to generate natural success message (NO TEMPLATES!)
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        
        booking_info = {
            "booking_id": booking_id,
            "confirmation_code": confirmation_code,
            "service": service_name,
            "doctor": doctor_name,
            "date": date,
            "time": time
        }
        
        success_message = await response_gen.confirm_booking_success(
            user_name=sender_name,
            booking_info=booking_info
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": success_message
        })
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [NODE:create_booking] Error: {e}", exc_info=True)
        state["step"] = "booking_creation_error"
        state["last_error"] = {"message": str(e), "node": "create_booking"}
        return state


async def send_confirmation_node(state: BookingState) -> BookingState:
    """
    Mark booking as completed and clean up state.
    """
    logger.info(f"✅ [NODE:send_confirmation] Booking flow completed")
    
    # Clear booking state but keep booking_id
    booking_id = state.get("booking_id")
    state["step"] = "completed"
    state["started"] = False
    state["last_booking_id"] = booking_id
    
    # Clear all booking details
    for key in list(state.keys()):
        if key not in ["session_id", "phone_number", "sender_name", "arabic_name", "messages", "step", "started", "last_booking_id", "booking_id"]:
            state.pop(key, None)
    
    return state
