"""
Resource selection nodes (doctor/device/specialist).

Extracted from BookingAgent and service_flow_helpers.
"""
from loguru import logger
from ..booking_state import BookingState
from ...service_flow_helpers import get_service_requirement
from ....utils.name_transliterator import transliterate_full_name


async def fetch_resources_node(
    state: BookingState,
    api_client
) -> BookingState:
    """
    Fetch appropriate resource (doctor/specialist/device) based on service.
    
    Maps to: fetch_resource_for_service() in service_flow_helpers
    """
    service_id = state.get("service_id")
    service_name = state.get("service_name")
    
    if not service_id:
        logger.error(f"⚠️ [NODE:fetch_resources] No service selected")
        state["step"] = "error_no_service"
        return state
    
    try:
        # Get service details to determine requirement
        service_result = await api_client.get(f"/services/{service_id}")
        service = service_result.get("data") or service_result
        
        requirement = get_service_requirement(service)
        logger.info(f"📋 [NODE:fetch_resources] Service requires: {requirement}")
        
        state["resource_type"] = requirement
        
        if requirement == "doctor":
            logger.info(f"📋 Fetching doctors for service {service_id}")
            result = await api_client.get(f"/services/{service_id}/doctors")
            resources = result.get("results") or result.get("data") or []
            
            if not resources:
                # Fallback: get all doctors
                result = await api_client.get("/doctors", params={"limit": 20})
                resources = result.get("results") or result.get("data") or []
            
            state["doctors"] = resources
            state["step"] = "awaiting_doctor"
            logger.info(f"✅ Fetched {len(resources)} doctors")
            
        elif requirement == "specialist":
            logger.info(f"📋 Fetching specialists")
            result = await api_client.get("/specialists", params={"limit": 20})
            resources = result.get("results") or result.get("data") or []
            state["specialists"] = resources
            state["step"] = "awaiting_specialist"
            logger.info(f"✅ Fetched {len(resources)} specialists")
            
        elif requirement == "device":
            logger.info(f"📋 Fetching devices (matching service name)")
            
            # Extract device name from service name
            service_name_lower = (service_name or "").lower()
            
            # Fetch all devices
            result = await api_client.get("/devices", params={"limit": 50})
            all_devices = result.get("results") or result.get("data") or []
            
            # Match devices whose name appears in service name
            matched_devices = []
            for device in all_devices:
                device_name = (device.get("name") or "").lower()
                device_name_ar = (device.get("name_ar") or "").lower()
                
                if (device_name and device_name in service_name_lower) or \
                   (device_name_ar and device_name_ar in service_name_lower):
                    matched_devices.append(device)
            
            # If no matches, show all devices
            if not matched_devices:
                logger.warning(f"⚠️ No devices matched service name '{service_name}' - showing all")
                matched_devices = all_devices
            
            state["devices"] = matched_devices
            state["step"] = "awaiting_device"
            logger.info(f"✅ Found {len(matched_devices)} devices")
            
        else:
            logger.error(f"❌ Unknown requirement: {requirement}")
            state["step"] = "unknown_requirement"
            state["last_error"] = {"message": f"Unknown requirement: {requirement}", "node": "fetch_resources"}
            return state
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [NODE:fetch_resources] Error: {e}", exc_info=True)
        state["step"] = "resource_fetch_error"
        state["last_error"] = {"message": str(e), "node": "fetch_resources"}
        return state


async def select_resource_node(state: BookingState, resource_type: str) -> BookingState:
    """
    Process user's resource selection (doctor/specialist/device).
    
    Args:
        state: Current state
        resource_type: "doctor", "specialist", or "device"
    """
    message = state["current_message"]
    arabic_name = state["arabic_name"]
    
    logger.info(f"🔍 [NODE:select_resource] Processing {resource_type} selection: {message[:30]}...")
    
    selected_resource = None
    
    if resource_type == "doctor":
        doctors = state.get("doctors", [])
        
        # Try number selection
        if message.strip().isdigit():
            index = int(message.strip()) - 1
            if 0 <= index < len(doctors):
                selected_resource = doctors[index]
        
        # Try name matching
        if not selected_resource:
            message_lower = message.lower()
            for doctor in doctors:
                doctor_name_ar = (doctor.get("name_ar") or "").lower()
                doctor_name_en = (doctor.get("name") or "").lower()
                if message_lower in doctor_name_ar or message_lower in doctor_name_en:
                    selected_resource = doctor
                    break
        
        if selected_resource:
            state["doctor_id"] = selected_resource.get("id")
            state["doctor_name"] = selected_resource.get("name_ar") or selected_resource.get("name")
            state["doctor_name_en"] = selected_resource.get("name")
            state["step"] = "doctor_selected"
            logger.info(f"✅ Selected doctor: {state['doctor_name']}")
        else:
            logger.warning(f"⚠️ Doctor not found: {message}")
            state["step"] = "doctor_not_found"
            state["doctor_search_failed"] = message
    
    elif resource_type == "specialist":
        specialists = state.get("specialists", [])
        
        # Try number selection
        if message.strip().isdigit():
            index = int(message.strip()) - 1
            if 0 <= index < len(specialists):
                selected_resource = specialists[index]
        
        # Try name matching
        if not selected_resource:
            message_lower = message.lower()
            for specialist in specialists:
                specialist_name_ar = (specialist.get("name_ar") or "").lower()
                specialist_name_en = (specialist.get("name") or "").lower()
                if message_lower in specialist_name_ar or message_lower in specialist_name_en:
                    selected_resource = specialist
                    break
        
        if selected_resource:
            state["specialist_id"] = selected_resource.get("id")
            state["specialist_name"] = selected_resource.get("name_ar") or selected_resource.get("name")
            state["step"] = "specialist_selected"
            logger.info(f"✅ Selected specialist: {state['specialist_name']}")
        else:
            logger.warning(f"⚠️ Specialist not found: {message}")
            state["step"] = "specialist_not_found"
    
    elif resource_type == "device":
        devices = state.get("devices", [])
        
        # Auto-select if only one device
        if len(devices) == 1:
            selected_resource = devices[0]
            logger.info(f"🚀 Auto-selected single device")
        else:
            # Try number selection
            if message.strip().isdigit():
                index = int(message.strip()) - 1
                if 0 <= index < len(devices):
                    selected_resource = devices[index]
            
            # Try name matching
            if not selected_resource:
                message_lower = message.lower()
                for device in devices:
                    device_name_ar = (device.get("name_ar") or "").lower()
                    device_name_en = (device.get("name") or "").lower()
                    if message_lower in device_name_ar or message_lower in device_name_en:
                        selected_resource = device
                        break
        
        if selected_resource:
            state["device_id"] = selected_resource.get("id")
            state["device_name"] = selected_resource.get("name_ar") or selected_resource.get("name")
            state["step"] = "device_selected"
            logger.info(f"✅ Selected device: {state['device_name']}")
        else:
            logger.warning(f"⚠️ Device not found: {message}")
            state["step"] = "device_not_found"
    
    return state
