"""
Service type and service selection nodes.

Extracted from BookingAgent service selection logic.
"""
import time
from loguru import logger
from ..booking_state import BookingState
from ...service_flow_helpers import fetch_service_types, fetch_services_by_type, format_service_types_list, format_services_list


# SHARED CACHE (Issue: Cache miss on every registration)
# Increased TTL to 30 minutes to reduce API calls
_SHARED_CACHE = {
    "services": {"data": None, "time": 0},
}

# Cache TTL in seconds (30 minutes)
_CACHE_TTL = 1800


async def warm_service_types_cache(api_client) -> None:
    """
    Pre-fetch and cache service types on startup to reduce first-request latency.
    
    Call this from application startup to avoid cache miss on first registration.
    """
    try:
        logger.info("🔥 Warming service types cache...")
        service_types = await fetch_service_types(api_client)
        _SHARED_CACHE["services"] = {"data": service_types, "time": time.time()}
        logger.info(f"✅ Service types cache warmed ({len(service_types)} types, TTL={_CACHE_TTL}s)")
    except Exception as e:
        logger.error(f"❌ Failed to warm service types cache: {e}")


async def fetch_service_types_node(
    state: BookingState,
    api_client
) -> BookingState:
    """
    Fetch and display service types.
    
    Maps to: Service type fetching in _handle_new_booking()
    Uses: fetch_service_types() from service_flow_helpers
    """
    arabic_name = state["arabic_name"]
    
    try:
        # CRITICAL: Check if service was already discussed in conversation
        discussed_service = state.get("last_discussed_service")
        if discussed_service:
            logger.info(f"🎯 [NODE:fetch_service_types] Service already discussed: '{discussed_service}' - skipping category selection")
            
            # Fetch service types to find matching one
            current_time = time.time()
            cached_data = _SHARED_CACHE["services"]["data"]
            cache_time = _SHARED_CACHE["services"]["time"]
            
            if cached_data and cache_time > 0 and (current_time - cache_time) < _CACHE_TTL:
                service_types = cached_data
            else:
                service_types = await fetch_service_types(api_client)
                _SHARED_CACHE["services"] = {"data": service_types, "time": current_time}
            
            # Find the matching service type
            selected_type = None
            discussed_lower = discussed_service.lower()
            for service_type in service_types:
                type_name = (service_type.get("name_ar") or service_type.get("name") or "").lower()
                if discussed_lower in type_name or type_name in discussed_lower:
                    selected_type = service_type
                    logger.info(f"✅ [NODE:fetch_service_types] Auto-selected service type: {service_type.get('name_ar')} (id={service_type.get('id')})")
                    break
            
            if selected_type:
                # Skip to showing variants for this service
                state["selected_service_type_id"] = selected_type.get("id")
                state["selected_service_type_name"] = selected_type.get("name_ar") or selected_type.get("name")
                state["step"] = "service_type_selected"
                logger.info(f"⏭️ [NODE:fetch_service_types] Skipping category list - going directly to {discussed_service} variants")
                return state
            else:
                logger.warning(f"⚠️ [NODE:fetch_service_types] Could not find service type matching '{discussed_service}' - showing full list")
        
        # Normal flow: Show all categories
        current_time = time.time()
        
        # Check SHARED cache with 30-minute TTL (Issue: Cache miss pattern)
        # CRITICAL FIX: Handle cache not initialized (time=0)
        cached_data = _SHARED_CACHE["services"]["data"]
        cache_time = _SHARED_CACHE["services"]["time"]
        
        if cached_data and cache_time > 0:
            cache_age = current_time - cache_time
            if cache_age < _CACHE_TTL:
                logger.info(f"⚡ [NODE:fetch_service_types] CACHE HIT (age: {int(cache_age)}s / {_CACHE_TTL}s TTL)")
                service_types = cached_data
            else:
                logger.info(f"🔀 [NODE:fetch_service_types] CACHE EXPIRED (age: {int(cache_age)}s) - refreshing")
                service_types = await fetch_service_types(api_client)
                _SHARED_CACHE["services"] = {"data": service_types, "time": current_time}
                logger.info(f"💾 [NODE:fetch_service_types] Cache refreshed ({len(service_types)} types)")
        else:
            logger.info(f"🔀 [NODE:fetch_service_types] CACHE EMPTY - first fetch")
            service_types = await fetch_service_types(api_client)
            _SHARED_CACHE["services"] = {"data": service_types, "time": current_time}
            logger.info(f"💾 [NODE:fetch_service_types] Cache initialized ({len(service_types)} types, TTL={_CACHE_TTL}s)")
        
        # VALIDATION (from original code)
        if len(service_types) == 0:
            logger.error(f"⚠️ [NODE:fetch_service_types] No service types available")
            state["step"] = "service_types_unavailable"
            state["last_error"] = {"message": "No service types", "node": "fetch_service_types"}
            return state
        
        state["service_types"] = service_types
        state["step"] = "awaiting_service_type"
        logger.info(f"✅ [NODE:fetch_service_types] Loaded {len(service_types)} types")
        
        # CRITICAL: DO NOT append templated message - let conversational agent respond naturally!
        # Set flag for router to use conversational agent
        state["needs_conversational_response"] = True
        state["context_for_agent"] = {
            "action": "show_service_categories",
            "service_types": service_types,
            "count": len(service_types)
        }
        logger.info(f"🤖 [NODE:fetch_service_types] Response will be generated by conversational agent")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [NODE:fetch_service_types] Error: {e}", exc_info=True)
        state["step"] = "service_types_fetch_error"
        state["last_error"] = {"message": str(e), "node": "fetch_service_types"}
        return state


async def select_service_type_node(state: BookingState) -> BookingState:
    """
    Process user's service type selection (number or name).
    """
    message = state["current_message"]
    service_types = state.get("service_types", [])
    
    logger.info(f"🔍 [NODE:select_service_type] Processing selection: {message[:30]}...")
    
    selected_type = None
    
    # Try number selection
    if message.strip().isdigit():
        index = int(message.strip()) - 1
        if 0 <= index < len(service_types):
            selected_type = service_types[index]
            logger.info(f"✅ [NODE:select_service_type] Selected by number: {selected_type.get('name_ar')}")
    
    # Try name matching
    if not selected_type:
        message_lower = message.lower()
        for service_type in service_types:
            type_name_ar = (service_type.get("name_ar") or "").lower()
            type_name_en = (service_type.get("name") or "").lower()
            if message_lower in type_name_ar or message_lower in type_name_en:
                selected_type = service_type
                logger.info(f"✅ [NODE:select_service_type] Selected by name: {selected_type.get('name_ar')}")
                break
    
    if selected_type:
        state["selected_service_type_id"] = selected_type.get("id")
        state["selected_service_type_name"] = selected_type.get("name_ar") or selected_type.get("name")
        state["step"] = "service_type_selected"
    else:
        logger.warning(f"⚠️ [NODE:select_service_type] Selection not found: {message}")
        state["step"] = "service_type_selection_failed"
    
    return state


async def fetch_services_node(
    state: BookingState,
    api_client
) -> BookingState:
    """
    Fetch services for selected service type.
    """
    service_type_id = state.get("selected_service_type_id")
    arabic_name = state["arabic_name"]
    type_name = state.get("selected_service_type_name")
    
    if not service_type_id:
        logger.error(f"⚠️ [NODE:fetch_services] No service type selected")
        state["step"] = "error_no_service_type"
        return state
    
    try:
        logger.info(f"🔀 [NODE:fetch_services] Fetching services for type {service_type_id}")
        services = await fetch_services_by_type(api_client, service_type_id)
        
        if not services:
            logger.warning(f"⚠️ [NODE:fetch_services] No services found for type {service_type_id}")
            state["step"] = "no_services_available"
            state["last_error"] = {"message": "No services available", "node": "fetch_services"}
            return state
        
        state["services"] = services
        state["displayed_services"] = services
        state["step"] = "awaiting_service"
        logger.info(f"✅ [NODE:fetch_services] Loaded {len(services)} services")
        
        # Add response message - CRITICAL: format_services_list is now async!
        response_text = await format_services_list(services, arabic_name, type_name)
        state["messages"].append({
            "role": "assistant",
            "content": response_text
        })
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [NODE:fetch_services] Error: {e}", exc_info=True)
        state["step"] = "services_fetch_error"
        state["last_error"] = {"message": str(e), "node": "fetch_services"}
        return state


async def select_service_node(state: BookingState) -> BookingState:
    """
    Process user's service selection (number or name).
    """
    message = state["current_message"]
    services = state.get("displayed_services", [])
    
    logger.info(f"🔍 [NODE:select_service] Processing selection: {message[:30]}...")
    
    selected_service = None
    
    # Try number selection
    if message.strip().isdigit():
        index = int(message.strip()) - 1
        if 0 <= index < len(services):
            selected_service = services[index]
            logger.info(f"✅ [NODE:select_service] Selected by number: {selected_service.get('name_ar')}")
    
    # Try name matching
    if not selected_service:
        message_lower = message.lower()
        for service in services:
            service_name_ar = (service.get("name_ar") or "").lower()
            service_name_en = (service.get("name") or "").lower()
            if message_lower in service_name_ar or message_lower in service_name_en:
                selected_service = service
                logger.info(f"✅ [NODE:select_service] Selected by name: {selected_service.get('name_ar')}")
                break
    
    if selected_service:
        state["service_id"] = selected_service.get("id")
        state["service_name"] = selected_service.get("name_ar") or selected_service.get("name")
        state["step"] = "service_selected"
    else:
        logger.warning(f"⚠️ [NODE:select_service] Selection not found: {message}")
        state["step"] = "service_selection_failed"
    
    return state
