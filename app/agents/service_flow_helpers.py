# -*- coding: utf-8 -*-
"""
Service Flow Helpers - New service type-based booking flow

Flow:
1. Show service types
2. User selects type → Show services for that type
3. User selects service → Check requirements (doctor/specialist/device)
4. Fetch appropriate resource and slots
"""

from typing import Dict, Any, Optional, List
from loguru import logger


async def fetch_service_types(api_client) -> List[Dict]:
    """
    Fetch all service types (categories).
    
    Note: The /api/services/ endpoint returns service types directly,
    not individual services. Use ?service_type_id=X to get services for a type.
    
    Returns:
        List of service type dictionaries
    """
    try:
        # Fetch service types (the API returns types directly, not services)
        result = await api_client.get("/services", params={"limit": 100})
        service_types = result.get("results") or result.get("data") or []
        
        # DEBUG: Log structure
        if service_types:
            logger.info(f"📋 API response format: {service_types[0]}")
        
        logger.info(f"✅ Fetched {len(service_types)} service types from API")
        return service_types
    except Exception as e:
        logger.error(f"❌ Failed to fetch service types: {e}")
        return []


async def fetch_services_by_type(api_client, service_type_id: int) -> List[Dict]:
    """
    Fetch services filtered by service type.
    
    Args:
        api_client: API client instance
        service_type_id: Service type ID to filter by
        
    Returns:
        List of service dictionaries
    """
    try:
        result = await api_client.get("/services", params={"service_type_id": service_type_id, "limit": 20})
        services = result.get("results") or result.get("data") or []
        logger.info(f"✅ Fetched {len(services)} services for type {service_type_id}")
        return services
    except Exception as e:
        logger.error(f"❌ Failed to fetch services for type {service_type_id}: {e}")
        return []


def get_service_requirement(service: Dict) -> str:
    """
    Determine what resource a service requires.
    
    Args:
        service: Service dictionary with requirement flags
        
    Returns:
        "doctor", "specialist", "device", or "unknown"
    """
    if service.get("requires_doctor"):
        return "doctor"
    elif service.get("requires_specialist"):
        return "specialist"
    elif service.get("requires_device"):
        return "device"
    else:
        logger.warning(f"⚠️ Service {service.get('id')} has no requirement flags set")
        return "unknown"


async def fetch_resource_for_service(api_client, service: Dict, booking_state: Dict) -> Dict:
    """
    Fetch the appropriate resource (doctor/specialist/device) based on service requirements.
    
    Args:
        api_client: API client instance
        service: Service dictionary
        booking_state: Current booking state
        
    Returns:
        Dictionary with resource info and updated booking state
    """
    requirement = get_service_requirement(service)
    service_id = service.get("id")
    
    try:
        if requirement == "doctor":
            logger.info(f"📋 Service requires DOCTOR - fetching doctors for service {service_id}")
            result = await api_client.get(f"/services/{service_id}/doctors")
            resources = result.get("results") or result.get("data") or []
            
            if not resources:
                # Fallback: get all doctors
                result = await api_client.get("/doctors", params={"limit": 20})
                resources = result.get("results") or result.get("data") or []
            
            booking_state["doctors"] = resources
            booking_state["resource_type"] = "doctor"
            
            # Clear any stale specialist/device data since this service requires doctor
            if "specialist_id" in booking_state:
                logger.info(f"🧹 Clearing stale specialist_id")
                booking_state.pop("specialist_id", None)
                booking_state.pop("specialist_name", None)
            if "device_id" in booking_state:
                logger.info(f"🧹 Clearing stale device_id")
                booking_state.pop("device_id", None)
                booking_state.pop("device_name", None)
            
            logger.info(f"✅ Fetched {len(resources)} doctors")
            
        elif requirement == "specialist":
            logger.info(f"📋 Service requires SPECIALIST - fetching specialists")
            result = await api_client.get("/specialists", params={"limit": 20})
            resources = result.get("results") or result.get("data") or []
            booking_state["specialists"] = resources
            booking_state["resource_type"] = "specialist"
            
            # Clear any stale doctor/device data since this service requires specialist
            if "doctor_id" in booking_state:
                logger.info(f"🧹 Clearing stale doctor_id")
                booking_state.pop("doctor_id", None)
                booking_state.pop("doctor_name", None)
                booking_state.pop("doctor_name_en", None)
                booking_state.pop("doctor_selected", None)
            if "device_id" in booking_state:
                logger.info(f"🧹 Clearing stale device_id")
                booking_state.pop("device_id", None)
                booking_state.pop("device_name", None)
            
            logger.info(f"✅ Fetched {len(resources)} specialists")
            
        elif requirement == "device":
            logger.info(f"📋 Service requires DEVICE - matching devices by name from service name")
            
            # Extract device name from service name
            # Service name format: "جلسة ليزر بوكسر 150" → device name is "بوكسر"
            service_name = service.get("name_ar") or service.get("name", "")
            service_name_lower = service_name.lower()
            
            logger.info(f"📋 Service name: '{service_name}'")
            
            # Fetch all devices
            result = await api_client.get("/devices", params={"limit": 50})
            all_devices = result.get("results") or result.get("data") or []
            
            # Match devices whose name appears in the service name
            matched_devices = []
            for device in all_devices:
                device_name = (device.get("name") or "").lower()
                device_name_ar = (device.get("name_ar") or "").lower()
                
                # Check if device name appears in service name (either English or Arabic)
                if (device_name and device_name in service_name_lower) or \
                   (device_name_ar and device_name_ar in service_name_lower):
                    matched_devices.append(device)
                    logger.info(f"✅ Matched device: {device.get('name_ar') or device.get('name')} (ID: {device.get('id')})")
            
            # If no matches found, assign ALL available devices (user can choose)
            if not matched_devices:
                logger.warning(f"⚠️ No devices matched service name '{service_name}' - showing all available devices")
                matched_devices = all_devices
            
            if matched_devices:
                booking_state["devices"] = matched_devices
                booking_state["resource_type"] = "device"
                
                # Clear any stale doctor/specialist data since this service requires device
                if "doctor_id" in booking_state:
                    logger.info(f"🧹 Clearing stale doctor_id from previous booking")
                    booking_state.pop("doctor_id", None)
                    booking_state.pop("doctor_name", None)
                    booking_state.pop("doctor_name_en", None)
                    booking_state.pop("doctor_selected", None)
                if "specialist_id" in booking_state:
                    logger.info(f"🧹 Clearing stale specialist_id from previous booking")
                    booking_state.pop("specialist_id", None)
                    booking_state.pop("specialist_name", None)
                
                logger.info(f"✅ Found {len(matched_devices)} available devices for service")
            else:
                # No devices at all - this is a system error
                logger.error(f"❌ No devices available in the system!")
                return {"error": "No devices available"}
            
        else:
            logger.error(f"❌ Unknown service requirement: {requirement}")
            return {"error": "unknown_requirement"}
        
        return {"success": True, "requirement": requirement}
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch resource for service: {e}")
        return {"error": str(e)}


async def format_service_types_list(service_types: List[Dict], arabic_name: str) -> str:
    """
    CRITICAL: Use LLM to generate natural service types list (NO TEMPLATES!)
    
    Args:
        service_types: List of service type dictionaries
        arabic_name: User's name in Arabic
        
    Returns:
        LLM-generated formatted message string
    """
    from app.core.response_generator import get_response_generator
    response_gen = get_response_generator()
    
    if not service_types:
        return await response_gen.handle_no_service_types_available(
            user_name=arabic_name
        )
    
    # Format types as simple list
    types_list = [
        f"{i+1}. {t.get('name_ar') or t.get('name')}"
        for i, t in enumerate(service_types)
    ]
    
    return await response_gen.present_service_types_list(
        user_name=arabic_name,
        service_types=types_list
    )


async def format_services_list(services: List[Dict], arabic_name: str, type_name: str = None) -> str:
    """
    CRITICAL: Use LLM to generate natural service list presentation (NO TEMPLATES!)
    
    Args:
        services: List of service dictionaries
        arabic_name: User's name in Arabic
        type_name: Optional service type name
        
    Returns:
        LLM-generated formatted message string
    """
    from app.core.response_generator import get_response_generator
    response_gen = get_response_generator()
    
    if not services:
        return await response_gen.handle_no_services_available(
            user_name=arabic_name,
            service_type=type_name
        )
    
    # Format services as simple list
    services_list = [
        f"{i+1}. {s.get('name_ar') or s.get('name')} - {s.get('price', 'حسب الاستشارة')} ريال"
        for i, s in enumerate(services)
    ]
    
    return await response_gen.present_services_list(
        user_name=arabic_name,
        services=services_list,
        service_type=type_name
    )
