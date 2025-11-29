"""
Complete state schema for LangGraph booking flow.

Maps directly to the booking_state dict from original BookingAgent.
"""
from typing import TypedDict, Optional, List, Annotated, Literal
from datetime import datetime
import operator


class BookingState(TypedDict, total=False):
    """
    Complete state for booking conversation flow.
    
    This state model captures ALL fields from the original booking_state dict
    in BookingAgent to ensure zero functionality loss during migration.
    """
    # ==========================================
    # SESSION IDENTIFIERS
    # ==========================================
    session_id: str
    phone_number: str
    sender_name: str
    arabic_name: str
    
    # ==========================================
    # PATIENT INFORMATION
    # ==========================================
    patient_id: Optional[int]
    patient_verified: bool
    patient_data: Optional[dict]  # CRITICAL: Full patient data from router (includes id, name, phone, already_registered)
    registration: Optional[dict]  # For registration sub-flow
    name: Optional[str]  # CRITICAL: Confirmed patient name (from router confirmation)
    national_id: Optional[str]  # CRITICAL: Confirmed national ID (from router confirmation)
    gender: Optional[str]  # Patient gender (male/female)
    
    # ==========================================
    # SERVICE SELECTION
    # ==========================================
    service_types: Optional[List[dict]]
    selected_service_type_id: Optional[int]
    selected_service_type_name: Optional[str]
    services: Optional[List[dict]]
    displayed_services: Optional[List[dict]]
    service_id: Optional[int]
    service_name: Optional[str]
    
    # CRITICAL (Bug 24, 28 fix): Router passes these for state sync
    selected_service_id: Optional[int]  # Pre-selected service from Router
    selected_service_name: Optional[str]  # Pre-selected service name from Router
    selected_service: Optional[str]  # Alias for selected_service_name (backward compat)
    last_discussed_service: Optional[str]  # Service mentioned in conversation
    conversation_history: Optional[List[dict]]  # Full conversation history from Router
    last_bot_message: Optional[str]  # Last message sent by bot
    
    # ==========================================
    # RESOURCE SELECTION (doctor/device/specialist)
    # ==========================================
    resource_type: Optional[Literal["doctor", "device", "specialist"]]
    doctors: Optional[List[dict]]
    specialists: Optional[List[dict]]
    devices: Optional[List[dict]]
    doctor_id: Optional[int]
    doctor_name: Optional[str]
    doctor_name_en: Optional[str]
    specialist_id: Optional[int]
    specialist_name: Optional[str]
    device_id: Optional[int]
    device_name: Optional[str]
    
    # ==========================================
    # BOOKING DETAILS
    # ==========================================
    preferred_date: Optional[str]
    preferred_time: Optional[str]
    available_slots: Optional[List[dict]]
    awaiting_confirmation: bool
    booking_id: Optional[int]
    
    # ==========================================
    # FLOW CONTROL
    # ==========================================
    step: str
    started: bool
    is_pure_intent: bool
    _resuming: Optional[bool]  # Internal: Flag for resuming conversation
    _previous_step: Optional[str]  # Internal: Previous step before resuming
    
    # ==========================================
    # ERROR HANDLING
    # ==========================================
    last_error: Optional[dict]
    critical_failures: int
    message_repeat_count: int
    last_user_message: str
    doctor_search_failed: Optional[str]
    
    # ==========================================
    # MESSAGES (for LangGraph conversation tracking)
    # ==========================================
    messages: Annotated[List[dict], operator.add]
    
    # ==========================================
    # CURRENT USER MESSAGE
    # ==========================================
    current_message: str
    
    # ==========================================
    # LANGUAGE & CONTEXT
    # ==========================================
    language: Optional[Literal["ar", "en"]]
    
    # ==========================================
    # ROUTER CONTEXT
    # ==========================================
    router_intent: Optional[str]
    router_confidence: Optional[float]
