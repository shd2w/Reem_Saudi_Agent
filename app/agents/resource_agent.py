"""
Resource Agent - Information about services, doctors, and facilities
=====================================================================
Provides information about medical services, doctors, specialists, and facilities.

Features:
- Service information and listings
- Doctor information and availability
- Specialist listings
- Device and facility information
- Natural language queries

Author: Agent Orchestrator Team
Version: 1.0.0
"""

import time
from typing import Dict, Any, Optional, List
from loguru import logger

from ..api.agent_api import AgentApiClient
from ..utils.language_detector import detect_language
from ..utils.api_normalizer import normalize_api_response
from ..services.llm_response_generator import get_llm_response_generator


class ResourceAgent:
    """
    Professional resource agent for medical information queries.
    
    Handles:
    - Service information
    - Doctor listings
    - Specialist information
    - Device/facility queries
    """
    
    def __init__(self, session_key: str):
        self.session_key = session_key
        self.api_client = AgentApiClient()
        try:
            self.llm_generator = get_llm_response_generator()
        except Exception as e:
            logger.error(f"Failed to initialize LLM generator: {e}")
            self.llm_generator = None
    
    async def handle(self, payload: dict, context: dict = None) -> dict:
        """
        Handle resource information requests with conversation context.
        
        Args:
            payload: Message payload with user query
            context: Conversation context (history, sender info, etc.)
            
        Returns:
            Response dictionary with resource information
        """
        try:
            message = payload.get("message", "").lower()
            
            logger.info(f"ℹ️ Resource agent processing: '{message[:50]}...'")
            
            # Determine resource type (check both Arabic and English keywords)
            # Check for offers/promotions first (most specific)
            if any(word in message for word in ["offer", "offers", "promotion", "promotions", "deal", "deals", "عرض", "عروض", "خصم", "خصومات", "تخفيض"]):
                logger.info("🎁 Detected offers/promotions query - fetching services as offers")
                return await self._handle_services_query(message, context)
            
            elif any(word in message for word in ["service", "services", "treatment", "procedure", "خدمة", "خدمات"]):
                return await self._handle_services_query(message, context)
            
            elif any(word in message for word in ["doctor", "doctors", "physician", "دكتور", "دكاترة", "طبيب"]):
                return await self._handle_doctors_query(message, context)
            
            elif any(word in message for word in ["specialist", "specialists", "specialization", "أخصائي", "أخصائيين"]):
                return await self._handle_specialists_query(message, context)
            
            elif any(word in message for word in ["device", "devices", "equipment", "facility", "facilities", "أجهزة", "معدات"]):
                return await self._handle_devices_query(message, context)
            
            elif any(word in message for word in ["slot", "slots", "available", "availability", "time", "موعد", "مواعيد"]):
                return await self._handle_slots_query(message, context)
            
            else:
                # General information
                return await self._handle_general_info(message, context)
                
        except Exception as exc:
            logger.error(f"Resource agent error: {exc}", exc_info=True)
            return {
                "response": "Sorry, I encountered an error retrieving information. Please try again.",
                "intent": "resource",
                "status": "error",
                "error": str(exc)
            }
    
    async def _handle_services_query(self, message: str, context: dict = None) -> dict:
        """Handle services query with LLM-generated response"""
        try:
            # PERFORMANCE: Cache services for 5 minutes (they don't change often)
            cached_services = getattr(self, '_services_cache', None)
            cache_time = getattr(self, '_services_cache_time', 0)
            
            if cached_services and (time.time() - cache_time) < 300:  # 5 min cache
                logger.info("⚡ Using CACHED services (instant)")
                services = cached_services
            else:
                logger.info("📋 Fetching services list from API...")
                services_result = await self.api_client.get("/services", params={"limit": 20})
                services = services_result.get("results") or services_result.get("data") or []
                
                # Cache it
                self._services_cache = services
                self._services_cache_time = time.time()
                logger.info(f"✅ Retrieved {len(services)} services from API (cached for 5 min)")
            
            if not services:
                services = []
            
            # CRITICAL: Detect query type
            # 1. "وش الخدمات" / "what services" → Show parent categories ONLY
            # 2. "وش العروض" / "what offers" → Show selected subservices (LLM chooses 5-8)
            # 3. User selected parent (e.g., "بوتوكس", "ليزر") → Show subservices of that category
            
            message_lower = message.lower()
            is_offers_query = any(word in message_lower for word in ["عرض", "عروض", "offer", "offers", "خصم", "تخفيض"])
            is_general_services_query = any(word in message_lower for word in ["خدمات", "services"]) and not is_offers_query
            
            # Check if user selected a specific parent category
            selected_parent = None
            parent_categories = ["بوتوكس", "ليزر", "تقشير", "بشرة", "نضارة", "النضارة", "البشرة", "التقشير"]
            for category in parent_categories:
                if category in message_lower:
                    selected_parent = category
                    logger.info(f"🎯 User selected parent category: {category}")
                    break
            
            # CASE 1: General services query → Show parent categories only (11 items)
            if is_general_services_query and not selected_parent:
                logger.info("📂 Showing PARENT CATEGORIES only (no expansion)")
                # Filter out test services
                parent_services = [s for s in services if s.get("name", "").lower() not in ["zzzz", "test"]]
                
                # Get patient data for personalization
                patient_data = context.get("patient_data") if context else None
                previous_bookings = await self._get_patient_bookings(patient_data)
                
                data = {
                    "services": parent_services,
                    "show_categories_only": True,  # Signal to LLM
                    "patient_data": patient_data,
                    "previous_bookings": previous_bookings
                }
                
                if self.llm_generator:
                    response_text = await self.llm_generator.generate_response(
                        intent="resource",
                        user_message=message,
                        data=data,
                        context=context
                    )
                    return {
                        "response": response_text,
                        "intent": "resource",
                        "status": "success",
                        "resource_type": "parent_categories"
                    }
            
            # CASE 2 & 3: Offers OR specific parent selected → Expand to subservices
            logger.info(f"🔍 Expanding {len(services)} parent services to subservices...")
            all_subservices = []
            
            for parent_service in services[:10]:  # Limit to 10 parents to avoid too many API calls
                parent_id = parent_service.get("id")
                parent_name = parent_service.get("name_ar") or parent_service.get("name", "Unknown")
                
                # Try to get subservices for this parent
                try:
                    subservices_result = await self.api_client.get("/services", params={"service_type_id": parent_id, "limit": 20})
                    subservices = subservices_result.get("results") or subservices_result.get("data") or []
                    
                    if subservices:
                        logger.info(f"  📦 '{parent_name}' has {len(subservices)} subservices")
                        # Add each subservice with parent context
                        for subsvc in subservices:
                            subsvc["parent_name"] = parent_name
                            all_subservices.append(subsvc)
                    else:
                        # No subservices - treat parent as standalone service
                        logger.info(f"  📦 '{parent_name}' is standalone (no subservices)")
                        all_subservices.append(parent_service)
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not fetch subservices for '{parent_name}': {e}")
                    # On error, use parent as fallback
                    all_subservices.append(parent_service)
            
            logger.info(f"✅ Expanded to {len(all_subservices)} total subservices")
            
            # Use subservices instead of parent services
            services = all_subservices
            
            # CASE 3: If user selected a specific parent, filter to show only that parent's subservices
            if selected_parent:
                logger.info(f"🔍 Filtering subservices for parent: {selected_parent}")
                filtered = [s for s in services if selected_parent in (s.get("parent_name", "").lower() or s.get("name", "").lower())]
                if filtered:
                    services = filtered
                    logger.info(f"✅ Showing {len(services)} subservices for '{selected_parent}'")
                else:
                    logger.warning(f"⚠️ No subservices found for '{selected_parent}', showing all")
            
            # Check if user mentioned a specific service name
            message_lower = message.lower()
            matched_service = None
            for service in services:
                service_name = service.get("name", "").lower()
                if service_name and service_name in message_lower:
                    matched_service = service
                    logger.info(f"🔍 Found specific service: {service.get('name')}")
                    break
            
            # Get patient data for personalization
            patient_data = context.get("patient_data") if context else None
            previous_bookings = await self._get_patient_bookings(patient_data)
            
            # Prepare data for LLM
            data = {
                "services": services,  # Now contains subservices with gender info
                "matched_service": matched_service,
                "patient_data": patient_data,
                "previous_bookings": previous_bookings
            }
            
            # Let LLM generate the response
            if self.llm_generator:
                response_text = await self.llm_generator.generate_response(
                    intent="resource",
                    user_message=message,
                    context=context,
                    data=data,
                    sender_name=context.get("sender_name") if context else None
                )
            else:
                # Fallback if LLM not available
                response_text = "عذراً، لا أستطيع الرد حالياً. يرجى المحاولة لاحقاً."
            
            return {
                "response": response_text,
                "intent": "resource",
                "status": "success",
                "data": data,
                "displayed_services": services  # CRITICAL: Track for numbered selection
            }
            
        except Exception as e:
            logger.error(f"Services query error: {e}")
            # Let LLM handle error response too
            if self.llm_generator:
                response_text = await self.llm_generator.generate_response(
                    intent="resource",
                    user_message=message,
                    context=context,
                    data={"error": str(e)},
                    sender_name=context.get("sender_name") if context else None
                )
            else:
                response_text = "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى."
            
            return {
                "response": response_text,
                "intent": "resource",
                "status": "error"
            }
    
    async def _handle_doctors_query(self, message: str, context: dict = None) -> dict:
        """Handle doctors query with LLM-generated response"""
        try:
            # PERFORMANCE: Cache doctors for 5 minutes
            cached_doctors = getattr(self, '_doctors_cache', None)
            cache_time = getattr(self, '_doctors_cache_time', 0)
            
            if cached_doctors and (time.time() - cache_time) < 300:  # 5 min cache
                logger.info("⚡ Using CACHED doctors (instant)")
                doctors = cached_doctors
            else:
                logger.info("📋 Fetching doctors list from API...")
                doctors_result = await self.api_client.get("/doctors", params={"limit": 20})
                doctors = doctors_result.get("results") or doctors_result.get("data") or []
                
                # Cache it
                self._doctors_cache = doctors
                self._doctors_cache_time = time.time()
                logger.info(f"✅ Retrieved {len(doctors)} doctors from API (cached for 5 min)")
            
            # Get patient data for personalization
            patient_data = context.get("patient_data") if context else None
            previous_bookings = await self._get_patient_bookings(patient_data)
            
            # Prepare data for LLM
            data = {
                "doctors": doctors[:10],
                "patient_data": patient_data,
                "previous_bookings": previous_bookings
            }
            
            # Let LLM generate the response
            if self.llm_generator:
                response_text = await self.llm_generator.generate_response(
                    intent="resource",
                    user_message=message,
                    context=context,
                    data=data,
                    sender_name=context.get("sender_name") if context else None
                )
            else:
                response_text = "عذراً، لا أستطيع الرد حالياً. يرجى المحاولة لاحقاً."
            
            return {
                "response": response_text,
                "intent": "resource",
                "status": "success",
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Doctors query error: {e}")
            # Let LLM handle error response
            if self.llm_generator:
                response_text = await self.llm_generator.generate_response(
                    intent="resource",
                    user_message=message,
                    context=context,
                    data={"error": str(e)},
                    sender_name=context.get("sender_name") if context else None
                )
            else:
                response_text = "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى."
            
            return {
                "response": response_text,
                "intent": "resource",
                "status": "error"
            }
    
    async def _handle_devices_query(self, message: str, context: dict = None) -> dict:
        """Handle devices/facilities information query"""
        try:
            result = await self.api_client.get("/devices", params={"limit": 10})
            
            if result and result.get("data"):
                devices = result["data"]
                
                if not devices:
                    return {
                        "response": "I couldn't find device information at the moment. Please contact support.",
                        "intent": "resource",
                        "status": "no_data"
                    }
                
                # Format devices list
                device_list = []
                for i, device in enumerate(devices[:8]):
                    name = device.get("name", "Unknown")
                    device_list.append(f"• {name}")
                
                devices_text = "\n".join(device_list)
                
                return {
                    "response": f"🔬 Our Equipment & Facilities:\n\n{devices_text}\n\nWe have state-of-the-art medical equipment to serve you better!",
                    "intent": "resource",
                    "status": "success",
                    "devices": devices
                }
            
            return {
                "response": "I'm having trouble loading facility information. Please try again.",
                "intent": "resource",
                "status": "error"
            }
            
        except Exception as exc:
            logger.error(f"Devices query error: {exc}")
            return {
                "response": "Sorry, I couldn't retrieve facility information. Please try again.",
                "intent": "resource",
                "status": "error"
            }
    
    async def _handle_slots_query(self, message: str, context: dict = None) -> dict:
        """Handle availability/slots query"""
        return {
            "response": "To check available time slots, please start a booking by telling me which service you'd like to book. I'll show you all available times!",
            "intent": "resource",
            "status": "redirect_to_booking"
        }
    
    async def _handle_general_info(self, message: str = "", context: dict = None) -> dict:
        """Handle general information request with LLM-generated personalized response"""
        try:
            # Get patient data and booking history for personalization
            patient_data = context.get("patient_data") if context else None
            logger.info(f"🔍 PERSONALIZATION CHECK: patient_data={patient_data}")
            previous_bookings = await self._get_patient_bookings(patient_data)
            logger.info(f"🔍 PERSONALIZATION CHECK: previous_bookings={len(previous_bookings) if previous_bookings else 0} found")
            
            # Prepare data for LLM with patient context
            data = {
                "patient_data": patient_data,
                "previous_bookings": previous_bookings,
                "is_returning_patient": bool(previous_bookings and len(previous_bookings) > 0),
                "is_registered": bool(patient_data and patient_data.get("already_registered"))
            }
            
            # Let LLM generate personalized response
            if self.llm_generator:
                try:
                    response_text = await self.llm_generator.generate_response(
                        intent="chitchat",  # Use chitchat for general greetings
                        user_message=message or "general info",
                        context=context,
                        data=data,
                        sender_name=context.get("sender_name") if context else None
                    )
                except Exception as llm_error:
                    logger.error(f"LLM generation failed for general info: {llm_error}")
                    response_text = "أهلاً وسهلاً! كيف أقدر أساعدك اليوم؟"
            else:
                response_text = "أهلاً وسهلاً! كيف أقدر أساعدك اليوم؟"
            
            return {
                "response": response_text,
                "intent": "resource",
                "status": "general_info"
            }
        except Exception as exc:
            logger.error(f"General info error: {exc}", exc_info=True)
            # Fallback to simple response
            language = detect_language(message) if message else "arabic"
            if language == "arabic":
                response = "🏥 أهلاً وسهلاً في مركز وجن الطبي! وش أقدر أساعدك فيه؟"
            else:
                response = "🏥 Welcome to Wajan Medical Center! How can I help you?"
            
            return {
                "response": response,
                "intent": "resource",
                "status": "general_info"
            }
    
    
    async def _get_patient_bookings(self, patient_data: dict = None) -> Optional[list]:
        """Get patient's previous bookings for personalization"""
        try:
            if not patient_data or not patient_data.get("id"):
                return None
            
            patient_id = patient_data.get("id")
            bookings_result = await self.api_client.get("/booking", params={"patient_id": patient_id, "limit": 5})
            
            if bookings_result and bookings_result.get("results"):
                previous_bookings = bookings_result["results"]
                logger.info(f"✅ Found {len(previous_bookings)} previous bookings for patient {patient_id}")
                return previous_bookings
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not fetch previous bookings: {e}")
            return None
    
