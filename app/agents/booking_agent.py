# -*- coding: utf-8 -*-
"""
Booking Agent - Professional appointment booking with conversational flow
=========================================================================
Handles complete booking workflow with backend API integration.

Features:
- Multi-turn conversation for booking details
- Service and doctor selection
- Available slot checking
- Booking creation and confirmation
- Booking rescheduling
- Booking retrieval and management
- Natural language processing
- Session state management

Author: Agent Orchestrator Team
Version: 1.0.0
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
import redis.asyncio
from loguru import logger
from ..config import settings
from ..api.agent_api import AgentApiClient
from ..api.api_operations import get_api_operations
from ..api.wasender_client import WaSenderClient
from ..memory.session_manager import SessionManager
from ..utils.language_detector import detect_language
from ..services.llm_response_generator import get_llm_response_generator
from ..utils.name_transliterator import get_arabic_name_or_fallback, transliterate_full_name
from ..core.llm_reasoner import get_llm_reasoner
from .booking_helpers import show_available_time_slots, request_booking_confirmation, complete_booking_with_details
from .service_flow_helpers import (
    fetch_service_types, 
    fetch_services_by_type, 
    get_service_requirement,
    fetch_resource_for_service,
    format_service_types_list,
    format_services_list
)

# SHARED CACHE: Persists across all BookingAgent instances for true caching
# This eliminates 0.7s API fetch time on subsequent requests
_SHARED_CACHE = {
    "services": {"data": None, "time": 0},
    "doctors": {"data": None, "time": 0}
}


class BookingAgent:
    """
    Handles all booking-related operations with natural language understanding.
    
    Architecture: Session-based Instance Pool (NOT true singleton)
    - One BookingAgent instance per session (e.g., per WhatsApp user)
    - Shared class-level resources (circuit breaker, cache, Redis client)
    - Automatic cleanup of stale sessions (30 min TTL)
    
    Flow:
    1. Identify/verify patient
    2. Collect booking requirements (service, doctor, date)
    3. Show available slots
    4. Create booking
    5. Send confirmation
    """
    
    # TIMEOUT CONFIGURATION (seconds)
    RESOURCE_FETCH_TIMEOUT = 10.0  # Timeout for fetching doctors/specialists/devices
    TIME_SLOTS_FETCH_TIMEOUT = 8.0  # Timeout for fetching available time slots
    API_CALL_TIMEOUT = 5.0  # Timeout for general API calls
    
    # Class-level shared state (shared across ALL instances)
    _instances = {}  # Session-based instance pool: {session_id: BookingAgent}
    _initialized = {}  # Track initialization per session: {session_id: True}
    _last_access = {}  # Track last access time per session for cleanup: {session_id: timestamp}
    _service_circuit_breaker = {}  # Shared circuit breaker for failing services
    _request_cache = {}  # Shared request deduplication cache
    _failure_counters = {}  # Track consecutive failures per session for recovery
    _shared_redis_client = None  # Shared Redis client (true singleton)
    
    # Session cleanup config
    SESSION_TTL = 1800  # 30 minutes - cleanup sessions older than this
    
    def __new__(cls, session_id: str):
        """
        Session-based instance pool: One instance per session ID.
        NOT a true singleton - creates multiple instances (one per user session).
        """
        # Cleanup stale sessions before creating new ones
        cls._cleanup_stale_sessions()
        
        if session_id not in cls._instances:
            cls._instances[session_id] = super(BookingAgent, cls).__new__(cls)
            logger.debug(f"📦 Created new BookingAgent instance for session {session_id} (pool size: {len(cls._instances)})")
        
        # Update last access time
        cls._last_access[session_id] = time.time()
        
        return cls._instances[session_id]
    
    @classmethod
    def _cleanup_stale_sessions(cls):
        """Remove instances that haven't been accessed in SESSION_TTL seconds"""
        current_time = time.time()
        stale_sessions = [
            session_id for session_id, last_time in cls._last_access.items()
            if current_time - last_time > cls.SESSION_TTL
        ]
        
        if stale_sessions:
            for session_id in stale_sessions:
                # Remove from all tracking dicts
                cls._instances.pop(session_id, None)
                cls._initialized.pop(session_id, None)
                cls._last_access.pop(session_id, None)
                cls._failure_counters.pop(session_id, None)
            
            logger.info(f"🧹 Cleaned up {len(stale_sessions)} stale session(s) (active sessions: {len(cls._instances)})")
    
    @classmethod
    def _get_shared_redis_client(cls):
        """Get or create shared Redis client (true singleton)"""
        if cls._shared_redis_client is None:
            # Use from_url() since settings only has redis_url
            cls._shared_redis_client = redis.asyncio.from_url(
                settings.redis_url,
                decode_responses=True
            )
            logger.info("✅ Created shared Redis client for all BookingAgent instances")
        return cls._shared_redis_client
    
    def __init__(self, session_id: str, api_client=None):
        """Initialize booking agent for a session"""
        # Only initialize if not already initialized
        if session_id in BookingAgent._initialized:
            logger.debug(f"♻️ Reusing existing BookingAgent for session {session_id}")
            self.session_id = session_id
            self.session_key = session_id  # Ensure session_key is always set
            # Ensure critical attributes exist (for instances created before these were added)
            if not hasattr(self, 'session_manager'):
                self.session_manager = SessionManager()
                logger.debug("⚙️ Added missing session_manager to existing instance")
            if not hasattr(self, 'llm_reasoner'):
                self.llm_reasoner = get_llm_reasoner()
                logger.debug("⚙️ Added missing llm_reasoner to existing instance")
            return
        
        logger.info(f"🆕 Initializing NEW BookingAgent for session {session_id}")
        BookingAgent._initialized[session_id] = True
        
        self.session_id = session_id
        self.session_key = session_id  # For compatibility with _save_booking_state
        self.api_client = api_client or AgentApiClient()  # Shared singleton
        self.llm_reasoner = get_llm_reasoner()  # Shared singleton
        self.session_manager = SessionManager()  # Shared singleton
        
        # Use shared Redis client (NOT per-instance)
        self.redis_client = BookingAgent._get_shared_redis_client()
        
        logger.info(f"✅ BookingAgent initialized for session {session_id} (session pool: {len(BookingAgent._instances)} active)")
    
    def _check_request_duplicate(self, phone_number: str, message: str) -> Optional[dict]:
        """Check if this is a duplicate request within 30 seconds"""
        import time
        import hashlib
        
        # Create cache key
        cache_key = hashlib.md5(f"{phone_number}:{message.lower().strip()}".encode()).hexdigest()
        
        # Check cache
        if cache_key in BookingAgent._request_cache:
            cached = BookingAgent._request_cache[cache_key]
            time_since = time.time() - cached["timestamp"]
            
            # If request is within 30 seconds, it's a duplicate
            if time_since < 30:
                logger.warning(f"🔁 DUPLICATE REQUEST detected: '{message}' from {phone_number} ({time_since:.1f}s ago)")
                return cached["response"]
        
        return None
    
    def _cache_request(self, phone_number: str, message: str, response: dict):
        """Cache request response for duplicate detection"""
        import time
        import hashlib
        
        cache_key = hashlib.md5(f"{phone_number}:{message.lower().strip()}".encode()).hexdigest()
        BookingAgent._request_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Clean old cache entries (older than 60 seconds)
        current_time = time.time()
        keys_to_delete = [
            key for key, value in BookingAgent._request_cache.items()
            if current_time - value["timestamp"] > 60
        ]
        for key in keys_to_delete:
            del BookingAgent._request_cache[key]
        
        logger.debug(f"💾 Cached request: '{message[:30]}...' (cache size: {len(BookingAgent._request_cache)})")
    
    def _check_service_circuit_breaker(self, service_id: int) -> tuple[bool, Optional[str]]:
        """
        Check if service is blocked by circuit breaker.
        Returns: (is_blocked, reason_message)
        """
        import time
        
        if service_id not in BookingAgent._service_circuit_breaker:
            return False, None
        
        breaker = BookingAgent._service_circuit_breaker[service_id]
        current_time = time.time()
        
        # Check if service is currently blocked
        if breaker.get("blocked_until", 0) > current_time:
            remaining = int(breaker["blocked_until"] - current_time)
            logger.warning(f"🚫 CIRCUIT BREAKER: Service {service_id} is blocked ({remaining}s remaining)")
            return True, f"هالخدمة مو متاحة مؤقتاً ({remaining} ثانية)\nجرب خدمة ثانية"
        
        # Auto-recovery: Reset if cooldown period passed
        if breaker.get("blocked_until", 0) < current_time and breaker.get("failures", 0) > 0:
            logger.info(f"✅ CIRCUIT BREAKER: Service {service_id} recovered (auto-reset)")
            del BookingAgent._service_circuit_breaker[service_id]
        
        return False, None
    
    def _record_service_failure(self, service_id: int):
        """Record service failure and trip circuit breaker if needed"""
        import time
        
        if service_id not in BookingAgent._service_circuit_breaker:
            BookingAgent._service_circuit_breaker[service_id] = {
                "failures": 0,
                "first_failure": time.time(),
                "last_failure": time.time()
            }
        
        breaker = BookingAgent._service_circuit_breaker[service_id]
        breaker["failures"] += 1
        breaker["last_failure"] = time.time()
        
        # Trip circuit breaker after 3 failures within 5 minutes
        time_since_first = time.time() - breaker["first_failure"]
        if breaker["failures"] >= 3 and time_since_first < 300:
            # Block service for 5 minutes
            breaker["blocked_until"] = time.time() + 300
            logger.error(f"🚫 CIRCUIT BREAKER TRIPPED: Service {service_id} blocked for 5 minutes ({breaker['failures']} failures)")
        else:
            logger.warning(f"⚠️ Service {service_id} failure recorded ({breaker['failures']} failures)")
    
    def _record_service_success(self, service_id: int):
        """Record service success and reset circuit breaker"""
        if service_id in BookingAgent._service_circuit_breaker:
            logger.info(f"✅ CIRCUIT BREAKER: Service {service_id} succeeded - resetting failure count")
            del BookingAgent._service_circuit_breaker[service_id]
    
    def _record_failure(self, session_id: str):
        """Record consecutive failure for this session"""
        if session_id not in BookingAgent._failure_counters:
            BookingAgent._failure_counters[session_id] = {"count": 0, "last_failure": 0}
        
        BookingAgent._failure_counters[session_id]["count"] += 1
        BookingAgent._failure_counters[session_id]["last_failure"] = time.time()
        
        failure_count = BookingAgent._failure_counters[session_id]["count"]
        logger.warning(f"⚠️ FAILURE TRACKER: Session {session_id} has {failure_count} consecutive failures")
        
        return failure_count
    
    def _reset_failure_counter(self, session_id: str):
        """Reset failure counter after success"""
        if session_id in BookingAgent._failure_counters:
            logger.info(f"✅ FAILURE TRACKER: Session {session_id} reset (was {BookingAgent._failure_counters[session_id]['count']} failures)")
            del BookingAgent._failure_counters[session_id]
    
    def _should_trigger_recovery(self, session_id: str) -> bool:
        """Check if we should trigger recovery mode (3+ failures)"""
        if session_id not in BookingAgent._failure_counters:
            return False
        
        failure_data = BookingAgent._failure_counters[session_id]
        failure_count = failure_data["count"]
        time_since_last = time.time() - failure_data["last_failure"]
        
        # Trigger recovery if 3+ failures within 5 minutes
        if failure_count >= 3 and time_since_last < 300:
            logger.error(f"🚨 RECOVERY MODE: Session {session_id} has {failure_count} failures - triggering recovery")
            return True
        
        return False
    
    def _trigger_recovery(self, session_id: str, arabic_name: str) -> dict:
        """
        Trigger recovery mode - reset everything and guide user.
        Note: Keeps phase as "booking" to maintain conversation context (Issue #12).
        """
        logger.error(f"🔧 RECOVERY TRIGGERED for session {session_id}")
        
        # Clear corrupted state but preserve phase
        try:
            booking_state = self._load_booking_state()
            booking_state.clear()
            booking_state["started"] = False
            
            # Save state and preserve booking phase (Issue #12)
            session_data = self.session_manager.get(self.session_key) or {}
            session_data["booking_state"] = booking_state
            session_data["booking_step"] = None
            session_data["current_phase"] = "booking"  # Keep in booking phase during recovery
            self.session_manager.put(self.session_key, session_data, ttl_minutes=120)
            
            logger.info("✅ Cleared corrupted booking state (phase preserved as 'booking')")
        except Exception as e:
            logger.error(f"Error clearing state during recovery: {e}", exc_info=True)
        
        # Reset failure counter
        self._reset_failure_counter(session_id)
        
        # Clear instance from memory (force re-initialization)
        if session_id in BookingAgent._instances:
            del BookingAgent._instances[session_id]
            logger.info("✅ Cleared agent instance from memory")
        
        if session_id in BookingAgent._initialized:
            del BookingAgent._initialized[session_id]
            logger.info("✅ Reset initialization flag")
        
        # Return helpful recovery message
        return {
            "response": f"""عزيزي {arabic_name} 🙏

حصلت مشكلة تقنية وأعدت ضبط النظام

**خياراتك الحين:**

🔹 **حجز موعد جديد**
   اكتب: حجز أو موعد

🔹 **شوف مواعيدك**
   اكتب: مواعيدي

🔹 **تواصل معنا**
   📞 اتصل: 920033304

آسف على الإزعاج، وجاهز أساعدك الحين 💪""",
            "intent": "booking",
            "status": "recovered"
        }
    
    async def _verify_patient_registration(
        self, phone_number: str, sender_name: str, arabic_name: str, booking_state: dict
    ) -> dict:
        """
        Verify patient is registered before allowing booking.
        If not registered, start registration flow.
        
        Returns:
            dict: Registration flow response or None if patient verified
        """
        try:
            # Check if patient exists in database
            logger.info(f"🔍 Checking if patient {phone_number} is registered...")
            
            # Search for patient by phone
            search_result = await self.api_client.get(f"/patients/search?phone={phone_number}")
            
            if search_result and search_result.get("data"):
                # Patient found - continue with booking
                patient = search_result["data"][0] if isinstance(search_result["data"], list) else search_result["data"]
                booking_state["patient_id"] = patient.get("id")
                booking_state["patient_verified"] = True
                booking_state["step"] = "initial"  # Continue to service selection
                self._save_booking_state(booking_state)
                logger.info(f"✅ Patient verified: {patient.get('name')} (ID: {patient.get('id')})")
                return None  # Continue with booking flow
            else:
                # Patient not found - start registration
                logger.info(f"⚠️ Patient not found - starting registration flow")
                booking_state["registration"] = {
                    "phone": phone_number,
                    "sender_name": sender_name
                }
                booking_state["step"] = "registration_name"
                self._save_booking_state(booking_state)
                
                return {
                    "response": f"""مرحباً {arabic_name}! 🤍

عشان أقدر أحجز لك، أحتاج أسجل معلوماتك الأساسية

**خليني أخذ منك:**

١. الاسم الكامل (الأول والأخير)
٢. رقم الهوية الوطنية
٣. تأكيد رقم جوالك: {phone_number}

ابدأ معي بالاسم الكامل؟ 📝""",
                    "intent": "booking",
                    "status": "registration_required"
                }
                
        except Exception as e:
            logger.error(f"❌ Error verifying patient: {e}", exc_info=True)
            # On error, allow booking to continue (fail open)
            booking_state["patient_verified"] = False
            booking_state["step"] = "initial"
            self._save_booking_state(booking_state)
            return None
    
    async def handle(self, payload: dict, context: dict = None) -> dict:
        """
        Handle booking-related requests with conversational flow and context.
        
        Args:
            payload: Message payload with user message and context
            context: Conversation context (history, sender info, etc.)
            
        Returns:
            Response dictionary with booking status and message
        """
        try:
            message = payload.get("message", "").strip()
            phone_number = payload.get("phone", "")
            sender_name = payload.get("sender_name", "Friend")
            session_id = payload.get("session_id", f"whatsapp:{phone_number}")
            is_pure_intent = payload.get("is_pure_intent", False)  # From router
            
            # Get Arabic transliteration of name
            arabic_name = transliterate_full_name(sender_name) if sender_name != "Friend" else "حبيبنا"
            
            # CRITICAL: Check if recovery mode should be triggered BEFORE any processing
            # This prevents infinite loops from corrupted state
            if self._should_trigger_recovery(self.session_id):
                logger.error(f"🚨 PRE-CHECK: Recovery mode triggered before processing")
                return self._trigger_recovery(self.session_id, arabic_name)
            
            # CRITICAL: Load booking state from session at the start
            booking_state = self._load_booking_state()
            
            # CRITICAL: Check if we're in an error state and handle recovery
            current_step = booking_state.get("step", "")
            if current_step and current_step.startswith("error_at_"):
                last_error = booking_state.get("last_error", {})
                previous_step = last_error.get("previous_step", "unknown")
                failure_count = last_error.get("failure_count", 0)
                
                logger.warning(f"🔄 RECOVERY: Detected error state (step={current_step}, failures={failure_count})")
                
                # Check if user is trying to cancel/restart
                if any(word in message.lower() for word in ["إلغاء", "cancel", "الغاء", "restart", "reset"]):
                    logger.info("✅ User requested restart - clearing error state")
                    booking_state.clear()
                    booking_state["started"] = False
                    self._save_booking_state(booking_state)
                    self._reset_failure_counter(self.session_id)
                    
                    return {
                        "response": f"تمام يا {arabic_name}! 🔄\n\nبدأنا من جديد\n\nاكتب 'حجز' عشان تحجز موعد 📅",
                        "intent": "booking",
                        "status": "reset"
                    }
                else:
                    # User is trying again - restore to previous step for retry
                    logger.info(f"🔄 Restoring from error state: {current_step} → {previous_step}")
                    booking_state["step"] = previous_step
                    booking_state.pop("last_error", None)
                    self._save_booking_state(booking_state)
                    # Continue processing with restored state
            
            # IDEMPOTENCY CHECK: Detect duplicate requests
            duplicate_response = self._check_request_duplicate(phone_number, message)
            if duplicate_response:
                logger.info(f"⚡ Returning cached response for duplicate request")
                return duplicate_response
            
            # Extract router intent and confidence (from context, not payload)
            router_intent = (context or {}).get('classified_intent') or payload.get('intent', 'booking')
            router_confidence = (context or {}).get('intent_confidence') or payload.get('confidence', 0.0)
            
            # LOG CONVERSATION CONTEXT for observability
            context_step = context.get("booking_step") if context else None
            state_step = booking_state.get("step")
            # Use context step if available (from router), otherwise use state step
            step = context_step if context_step is not None else state_step
            
            # CLEAR LOGGING: Distinguish between router-context, booking-state, and resolved values
            logger.info(f"📅 Booking agent processing: '{message[:50]}...' | ROUTER_INTENT: {router_intent} ({router_confidence:.2f}) | STEP: {step} [from_router={context_step}, from_state={state_step}], booking_active={booking_state.get('started', False)}")
            
            # Determine booking action (Arabic + English keywords)
            if any(word in message for word in ["reschedule", "change", "modify", "تعديل", "غير", "عدل"]):
                return await self._handle_reschedule(payload, booking_state)
            
            elif any(word in message for word in ["cancel", "delete", "إلغاء", "ألغي", "احذف"]):
                return await self._handle_cancel(payload, booking_state)
            
            elif any(word in message for word in ["check", "view", "show", "my booking", "مواعيدي", "موعدي", "شوف"]):
                return await self._handle_view_bookings(phone_number)
            
            else:
                # LOOP DETECTION: Check if user is stuck (sending same message repeatedly)
                repeat_count = booking_state.get("message_repeat_count", 0)
                if repeat_count >= 3:
                    logger.warning(f"🔁 User is stuck in a loop - offering help")
                    
                    current_step = booking_state.get("step")
                    
                    # Clear loop counter and update step to indicate help was offered
                    booking_state["message_repeat_count"] = 0
                    booking_state["step"] = f"{current_step}_help"  # Mark that help was offered
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"🔄 State updated: step={current_step}_help (loop detected, help offered)")
                    
                    # Offer contextual help based on current step
                    if current_step == "awaiting_service":
                        return {
                            "response": f"يا {arabic_name}، شكلك محتار 😅\n\nاكتب رقم الخدمة اللي تبغاها\nمثلاً: 1 أو 2\n\nأو اكتب اسم الخدمة مباشرة 📋",
                            "intent": "booking",
                            "status": f"{current_step}_help"  # Match the step
                        }
                    elif current_step in ["awaiting_doctor", "awaiting_device", "awaiting_specialist"]:
                        return {
                            "response": f"يا {arabic_name}، شكلك محتار 😅\n\nاكتب رقم أو اسم الخيار اللي تبغاه\nمثلاً: 1 أو 2\n\nأو اكتب 'أي واحد' وأنا أختار لك 👌",
                            "intent": "booking",
                            "status": f"{current_step}_help"  # Match the step
                        }
                    else:
                        booking_state["step"] = "started"  # Reset to beginning
                        self._save_booking_state(booking_state)
                        logger.info(f"🔄 State reset: step=started (user stuck, offering restart)")
                        
                        return {
                            "response": f"يا {arabic_name}، شكلك محتار 😅\n\nاكتب 'خدمات' عشان تشوف الخدمات\nأو اكتب 'إلغاء' عشان نبدأ من جديد 🙏",
                            "intent": "booking",
                            "status": "started"  # Match the step
                        }
            
            # Extract information from natural language
            await self._extract_and_update_booking_info(message, booking_state, phone_number, is_pure_intent)
            
            # Continue with new booking flow (pass payload and context)
            response = await self._handle_new_booking(payload, booking_state, phone_number, sender_name, context)
            
            # Cache response for duplicate detection
            self._cache_request(phone_number, message, response)
            
            # Reset failure counter on success
            self._reset_failure_counter(self.session_id)
            
            return response
                
        except Exception as exc:
            # COMPREHENSIVE ERROR LOGGING (Issue #13)
            import traceback
            import sys
            
            # Get full exception info
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # Generate error ID for correlation (Issue #32)
            import hashlib
            error_id = hashlib.md5(f"{exc_type.__name__}:{str(exc)[:30]}:{time.time()}".encode()).hexdigest()[:8]
            
            # Log comprehensive error details with structured markers (Issue #31)
            logger.error(f"🚨 ERROR_START [ID:{error_id}] ═══════════════════════════════")
            logger.error(f"📍 Error Type: {exc_type.__name__}")
            logger.error(f"📍 Error Message: {str(exc)}")
            logger.error(f"📍 Error ID: {error_id} (for correlation)")
            logger.error(f"📍 Session ID: {self.session_id}")
            logger.error(f"📍 Message: '{message[:100]}...'")
            logger.error(f"📍 Phone: {phone_number}")
            logger.error(f"📍 Sender: {sender_name}")
            
            # Log local variables at error point
            try:
                frame = exc_traceback.tb_frame
                local_vars = {k: str(v)[:200] for k, v in frame.f_locals.items() 
                             if not k.startswith('_') and k not in ['self', 'cls']}
                logger.error(f"📍 Local Variables: {local_vars}")
            except:
                logger.error(f"📍 Could not extract local variables")
            
            # Log booking state at error
            try:
                error_booking_state = self._load_booking_state()
                logger.error(f"📍 Booking State: step={error_booking_state.get('step')}, started={error_booking_state.get('started')}")
                logger.error(f"📍 Booking Progress: service={error_booking_state.get('service_name')}, doctor={error_booking_state.get('doctor_name')}")
            except:
                logger.error(f"📍 Could not load booking state")
            
            # Check for related errors (Issue #32)
            related_errors = []
            if "not defined" in str(exc) or "has no attribute" in str(exc):
                logger.error(f"📍 Error Category: INITIALIZATION/ATTRIBUTE ERROR")
                related_errors.append("Possible initialization failure cascade")
            
            # Log full stack trace
            logger.error(f"📍 Stack Trace:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logger.error(f"  {line}")
            
            if related_errors:
                logger.error(f"📍 Related Issues: {', '.join(related_errors)}")
            
            logger.error(f"🚨 ERROR_END [ID:{error_id}] ═══════════════════════════════")
            
            # Record failure
            failure_count = self._record_failure(self.session_id)
            
            # Check if we should trigger recovery mode
            if self._should_trigger_recovery(self.session_id):
                logger.error(f"🚨 CATASTROPHIC FAILURE: {failure_count} consecutive failures - triggering full recovery")
                return self._trigger_recovery(self.session_id, arabic_name)
            
            # CRITICAL: Preserve state context AND phase on error (Issue #12)
            # Load current state to see where we were
            try:
                error_booking_state = self._load_booking_state()
                previous_step = error_booking_state.get("step", "unknown")
                
                # Set error step that preserves context
                error_booking_state["step"] = f"error_at_{previous_step}"
                error_booking_state["last_error"] = {
                    "message": str(exc),
                    "timestamp": time.time(),
                    "failure_count": failure_count,
                    "previous_step": previous_step,
                    "error_type": exc_type.__name__,
                    "stack_trace": traceback.format_exc()[:500]  # First 500 chars
                }
                
                # CRITICAL: Preserve phase as "booking" - don't reset to discovery (Issue #12)
                session_data = self.session_manager.get(self.session_key) or {}
                session_data["current_phase"] = "booking"  # Keep in booking phase
                session_data["booking_state"] = error_booking_state
                session_data["booking_step"] = error_booking_state["step"]
                # Preserve topic if it exists
                if error_booking_state.get("service_name"):
                    session_data["current_topic"] = error_booking_state["service_name"]
                
                self.session_manager.put(self.session_key, session_data, ttl_minutes=120)
                
                logger.warning(f"💾 Error state saved: step=error_at_{previous_step}, phase=booking (preserved), failure_count={failure_count}")
                
            except Exception as state_exc:
                logger.error(f"Failed to save error state: {state_exc}", exc_info=True)
                # Even if we can't save state, continue with error response
            
            # Return specific error message with action based on failure count
            # CRITICAL: Detect what action user was trying to avoid suggesting same action
            message_lower = message.lower()
            was_booking_attempt = any(word in message_lower for word in ['حجز', 'احجز', 'أحجز', 'موعد', 'book', 'booking'])
            was_viewing_attempt = any(word in message_lower for word in ['مواعيدي', 'موعدي', 'حجوزاتي', 'my bookings', 'appointments'])
            was_cancel_attempt = any(word in message_lower for word in ['إلغاء', 'الغاء', 'cancel', 'ألغي'])
            
            language = detect_language(message)
            if language == "arabic":
                if failure_count == 1:
                    # First failure - gentle with SAFE alternatives (not what they just tried)
                    if was_booking_attempt:
                        response = f"يا عيني يا {arabic_name} 😅\n\nصار خطأ بسيط\n\n**جرب بدل كذا:**\n• اكتب 'خدمات' عشان تشوف الخدمات\n• أو اكتب 'مواعيدي' عشان تشوف حجوزاتك\n• أو اتصل: 920033304 📞"
                    elif was_viewing_attempt:
                        response = f"يا عيني يا {arabic_name} 😅\n\nصار خطأ بسيط\n\n**جرب بدل كذا:**\n• اكتب 'خدمات' للاستفسار\n• أو اتصل: 920033304 📞"
                    else:
                        response = f"يا عيني يا {arabic_name} 😅\n\nصار خطأ بسيط\n\nجرب مرة ثانية أو اتصل: 920033304 📞"
                        
                elif failure_count == 2:
                    # Second failure - focus on alternatives, DON'T suggest retry
                    response = f"يا {arabic_name}، آسف على المشكلة 😞\n\nالنظام عنده مشكلة مؤقتة\n\n**أفضل الخيارات:**\n• اتصل مباشرة: 920033304 📞\n• أو انتظر 5 دقائق وجرب مرة ثانية\n\n⚠️ لا تعيد نفس الطلب الحين"
                else:
                    # Third failure - will trigger recovery on next attempt, ONLY support
                    response = f"يا {arabic_name}، في مشكلة مستمرة 😔\n\n**الحل الأفضل:**\n📞 اتصل: 920033304\n\nفريقنا جاهز يساعدك\nمن ٩ صباحاً إلى ٩ مساءً\n\nمعذرة على الإزعاج 🙏"
            else:
                # English - also avoid circular suggestions
                if was_booking_attempt:
                    response = "Sorry, booking isn't working right now.\n\nTry instead:\n• Type 'services' to view services\n• Type 'my bookings' to view your bookings\n• Call: 920033304 📞"
                else:
                    response = "Sorry, something went wrong.\n\nTry:\n• Type 'services' to view services\n• Call: 920033304 📞"
            
            return {
                "response": response,
                "intent": "booking",
                "status": "error",
                "error": str(exc),
                "failure_count": failure_count
            }
    
    async def _handle_new_booking(
        self,
        payload: dict,
        booking_state: dict,
        phone_number: str,
        sender_name: str,
        context: dict = None
    ) -> dict:
        """
        Handle new booking creation with multi-turn conversation.
        
        Collects:
        1. Patient verification
        2. Service selection
        3. Doctor/specialist selection (optional)
        4. Date preference
        5. Time slot selection
        """
        try:
            message = payload.get("message", "").lower()
            language = detect_language(message)
            
            # CRITICAL: Convert English names to Arabic or use fallback
            # This maintains professionalism - no English in Arabic text!
            arabic_name = get_arabic_name_or_fallback(sender_name, use_generic=True)
            logger.info(f"🔤 Name: '{sender_name}' → Arabic: '{arabic_name}'")
            
            # CRITICAL: Get router intent to avoid re-extracting keywords as data
            router_intent = context.get("classified_intent") if context else None
            is_pure_intent_keyword = router_intent == "booking" and message.strip() in ['حجز', 'احجز', 'أحجز', 'موعد', 'book', 'booking', 'appointment']
            
            # CRITICAL: Check if user is confirming/restarting booking in active session
            booking_active = booking_state.get("started", False)
            current_step = booking_state.get("step", "unknown")
            conversation_turn = context.get("conversation_turn", 0) if context else 0
            
            if is_pure_intent_keyword and booking_active and conversation_turn > 1:
                logger.info(f"✅ STATE MACHINE: User said '{message}' in active booking (step={current_step}, turn={conversation_turn})")
                logger.info(f"✅ INTERPRETATION: Confirmation/continuation, NOT entity extraction")
                # User is confirming they want to proceed with booking
                # Don't treat this as doctor/service name
            
            # Initialize booking state if empty OR if step is missing
            is_new_booking = False
            if not booking_state or len(booking_state) == 0:
                booking_state = {"started": True, "step": "checking_patient"}
                self._save_booking_state(booking_state)
                is_new_booking = True
                logger.info("⚡ New booking - checking patient registration first")
            elif booking_state.get("started") and not booking_state.get("step"):
                # CRITICAL: Fix state inconsistency - active booking must have a step
                logger.warning(f"⚠️ STATE FIX: Active booking has no step, setting to 'checking_patient'")
                booking_state["step"] = "checking_patient"
                self._save_booking_state(booking_state)
            
            # PATIENT VERIFICATION: Check if patient is registered before booking
            if is_new_booking or current_step == "checking_patient":
                patient_check = await self._verify_patient_registration(
                    phone_number, sender_name, arabic_name, booking_state
                )
                if patient_check:
                    return patient_check  # Return registration flow response
            
            # NEW FLOW: Fetch service types for new bookings
            if is_new_booking:
                try:
                    current_time = time.time()
                    
                    # Check SHARED cache for service types
                    if _SHARED_CACHE["services"]["data"] and (current_time - _SHARED_CACHE["services"]["time"]) < 300:
                        logger.info("⚡ SHARED CACHE HIT: Service types (instant - 0.0s)")
                        service_types = _SHARED_CACHE["services"]["data"]
                    else:
                        logger.info("🔀 CACHE MISS: Fetching service types from API...")
                        service_types = await fetch_service_types(self.api_client)
                        _SHARED_CACHE["services"] = {"data": service_types, "time": current_time}
                        logger.info(f"✅ Fetched {len(service_types)} service types from API (cached globally for 5 min)")
                    
                    booking_state["service_types"] = service_types
                    booking_state["step"] = "awaiting_service_type"
                    self._save_booking_state(booking_state)
                except Exception as e:
                    logger.error(f"❌ Failed to load service types: {e}", exc_info=True)
                    return {
                        "response": f"يا عيني يا {arabic_name} 😅\n\nما قدرت أحمّل قائمة الخدمات\n\nجرب كذا:\n• انتظر 10 ثواني وجرب مرة ثانية\n• أو اكتب اسم الخدمة مباشرة (مثلاً: ليزر، فيلر) 💡",
                        "intent": "booking",
                        "status": "error",
                        "error": "Failed to fetch service types"
                    }
                
                # CRITICAL: Validate data before proceeding
                service_types_count = len(booking_state.get("service_types", []))
                
                if service_types_count == 0:
                    logger.error(f"⚠️ VALIDATION FAILED: service_types={service_types_count}")
                    # CRITICAL: Use LLM to generate natural error message (NO TEMPLATES!)
                    from app.core.response_generator import get_response_generator
                    response_gen = get_response_generator()
                    error_msg = await response_gen.handle_no_service_types_available(user_name=arabic_name)
                    
                    return {
                        "response": error_msg,
                        "intent": "booking",
                        "status": "error",
                        "error": "Empty services or doctors list"
                    }
                
                # Show service types list - CRITICAL: format_service_types_list is now async!
                service_types = booking_state.get("service_types", [])
                response_text = await format_service_types_list(service_types, arabic_name)
                
                logger.info(f"⚡ Showing {len(service_types)} service types")
                
                return {
                    "response": response_text,
                    "intent": "booking",
                    "status": "awaiting_service_type"
                }
            
            # FAST PATH: Handle progression keywords in active booking session (skip LLM)
            PROGRESSION_KEYWORDS = ["احجز", "أحجز", "حجز", "يلا", "يالله", "تمام", "اوكي", "ماشي", "ok", "okay", "go", "next"]
            is_progression = any(keyword in message.lower() for keyword in PROGRESSION_KEYWORDS)
            
            if is_progression and booking_state.get("started") and not is_new_booking:
                current_step = booking_state.get("step")
                logger.info(f"⚡ Fast path: Progression keyword '{message}' in active booking (step={current_step})")
                
                # Return appropriate template based on current step
                if current_step == "awaiting_service_type":
                    service_types = booking_state.get("service_types", [])
                    
                    # If service types not in state, fetch them
                    if not service_types:
                        logger.info("📋 Service types missing, fetching...")
                        service_types = await fetch_service_types(self.api_client)
                        booking_state["service_types"] = service_types
                        self._save_booking_state(booking_state)
                    
                    response_text = await format_service_types_list(service_types, arabic_name)
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": "awaiting_service_type"
                    }
                
                elif current_step == "awaiting_service":
                    services = booking_state.get("displayed_services", [])
                    type_name = booking_state.get("selected_service_type_name")
                    
                    # If services not in state, try to fetch them if we have a type selected
                    if not services:
                        service_type_id = booking_state.get("selected_service_type_id")
                        if service_type_id:
                            logger.info(f"📋 Services missing, fetching for type {service_type_id}...")
                            services = await fetch_services_by_type(self.api_client, service_type_id)
                            booking_state["services"] = services
                            booking_state["displayed_services"] = services
                            self._save_booking_state(booking_state)
                    
                    response_text = format_services_list(services, arabic_name, type_name)
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": "awaiting_service"
                    }
                
                elif current_step in ["awaiting_doctor", "doctor_selection"]:
                    doctors = booking_state.get("doctors", [])
                    
                    # If doctors not in state, try to fetch them
                    if not doctors:
                        service_id = booking_state.get("service_id")
                        if service_id:
                            logger.info(f"📋 Doctors missing, fetching for service {service_id}...")
                            try:
                                doctors_result = await self.api_client.get(f"/services/{service_id}/doctors")
                                doctors = doctors_result.get("results") or doctors_result.get("data") or []
                                if not doctors:
                                    # Fallback to all doctors
                                    doctors_result = await self.api_client.get("/doctors", params={"limit": 20})
                                    doctors = doctors_result.get("results") or doctors_result.get("data") or []
                                booking_state["doctors"] = doctors
                                self._save_booking_state(booking_state)
                            except Exception as e:
                                logger.error(f"❌ Failed to fetch doctors: {e}")
                    
                    if doctors:
                        doctors_list = "\n".join([
                            f"{i+1}. د. {transliterate_full_name(d.get('name_ar') or d.get('name', ''))} - {d.get('specialty_ar') or d.get('specialty', 'عام')}"
                            for i, d in enumerate(doctors[:10])
                        ])
                        response_text = f"تمام يا {arabic_name}! 👨‍⚕️\n\nعندنا هالدكاترة:\n\n{doctors_list}\n\nمع أي دكتور تحب تحجز؟"
                        return {
                            "response": response_text,
                            "intent": "booking",
                            "status": "awaiting_doctor"
                        }
                    else:
                        response_text = f"يا عيني يا {arabic_name} 😅\nما قدرت أحصل على قائمة الدكاترة\nخليني أساعدك بطريقة ثانية"
                        return {
                            "response": response_text,
                            "intent": "booking",
                            "status": "error"
                        }
                
                elif current_step in ["service_selected", "device_selected", "doctor_selected", "specialist_selected"]:
                    # Check if required resource is actually selected
                    resource_type = booking_state.get("resource_type")
                    doctor_id = booking_state.get("doctor_id")
                    specialist_id = booking_state.get("specialist_id")
                    device_id = booking_state.get("device_id")
                    
                    # Only fetch slots if the required resource exists
                    resource_exists = (
                        (resource_type == "doctor" and doctor_id) or
                        (resource_type == "specialist" and specialist_id) or
                        (resource_type == "device" and device_id)
                    )
                    
                    if resource_exists:
                        # Resource is actually selected, fetch slots
                        service_name = booking_state.get("service_name", "الخدمة")
                        
                        logger.info(f"📅 Resource selected (step={current_step}, type={resource_type}) - fetching available time slots")
                        
                        # Fetch and show available time slots
                        from .booking_helpers import show_available_time_slots
                        slots_response = await show_available_time_slots(
                            api_client=self.api_client,
                            booking_state=booking_state,
                            sender_name=arabic_name
                        )
                        
                        return slots_response
                    else:
                        # Service selected but resource not selected yet - need to select resource
                        logger.info(f"📅 Service selected but {resource_type} not selected yet - showing selection")
                        
                        # Re-run the service selection flow to show the resource list
                        service_id = booking_state.get("service_id")
                        if service_id:
                            # Re-fetch service to get resources
                            try:
                                service_result = await self.api_client.get(f"/services/{service_id}")
                                service = service_result.get("data") or service_result
                                
                                # PRE-VALIDATION: Quick check if resources exist before fetching
                                logger.info(f"🔍 PRE-VALIDATION: Checking {resource_type} availability (fast path)...")
                                
                                if resource_type == "doctor":
                                    quick_check = await self.api_client.get("/doctors", params={"limit": 1})
                                elif resource_type == "specialist":
                                    quick_check = await self.api_client.get("/specialists", params={"limit": 1})
                                elif resource_type == "device":
                                    quick_check = await self.api_client.get("/devices", params={"limit": 1})
                                else:
                                    quick_check = {"results": []}
                                
                                if not (quick_check.get("results") or quick_check.get("data")):
                                    logger.error(f"❌ PRE-VALIDATION FAILED: No {resource_type}s in system")
                                    raise ValueError(f"No {resource_type}s available")
                                
                                logger.info(f"✅ PRE-VALIDATION PASSED (fast path)")
                                
                                # Fetch resources again (now validated)
                                # TIMEOUT PROTECTION: 10 second timeout
                                import asyncio
                                try:
                                    await asyncio.wait_for(
                                        fetch_resource_for_service(
                                            api_client=self.api_client,
                                            booking_state=booking_state,
                                            service=service
                                        ),
                                        timeout=self.RESOURCE_FETCH_TIMEOUT
                                    )
                                except asyncio.TimeoutError:
                                    logger.error(f"⏱️ TIMEOUT: Fast path resource fetch exceeded {self.RESOURCE_FETCH_TIMEOUT}s")
                                    raise ValueError("Resource fetch timeout")
                                
                                # Now show the appropriate selection
                                if resource_type == "doctor":
                                    doctors = booking_state.get("doctors", [])
                                    if doctors:
                                        doctors_list = "\n".join([
                                            f"{i+1}. د. {transliterate_full_name(d.get('name_ar') or d.get('name', ''))} - {d.get('specialty_ar') or d.get('specialty', 'عام')}"
                                            for i, d in enumerate(doctors[:10])
                                        ])
                                        response_text = f"تمام يا {arabic_name}! 👨‍⚕️\n\nعندنا هالدكاترة:\n\n{doctors_list}\n\nمع أي دكتور تحب تحجز؟"
                                    else:
                                        response_text = f"يا عيني يا {arabic_name} 😅\nما في دكاترة متاحين لهالخدمة الحين"
                                elif resource_type == "specialist":
                                    specialists = booking_state.get("specialists", [])
                                    if specialists:
                                        specialists_list = "\n".join([
                                            f"{i+1}. {s.get('name_ar') or s.get('name')} - {s.get('specialty_ar') or s.get('specialty', 'عام')}"
                                            for i, s in enumerate(specialists[:10])
                                        ])
                                        response_text = f"تمام يا {arabic_name}! 👨‍⚕️\n\nعندنا هالمتخصصين:\n\n{specialists_list}\n\nمع أي متخصص تحب تحجز؟"
                                    else:
                                        response_text = f"يا عيني يا {arabic_name} 😅\nما في متخصصين متاحين لهالخدمة الحين"
                                elif resource_type == "device":
                                    devices = booking_state.get("devices", [])
                                    if devices:
                                        if len(devices) == 1:
                                            # Auto-select single device
                                            device = devices[0]
                                            booking_state["device_id"] = device.get("id")
                                            booking_state["device_name"] = device.get("name_ar") or device.get("name")
                                            booking_state["step"] = "device_selected"
                                            self._save_booking_state(booking_state)
                                            
                                            logger.info(f"✅ Auto-selected single device: {booking_state['device_name']}")
                                            
                                            # Fetch slots immediately
                                            from .booking_helpers import show_available_time_slots
                                            return await show_available_time_slots(
                                                api_client=self.api_client,
                                                booking_state=booking_state,
                                                sender_name=arabic_name
                                            )
                                        else:
                                            devices_list = "\n".join([
                                                f"{i+1}. {d.get('name_ar') or d.get('name')} 🔬"
                                                for i, d in enumerate(devices[:10])
                                            ])
                                            response_text = f"تمام يا {arabic_name}! 🔬\n\nعندنا هالأجهزة:\n\n{devices_list}\n\nوش الجهاز اللي تبغاه؟"
                                            booking_state["step"] = "awaiting_device"
                                            self._save_booking_state(booking_state)
                                    else:
                                        response_text = f"يا عيني يا {arabic_name} 😅\nما في أجهزة متاحة لهالخدمة الحين"
                                else:
                                    response_text = f"تمام يا {arabic_name}! 😊\nخليني أساعدك"
                                
                                return {
                                    "response": response_text,
                                    "intent": "booking",
                                    "status": f"awaiting_{resource_type}"
                                }
                            except Exception as e:
                                logger.error(f"❌ Failed to re-fetch service resources: {e}")
                                response_text = f"يا عيني يا {arabic_name} 😅\n\nما قدرت أجيب تفاصيل الخدمة\n\nجرب:\n• اختر خدمة ثانية\n• أو اكتب 'خدمات' عشان تشوف القائمة من جديد 🔄"
                                return {
                                    "response": response_text,
                                    "intent": "booking",
                                    "status": "error"
                                }
                        else:
                            response_text = f"يا عيني يا {arabic_name} 😅\n\nما قدرت أكمل الحجز\n\nعشان نبدأ من جديد:\n• اكتب 'أبغى أحجز' 🔄"
                            return {
                                "response": response_text,
                                "intent": "booking",
                                "status": "error"
                            }
                
                # For other steps, provide a generic progression response
                response_text = f"تمام يا {arabic_name}! 👌\nخليني أساعدك في الحجز\nوش المعلومة اللي تبغى تعطيني إياها؟"
                return {
                    "response": response_text,
                    "intent": "booking",
                    "status": current_step or "in_progress"
                }
            
            # REGISTRATION FLOW HANDLERS
            if current_step == "registration_name":
                return await self._handle_registration_name(message, booking_state, phone_number, arabic_name)
            elif current_step == "registration_national_id":
                return await self._handle_registration_national_id(message, booking_state, arabic_name)
            elif current_step == "registration_phone_confirm":
                return await self._handle_registration_phone_confirm(message, booking_state, phone_number, arabic_name)
            elif current_step == "registration_optional":
                return await self._handle_registration_optional(message, booking_state, arabic_name)
            elif current_step == "registration_confirm":
                return await self._handle_registration_confirm(message, booking_state, arabic_name, context)
            
            # Handle "show doctors" request - OPTIMIZED with template
            if any(word in message for word in ["دكتور", "دكاترة", "الدكاترة", "طبيب", "doctors", "doctor", "منو الدكتور", "show doctor"]):
                doctors = booking_state.get("doctors", [])
                if not doctors:
                    try:
                        doctors_result = await self.api_client.get("/doctors", params={"limit": 10})
                        doctors = doctors_result.get("results") or doctors_result.get("data") or []
                        booking_state["doctors"] = doctors
                        self._save_booking_state(booking_state)
                    except Exception as e:
                        logger.error(f"Failed to get doctors: {e}")
                
                if doctors:
                    # PERFORMANCE: Use TEMPLATE response (3.5s → 0.1s)
                    doctors_list = "\n".join([
                        f"{i+1}. د. {doc.get('name_ar') or doc.get('name')} - {doc.get('specialty_ar') or doc.get('specialty', 'عام')}" 
                        for i, doc in enumerate(doctors[:10])
                    ])
                    
                    response_text = f"""حياك الله يا {arabic_name}! 😊

عندنا أفضل الدكاترة في مركز وجن الطبي:

{doctors_list}

مع أي دكتور تحب تحجز؟ اختر الرقم أو الاسم 👨‍⚕️"""
                    
                    booking_state["step"] = "doctor_selection"
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"⚡ TEMPLATE response for doctors (instant) - no LLM call")
                    
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": "showing_doctors"
                    }
            
            # CRITICAL: Handle confirmation keywords in active booking
            CONFIRMATION_KEYWORDS = ['يلا', 'تمام', 'اوك', 'ok', 'yes', 'نعم', 'ماشي', 'زين', 'طيب']
            if booking_active and any(word in message for word in CONFIRMATION_KEYWORDS):
                logger.info(f"✅ STATE MACHINE: User confirmed with '{message}' (step={current_step})")
                # User is confirming - proceed with current step
                # Don't extract this as entity
            
            # Handle state: service_selected_no_resources - allow user to change service
            current_step = booking_state.get("step")
            if current_step == "service_selected_no_resources":
                # User wants to try different service
                logger.info(f"📋 User in 'no_resources' state - allowing service change")
                
                # Clear previous service selection but keep booking active
                booking_state.pop("service_id", None)
                booking_state.pop("service_name", None)
                booking_state.pop("doctors", None)
                booking_state.pop("specialists", None)
                booking_state.pop("devices", None)
                booking_state.pop("resource_type", None)
                booking_state["step"] = "awaiting_service"
                self._save_booking_state(booking_state)
                
                logger.info(f"🔄 Cleared previous service, step=awaiting_service (ready for new selection)")
            
            # Handle "show services" request - Use LLM
            if any(word in message for word in ["خدمة", "خدمات", "services", "service", "وش عندكم", "show me", "ليزر", "استشارة", "فحص"]):
                services = booking_state.get("services", [])
                if not services:
                    try:
                        # PERFORMANCE: Use cache
                        cached_services = getattr(self, '_services_cache', None)
                        cache_time = getattr(self, '_services_cache_time', 0)
                        
                        if cached_services and (time.time() - cache_time) < 300:
                            logger.info("⚡ Using CACHED services")
                            services = cached_services
                        else:
                            services_result = await self.api_client.get("/services", params={"limit": 10})
                            services = services_result.get("results") or services_result.get("data") or []
                            self._services_cache = services
                            self._services_cache_time = time.time()
                            logger.info(f"✅ Fetched {len(services)} services from API (cached)")
                        
                        booking_state["services"] = services
                        self._save_booking_state(booking_state)
                    except Exception as e:
                        logger.error(f"❌ Failed to get services: {e}")
                        # Return specific error with action
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\n\nما قدرت أوصل لقائمة الخدمات\n\nجرب:\n• انتظر 10 ثواني وأعد المحاولة\n• أو اتصل على: 920033304 📞",
                            "intent": "booking",
                            "status": "api_error",
                            "error": "services_unavailable"
                        }
                
                if services:
                    # PERFORMANCE: Use TEMPLATE response (3.5s → 0.1s)
                    services_list = "\n".join([
                        f"{i+1}. {svc.get('name_ar') or svc.get('name')} - {svc.get('price', 'حسب الاستشارة')} ريال" 
                        for i, svc in enumerate(services[:10])
                    ])
                    
                    response_text = f"""حياك الله يا {arabic_name}! 😊

عندنا خدمات متنوعة في مركز وجن الطبي:

{services_list}

اختر الخدمة التي تحتاجها. يمكنك اختيار الرقم أو الاسم 📋"""
                    
                    # TRACK displayed services in booking state for context
                    booking_state["displayed_services"] = services[:10]
                    booking_state["last_displayed_list"] = services_list
                    booking_state["step"] = "service_selection"
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"⚡ TEMPLATE response for services (instant) - no LLM call")
                    
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": "showing_services"
                    }
            
            # CRITICAL: Check current booking state to determine context
            has_service = booking_state.get("service_id") or booking_state.get("service_name")
            has_doctor = booking_state.get("doctor_id") or booking_state.get("doctor_name")
            current_step = booking_state.get("step", "initial")
            
            # Handle "first", "second", "third" selection - CONTEXT AWARE
            # Also handles "اول واحد" (first one), "ثاني واحد" (second one), etc.
            selection_keywords = {
                'أول': 0, 'الأول': 0, 'الاول': 0, 'اول': 0, 'first': 0, '1': 0,
                'اول واحد': 0, 'أول واحد': 0, 'الاول واحد': 0,
                'ثاني': 1, 'الثاني': 1, 'ثانية': 1, 'second': 1, '2': 1,
                'ثاني واحد': 1, 'الثاني واحد': 1,
                'ثالث': 2, 'الثالث': 2, 'ثالثة': 2, 'third': 2, '3': 2,
                'ثالث واحد': 2, 'الثالث واحد': 2,
                'رابع': 3, 'الرابع': 3, 'fourth': 3, '4': 3,
                'خامس': 4, 'الخامس': 4, 'fifth': 4, '5': 4
            }
            
            selected_index = None
            # Match longest keywords first to handle "اول واحد" before "اول"
            for keyword in sorted(selection_keywords.keys(), key=len, reverse=True):
                if keyword in message:
                    selected_index = selection_keywords[keyword]
                    break
            
            # CRITICAL: Apply selection based on CURRENT STEP
            # If service already selected but doctor not selected → select doctor
            # If service not selected → select service
            
            if selected_index is not None:
                logger.info(f"🔢 NUMBER/ORDINAL DETECTED: index={selected_index} | has_service={has_service} | has_doctor={has_doctor} | step={current_step}")
            
            # CRITICAL: Check if user is deferring choice to system ("اختار", "أي واحد", "ما يهم")
            DEFER_CHOICE_KEYWORDS = [
                'اختار', 'أختار', 'اخترلي', 'أخترلي', 'انت اختار', 'choose', 'select',
                'أي واحد', 'اي واحد', 'ما يهم', 'مايهم', 'ما يهمني', 'انت شوف',
                'انت اشوف', 'أنت شوف', 'على راحتك', 'عادي', 'any', 'whatever', 'you choose',
                'you decide', 'انت قرر', 'أنت قرر'
            ]
            
            is_defer_choice = any(keyword in message for keyword in DEFER_CHOICE_KEYWORDS)
            
            if is_defer_choice and has_service and not has_doctor:
                # User wants system to choose doctor
                doctors = booking_state.get("doctors", [])
                logger.info(f"🎯 DEFER CHOICE DETECTED: User asked system to choose doctor | Available: {len(doctors)}")
                
                if doctors:
                    recommended_doctor = doctors[0]
                    booking_state["doctor_id"] = recommended_doctor.get("id")
                    booking_state["doctor_name"] = recommended_doctor.get("name_ar") or recommended_doctor.get("name")
                    booking_state["doctor_name_en"] = recommended_doctor.get("name")
                    booking_state["doctor_selected"] = True
                    booking_state["step"] = "doctor_selected"
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"✅ AUTO-SELECTED doctor per user request: {booking_state['doctor_name']}")
                    
                    # Fetch and show available time slots immediately
                    from .booking_helpers import show_available_time_slots
                    return await show_available_time_slots(
                        api_client=self.api_client,
                        booking_state=booking_state,
                        sender_name=arabic_name
                    )
                else:
                    logger.warning("⚠️ User asked to choose but no doctors available")
            
            elif is_defer_choice and not has_service:
                # User wants system to choose service
                services = booking_state.get("services", [])
                logger.info(f"🎯 DEFER CHOICE DETECTED: User asked system to choose service | Available: {len(services)}")
                
                if services:
                    # Pick first popular service or most common
                    recommended_service = services[0]
                    booking_state["service_id"] = recommended_service.get("id")
                    booking_state["service_name"] = recommended_service.get("name")
                    booking_state["step"] = "service_selected"
                    self._save_booking_state(booking_state)
                    
                    # Fetch doctors for this service
                    try:
                        service_id = recommended_service.get("id")
                        logger.info(f"🔍 Fetching doctors for auto-selected service_id={service_id}...")
                        doctors_result = await self.api_client.get(f"/services/{service_id}/doctors")
                        service_doctors = doctors_result.get("results") or doctors_result.get("data") or []
                        
                        if not service_doctors:
                            service_doctors = booking_state.get("doctors", [])
                        
                        booking_state["doctors"] = service_doctors
                        self._save_booking_state(booking_state)
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch doctors: {e}")
                    
                    logger.info(f"✅ AUTO-SELECTED service: {booking_state['service_name']}")
                    
                    polite_greeting = f"ممتاز يا {arabic_name}! 😊"
                    return {
                        "response": f"{polite_greeting}\n\nتم اختيار خدمة {booking_state['service_name']} - من خدماتنا المميزة 📋\n\nمع أي دكتور تفضّل تحجز؟",
                        "intent": "booking",
                        "status": "service_selected"
                    }
                else:
                    logger.warning("⚠️ User asked to choose but no services available")
            
            # Handle specialist selection
            if selected_index is not None and current_step == "awaiting_specialist" and booking_state.get("specialists"):
                specialists = booking_state["specialists"]
                logger.info(f"👨‍⚕️ CONTEXT: Selecting SPECIALIST | Available: {len(specialists)}")
                if selected_index < len(specialists):
                    selected_specialist = specialists[selected_index]
                    booking_state["specialist_id"] = selected_specialist.get("id")
                    booking_state["specialist_name"] = selected_specialist.get("name_ar") or selected_specialist.get("name")
                    booking_state["step"] = "specialist_selected"
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"✅ SPECIALIST SELECTED: {booking_state['specialist_name']} (ID: {selected_specialist.get('id')})")
                    
                    # Fetch and show available time slots immediately
                    from .booking_helpers import show_available_time_slots
                    return await show_available_time_slots(
                        api_client=self.api_client,
                        booking_state=booking_state,
                        sender_name=arabic_name
                    )
            
            # Handle "any one" auto-selection for resources
            if current_step in ["awaiting_device", "awaiting_doctor", "awaiting_specialist"]:
                if any(keyword in message.lower() for keyword in ["أي واحد", "اي واحد", "any", "أي شي", "اي شي"]):
                    logger.info(f"✅ User requested auto-selection: '{message}'")
                    
                    # Auto-select first available resource
                    if current_step == "awaiting_device" and booking_state.get("devices"):
                        first_device = booking_state["devices"][0]
                        booking_state["device_id"] = first_device.get("id")
                        booking_state["device_name"] = first_device.get("name_ar") or first_device.get("name")
                        booking_state["step"] = "device_selected"
                        resource_name = booking_state["device_name"]
                        
                    elif current_step == "awaiting_doctor" and booking_state.get("doctors"):
                        first_doctor = booking_state["doctors"][0]
                        booking_state["doctor_id"] = first_doctor.get("id")
                        booking_state["doctor_name"] = first_doctor.get("name_ar") or first_doctor.get("name")
                        booking_state["step"] = "doctor_selected"
                        resource_name = f"د. {booking_state['doctor_name']}"
                        
                    elif current_step == "awaiting_specialist" and booking_state.get("specialists"):
                        first_specialist = booking_state["specialists"][0]
                        booking_state["specialist_id"] = first_specialist.get("id")
                        booking_state["specialist_name"] = first_specialist.get("name_ar") or first_specialist.get("name")
                        booking_state["step"] = "specialist_selected"
                        resource_name = booking_state["specialist_name"]
                    else:
                        resource_name = "الخيار المتاح"
                    
                    self._save_booking_state(booking_state)
                    logger.info(f"✅ AUTO-SELECTED: {resource_name}")
                    
                    # Fetch and show available time slots immediately
                    from .booking_helpers import show_available_time_slots
                    return await show_available_time_slots(
                        api_client=self.api_client,
                        booking_state=booking_state,
                        sender_name=arabic_name
                    )
            
            # Handle device selection  
            elif selected_index is not None and current_step == "awaiting_device" and booking_state.get("devices"):
                devices = booking_state["devices"]
                logger.info(f"🔬 CONTEXT: Selecting DEVICE | Available: {len(devices)}")
                if selected_index < len(devices):
                    selected_device = devices[selected_index]
                    booking_state["device_id"] = selected_device.get("id")
                    booking_state["device_name"] = selected_device.get("name_ar") or selected_device.get("name")
                    booking_state["step"] = "device_selected"
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"✅ DEVICE SELECTED: {booking_state['device_name']} (ID: {selected_device.get('id')})")
                    
                    # Fetch and show available time slots immediately
                    from .booking_helpers import show_available_time_slots
                    return await show_available_time_slots(
                        api_client=self.api_client,
                        booking_state=booking_state,
                        sender_name=arabic_name
                    )
            
            elif selected_index is not None and has_service and not has_doctor:
                # User is selecting DOCTOR by number/ordinal
                doctors = booking_state.get("doctors", [])
                logger.info(f"👨‍⚕️ CONTEXT: Selecting DOCTOR (service already selected) | Available doctors: {len(doctors)}")
                if selected_index < len(doctors):
                    selected_doctor = doctors[selected_index]
                    booking_state["doctor_id"] = selected_doctor.get("id")
                    booking_state["doctor_name"] = selected_doctor.get("name_ar") or selected_doctor.get("name")
                    booking_state["doctor_name_en"] = selected_doctor.get("name")
                    booking_state["doctor_selected"] = True
                    booking_state["step"] = "doctor_selected"
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"✅ DOCTOR SELECTED: Option #{selected_index + 1} → '{booking_state['doctor_name']}' (ID: {selected_doctor.get('id')})")
                    
                    # Fetch and show available time slots immediately
                    from .booking_helpers import show_available_time_slots
                    return await show_available_time_slots(
                        api_client=self.api_client,
                        booking_state=booking_state,
                        sender_name=arabic_name
                    )
            
            elif selected_index is not None and has_service and current_step == "device_selection" and booking_state.get("devices"):
                # User is selecting DEVICE by number
                devices = booking_state["devices"]
                logger.info(f"🔬 CONTEXT: Selecting DEVICE (service selected, device needed) | Available devices: {len(devices)}")
                if selected_index < len(devices):
                    selected_device = devices[selected_index]
                    booking_state["device_id"] = selected_device.get("id")
                    booking_state["device_name"] = selected_device.get("name_ar") or selected_device.get("name")
                    booking_state["step"] = "device_selected"
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"✅ DEVICE SELECTED: Option #{selected_index + 1} → '{booking_state['device_name']}' (ID: {selected_device.get('id')})")
                    
                    # Now fetch doctors for this service
                    try:
                        service_id = booking_state.get("service_id")
                        logger.info(f"🔍 Fetching doctors for service_id={service_id}...")
                        doctors_result = await self.api_client.get(f"/services/{service_id}/doctors")
                        service_doctors = doctors_result.get("results") or doctors_result.get("data") or []
                        
                        if not service_doctors:
                            logger.warning(f"⚠️ No doctors for service {service_id}, using all doctors")
                            doctors_result = await self.api_client.get("/doctors", params={"limit": 10})
                            service_doctors = doctors_result.get("results") or doctors_result.get("data") or []
                        
                        booking_state["doctors"] = service_doctors
                        self._save_booking_state(booking_state)
                        logger.info(f"✅ Loaded {len(service_doctors)} doctors for service")
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch doctors: {e}")
                    
                    # Ask for doctor
                    return {
                        "response": f"تمام يا بعدي 🙌\n\nاخترت {booking_state['device_name']} 🔬\n\nمع أي دكتور تبغى تحجز؟",
                        "intent": "booking",
                        "status": "device_selected"
                    }
            
            # NEW FLOW: Handle service type selection
            elif selected_index is not None and current_step == "awaiting_service_type" and booking_state.get("service_types"):
                # User is selecting SERVICE TYPE by number
                service_types = booking_state["service_types"]
                logger.info(f"📂 CONTEXT: Selecting SERVICE TYPE | Available types: {len(service_types)}")
                if selected_index < len(service_types):
                    selected_type = service_types[selected_index]
                    type_id = selected_type.get("id")
                    type_name = selected_type.get("name_ar") or selected_type.get("name")
                    
                    logger.info(f"✅ SERVICE TYPE SELECTED: #{selected_index + 1} → '{type_name}' (ID: {type_id})")
                    
                    # Fetch services for this type
                    services = await fetch_services_by_type(self.api_client, type_id)
                    
                    if services:
                        booking_state["selected_service_type_id"] = type_id
                        booking_state["selected_service_type_name"] = type_name
                        booking_state["services"] = services
                        booking_state["displayed_services"] = services
                        booking_state["step"] = "awaiting_service"
                        self._save_booking_state(booking_state)
                        
                        response_text = format_services_list(services, arabic_name, type_name)
                        
                        return {
                            "response": response_text,
                            "intent": "booking",
                            "status": "awaiting_service"
                        }
                    else:
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\nما في خدمات متاحة لهالنوع الحين\nتبغى تختار نوع ثاني؟",
                            "intent": "booking",
                            "status": "awaiting_service_type"
                        }
            
            elif selected_index is not None and current_step == "awaiting_service" and booking_state.get("displayed_services"):
                # User is selecting SERVICE by number/ordinal (NEW FLOW)
                services = booking_state["displayed_services"]
                logger.info(f"📋 CONTEXT: Selecting SERVICE (no service yet) | Available services: {len(services)}")
                if selected_index < len(services):
                    selected_service = services[selected_index]
                    service_id = selected_service.get("id")
                    
                    # CIRCUIT BREAKER: Check if service is blocked
                    is_blocked, block_message = self._check_service_circuit_breaker(service_id)
                    if is_blocked:
                        # Clean up state since this service is unavailable
                        self._cleanup_failed_booking_state(booking_state, keep_service=False)
                        
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\n\n{block_message}",
                            "intent": "booking",
                            "status": "service_blocked"
                        }
                    
                    booking_state["service_id"] = service_id
                    booking_state["service_name"] = selected_service.get("name_ar") or selected_service.get("name")
                    booking_state["step"] = "service_selected"
                    
                    logger.info(f"✅ SERVICE SELECTED: #{selected_index + 1} → '{booking_state['service_name']}' (ID: {service_id})")
                    
                    # NEW FLOW: Check service requirements and fetch appropriate resource
                    requirement = get_service_requirement(selected_service)
                    logger.info(f"📋 Service requires: {requirement}")
                    
                    # PRE-VALIDATION: Check if requirement is supported
                    if requirement not in ["doctor", "specialist", "device"]:
                        logger.error(f"❌ UNSUPPORTED REQUIREMENT: {requirement}")
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\n\nهالخدمة تحتاج إعداد خاص\n\nاتصل على: 920033304 عشان نساعدك 📞",
                            "intent": "booking",
                            "status": "unsupported_requirement"
                        }
                    
                    # PRE-VALIDATION: Quick health check before attempting fetch
                    try:
                        logger.info(f"🔍 PRE-VALIDATION: Checking {requirement} availability...")
                        
                        # Quick lightweight check - just count available resources
                        if requirement == "doctor":
                            check_result = await self.api_client.get("/doctors", params={"limit": 1})
                        elif requirement == "specialist":
                            check_result = await self.api_client.get("/specialists", params={"limit": 1})
                        elif requirement == "device":
                            check_result = await self.api_client.get("/devices", params={"limit": 1})
                        
                        available_count = len(check_result.get("results") or check_result.get("data") or [])
                        
                        if available_count == 0:
                            logger.error(f"❌ PRE-VALIDATION FAILED: No {requirement}s available in system")
                            
                            # Clear service selection since we can't proceed
                            booking_state.pop("service_id", None)
                            booking_state.pop("service_name", None)
                            booking_state["step"] = "awaiting_service"
                            self._save_booking_state(booking_state)
                            
                            return {
                                "response": f"يا عيني يا {arabic_name} 😅\n\nما في {requirement} متاح لهالخدمة الحين\n\nجرب:\n• اختر خدمة ثانية\n• أو اتصل على: 920033304 📞",
                                "intent": "booking",
                                "status": "no_resources_available"
                            }
                        
                        logger.info(f"✅ PRE-VALIDATION PASSED: {available_count}+ {requirement}(s) available")
                        
                    except Exception as validation_error:
                        logger.error(f"❌ PRE-VALIDATION ERROR: {validation_error}", exc_info=True)
                        
                        # API is down or inaccessible
                        booking_state.pop("service_id", None)
                        booking_state.pop("service_name", None)
                        booking_state["step"] = "awaiting_service"
                        self._save_booking_state(booking_state)
                        
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\n\nما قدرت أتحقق من توفر المواعيد\n\nجرب:\n• انتظر شوي وجرب مرة ثانية\n• أو اتصل على: 920033304 📞",
                            "intent": "booking",
                            "status": "api_unavailable"
                        }
                    
                    # Now proceed with full resource fetch (validated to be safe)
                    # TIMEOUT PROTECTION: Prevent hanging on slow API calls
                    import asyncio
                    try:
                        result = await asyncio.wait_for(
                            fetch_resource_for_service(self.api_client, selected_service, booking_state),
                            timeout=self.RESOURCE_FETCH_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"⏱️ TIMEOUT: Resource fetch exceeded {self.RESOURCE_FETCH_TIMEOUT}s for service {service_id}")
                        
                        # Record as failure for circuit breaker
                        self._record_service_failure(service_id)
                        
                        # Clean up stale state (keep service for retry)
                        self._cleanup_failed_booking_state(booking_state, keep_service=True)
                        
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\n\nاستغرق الوقت طويل جداً\n\nجرب:\n• اختر خدمة ثانية\n• أو انتظر شوي واتصل على: 920033304 📞",
                            "intent": "booking",
                            "status": "timeout_error"
                        }
                    except Exception as fetch_error:
                        logger.error(f"❌ Resource fetch exception: {fetch_error}", exc_info=True)
                        result = {"error": str(fetch_error)}
                    
                    if result.get("error"):
                        # CIRCUIT BREAKER: Record failure
                        self._record_service_failure(service_id)
                        
                        # Track failure count for this service
                        failure_key = f"fetch_resource_failures_{service_id}"
                        failure_count = booking_state.get(failure_key, 0) + 1
                        booking_state[failure_key] = failure_count
                        
                        logger.error(f"❌ Failed to fetch resources for service {service_id} (attempt {failure_count}): {result.get('error')}")
                        
                        # After 2 failures, offer alternative path
                        if failure_count >= 2:
                            logger.warning(f"⚠️ Resource fetch failed {failure_count} times - offering workaround")
                            
                            # DON'T clear the service - it was successfully selected!
                            # Only clear the failed resource fetch attempts
                            booking_state.pop(failure_key, None)
                            
                            service_name = booking_state.get("service_name", "الخدمة")
                            
                            # Check if there are any resources available at all
                            has_resources = (
                                booking_state.get("doctors") or 
                                booking_state.get("specialists") or 
                                booking_state.get("devices")
                            )
                            
                            if has_resources:
                                # Resources exist but selection is failing - offer MANUAL selection
                                logger.info(f"✅ Resources exist - offering MANUAL selection as fallback")
                                
                                # Prepare manual selection list
                                if requirement == "doctor" and booking_state.get("doctors"):
                                    doctors = booking_state["doctors"]
                                    resources_list = "\n".join([
                                        f"{i+1}. د. {d.get('name_ar') or d.get('name')} - {d.get('specialty_ar') or d.get('specialty', 'عام')}"
                                        for i, d in enumerate(doctors[:10])
                                    ])
                                    response_text = f"تمام يا {arabic_name}! 👨‍⚕️\n\nعندنا هالدكاترة لخدمة {service_name}:\n\n{resources_list}\n\nاختر رقم الدكتور اللي تبغاه؟"
                                    booking_state["step"] = "awaiting_doctor"
                                    
                                elif requirement == "specialist" and booking_state.get("specialists"):
                                    specialists = booking_state["specialists"]
                                    resources_list = "\n".join([
                                        f"{i+1}. {s.get('name_ar') or s.get('name')} - {s.get('specialty_ar') or s.get('specialty', 'عام')}"
                                        for i, s in enumerate(specialists[:10])
                                    ])
                                    response_text = f"تمام يا {arabic_name}! 👨‍⚕️\n\nعندنا هالمتخصصين لخدمة {service_name}:\n\n{resources_list}\n\nاختر رقم المتخصص اللي تبغاه؟"
                                    booking_state["step"] = "awaiting_specialist"
                                    
                                elif requirement == "device" and booking_state.get("devices"):
                                    devices = booking_state["devices"]
                                    resources_list = "\n".join([
                                        f"{i+1}. {d.get('name_ar') or d.get('name')} 🔬"
                                        for i, d in enumerate(devices[:10])
                                    ])
                                    response_text = f"تمام يا {arabic_name}! 🔬\n\nعندنا هالأجهزة لخدمة {service_name}:\n\n{resources_list}\n\nاختر رقم الجهاز اللي تبغاه؟\n\nأو اكتب 'أي واحد' وأنا أختار لك 👌"
                                    booking_state["step"] = "awaiting_device"
                                    
                                else:
                                    # Fallback: auto-select if we can't show manual list
                                    logger.warning(f"⚠️ Can't show manual list - falling back to auto-selection")
                                    if requirement == "doctor":
                                        first = booking_state["doctors"][0]
                                        booking_state["doctor_id"] = first.get("id")
                                        booking_state["step"] = "doctor_selected"
                                    elif requirement == "specialist":
                                        first = booking_state["specialists"][0]
                                        booking_state["specialist_id"] = first.get("id")
                                        booking_state["step"] = "specialist_selected"
                                    elif requirement == "device":
                                        first = booking_state["devices"][0]
                                        booking_state["device_id"] = first.get("id")
                                        booking_state["step"] = "device_selected"
                                    
                                    response_text = f"تمام يا {arabic_name}! 👌\nجاري التجهيز...\nاكتب 'يلا' عشان نكمل 📅"
                                
                                self._save_booking_state(booking_state)
                                logger.info(f"✅ Showing manual {requirement} selection ({len(booking_state.get(f'{requirement}s', []))} options)")
                                
                                return {
                                    "response": response_text,
                                    "intent": "booking",
                                    "status": booking_state["step"]
                                }
                            else:
                                # No resources available - offer to change service
                                logger.warning(f"⚠️ No resources available for service {service_id}")
                                
                                booking_state["step"] = "service_selected_no_resources"
                                self._save_booking_state(booking_state)
                                
                                return {
                                    "response": f"يا عيني يا {arabic_name} 😅\n\nما في مواعيد متاحة لـ {service_name} الحين\n\nتبغى تختار خدمة ثانية؟\nاكتب 'خدمات' عشان تشوف الخيارات 📋",
                                    "intent": "booking",
                                    "status": "service_selected_no_resources"
                                }
                        
                        # First failure - keep state but mark as error_retry
                        booking_state["step"] = "awaiting_service_retry"  # New step to indicate retry state
                        self._save_booking_state(booking_state)
                        
                        logger.info(f"🔄 State updated: step=awaiting_service_retry (first error)")
                        
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\n\nما قدرت أجهز هالخدمة\n\nخياراتك:\n• اختر خدمة ثانية (اكتب رقمها)\n• أو اكتب 'خدمات' عشان تشوف القائمة كاملة 📋",
                            "intent": "booking",
                            "status": "awaiting_service_retry"
                        }
                    
                    # CIRCUIT BREAKER: Record success
                    self._record_service_success(service_id)
                    
                    # Clear failure counters on success
                    if f"fetch_resource_failures_{service_id}" in booking_state:
                        booking_state.pop(f"fetch_resource_failures_{service_id}")
                    if "critical_failures" in booking_state:
                        booking_state["critical_failures"] = 0
                    
                    self._save_booking_state(booking_state)
                    
                    # Show appropriate selection based on requirement
                    if requirement == "doctor":
                        doctors = booking_state.get("doctors", [])
                        if doctors:
                            doctors_list = "\n".join([
                                f"{i+1}. د. {transliterate_full_name(d.get('name_ar') or d.get('name', ''))} - {d.get('specialty_ar') or d.get('specialty', 'عام')}"
                                for i, d in enumerate(doctors[:10])
                            ])
                            response_text = f"تمام يا {arabic_name}! 👨‍⚕️\n\nعندنا هالدكاترة:\n\n{doctors_list}\n\nمع أي دكتور تحب تحجز؟"
                        else:
                            response_text = f"يا عيني يا {arabic_name} 😅\nما في دكاترة متاحين لهالخدمة الحين"
                    
                    elif requirement == "specialist":
                        specialists = booking_state.get("specialists", [])
                        if specialists:
                            specialists_list = "\n".join([
                                f"{i+1}. {s.get('name_ar') or s.get('name')} - {s.get('specialty_ar') or s.get('specialty', 'عام')}"
                                for i, s in enumerate(specialists[:10])
                            ])
                            response_text = f"تمام يا {arabic_name}! 👨‍⚕️\n\nعندنا هالمتخصصين:\n\n{specialists_list}\n\nمع أي متخصص تحب تحجز؟"
                        else:
                            response_text = f"يا عيني يا {arabic_name} 😅\nما في متخصصين متاحين لهالخدمة الحين"
                    
                    elif requirement == "device":
                        # Show matched devices for user to select
                        devices = booking_state.get("devices", [])
                        
                        if devices:
                            if len(devices) == 1:
                                # Only one device - auto-select it
                                device = devices[0]
                                booking_state["device_id"] = device.get("id")
                                booking_state["device_name"] = device.get("name_ar") or device.get("name")
                                booking_state["step"] = "device_selected"
                                self._save_booking_state(booking_state)
                                
                                logger.info(f"✅ Auto-selected single device: {booking_state['device_name']}")
                                
                                # Fetch and show available time slots immediately
                                from .booking_helpers import show_available_time_slots
                                return await show_available_time_slots(
                                    api_client=self.api_client,
                                    booking_state=booking_state,
                                    sender_name=arabic_name
                                )
                            else:
                                # Multiple devices - let user choose
                                devices_list = "\n".join([
                                    f"{i+1}. {d.get('name_ar') or d.get('name')} 🔬"
                                    for i, d in enumerate(devices[:10])
                                ])
                                response_text = f"تمام يا {arabic_name}! 🔬\n\nعندنا هالأجهزة:\n\n{devices_list}\n\nوش الجهاز اللي تبغاه؟"
                                
                                booking_state["step"] = "awaiting_device"
                                self._save_booking_state(booking_state)
                        else:
                            response_text = f"يا عيني يا {arabic_name} 😅\nما في أجهزة متاحة لهالخدمة الحين"
                    
                    else:
                        response_text = f"تمام يا {arabic_name}! 😊\nاخترت {booking_state['service_name']}"
                    
                    # Return appropriate status based on requirement and device count
                    if requirement == "device" and devices and len(devices) > 1:
                        status = "awaiting_device"
                    elif requirement == "device" and devices and len(devices) == 1:
                        status = "device_selected"  # Already handled above, won't reach here
                    else:
                        status = f"awaiting_{requirement}"
                    
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": status
                    }
            
            # Handle "show available times" request - Use LLM
            if any(word in message for word in ["مواعيد", "وقت", "متى", "slots", "times", "available", "متاح"]):
                from ..core.llm_reasoner import get_llm_reasoner
                llm = get_llm_reasoner()
                
                llm_context = {
                    "sender_name": sender_name,
                    "intent": "booking_request_times",
                    "booking_step": "need_details_for_slots",
                    "has_service": booking_state.get("service_id") is not None,
                    "has_date": booking_state.get("preferred_date") is not None
                }
                
                response_text = llm.generate_reply(
                    user_id=phone_number,
                    user_message=message,
                    context=llm_context
                )
                
                return {
                    "response": response_text,
                    "intent": "booking",
                    "status": "requesting_details"
                }
            
            # CRITICAL: Active booking flow - guide user through missing fields
            if booking_active and is_pure_intent_keyword:
                # User said "حجز" or "موعد" in active booking - guide them
                logger.info(f"✅ ACTIVE BOOKING FLOW: Guiding user through missing fields")
                
                has_service = booking_state.get("service_id") or booking_state.get("service_name")
                has_doctor = booking_state.get("doctor_id") or booking_state.get("doctor_name")
                has_date = booking_state.get("preferred_date")
                
                # Determine what to ask for next
                if not has_service:
                    services = booking_state.get("services", [])
                    if services:
                        services_list = "\n".join([
                            f"{i+1}. {s.get('name_ar') or s.get('name')} - {s.get('price', 'حسب الاستشارة')} ريال"
                            for i, s in enumerate(services[:10])
                        ])
                        response_text = f"""حياك الله يا {arabic_name}! 😊

يسعدنا نساعدك بالحجز

عندنا خدمات متنوعة:

{services_list}

أي خدمة تحتاجها؟ اختر الرقم أو الاسم 📋"""
                        booking_state["step"] = "awaiting_service"
                        self._save_booking_state(booking_state)
                        return {
                            "response": response_text,
                            "intent": "booking",
                            "status": "awaiting_service"
                        }
                
                elif not has_doctor:
                    doctors = booking_state.get("doctors", [])
                    if doctors:
                        doctors_list = "\n".join([
                            f"{i+1}. د. {d.get('name_ar') or transliterate_full_name(d.get('name', ''))} - {d.get('specialty_ar') or d.get('specialty', 'عام')}"
                            for i, d in enumerate(doctors[:10])
                        ])
                        response_text = f"""زين يا {arabic_name}! 👍

عندنا دكاترة متخصصين:

{doctors_list}

تبي تحجز عند أي دكتور؟ قول الرقم أو الاسم 👨‍⚕️"""
                        booking_state["step"] = "awaiting_doctor"
                        self._save_booking_state(booking_state)
                        return {
                            "response": response_text,
                            "intent": "booking",
                            "status": "awaiting_doctor"
                        }
                
                elif not has_date:
                    response_text = f"""ممتاز يا {arabic_name}! 🎉

متى تبي الموعد؟
• بكرة (غداً)
• الأحد
• الاثنين
• أو قول التاريخ اللي يناسبك 📅"""
                    booking_state["step"] = "awaiting_date"
                    self._save_booking_state(booking_state)
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": "awaiting_date"
                    }
            
            # Extract booking info from user message
            # CRITICAL: Pass router context to avoid extracting intent keywords as data
            await self._extract_and_update_booking_info(
                message, 
                booking_state, 
                phone_number,
                is_pure_intent=is_pure_intent_keyword
            )
            
            # NEW CONVERSATIONAL FLOW - Let LLM collect info naturally
            from ..core.llm_reasoner import get_llm_reasoner
            llm = get_llm_reasoner()
            
            # Build comprehensive context for LLM
            llm_context = {
                "sender_name": sender_name,
                "phone_number": phone_number,
                "intent": "booking_conversation",
                "booking_state": booking_state,
                "current_step": booking_state.get("step", "initial"),
                "collected_info": {
                    "service": booking_state.get("service_name"),
                    "doctor": booking_state.get("doctor_name"),
                    "date": booking_state.get("preferred_date"),
                    "time": booking_state.get("preferred_time")
                },
                "available_services": [s.get("name") for s in booking_state.get("services", [])[:10]],
                "available_doctors": [d.get("name") for d in booking_state.get("doctors", [])[:10]]
            }
            
            # Let LLM understand what user wants
            response_text = llm.generate_reply(
                user_id=phone_number,
                user_message=message,
                context=llm_context
            )
            
            # CRITICAL: Check if doctor search failed
            if booking_state.get("step") == "doctor_not_found":
                failed_name = booking_state.get("doctor_search_failed", "")
                doctors = booking_state.get("doctors", [])
                
                # CRITICAL: Check if failed_name is actually an intent keyword or action request
                INTENT_KEYWORDS = ['حجز', 'احجز', 'أحجز', 'موعد', 'book', 'booking', 'appointment',
                                   'اختار', 'أختار', 'اخترلي', 'أخترلي', 'انت اختار', 'choose', 'select',
                                   'أي واحد', 'اي واحد', 'ما يهم', 'مايهم', 'ما يهمني', 'انت شوف',
                                   'انت اشوف', 'أنت شوف', 'على راحتك', 'عادي', 'any', 'whatever', 'you choose']
                is_intent_keyword = failed_name.strip() in INTENT_KEYWORDS
                
                if is_intent_keyword:
                    # User didn't ask for a doctor - they're confirming booking
                    logger.info(f"✅ ERROR RECOVERY: '{failed_name}' is intent keyword, not doctor name - graceful response")
                    
                    # Check if user asked system to choose (defer choice)
                    CHOOSE_KEYWORDS = ['اختار', 'أختار', 'اخترلي', 'أخترلي', 'انت اختار', 'choose', 'select',
                                      'أي واحد', 'اي واحد', 'ما يهم', 'مايهم', 'ما يهمني', 'انت شوف',
                                      'انت اشوف', 'أنت شوف', 'على راحتك', 'عادي', 'any', 'whatever', 'you choose']
                    if failed_name.strip() in CHOOSE_KEYWORDS:
                        # User wants system to recommend - pick first available doctor
                        if doctors:
                            recommended_doctor = doctors[0]
                            booking_state["doctor_id"] = recommended_doctor.get("id")
                            booking_state["doctor_name"] = recommended_doctor.get("name_ar") or recommended_doctor.get("name")
                            booking_state["doctor_name_en"] = recommended_doctor.get("name")
                            booking_state["doctor_selected"] = True
                            booking_state.pop("doctor_search_failed", None)
                            booking_state["step"] = "doctor_selected"
                            self._save_booking_state(booking_state)
                            
                            logger.info(f"✅ AUTO-SELECTED doctor per user request: {booking_state['doctor_name']}")
                            
                            # Fetch and show available time slots immediately
                            from .booking_helpers import show_available_time_slots
                            return await show_available_time_slots(
                                api_client=self.api_client,
                                booking_state=booking_state,
                                sender_name=arabic_name
                            )
                    
                    # Show available doctors without confusing error message
                    doctor_list = "\n".join([
                        f"{i+1}. د. {d.get('name_ar') or transliterate_full_name(d.get('name', ''))} - {d.get('specialty_ar') or d.get('specialty', 'عام')}" 
                        for i, d in enumerate(doctors[:5])
                    ])
                    
                    booking_state.pop("doctor_search_failed", None)
                    booking_state["step"] = "awaiting_doctor_selection"
                    self._save_booking_state(booking_state)
                    
                    return {
                        "response": f"حياك الله يا {arabic_name}! 😊\n\nيسعدنا نساعدك بالحجز\n\nعندنا هؤلاء الدكاترة:\n\n{doctor_list}\n\nمع أي دكتور تحب تحجز؟ اختر الرقم أو الاسم 👨‍⚕️",
                        "intent": "booking",
                        "status": "showing_doctors"
                    }
                else:
                    # User actually mentioned a name that doesn't exist
                    logger.warning(f"⚠️ Doctor '{failed_name}' not found - showing available doctors")
                    
                    doctor_list = "\n".join([
                        f"{i+1}. د. {d.get('name_ar') or transliterate_full_name(d.get('name', ''))}" 
                        for i, d in enumerate(doctors[:5])
                    ])
                    
                    booking_state.pop("doctor_search_failed", None)
                    booking_state["step"] = "awaiting_doctor_selection"
                    self._save_booking_state(booking_state)
                    
                    # CRITICAL: Polite error message with buffer (Saudi-neutral professional tone)
                    polite_greeting = f"عذرًا يا {arabic_name}"
                    return {
                        "response": f"""{polite_greeting}، ما لقيت دكتور بالاسم اللي ذكرته.

هذي قائمة الأطباء المتاحين، تقدر تختار الرقم المناسب:

{doctor_list}

وش الدكتور اللي تفضّله؟ 👨‍⚕️""",
                        "intent": "booking",
                        "status": "doctor_not_found"
                    }
                        
            # Check if we have all required info
            has_service = booking_state.get("service_id") or booking_state.get("service_name")
            has_doctor = booking_state.get("doctor_id") or booking_state.get("doctor_name")
            has_specialist = booking_state.get("specialist_id")
            has_device = booking_state.get("device_id")
            has_resource = has_doctor or has_specialist or has_device
            has_date = booking_state.get("preferred_date")  
            has_time = booking_state.get("preferred_time")
            
            # STEP 1: Service & resource selected, now show time slots
            if has_service and has_resource and not has_date:
                logger.info("📅 Service & resource selected - fetching available time slots")
                from .booking_helpers import show_available_time_slots
                return await show_available_time_slots(
                    api_client=self.api_client,
                    booking_state=booking_state,
                    sender_name=arabic_name
                )
            
            # STEP 2: Time slot selected, show confirmation
            if has_service and has_resource and has_date and has_time:
                # Check if already confirmed
                if booking_state.get("awaiting_confirmation"):
                    # User is responding to confirmation request
                    if any(word in message for word in ["نعم", "yes", "تأكيد", "confirm", "موافق", "ok"]):
                        logger.info("✅ User confirmed booking - proceeding to complete")
                        return await complete_booking_with_details(
                            api_client=self.api_client,
                            booking_state=booking_state,
                            phone_number=phone_number,
                            sender_name=arabic_name
                        )
                    elif any(word in message for word in ["لا", "no", "إلغاء", "cancel"]):
                        logger.info("❌ User cancelled booking confirmation")
                        booking_state.clear()
                        booking_state["started"] = False
                        self._save_booking_state(booking_state)
                        
                        lang = detect_language(message)
                        if lang == "arabic":
                            return {
                                "response": "تمام، تم إلغاء الحجز. إذا تبي تحجز مرة ثانية، قول لي! 😊",
                                "intent": "booking",
                                "status": "cancelled"
                            }
                        else:
                            return {
                                "response": "Okay, booking cancelled. Let me know if you'd like to book again! 😊",
                                "intent": "booking",
                                "status": "cancelled"
                            }
                else:
                    # Show confirmation request
                    logger.info("📋 All info collected - requesting user confirmation")
                    return await request_booking_confirmation(
                        booking_state=booking_state,
                        sender_name=arabic_name,
                        api_client=self.api_client
                    )

            # CRITICAL: Log what's missing and guide user to next step
            logger.info(f"🔍 BOOKING PROGRESS: service={'✓' if has_service else '✗'}, doctor={'✓' if has_doctor else '✗'}, date={'✗' if not has_date else '✓'}, time={'✗' if not has_time else '✓'}")
            
            # Determine what to ask for next
            if not has_service:
                next_step = "service selection"
                logger.info("📋 NEXT STEP: Ask user to select a service")
            elif not has_doctor:
                next_step = "doctor selection"
                logger.info("📋 NEXT STEP: Ask user to select a doctor")
            elif not has_date:
                next_step = "date selection"
                logger.info("📋 NEXT STEP: Ask user to select a date")
            elif not has_time:
                next_step = "time selection"
                logger.info("📋 NEXT STEP: Ask user to select a time")
            else:
                next_step = "unknown"
                logger.warning("⚠️ All fields present but not in confirmation - logic error?")
            
            # CRITICAL: Detect premature confirmation by LLM and override
            if "حجز" in response_text or "موعد" in response_text or "تم" in response_text:
                # LLM might be prematurely confirming - override with explicit question
                logger.warning(f"⚠️ PREMATURE CONFIRMATION DETECTED: LLM used confirmation words but {next_step} is missing")
                logger.info(f"✅ CORRECTIVE ACTION: Overriding LLM response with explicit question for {next_step}")
                
                if not has_service:
                    response_text = f"حياك الله يا {arabic_name}! 😊 وش الخدمة اللي تحتاجها؟"
                    logger.info(f"✅ OVERRIDE: Asking for service selection")
                elif not has_doctor:
                    # Show doctor list if available
                    doctors = booking_state.get("doctors", [])
                    if doctors:
                        doctor_list = "\n".join([
                            f"{i+1}. د. {d.get('name_ar') or transliterate_full_name(d.get('name', ''))} - {d.get('specialty_ar') or d.get('specialty', 'عام')}"
                            for i, d in enumerate(doctors[:10])
                        ])
                        response_text = f"ممتاز يا {arabic_name}! 👨‍⚕️\n\nعندنا هالدكاترة:\n\n{doctor_list}\n\nمع أي دكتور تحب تحجز؟"
                    else:
                        response_text = f"ممتاز يا {arabic_name}! مع أي دكتور تحب تحجز؟"
                    logger.info(f"✅ OVERRIDE: Asking for doctor selection")
                elif not has_date or not has_time:
                    response_text = f"ممتاز يا {arabic_name}! متى يناسبك الموعد؟ (مثلاً: باكر الساعة 3 العصر)"
                    logger.info(f"✅ OVERRIDE: Asking for date/time selection")
            
            # Otherwise, return LLM response asking for more info
            return {
                "response": response_text,
                "intent": "booking",
                "status": "collecting_info",
                "next_step": next_step
            }
            
        except Exception as exc:
            logger.error(f"New booking error: {exc}", exc_info=True)
            
            # Track critical failures and offer recovery
            failure_count = booking_state.get("critical_failures", 0) + 1
            booking_state["critical_failures"] = failure_count
            
            # After 3 critical failures, offer to restart booking
            if failure_count >= 3:
                logger.error(f"🚨 CRITICAL: Booking failed {failure_count} times - offering full restart")
                
                # Clean up entire state (full reset)
                self._cleanup_failed_booking_state(booking_state, keep_service=False)
                
                logger.info(f"🔄 FULL RESET: Cleared all booking state, step=started")
                
                return {
                    "response": f"يا عيني يا {arabic_name} 😅\n\nشكله صار خطأ كبير\nخليني أبدأ معك من جديد\n\nاكتب 'أبغى أحجز' عشان نبدأ 🙏",
                    "intent": "booking",
                    "status": "started"  # Match the step
                }
            
            # Save failure count and update step to indicate error state
            current_step = booking_state.get("step", "in_progress")
            booking_state["step"] = f"{current_step}_error"  # Add error suffix
            self._save_booking_state(booking_state)
            
            logger.info(f"🔄 State updated: step={current_step}_error (error occurred)")
            
            return {
                "response": f"يا عيني يا {arabic_name} 😅\n\nصار خطأ في الحجز\n\nخياراتك:\n• جرب مرة ثانية (كرر آخر خطوة)\n• اكتب 'إلغاء' عشان نبدأ من جديد\n• أو اكتب 'خدمات' عشان تشوف الخدمات 🔄",
                "intent": "booking",
                "status": "error"
            }
    
    async def _verify_patient(self, phone_number: str) -> Optional[dict]:
        """Verify patient exists in system"""
        try:
            # Search for patient by phone
            result = await self.api_client.get("/patients", params={"q": phone_number})
            
            if result and result.get("data"):
                patients = result["data"]
                if patients:
                    logger.info(f"✓ Patient found: {patients[0].get('name')}")
                    return patients[0]
            
            logger.warning(f"Patient not found for phone: {phone_number}")
            return None
            
        except Exception as exc:
            logger.error(f"Patient verification error: {exc}")
            return None
    
    async def _request_service_selection(self, booking_state: dict) -> dict:
        """Request service selection from user"""
        try:
            # Get available services
            services_result = await self.api_client.get("/services", params={"limit": 10})
            
            if services_result and services_result.get("data"):
                services = services_result["data"]
                
                # Format services list
                service_list = "\n".join([
                    f"{i+1}. {service.get('name', 'Unknown')}"
                    for i, service in enumerate(services[:10])
                ])
                
                # Save services to state for reference
                booking_state["available_services"] = services
                self._save_booking_state(booking_state)
                
                return {
                    "response": f"Great! What service would you like to book?\n\n{service_list}\n\nPlease reply with the service name or number.",
                    "intent": "booking",
                    "status": "awaiting_service"
                }
            else:
                return {
                    "response": "I'm having trouble loading our services. Please try again or contact support.",
                    "intent": "booking",
                    "status": "error"
                }
                
        except Exception as exc:
            logger.error(f"Service request error: {exc}")
            return {
                "response": "What service would you like to book? (e.g., General Consultation, X-Ray, Blood Test)",
                "intent": "booking",
                "status": "awaiting_service"
            }
    
    async def _process_service_selection(self, message: str, booking_state: dict) -> Optional[dict]:
        """Process user's service selection"""
        try:
            available_services = booking_state.get("available_services", [])
            
            if not available_services:
                # Fetch services if not in state
                services_result = await self.api_client.get("/services", params={"limit": 20})
                if services_result and services_result.get("data"):
                    available_services = services_result["data"]
            
            # Try to match service by number
            if message.isdigit():
                service_num = int(message) - 1
                if 0 <= service_num < len(available_services):
                    selected_service = available_services[service_num]
                    booking_state["service_id"] = selected_service.get("id")
                    booking_state["service_name"] = selected_service.get("name")
                    self._save_booking_state(booking_state)
                    
                    return await self._request_date_selection(booking_state)
            
            # Try to match service by name
            for service in available_services:
                service_name = service.get("name", "").lower()
                if service_name in message or message in service_name:
                    booking_state["service_id"] = service.get("id")
                    booking_state["service_name"] = service.get("name")
                    self._save_booking_state(booking_state)
                    
                    return await self._request_date_selection(booking_state)
            
            # No match found
            return {
                "response": "I couldn't find that service. Please choose from the list or describe the service you need.",
                "intent": "booking",
                "status": "awaiting_service"
            }
            
        except Exception as exc:
            logger.error(f"Service selection error: {exc}")
            return None
    
    async def _process_doctor_selection(self, message: str, booking_state: dict) -> Optional[dict]:
        """Process doctor selection (optional step)"""
        # For now, skip doctor selection and mark as completed
        booking_state["doctor_selected"] = True
        self._save_booking_state(booking_state)
        return None
    
    async def _request_date_selection(self, booking_state: dict) -> dict:
        """Request preferred date from user"""
        service_name = booking_state.get("service_name", "your appointment")
        
        return {
            "response": f"Perfect! When would you like to schedule your {service_name}?\n\nPlease provide a preferred date (e.g., 'tomorrow', 'next Monday', '2025-10-15').",
            "intent": "booking",
            "status": "awaiting_date"
        }
    
    async def _process_date_selection(self, message: str, booking_state: dict) -> Optional[dict]:
        """Process user's date selection"""
        try:
            # Parse date from message
            preferred_date = self._parse_date_from_message(message)
            
            if preferred_date:
                booking_state["preferred_date"] = preferred_date.strftime("%Y-%m-%d")
                self._save_booking_state(booking_state)
                
                # Move to slot selection
                return None  # Will proceed to show_slots_and_book
            else:
                return {
                    "response": "I couldn't understand that date. Please provide a date like 'tomorrow', 'October 15', or '2025-10-15'.",
                    "intent": "booking",
                    "status": "awaiting_date"
                }
                
        except Exception as exc:
            logger.error(f"Date selection error: {exc}")
            return None
    
    async def _show_slots_and_book(self, booking_state: dict, message: str) -> dict:
        """Show available slots and create booking"""
        try:
            service_id = booking_state.get("service_id")
            preferred_date = booking_state.get("preferred_date")
            patient_id = booking_state.get("patient_id")
            
            # Get device_id from booking state (REQUIRED for slots API)
            device_id = booking_state.get("device_id")
            
            # CRITICAL: device_id is REQUIRED for /api/slots endpoint
            if not device_id:
                logger.error(f"⚠️ Missing device_id - cannot fetch slots!")
                return {
                    "response": "لحظة شوي 😅\nتبغى تحجز على أي جهاز؟",
                    "intent": "booking",
                    "status": "missing_device"
                }
            
            # Get available slots (REQUIRES: service_id, date, device_id)
            slots_result = await self.api_client.get("/slots", params={
                "service_id": service_id,
                "date": preferred_date,
                "device_id": device_id
            })
            
            if slots_result and slots_result.get("data"):
                slots = slots_result["data"]
                
                if not slots:
                    return {
                        "response": f"عذراً، ما في مواعيد متاحة في {preferred_date}. تبي تجرب تاريخ ثاني؟ 📅",
                        "intent": "booking",
                        "status": "no_slots"
                    }
                
                # If user hasn't selected a slot yet, show options
                if not booking_state.get("slot_selected"):
                    slot_list = "\n".join([
                        f"{i+1}. {slot.get('start_time', 'Unknown time')}"
                        for i, slot in enumerate(slots[:10])
                    ])
                    
                    booking_state["available_slots"] = slots
                    self._save_booking_state(booking_state)
                    
                    return {
                        "response": f"Available time slots on {preferred_date}:\n\n{slot_list}\n\nPlease reply with the slot number you prefer.",
                        "intent": "booking",
                        "status": "awaiting_slot"
                    }
                
                # User has selected a slot, create booking
                slot_num = self._extract_number_from_message(message)
                if slot_num and 0 < slot_num <= len(slots):
                    selected_slot = slots[slot_num - 1]
                    
                    # Create booking
                    booking_payload = {
                        "patient_id": patient_id,
                        "service_id": service_id,
                        "slot_id": selected_slot.get("id"),
                        "date": preferred_date
                    }
                    
                    booking_result = await self.api_client.post("/booking/create", json_body=booking_payload)
                    
                    if booking_result and not booking_result.get("error"):
                        # Clear booking state
                        self._clear_booking_state()
                        
                        service_name = booking_state.get("service_name", "appointment")
                        slot_time = selected_slot.get("start_time", "")
                        
                        return {
                            "response": f"✅ Booking confirmed!\n\n📅 Service: {service_name}\n🕐 Date & Time: {preferred_date} at {slot_time}\n\nWe look forward to seeing you! You'll receive a confirmation message shortly.",
                            "intent": "booking",
                            "status": "confirmed",
                            "booking_details": booking_result
                        }
                    else:
                        return {
                            "response": f"يا عيني يا {arabic_name} 😅\n\nما قدرت أكمل الحجز\n\nجرب:\n• تأكد من المعلومات وجرب مرة ثانية\n• أو اكتب 'أبغى أحجز' عشان نبدأ من جديد 🔄",
                            "intent": "booking",
                            "status": "failed"
                        }
            
            # No slots available
            return {
                "response": "I'm having trouble finding available slots. Please try a different date or contact support.",
                "intent": "booking",
                "status": "error"
            }
            
        except Exception as exc:
            logger.error(f"Slot booking error: {exc}", exc_info=True)
            return {
                "response": f"يا عيني 😅\n\nما قدرت أحجز الموعد\n\nجرب:\n• جرب موعد ثاني\n• أو اكتب 'خدمات' عشان تختار خدمة ثانية\n• أو اتصل على: 920033304 📞",
                "intent": "booking",
                "status": "error"
            }
    
    async def _handle_view_bookings(self, phone_number: str) -> dict:
        """
        View user's existing bookings with proper Arabic formatting.
        Shows actual appointment data from the API.
        Includes validation before promises (Issue #39).
        """
        try:
            logger.info(f"📋 [VALIDATION] Starting appointment view for phone: {phone_number}")
            
            # VALIDATION STEP 1: Check if user exists (Issue #39)
            logger.info(f"📋 [VALIDATION] Step 1/3: Verifying patient exists...")
            patient = await self._verify_patient(phone_number)
            
            if not patient:
                logger.warning(f"⚠️ Patient not found for phone: {phone_number}")
                return {
                    "response": """ما لقيت ملفك الطبي 🔍

عشان تشوف مواعيدك، لازم تكون مسجل عندنا

اكتب 'حجز' عشان نسجلك ونحجز لك موعد 📅""",
                    "intent": "booking",
                    "status": "patient_not_found"
                }
            
            patient_id = patient.get("id")
            patient_name = patient.get("name", "")
            logger.info(f"✅ [VALIDATION] Step 1/3 PASSED: Patient found - {patient_name} (ID: {patient_id})")
            
            # VALIDATION STEP 2: Check if API is available (Issue #39)
            logger.info(f"📋 [VALIDATION] Step 2/3: Checking API availability...")
            try:
                bookings_result = await self.api_client.get("/booking", params={
                    "patient_id": patient_id,
                    "limit": 10
                })
                logger.info(f"✅ [VALIDATION] Step 2/3 PASSED: API responded successfully")
            except Exception as api_error:
                logger.error(f"❌ [VALIDATION] Step 2/3 FAILED: API error - {api_error}")
                return {
                    "response": """يا عيني 😅

ما قدرت أوصل لقاعدة البيانات الحين

جرب:
• انتظر شوي وجرب مرة ثانية
• أو اتصل على: 920033304 📞""",
                    "intent": "booking",
                    "status": "api_error"
                }
            
            # VALIDATION STEP 3: Check if user has appointments (Issue #39)
            logger.info(f"📋 [VALIDATION] Step 3/3: Checking if appointments exist...")
            
            if bookings_result and bookings_result.get("data"):
                bookings = bookings_result["data"]
                
                if not bookings or len(bookings) == 0:
                    logger.info(f"✅ [VALIDATION] Step 3/3 PASSED: No appointments found (valid state)")
                    logger.info(f"📭 Patient {patient_id} has no bookings")
                    return {
                        "response": """ما عندك مواعيد محجوزة الحين 📭

تبغى تحجز موعد جديد؟
اكتب 'حجز' وأساعدك 📅""",
                        "intent": "booking",
                        "status": "no_bookings"
                    }
                
                logger.info(f"✅ [VALIDATION] Step 3/3 PASSED: {len(bookings)} appointments found")
                logger.info(f"✅ [VALIDATION] ALL CHECKS PASSED - Safe to show appointments")
                logger.info(f"📋 Found {len(bookings)} bookings for patient {patient_id}")
                
                # Format bookings with Arabic status translations and proper structure
                booking_list = []
                for i, booking in enumerate(bookings[:5]):  # Show max 5 bookings
                    # Extract booking details
                    service_name = booking.get("service", {}).get("name_ar") or booking.get("service", {}).get("name") or "خدمة غير محددة"
                    date = booking.get("date", "تاريخ غير محدد")
                    time = booking.get("time", "")
                    status = booking.get("state", "Unknown")
                    booking_id = booking.get("id", "")
                    
                    # Translate status to Arabic
                    status_ar = {
                        "confirmed": "✅ مؤكد",
                        "pending": "⏳ بالانتظار",
                        "completed": "✔️ مكتمل",
                        "cancelled": "❌ ملغي",
                        "rescheduled": "🔄 معاد جدولته"
                    }.get(status.lower(), status)
                    
                    # Format: 1️⃣ Service - Date Time (Status)
                    booking_entry = f"{i+1}️⃣ **{service_name}**\n   📅 {date}"
                    if time:
                        booking_entry += f" | ⏰ {time}"
                    booking_entry += f"\n   {status_ar}"
                    if booking_id:
                        booking_entry += f" | 🆔 #{booking_id}"
                    
                    booking_list.append(booking_entry)
                
                bookings_text = "\n\n".join(booking_list)
                
                # Build response with header
                response = f"""مواعيدك المحجوزة 📋

{bookings_text}

**خياراتك:**
• اكتب 'حجز' للحجز موعد جديد
• اكتب 'إلغاء #رقم' لإلغاء موعد
• أو اتصل: 920033304 📞"""
                
                return {
                    "response": response,
                    "intent": "booking",
                    "status": "success",
                    "bookings": bookings,
                    "count": len(bookings)
                }
            else:
                logger.warning(f"⚠️ API returned no data for patient {patient_id}")
                return {
                    "response": """ما عندك مواعيد محجوزة 📭

تبغى تحجز موعد؟
اكتب 'حجز' وأساعدك 📅""",
                    "intent": "booking",
                    "status": "no_bookings"
                }
            
        except Exception as exc:
            logger.error(f"❌ View bookings error: {exc}", exc_info=True)
            return {
                "response": f"""يا عيني 😅

ما قدرت أجيب مواعيدك الحين

جرب:
• انتظر شوي وجرب مرة ثانية
• أو اتصل على: 920033304 📞""",
                "intent": "booking",
                "status": "error",
                "error": str(exc)
            }
    
    async def _handle_reschedule(self, payload: dict, booking_state: dict) -> dict:
        """Handle booking rescheduling"""
        return {
            "response": "To reschedule, please provide your booking ID and preferred new date. You can also call our support team for assistance.",
            "intent": "booking",
            "status": "reschedule_info"
        }
    
    async def _handle_cancel(self, payload: dict, booking_state: dict) -> dict:
        """Handle booking cancellation"""
        try:
            message = payload.get("message", "").lower()
            phone_number = payload.get("phone_number")
            
            # Check if booking ID is provided
            booking_id = self._extract_booking_id(message)
            
            if not booking_id:
                # No booking ID - show user's bookings
                patient = await self._verify_patient(phone_number)
                
                if not patient:
                    return {
                        "response": "I couldn't find your patient record. Please register first.",
                        "intent": "booking",
                        "status": "not_found"
                    }
                
                # Get active bookings
                bookings_result = await self.api_client.get("/booking", params={
                    "patient_id": patient.get("id"),
                    "state": "confirmed",
                    "limit": 10
                })
                
                if bookings_result and bookings_result.get("data"):
                    bookings = bookings_result["data"]
                    
                    if not bookings:
                        return {
                            "response": "You don't have any active bookings to cancel.",
                            "intent": "booking",
                            "status": "no_bookings"
                        }
                    
                    # Format bookings list
                    booking_list = []
                    for i, booking in enumerate(bookings[:5]):
                        bid = booking.get("id", "N/A")
                        service = booking.get("service", {}).get("name", "Unknown")
                        date = booking.get("date", "Unknown")
                        booking_time = booking.get("time", "")
                        booking_list.append(f"{i+1}. ID: {bid} - {service} on {date} {booking_time}")
                    
                    bookings_text = "\n".join(booking_list)
                    
                    # Save bookings to state for next interaction
                    booking_state["cancel_bookings"] = bookings
                    self._save_booking_state(booking_state)
                    
                    return {
                        "response": f"Here are your active bookings:\n\n{bookings_text}\n\nReply with the booking number (1-{len(bookings)}) to cancel it.",
                        "intent": "booking",
                        "status": "awaiting_selection"
                    }
                
                return {
                    "response": "You don't have any active bookings.",
                    "intent": "booking",
                    "status": "no_bookings"
                }
            
            # Booking ID provided - proceed with cancellation
            return await self._cancel_booking_by_id(booking_id, phone_number)
            
        except Exception as exc:
            logger.error(f"Cancel booking error: {exc}", exc_info=True)
            return {
                "response": f"يا عيني 😅\n\nما قدرت ألغي الموعد\n\nجرب:\n• جرب مرة ثانية بعد شوي\n• أو اتصل على: 920033304 عشان نلغيه لك 📞",
                "intent": "booking",
                "status": "error"
            }
    
    async def _cancel_booking_by_id(self, booking_id: str, phone_number: str) -> dict:
        """Cancel a specific booking by ID"""
        try:
            # Verify patient owns this booking
            patient = await self._verify_patient(phone_number)
            
            if not patient:
                return {
                    "response": "I couldn't verify your patient record.",
                    "intent": "booking",
                    "status": "error"
                }
            
            # Get booking details
            booking_result = await self.api_client.get(f"/booking/{booking_id}")
            
            if not booking_result or booking_result.get("error"):
                return {
                    "response": f"I couldn't find booking ID: {booking_id}. Please check the ID and try again.",
                    "intent": "booking",
                    "status": "not_found"
                }
            
            booking = booking_result.get("data", booking_result)
            
            # Verify ownership
            if booking.get("patient_id") != patient.get("id"):
                return {
                    "response": "This booking doesn't belong to you.",
                    "intent": "booking",
                    "status": "unauthorized"
                }
            
            # Cancel the booking (update status to cancelled)
            cancel_result = await self.api_client.put(f"/booking/{booking_id}", json_body={
                "state": "cancelled",
                "cancelled_at": datetime.now().isoformat()
            })
            
            if cancel_result and not cancel_result.get("error"):
                service_name = booking.get("service", {}).get("name", "appointment")
                date = booking.get("date", "")
                booking_time = booking.get("time", "")
                
                return {
                    "response": f"✅ Booking cancelled successfully!\n\n📅 {service_name}\n🕐 {date} {booking_time}\n\nYour booking has been cancelled. You can book a new appointment anytime!",
                    "intent": "booking",
                    "status": "cancelled",
                    "booking_id": booking_id
                }
            else:
                return {
                    "response": f"يا عيني 😅\n\nما قدرت ألغي الحجز\n\nجرب:\n• تأكد من رقم الحجز وجرب مرة ثانية\n• أو اتصل على: 920033304 📞",
                    "intent": "booking",
                    "status": "error"
                }
                
        except Exception as exc:
            logger.error(f"Cancel booking by ID error: {exc}", exc_info=True)
            return {
                "response": f"يا عيني 😅\n\nما قدرت ألغي الحجز\n\nتأكد من:\n• رقم الحجز صحيح\n• الموعد لسه ما مر\n\nأو اتصل على: 920033304 📞",
                "intent": "booking",
                "status": "error"
            }
    
    def _extract_booking_id(self, message: str) -> Optional[str]:
        """Extract booking ID from message"""
        try:
            import re
            # Look for patterns like "ID: 123", "booking 123", "#123"
            patterns = [
                r'id[:\s]+(\d+)',
                r'booking[:\s]+(\d+)',
                r'#(\d+)',
                r'\b(\d{3,})\b'  # Any 3+ digit number
            ]
            
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return None
        except:
            return None
    
    def _parse_date_from_message(self, message: str) -> Optional[datetime]:
        """Parse date from natural language message"""
        try:
            message = message.lower()
            today = datetime.now()
            
            if "today" in message:
                return today
            elif "tomorrow" in message:
                return today + timedelta(days=1)
            elif "next week" in message:
                return today + timedelta(days=7)
            elif "monday" in message:
                days_ahead = 0 - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                return today + timedelta(days=days_ahead)
            # Add more date parsing logic as needed
            
            # Try to parse ISO format (YYYY-MM-DD)
            try:
                return datetime.strptime(message, "%Y-%m-%d")
            except:
                pass
            
            return None
            
        except Exception as exc:
            logger.error(f"Date parsing error: {exc}")
            return None
    
    def _extract_number_from_message(self, message: str) -> Optional[int]:
        """Extract number from message"""
        try:
            # Find first number in message
            import re
            numbers = re.findall(r'\d+', message)
            if numbers:
                return int(numbers[0])
            return None
        except:
            return None
    
    def _load_booking_state(self) -> dict:
        """Load booking state from session"""
        try:
            session_data = self.session_manager.get(self.session_key) or {}
            booking_state = session_data.get("booking_state", {})
            logger.debug(f"📥 Loaded booking state: step={booking_state.get('step')}, started={booking_state.get('started')}")
            return booking_state
        except Exception as exc:
            logger.error(f"Error loading booking state: {exc}", exc_info=True)
            return {}
    
    def _save_booking_state(self, booking_state: dict) -> None:
        """Save booking state to session with progress tracking and context sync"""
        try:
            session_data = self.session_manager.get(self.session_key) or {}
            session_data["booking_state"] = booking_state
            
            # CRITICAL: Sync step to context for consistency (Issue #9)
            # Ensures step is available in both booking_state and top-level context
            # Always sync, even if None, to ensure consistency
            session_data["booking_step"] = booking_state.get("step")
            
            # CRITICAL: Preserve booking phase and specific topic across greetings
            # Don't reset to discovery if booking is active
            if booking_state.get("started"):
                session_data["current_phase"] = "booking"
                # CRITICAL: Use specific service name as topic, not generic "booking"
                # This maintains context: "ليزر" instead of "booking"
                service_name = booking_state.get("service_name")
                if service_name:
                    session_data["current_topic"] = service_name
                else:
                    session_data["current_topic"] = "booking"
            
            self.session_manager.put(self.session_key, session_data, ttl_minutes=120)
            
            # LOG BOOKING PROGRESS for observability
            step = booking_state.get("step", "unknown")
            has_service = "✓" if booking_state.get("service_name") else "✗"
            has_doctor = "✓" if booking_state.get("doctor_name") else "✗"
            has_date = "✓" if booking_state.get("preferred_date") else "✗"
            has_time = "✓" if booking_state.get("preferred_time") else "✗"
            
            logger.info(
                f"💾 BOOKING STATE SAVED: step={step} | "
                f"service{has_service} doctor{has_doctor} date{has_date} time{has_time} | "
                f"phase={session_data.get('current_phase', 'unknown')} | "
                f"topic={session_data.get('current_topic', 'none')}"
            )
        except Exception as exc:
            logger.error(f"Failed to save booking state: {exc}")
    
    def _clear_booking_state(self, reset_phase: bool = True) -> None:
        """
        Clear booking state from session.
        
        Args:
            reset_phase: If True, reset phase to discovery. If False, keep current phase (Issue #12)
        """
        try:
            session_data = self.session_manager.get(self.session_key) or {}
            session_data["booking_state"] = {}
            session_data["booking_step"] = None
            
            # CRITICAL: Only reset phase if explicitly requested (Issue #12)
            # During errors, we want to keep phase="booking" for context
            if reset_phase:
                session_data["current_phase"] = "discovery"
                session_data["current_topic"] = None
                logger.info("🗑️ BOOKING STATE CLEARED: phase reset to discovery")
            else:
                # Keep current phase (usually "booking") - user is still in booking context
                logger.info(f"🗑️ BOOKING STATE CLEARED: phase kept as '{session_data.get('current_phase', 'unknown')}'")
            
            self.session_manager.put(self.session_key, session_data, ttl_minutes=120)
            
        except Exception as exc:
            logger.error(f"Clear booking state error: {exc}", exc_info=True)
    
    async def _extract_and_update_booking_info(
        self, 
        message: str, 
        booking_state: dict, 
        phone_number: str,
        is_pure_intent: bool = False
    ) -> None:
        """Extract booking information from natural language and update state"""
        try:
            message_lower = message.lower()
            
            # CRITICAL: Skip extraction if this is a pure intent keyword from router
            if is_pure_intent:
                logger.info(f"⏭️ SKIPPING entity extraction - '{message}' is pure booking intent (router confirmed)")
                return
            
            # CRITICAL: Skip extraction for confirmation keywords in active booking
            CONFIRMATION_KEYWORDS = ['يلا', 'تمام', 'اوك', 'ok', 'yes', 'نعم', 'ماشي', 'زين', 'طيب', 'لا', 'no']
            booking_active = booking_state.get("started", False)
            
            if booking_active and message_lower.strip() in CONFIRMATION_KEYWORDS:
                logger.info(f"⏭️ SKIPPING entity extraction - '{message}' is confirmation keyword in active booking")
                return
            
            # LOOP DETECTION: Check if user is sending the same message repeatedly
            last_message = booking_state.get("last_user_message", "")
            repeat_count = booking_state.get("message_repeat_count", 0)
            
            if message_lower == last_message.lower():
                repeat_count += 1
                booking_state["message_repeat_count"] = repeat_count
                
                if repeat_count >= 3:
                    logger.warning(f"🔁 LOOP DETECTED: User sent '{message}' {repeat_count} times - they're stuck!")
                    # This will be handled in _handle_new_booking with a helpful message
            else:
                # Different message - reset counter
                booking_state["last_user_message"] = message_lower
                booking_state["message_repeat_count"] = 1
            
            # CRITICAL: STEP-AWARE EXTRACTION - only extract what user should provide now
            # This prevents extracting date when service is still missing
            has_service = booking_state.get("service_id") or booking_state.get("service_name")
            has_doctor = booking_state.get("doctor_id") or booking_state.get("doctor_name")
            has_specialist = booking_state.get("specialist_id")
            has_device = booking_state.get("device_id")
            has_resource = has_doctor or has_specialist or has_device
            has_date = booking_state.get("preferred_date")
            
            # CRITICAL: State consistency check - fix mismatches between resource_type and actual resource IDs
            resource_type = booking_state.get("resource_type")
            if resource_type == "device" and has_doctor and not has_device:
                logger.warning(f"🧹 STATE MISMATCH: resource_type=device but doctor_id exists without device_id - cleaning up")
                booking_state.pop("doctor_id", None)
                booking_state.pop("doctor_name", None)
                booking_state.pop("doctor_name_en", None)
                booking_state.pop("doctor_selected", None)
                if booking_state.get("step") == "doctor_selected":
                    booking_state["step"] = "service_selected"
                    logger.info(f"🔄 Reset step from doctor_selected → service_selected")
                has_doctor = None
                has_resource = has_device
            elif resource_type == "doctor" and (has_specialist or has_device) and not has_doctor:
                logger.warning(f"🧹 STATE MISMATCH: resource_type=doctor but has specialist/device - cleaning up")
                booking_state.pop("specialist_id", None)
                booking_state.pop("specialist_name", None)
                booking_state.pop("device_id", None)
                booking_state.pop("device_name", None)
                has_specialist = None
                has_device = None
                has_resource = has_doctor
            elif resource_type == "specialist" and (has_doctor or has_device) and not has_specialist:
                logger.warning(f"🧹 STATE MISMATCH: resource_type=specialist but has doctor/device - cleaning up")
                booking_state.pop("doctor_id", None)
                booking_state.pop("doctor_name", None)
                booking_state.pop("doctor_name_en", None)
                booking_state.pop("doctor_selected", None)
                booking_state.pop("device_id", None)
                booking_state.pop("device_name", None)
                has_doctor = None
                has_device = None
                has_resource = has_specialist
            
            logger.debug(f"🔍 STEP-AWARE EXTRACTION: has_service={bool(has_service)}, has_doctor={bool(has_doctor)}, has_date={bool(has_date)}")
            
            # Extract service (match against available services) - flexible matching
            # ONLY if service is not already selected
            if not has_service:
                services = booking_state.get("services", [])
                logger.debug(f"🔍 Extracting SERVICE from: '{message_lower}' | Available: {len(services)}")
                
                for service in services:
                    service_name = service.get("name", "").lower()
                    service_keywords = service_name.split()
                    
                    # Check multiple matching strategies:
                    # 1. Exact service name in message
                    # 2. Any word from service name in message
                    # 3. Any word from message in service name (reverse match)
                    message_words = message_lower.split()
                    
                    if (service_name in message_lower or 
                        any(word in message_lower for word in service_keywords) or
                        any(word in service_name for word in message_words if len(word) > 2)):
                        
                        booking_state["service_id"] = service.get("id")
                        booking_state["service_name"] = service.get("name")
                        booking_state["step"] = "service_selected"
                        logger.info(f"✅ Extracted service: {service.get('name')}")
                        
                        # CRITICAL: Fetch doctors for this service
                        try:
                            service_id = service.get("id")
                            logger.info(f"🔍 Fetching doctors for extracted service_id={service_id}...")
                            doctors_result = await self.api_client.get(f"/services/{service_id}/doctors")
                            service_doctors = doctors_result.get("results") or doctors_result.get("data") or []
                            
                            # If no service-specific doctors, FETCH all doctors
                            if not service_doctors:
                                logger.warning(f"⚠️ No doctors found for service {service_id}, fetching all doctors as fallback...")
                                all_doctors_result = await self.api_client.get("/doctors", params={"limit": 20})
                                service_doctors = all_doctors_result.get("results") or all_doctors_result.get("data") or []
                                logger.info(f"✅ Loaded {len(service_doctors)} general doctors as fallback")
                            
                            # Update booking state with doctors (service-specific or all)
                            booking_state["doctors"] = service_doctors
                            logger.info(f"✅ Final doctor list: {len(service_doctors)} doctors available: {[d.get('name') for d in service_doctors[:5]]}")
                        except Exception as e:
                            logger.error(f"❌ Failed to fetch doctors for service: {e}")
                            # Try to fetch all doctors as last resort
                            try:
                                logger.info("🔄 Attempting to fetch all doctors as last resort...")
                                all_doctors_result = await self.api_client.get("/doctors", params={"limit": 20})
                                booking_state["doctors"] = all_doctors_result.get("results") or all_doctors_result.get("data") or []
                                logger.info(f"✅ Emergency fallback: Loaded {len(booking_state['doctors'])} doctors")
                            except Exception as fallback_error:
                                logger.error(f"❌ Failed to fetch doctors even as fallback: {fallback_error}")
                        
                        break
            else:
                logger.debug(f"⏭️ SKIPPING service extraction - already selected")
            
            # CRITICAL: Check for date/time keywords BEFORE doctor matching
            # Prevents matching "باكر" (tomorrow) as doctor name
            DATE_KEYWORDS = ["باكر", "بكرا", "بكره", "اليوم", "tomorrow", "today", "الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"]
            is_date_keyword = any(kw in message_lower for kw in DATE_KEYWORDS)
            
            # Check if service requires doctor (skip doctor matching for device-only services)
            resource_type = booking_state.get("resource_type", "doctor")
            requires_doctor_or_specialist = resource_type in ["doctor", "specialist"]
            
            if is_date_keyword:
                logger.info(f"⏭️ SKIPPING doctor matching - '{message}' is a date keyword")
            elif not requires_doctor_or_specialist:
                logger.info(f"⏭️ SKIPPING doctor matching - service requires {resource_type}, not doctor")
            
            # Extract doctor preference - ENHANCED MATCHING
            # CRITICAL: ONLY extract doctor if service is already selected AND requires doctor
            # This enforces correct flow: Service → Doctor → Date → Time
            if has_service and not has_doctor and not is_date_keyword and requires_doctor_or_specialist:
                # CRITICAL: Filter out NON-DATA keywords before matching
                # These are social/intent words that should NOT be matched as doctor names
                SOCIAL_AND_INTENT_KEYWORDS = [
                    # Booking intents
                    'حجز', 'احجز', 'أحجز', 'موعد', 'حجز موعد', 'احجز موعد',
                    'book', 'booking', 'appointment', 'reserve', 'reservation',
                    
                    # Confirmation/acknowledgment
                    'يلا', 'تمام', 'اوك', 'ok', 'okay', 'yes', 'نعم', 'لا', 'no',
                    'ماشي', 'زين', 'طيب', 'خلاص', 'أوكي',
                    
                    # GREETINGS & SOCIAL (CRITICAL - was missing!)
                    'هلا', 'هلو', 'اهلا', 'أهلا', 'مرحبا', 'مراحب', 'مرحبتين',
                    'hello', 'hi', 'hey', 'greetings',
                    'السلام عليكم', 'سلام', 'صباح الخير', 'مساء الخير',
                    
                    # Thanks & politeness
                    'شكرا', 'شكراً', 'شكرً', 'thank you', 'thanks', 'thx',
                    'يعطيك العافية', 'الله يعطيك العافية', 'تسلم', 'ماقصرت',
                    'sorry', 'excuse me', 'آسف', 'معذرة', 'عذراً',
                    
                    # Questions
                    'خدمة', 'خدمات', 'service', 'services', 'دكتور', 'دكاترة',
                    'متى', 'when', 'وين', 'where', 'كيف', 'how', 'وش', 'what',
                    'ليه', 'why', 'مين', 'who', 'كم', 'how much',
                    
                    # Choice delegation
                    'اختار', 'أختار', 'اخترلي', 'أخترلي', 'انت اختار', 'choose', 'select',
                    'أي واحد', 'اي واحد', 'ما يهم', 'مايهم', 'ما يهمني', 'انت شوف',
                    'انت اشوف', 'أنت شوف', 'على راحتك', 'عادي', 'any', 'whatever', 'you choose'
                ]
                
                # Skip doctor matching if message is a social/intent keyword (not actual data)
                is_social_or_intent = message_lower.strip() in SOCIAL_AND_INTENT_KEYWORDS
                
                if is_social_or_intent:
                    logger.info(f"⏭️ SKIPPED doctor matching - '{message}' is social/intent keyword, not a doctor name")
                
                doctors = booking_state.get("doctors", [])
                
                # Validate doctors list - fetch if missing
                if not doctors:
                    logger.warning(f"⚠️ Doctor list missing from state - fetching from API...")
                    logger.debug(f"Service ID: {booking_state.get('service_id')}, Service Name: {booking_state.get('service_name')}")
                    
                    # Fetch doctors (normal operation, not emergency)
                    try:
                        emergency_doctors = await self.api_client.get("/doctors", params={"limit": 20})
                        doctors = emergency_doctors.get("results") or emergency_doctors.get("data") or []
                        if doctors:
                            booking_state["doctors"] = doctors
                            logger.info(f"✅ Fetched {len(doctors)} doctors from API")
                        else:
                            logger.error("❌ CRITICAL: API returned empty doctor list - cannot proceed")
                    except Exception as recovery_error:
                        logger.error(f"❌ Failed to fetch doctors: {recovery_error}")
                
                logger.debug(f"🔍 Extracting DOCTOR from: '{message_lower}' | Available doctors: {[d.get('name', 'Unknown') for d in doctors]}")
                
                matched_doctor = None
                
                # ONLY match doctor if NOT a social/intent keyword AND doctors list is not empty
                if not is_social_or_intent and doctors:
                    for doctor in doctors:
                        # Get both English and Arabic names for matching
                        doctor_name_en = doctor.get("name", "").lower()
                        doctor_name_ar = doctor.get("name_ar", "").lower()
                        doctor_first_name_en = doctor_name_en.split()[0] if doctor_name_en else ""
                        doctor_first_name_ar = doctor_name_ar.split()[0] if doctor_name_ar else ""
                        
                        # Match strategies - check BOTH English and Arabic names:
                        # 1. Full name match (English or Arabic)
                        # 2. First name match (English or Arabic)
                        # 3. Partial name match (any word in either language)
                        if (doctor_name_en in message_lower or 
                            message_lower in doctor_name_en or
                            doctor_name_ar in message_lower or
                            message_lower in doctor_name_ar or
                            doctor_first_name_en in message_lower or
                            message_lower in doctor_first_name_en or
                            doctor_first_name_ar in message_lower or
                            message_lower in doctor_first_name_ar):
                            
                            matched_doctor = doctor
                            booking_state["doctor_id"] = doctor.get("id")
                            # ALWAYS use Arabic name for display
                            booking_state["doctor_name"] = doctor.get("name_ar") or doctor.get("name")
                            booking_state["doctor_name_en"] = doctor.get("name")  # Keep English for API
                            booking_state["doctor_selected"] = True
                            booking_state["step"] = "doctor_selected"
                            display_name = doctor.get("name_ar") or doctor.get("name")
                            logger.info(f"✅ DOCTOR MATCHED: '{message}' → {display_name} (ID: {doctor.get('id')})")
                            break
                else:
                    logger.info(f"⏭️ SKIPPED doctor matching - '{message}' is a social/intent keyword, not a doctor name")
                
                # CRITICAL: If user mentioned a name but no match found, flag it
                # BUT exclude social/intent keywords from "not found" logic
                # AND only flag if doctors list was actually populated
                if (not matched_doctor and 
                    not is_social_or_intent and 
                    len(message.split()) <= 2 and 
                    not any(char.isdigit() for char in message) and
                    doctors):  # Only flag if there ARE doctors to search through
                    # User likely mentioned a doctor name that doesn't exist (not a greeting/social message)
                    logger.warning(f"⚠️ DOCTOR NOT FOUND: User said '{message}' but no match in {len(doctors)} available doctors: {[d.get('name') for d in doctors]}")
                    booking_state["doctor_search_failed"] = message
                    booking_state["step"] = "doctor_not_found"
                elif not matched_doctor and not doctors:
                    # No doctors available at all - this is a system issue, not user error
                    logger.error(f"❌ SYSTEM ISSUE: Cannot extract doctor - doctor list is empty! User said: '{message}'")
                    logger.error("This booking cannot proceed without doctors. Check API or service configuration.")
            else:
                logger.debug(f"⏭️ SKIPPING doctor extraction - service not selected yet or doctor already selected")
            
            # Extract date (Arabic keywords)
            # CRITICAL: ONLY extract date if service AND resource are already selected
            # This enforces correct flow: Service → Resource → Date → Time
            if has_service and has_resource and not has_date:
                from datetime import datetime, timedelta
                today = datetime.now()
                logger.debug(f"🔍 Extracting DATE from: '{message_lower}'")
                
                if any(word in message_lower for word in ["بكرة", "باكر", "بكره", "غداً", "غدا", "tomorrow"]):
                    booking_state["preferred_date"] = (today + timedelta(days=1)).strftime("%Y-%m-%d")
                    booking_state["step"] = "date_selected"
                    logger.info(f"✅ Extracted date: tomorrow")
                elif any(word in message_lower for word in ["اليوم", "today"]):
                    booking_state["preferred_date"] = today.strftime("%Y-%m-%d")
                    booking_state["step"] = "date_selected"
                    logger.info(f"✅ Extracted date: today")
                elif any(word in message_lower for word in ["بعد بكرة", "بعد غد"]):
                    booking_state["preferred_date"] = (today + timedelta(days=2)).strftime("%Y-%m-%d")
                    booking_state["step"] = "date_selected"
                    logger.info(f"✅ Extracted date: day after tomorrow")
            else:
                logger.debug(f"⏭️ SKIPPING date extraction - service or resource not selected yet, or date already selected")
            
            # Extract time (Arabic keywords + numbers) - with spelling variations
            # CRITICAL: ONLY extract time if service, resource, AND date are already selected
            # This enforces correct flow: Service → Resource → Date → Time
            has_time = booking_state.get("preferred_time")
            if has_service and has_resource and has_date and not has_time:
                logger.debug(f"🔍 Extracting TIME from: '{message_lower}'")
                
                if any(word in message_lower for word in ["صباح", "الصباح", "صبح", "الصبح", "صبح", "morning"]):
                    booking_state["preferred_time"] = "10:00"  # Default morning time
                    booking_state["step"] = "time_selected"
                    logger.info(f"✅ Extracted time: morning (10:00)")
                elif any(word in message_lower for word in ["ظهر", "الظهر", "ظهر", "الظهر", "noon", "دهر"]):
                    booking_state["preferred_time"] = "12:00"
                    booking_state["step"] = "time_selected"
                    logger.info(f"✅ Extracted time: noon (12:00)")
                elif any(word in message_lower for word in ["عصر", "العصر", "عصر", "afternoon", "بعد الظهر"]):
                    booking_state["preferred_time"] = "15:00"  # 3 PM
                    booking_state["step"] = "time_selected"
                    logger.info(f"✅ Extracted time: afternoon (15:00)")
                elif any(word in message_lower for word in ["مساء", "المساء", "مسا", "المسا", "evening", "ليل"]):
                    booking_state["preferred_time"] = "18:00"  # 6 PM
                    booking_state["step"] = "time_selected"
                    logger.info(f"✅ Extracted time: evening (18:00)")
            else:
                logger.debug(f"⏭️ SKIPPING time extraction - service, doctor or date not selected yet, or time already selected")
            
            # Extract specific hour (e.g., "10", "3", "الساعة 10")
            import re
            time_match = re.search(r'(\d{1,2})\s*(am|pm|ص|م)?', message_lower)
            if time_match:
                hour = int(time_match.group(1))
                period = time_match.group(2)
                
                # Convert to 24-hour format
                if period in ["pm", "م"] and hour < 12:
                    hour += 12
                elif period in ["am", "ص"] and hour == 12:
                    hour = 0
                
                booking_state["preferred_time"] = f"{hour:02d}:00"
                booking_state["step"] = "time_selected"
                logger.info(f"✅ Extracted specific time: {hour:02d}:00")
            
            # Save updated state
            self._save_booking_state(booking_state)
            
        except Exception as exc:
            logger.error(f"Extract booking info error: {exc}", exc_info=True)
    
    async def _handle_registration_name(
        self, message: str, booking_state: dict, phone_number: str, arabic_name: str
    ) -> dict:
        """Handle name collection during registration"""
        # Extract full name (must have at least first and last name)
        name = message.strip()
        name_parts = name.split()
        
        if len(name_parts) < 2:
            return {
                "response": f"يا {arabic_name}، أحتاج الاسم الكامل (الأول والأخير) 📝\n\nمثلاً: أحمد محمد",
                "intent": "booking",
                "status": "registration_name"
            }
        
        # Store name and move to national ID
        booking_state["registration"]["name"] = name
        booking_state["step"] = "registration_national_id"
        self._save_booking_state(booking_state)
        
        return {
            "response": f"ممتاز يا {name_parts[0]}! 👍\n\nالحين أحتاج رقم الهوية الوطنية؟ 🆔",
            "intent": "booking",
            "status": "registration_national_id"
        }
    
    async def _handle_registration_national_id(
        self, message: str, booking_state: dict, arabic_name: str
    ) -> dict:
        """Handle national ID collection"""
        import re
        
        # Extract ID number (10 digits for Saudi ID)
        id_match = re.search(r'\d{10}', message)
        
        if not id_match:
            return {
                "response": f"يا {arabic_name}، رقم الهوية لازم يكون 10 أرقام 🔢\n\nجرب مرة ثانية؟",
                "intent": "booking",
                "status": "registration_national_id"
            }
        
        national_id = id_match.group(0)
        
        # Check if ID already exists
        try:
            search_result = await self.api_client.get(f"/patients/search?identification_id={national_id}")
            if search_result and search_result.get("data"):
                return {
                    "response": f"يا {arabic_name}، رقم الهوية هذا مسجل عندنا من قبل! ✅\n\nتقدر تحجز مباشرة\nاكتب 'حجز' عشان نكمل 📅",
                    "intent": "booking",
                    "status": "already_registered"
                }
        except Exception as e:
            logger.error(f"Error checking ID: {e}")
        
        # Store ID and move to phone confirmation
        booking_state["registration"]["identification_id"] = national_id
        booking_state["step"] = "registration_phone_confirm"
        self._save_booking_state(booking_state)
        
        phone = booking_state["registration"]["phone"]
        return {
            "response": f"تمام! 👌\n\nرقم جوالك هو: {phone}\n\nصحيح؟ (اكتب: نعم أو لا) 📱",
            "intent": "booking",
            "status": "registration_phone_confirm"
        }
    
    async def _handle_registration_phone_confirm(
        self, message: str, booking_state: dict, phone_number: str, arabic_name: str
    ) -> dict:
        """Handle phone number confirmation"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["نعم", "صحيح", "اي", "yes", "correct", "تمام", "أكيد"]):
            # Phone confirmed, ask for optional info
            booking_state["registration"]["patient_phone"] = phone_number
            booking_state["step"] = "registration_optional"
            self._save_booking_state(booking_state)
            
            return {
                "response": f"""عظيم! 🎉

**معلومات اختيارية** (تقدر تتخطاها):

• عندك أي حساسية؟ (مثلاً: بنسلين)
• أي ملاحظات صحية تحب نعرفها؟

اكتب المعلومات أو اكتب 'لا' عشان نكمل ⏭️""",
                "intent": "booking",
                "status": "registration_optional"
            }
        else:
            # Phone not confirmed, ask for correct number
            return {
                "response": f"طيب، اكتب رقم الجوال الصحيح (05xxxxxxxx) 📱",
                "intent": "booking",
                "status": "registration_phone_update"
            }
    
    async def _handle_registration_optional(
        self, message: str, booking_state: dict, arabic_name: str
    ) -> dict:
        """Handle optional information (allergies, notes)"""
        message_lower = message.lower()
        
        # Check if user wants to skip
        if any(word in message_lower for word in ["لا", "no", "skip", "تخطي", "ما في", "مافي"]):
            booking_state["registration"]["allergies"] = None
            booking_state["registration"]["notes"] = None
        else:
            # Store as notes (can be parsed later)
            booking_state["registration"]["allergies"] = message if "حساسية" in message_lower or "allerg" in message_lower else None
            booking_state["registration"]["notes"] = message
        
        # Move to confirmation
        booking_state["step"] = "registration_confirm"
        self._save_booking_state(booking_state)
        
        # Show confirmation
        reg = booking_state["registration"]
        confirmation_text = f"""تمام! خليني أتأكد من المعلومات 📋

**الاسم:** {reg['name']}
**الهوية:** {reg['identification_id']}
**الجوال:** {reg['patient_phone']}"""
        
        if reg.get("allergies"):
            confirmation_text += f"\n**حساسية:** {reg['allergies']}"
        if reg.get("notes"):
            confirmation_text += f"\n**ملاحظات:** {reg['notes']}"
        
        confirmation_text += "\n\n**كل شي صحيح؟**\n\n• اكتب 'نعم' للتأكيد\n• اكتب 'لا' للتعديل"
        
        return {
            "response": confirmation_text,
            "intent": "booking",
            "status": "registration_confirm"
        }
    
    async def _handle_registration_confirm(
        self, message: str, booking_state: dict, arabic_name: str, context: dict
    ) -> dict:
        """Handle final confirmation and create patient"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["نعم", "yes", "صحيح", "تمام", "أكيد", "confirm"]):
            try:
                reg = booking_state["registration"]
                
                # Detect gender from conversation context (use LLM)
                llm = get_llm_reasoner()
                
                # Get conversation history for gender detection
                conversation_history = context.get("conversation_history", []) if context else []
                history_text = "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in conversation_history[-10:]])
                
                gender_prompt = f"""Based on the Arabic name and conversation, determine the gender.

Name: {reg['name']}
Conversation:
{history_text}

Respond with ONLY one word: "male" or "female"
Common male names: أحمد، محمد، خالد، عبدالله، فهد، سلطان
Common female names: فاطمة، نورة، سارة، مريم، هند، لطيفة

Gender:"""
                
                gender_response = llm.generate_reply(
                    user_id=reg['patient_phone'],
                    user_message=gender_prompt,
                    context={"intent": "gender_detection"}
                ).strip().lower()
                
                gender = "male" if "male" in gender_response or "ذكر" in gender_response else "female"
                
                # Extract birth date from national ID
                from app.utils.national_id_parser import get_birth_date_from_national_id
                birth_date = get_birth_date_from_national_id(
                    reg["identification_id"],
                    fallback="1990-01-01"
                )
                
                # Create patient record - MATCH EXACT BACKEND STRUCTURE
                # Required fields: name, identification_id, gender, patient_phone, birth_date
                # Optional fields: Pass as empty strings (backend expects all fields)
                patient_data = {
                    "name": reg["name"],  # REQUIRED: Collected from user
                    "identification_id": reg["identification_id"],  # REQUIRED: National ID
                    "gender": gender,  # REQUIRED: Collected
                    "patient_phone": reg["patient_phone"],  # REQUIRED: From registration
                    "birth_date": birth_date,  # REQUIRED: Extracted from national ID
                    "city": "الرياض",  # Default city
                    "country_code": "SA",  # Default country
                    "email": "",  # Optional - empty
                    "blood_type": "",  # Optional - empty
                    "chronic_diseases": "",  # Optional - empty
                    "allergies": "",  # Optional - empty
                    "notes": "",  # Optional - empty
                    "registration_type": "agent",  # Backend only accepts: 'agent' (NOT 'whatsapp_bot'!)
                    "reference_by": "whatsapp"  # Channel: WhatsApp
                }
                
                logger.info(f"Creating patient: {patient_data['name']}")
                # CRITICAL: Use /customer/create endpoint, not /patients (405 error fix)
                result = await self.api_client.post("/customer/create", json_body=patient_data)
                
                if result and result.get("id"):
                    patient_id = result["id"]
                    booking_state["patient_id"] = patient_id
                    booking_state["patient_verified"] = True
                    booking_state["step"] = "initial"  # Move to booking flow
                    booking_state.pop("registration", None)  # Clear registration data
                    self._save_booking_state(booking_state)
                    
                    logger.info(f"✅ Patient created successfully: ID={patient_id}")
                    
                    return {
                        "response": f"""ممتاز {arabic_name}! تم التسجيل بنجاح ✅

الحين تقدر تحجز موعدك 📅

اكتب 'حجز' عشان نبدأ 🚀""",
                        "intent": "booking",
                        "status": "registration_complete"
                    }
                else:
                    raise Exception("Failed to create patient")
                    
            except Exception as e:
                logger.error(f"❌ Error creating patient: {e}", exc_info=True)
                return {
                    "response": f"""يا عيني {arabic_name} 😅

ما قدرت أكمل التسجيل

جرب:
• انتظر شوي وجرب مرة ثانية
• أو اتصل على: 920033304 📞""",
                    "intent": "booking",
                    "status": "registration_error"
                }
        else:
            # User wants to edit
            booking_state["step"] = "registration_name"
            booking_state.pop("registration", None)
            self._save_booking_state(booking_state)
            
            return {
                "response": f"تمام {arabic_name}! خليني ناخذ المعلومات من جديد 🔄\n\nابدأ معي بالاسم الكامل؟ 📝",
                "intent": "booking",
                "status": "registration_restart"
            }
    
    @staticmethod
    def _validate_error_response(response: str) -> bool:
        """
        Validate that error response meets quality standards.
        
        Requirements:
        1. Has actionable guidance (keywords: جرب, اكتب, اختر, اتصل)
        2. Not too long (< 300 chars for WhatsApp readability)
        3. Has friendly tone (emoji, يا عيني)
        4. Has contact info OR alternative action
        
        Returns:
            True if response meets standards
        """
        # Check for actionable keywords
        action_keywords = ['جرب', 'اكتب', 'اختر', 'اتصل', 'انتظر', 'try', 'type', 'call', 'wait']
        has_action = any(keyword in response.lower() for keyword in action_keywords)
        
        # Check for contact info or alternative
        has_contact = '920033304' in response or 'خدمات' in response or 'إلغاء' in response
        
        # Check for friendly tone
        has_friendly_tone = '😅' in response or 'يا عيني' in response or 'يا' in response
        
        # Length check
        reasonable_length = len(response) < 400
        
        is_valid = has_action and has_contact and has_friendly_tone and reasonable_length
        
        if not is_valid:
            logger.warning(f"⚠️ Error response quality check FAILED: action={has_action}, contact={has_contact}, friendly={has_friendly_tone}, length={len(response)}")
        
        return is_valid
    
    def _build_error_response(self, error_type: str, user_name: str = "حبيبنا", context: dict = None) -> str:
        """
        Build user-friendly error response with clear next steps.
        
        IMPORTANT: All error responses MUST include:
        1. Friendly greeting (يا عيني يا {name})
        2. Brief explanation of what went wrong
        3. Actionable next steps (numbered list with • bullets)
        4. Support phone number OR alternative commands
        5. Appropriate emoji (😅 for errors, 📞 for contact, 🔄 for retry)
        
        Args:
            error_type: Type of error (timeout, api_error, etc.)
            user_name: User's name for personalization
            context: Additional context (service_name, etc.)
        
        Returns:
            Error message with actionable guidance
        """
        context = context or {}
        
        error_templates = {
            "timeout": f"""يا عيني يا {user_name} 😅

استغرق الوقت طويل جداً

جرب:
• انتظر 10 ثواني وجرب مرة ثانية
• أو اختر خدمة ثانية
• أو اتصل على: 920033304 📞""",
            
            "api_error": f"""يا عيني يا {user_name} 😅

ما قدرت أوصل للنظام

جرب:
• انتظر شوي وجرب مرة ثانية
• أو اتصل على: 920033304 📞""",
            
            "no_resources": f"""يا عيني يا {user_name} 😅

ما في {context.get('resource_type', 'مواعيد')} متاحة حالياً

خياراتك:
• جرب خدمة ثانية (اكتب 'خدمات')
• جرب موعد ثاني
• أو اتصل على: 920033304 📞""",
            
            "validation_error": f"""يا عيني يا {user_name} 😅

فيه خطأ في المعلومات

جرب:
• تأكد من البيانات وجرب مرة ثانية
• أو اكتب 'إلغاء' عشان نبدأ من جديد 🔄""",
            
            "service_unavailable": f"""يا عيني يا {user_name} 😅

{context.get('service_name', 'هالخدمة')} مو متاحة مؤقتاً

خياراتك:
• اختر خدمة ثانية (اكتب 'خدمات')
• جرب بعد شوي
• أو اتصل على: 920033304 📞""",
            
            "generic": f"""يا عيني يا {user_name} 😅

صار خطأ مؤقت

جرب هالخيارات:
• اكتب 'حجز' عشان تحجز موعد
• اكتب 'مواعيدي' عشان تشوف حجوزاتك
• اكتب 'إلغاء' عشان تلغي حجز
• أو اتصل على: 920033304 📋"""
        }
        
        response = error_templates.get(error_type, error_templates["generic"])
        
        # QUALITY CHECK: Validate response meets standards
        if not self._validate_error_response(response):
            logger.error(f"🚨 ERROR RESPONSE QUALITY FAILURE: type={error_type}")
            # Fallback to generic template (already validated)
            response = error_templates["generic"]
        
        logger.info(f"📝 Built error response: type={error_type}, has_guidance=True, validated=True")
        return response
    
    def _clear_booking_state(self):
        """Clear booking state"""
        try:
            booking_state = self._load_booking_state()
            booking_state.clear()
            booking_state["started"] = False
            self._save_booking_state(booking_state)
            logger.info("🗑️ Booking state cleared")
        except Exception as exc:
            logger.error(f"Clear booking state error: {exc}")
    
    def _cleanup_failed_booking_state(self, booking_state: dict, keep_service: bool = False):
        """
        Clean up booking state after error while preserving session continuity.
        
        Args:
            booking_state: Current booking state
            keep_service: If True, keep service selection (for retries)
        """
        logger.info(f"🧹 Cleaning up failed booking state (keep_service={keep_service})")
        
        # What to clear
        keys_to_clear = [
            "doctor_id", "doctor_name", "doctor_name_en", "doctor_selected",
            "specialist_id", "specialist_name",
            "device_id", "device_name",
            "preferred_date", "preferred_time",
            "awaiting_confirmation",
            "doctors", "specialists", "devices",
            "displayed_services",  # Temporary UI state
            "last_user_message",  # Loop detection state
            "message_repeat_count"
        ]
        
        # Clear failure tracking if service is being removed
        if not keep_service:
            keys_to_clear.extend([
                "service_id", "service_name",
                "resource_type",
                "services",
                "service_types"
            ])
            # Also clear all failure counters
            failure_keys = [k for k in booking_state.keys() if k.startswith("fetch_resource_failures_")]
            keys_to_clear.extend(failure_keys)
        
        for key in keys_to_clear:
            booking_state.pop(key, None)
        
        # Reset to safe state
        if keep_service and booking_state.get("service_id"):
            booking_state["step"] = "service_selected"
            logger.info(f"✅ State cleaned, preserved service: {booking_state.get('service_name')}")
        else:
            booking_state["step"] = "started"
            booking_state["started"] = True
            logger.info(f"✅ State cleaned, reset to started")
        
        # Reset critical failure counter if present
        if "critical_failures" in booking_state:
            booking_state["critical_failures"] = 0
        
        self._save_booking_state(booking_state)
        logger.info(f"💾 Clean state saved: step={booking_state['step']}")

    async def _complete_booking(
        self, 
        booking_state: dict, 
        phone_number: str, 
        sender_name: str,
        llm_response: str
    ) -> dict:
        """Complete the booking - verify patient and create appointment"""
        try:
            # CRITICAL VALIDATION: Ensure all required info is present
            required_fields = {
                "service_id": booking_state.get("service_id"),
                "service_name": booking_state.get("service_name"),
                "preferred_date": booking_state.get("preferred_date"),
                "preferred_time": booking_state.get("preferred_time")
            }
            
            missing_fields = [k for k, v in required_fields.items() if not v]
            
            if missing_fields:
                logger.error(f"🚫 BOOKING VALIDATION FAILED: Missing required fields: {missing_fields}")
                logger.error(f"🚫 Current state: service={booking_state.get('service_name')}, doctor={booking_state.get('doctor_name')}, date={booking_state.get('preferred_date')}, time={booking_state.get('preferred_time')}")
                
                # Return error - DO NOT confirm booking
                return {
                    "response": "لحظة شوي 😅\nتبغى تحجز أي خدمة بالضبط؟",
                    "intent": "booking",
                    "status": "missing_info",
                    "missing_fields": missing_fields
                }
            
            logger.info(f"✅ BOOKING VALIDATION PASSED: All required fields present")
            
            from ..core.llm_reasoner import get_llm_reasoner
            llm = get_llm_reasoner()
            
            # First, verify/get patient
            patient = await self._verify_patient(phone_number)
            
            if not patient:
                # Patient doesn't exist - ask to register
                llm_context = {
                    "sender_name": sender_name,
                    "intent": "booking_needs_registration",
                    "booking_step": "patient_registration_required",
                    "collected_booking_info": {
                        "service": booking_state.get("service_name"),
                        "date": booking_state.get("preferred_date"),
                        "time": booking_state.get("preferred_time")
                    }
                }
                
                response_text = llm.generate_reply(
                    user_id=phone_number,
                    user_message="",
                    context=llm_context
                )
                
                return {
                    "response": response_text,
                    "intent": "booking",
                    "status": "needs_registration"
                }
            
            # Patient exists - try to create booking
            try:
                # Use new API operations for booking creation
                result = await self.api_ops.create_booking(
                    patient_id=patient.get("id"),
                    service_id=booking_state.get("service_id"),
                    appointment_date=booking_state.get("preferred_date"),
                    appointment_time=booking_state.get("preferred_time"),
                    doctor_id=booking_state.get("doctor_id"),
                    notes=f"Booked via WhatsApp by {sender_name}"
                )
                
                if result:
                    # Booking created successfully!
                    llm_context = {
                        "sender_name": sender_name,
                        "intent": "booking_confirmed",
                        "booking_step": "completed",
                        "booking_details": {
                            "service": booking_state.get("service_name"),
                            "doctor": booking_state.get("doctor_name", "متاح"),
                            "date": booking_state.get("preferred_date"),
                            "time": booking_state.get("preferred_time"),
                            "patient_name": patient.get("name")
                        },
                        "booking_id": result.get("id")
                    }
                    
                    response_text = llm.generate_reply(
                        user_id=phone_number,
                        user_message="",
                        context=llm_context
                    )
                    
                    # Clear booking state
                    self._clear_booking_state()
                    
                    logger.info(f"✅ Booking completed successfully for {sender_name}")
                    
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": "completed",
                        "booking_id": result.get("id")
                    }
                else:
                    # Booking API failed
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Booking API failed: {error_msg}")
                    
                    llm_context = {
                        "sender_name": sender_name,
                        "intent": "booking_failed",
                        "booking_step": "error",
                        "error_message": error_msg
                    }
                    
                    response_text = llm.generate_reply(
                        user_id=phone_number,
                        user_message="",
                        context=llm_context
                    )
                    
                    return {
                        "response": response_text,
                        "intent": "booking",
                        "status": "error"
                    }
                    
            except Exception as api_error:
                logger.error(f"Booking API error: {api_error}", exc_info=True)
                
                llm_context = {
                    "sender_name": sender_name,
                    "intent": "booking_api_error",
                    "booking_step": "error"
                }
                
                response_text = llm.generate_reply(
                    user_id=phone_number,
                    user_message="",
                    context=llm_context
                )
                
                return {
                    "response": response_text,
                    "intent": "booking",
                    "status": "error"
                }
                
        except Exception as exc:
            logger.error(f"Complete booking error: {exc}", exc_info=True)
            
            language = detect_language(booking_state.get("service_name", ""))
            if language == "arabic":
                response = f"يا عيني 😅\n\nما قدرت أكمل الحجز\n\nجرب:\n• جرب مرة ثانية\n• أو اكتب 'أبغى أحجز' عشان نبدأ من جديد\n• أو اتصل على: 920033304 📞"
            else:
                response = "Sorry, I couldn't complete your booking.\n\nYou can:\n• Try again\n• Type 'book' to start over\n• Call us: 920033304 📞"
            
            return {
                "response": response,
                "intent": "booking",
                "status": "error"
            }
