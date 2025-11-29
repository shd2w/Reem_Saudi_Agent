"""
LangGraph-based Booking Agent

Drop-in replacement for BookingAgent with same interface.
Internally uses LangGraph for state management.
"""
import time
import hashlib
import redis.asyncio
from typing import Dict, Any, Optional
from loguru import logger

from ..config import settings
from ..api.agent_api import AgentApiClient
from ..memory.session_manager import SessionManager
from ..utils.language_detector import detect_language
from ..utils.name_transliterator import transliterate_full_name
from ..utils.phone_parser import extract_saudi_phone  # Issue #41
from ..core.llm_reasoner import get_llm_reasoner

from .langgraph.graph import create_booking_graph
from .langgraph.booking_state import BookingState


class BookingAgentLangGraph:
    """
    LangGraph-based booking agent.
    
    Drop-in replacement for BookingAgent with same interface.
    Internally uses LangGraph for state management.
    
    Architecture: Session-based Instance Pool (preserves original pattern)
    - One instance per session
    - Shared class-level resources
    - Automatic cleanup of stale sessions
    """
    
    # Class-level shared state (PRESERVE FROM ORIGINAL)
    _instances = {}
    _initialized = {}
    _last_access = {}
    _service_circuit_breaker = {}
    _request_cache = {}
    _failure_counters = {}
    _shared_redis_client = None
    _conversation_states = {}  # In-memory state storage (temporary until Redis checkpointing works)
    _graph = None  # Shared, compiled graph (Issue #29)
    
    SESSION_TTL = 1800  # 30 minutes
    
    def __new__(cls, session_id: str):
        """Session-based instance pool (PRESERVE EXACT LOGIC)"""
        cls._cleanup_stale_sessions()
        
        if session_id not in cls._instances:
            cls._instances[session_id] = super(BookingAgentLangGraph, cls).__new__(cls)
            logger.debug(f"📦 Created new LangGraph BookingAgent for {session_id}")
        
        cls._last_access[session_id] = time.time()
        return cls._instances[session_id]
    
    @classmethod
    def _cleanup_stale_sessions(cls):
        """PRESERVE EXACT CLEANUP LOGIC FROM ORIGINAL"""
        current_time = time.time()
        stale_sessions = [
            session_id for session_id, last_time in cls._last_access.items()
            if current_time - last_time > cls.SESSION_TTL
        ]
        
        if stale_sessions:
            for session_id in stale_sessions:
                cls._instances.pop(session_id, None)
                cls._initialized.pop(session_id, None)
                cls._last_access.pop(session_id, None)
                cls._failure_counters.pop(session_id, None)
            logger.info(f"🧹 Cleaned up {len(stale_sessions)} stale LangGraph sessions")
    
    @classmethod
    def _get_shared_redis_client(cls):
        """
        Get or create shared Redis client for LangGraph checkpointing.
        Only create when checkpointing is enabled to avoid unused connections (#28).
        """
        use_checkpoints = getattr(settings, 'langgraph_use_checkpointing', False)
        if not use_checkpoints:
            logger.debug("LangGraph checkpointing disabled; not creating Redis client")
            return None
        if cls._shared_redis_client is None:
            cls._shared_redis_client = redis.asyncio.from_url(
                settings.redis_url,
                decode_responses=False  # LangGraph needs bytes, not strings
            )
            logger.info("✅ Created shared Redis client for LangGraph BookingAgent (checkpointing enabled)")
        return cls._shared_redis_client
    
    def __init__(self, session_id: str, api_client=None):
        """Initialize LangGraph booking agent"""
        if session_id in BookingAgentLangGraph._initialized:
            self.session_id = session_id
            self.session_key = session_id
            # CRITICAL: Always set graph even for existing sessions
            self.graph = self._get_or_create_graph()
            self.api_client = api_client or AgentApiClient()
            self.llm_reasoner = get_llm_reasoner()
            self.session_manager = SessionManager()
            self.redis_client = BookingAgentLangGraph._get_shared_redis_client()
            return
        
        logger.info(f"🆕 Initializing LangGraph BookingAgent for {session_id}")
        BookingAgentLangGraph._initialized[session_id] = True
        
        self.session_id = session_id
        self.session_key = session_id
        self.api_client = api_client or AgentApiClient()
        self.llm_reasoner = get_llm_reasoner()
        self.session_manager = SessionManager()
        
        # Get shared Redis client
        self.redis_client = BookingAgentLangGraph._get_shared_redis_client()
        
        # Get or create the shared, compiled graph (Issue #29)
        self.graph = self._get_or_create_graph()
        
        logger.info(f"✅ LangGraph BookingAgent initialized for {session_id}")
    
    @classmethod
    def _get_or_create_graph(cls):
        """Compile the graph once and reuse it across all instances (🧠 with LLM brain)."""
        if cls._graph is None:
            logger.info("⚙️ Compiling LangGraph booking workflow for the first time...")
            # These clients are singletons, so it's safe to create them here once.
            api_client = AgentApiClient()
            session_manager = SessionManager()
            llm_reasoner = get_llm_reasoner()  # 🧠 LLM brain for intelligent routing
            
            # Create graph with LLM-driven routing
            cls._graph = create_booking_graph(
                api_client=api_client,
                session_manager=session_manager,
                redis_client=cls._get_shared_redis_client(),
                llm_reasoner=llm_reasoner  # 🧠 Pass LLM brain to graph
            )
            logger.info("✅ LangGraph workflow compiled with LLM-Brain architecture")
        return cls._graph

    async def handle(self, payload: dict, context: dict = None) -> dict:
        """
        Handle booking request using LangGraph.
        
        MAINTAINS SAME INTERFACE as original BookingAgent.handle()
        """
        try:
            message = payload.get("message", "").strip()
            # CRITICAL FIX: Accept both "phone_number" and "phone" keys (Issue #1)
            phone_number = payload.get("phone_number") or payload.get("phone", "")
            sender_name = payload.get("sender_name", "Friend")
            session_id = payload.get("session_id", f"whatsapp:{phone_number}")
            is_pure_intent = payload.get("is_pure_intent", False)
            
            # CRITICAL: Log phone number for debugging
            logger.info(f"📞 [LANGGRAPH] Session phone number: '{phone_number}' (from payload)")

            # Issue #37 & #41: Extract phone from message and compare
            message_phone = extract_saudi_phone(message)
            if message_phone and message_phone != phone_number:
                logger.warning(f"📱 Phone number mismatch! Session: {phone_number}, Message: {message_phone}")
            # If payload phone is missing, fall back to message-extracted phone (Issue #45)
            if (not phone_number or phone_number.strip() == "") and message_phone:
                phone_number = message_phone
                # If session_id was derived from phone, update it too
                if payload.get("session_id") is None:
                    session_id = f"whatsapp:{phone_number}"
                logger.info(f"📲 Using phone from message as fallback: {phone_number}")
            
            # Get Arabic name (PRESERVE HELPER USAGE)
            logger.debug(f"🔍 RAW sender_name from WhatsApp: '{sender_name}' (len={len(sender_name)})")
            arabic_name = transliterate_full_name(sender_name) if sender_name != "Friend" else "حبيبنا"
            logger.debug(f"🔍 After transliteration: '{arabic_name}' (len={len(arabic_name)})")
            
            # IDEMPOTENCY CHECK (PRESERVE FROM ORIGINAL)
            # Skip duplicate check for confirmation steps - LLM needs to understand naturally
            duplicate_response = await self._check_request_duplicate(phone_number, message, session_id)
            if duplicate_response:
                logger.info(f"⚡ Returning cached response for duplicate request")
                return duplicate_response
            
            # Extract router intent - use primary 'intent' field from payload
            # Priority: payload['intent'] (user's actual intent) > context['classified_intent'] (fallback)
            router_intent = payload.get('intent') or (context or {}).get('classified_intent', 'booking')
            router_confidence = (context or {}).get('intent_confidence') or payload.get('confidence', 0.0)
            routing_intent = payload.get('routing_intent', router_intent)
            
            # CRITICAL: Extract patient_data from context (already loaded by router)
            patient_data = (context or {}).get('patient_data')
            if patient_data and patient_data.get('already_registered'):
                logger.info(f"✅ [LANGGRAPH] Patient already registered: {patient_data.get('name')} (ID: {patient_data.get('id')})")
            else:
                logger.info(f"⚠️ [LANGGRAPH] No registered patient data in context")
            
            # Log intent tracking (simplified)
            if routing_intent != router_intent:
                logger.info(f"🎯 [LANGGRAPH] Intent: {router_intent} (confidence: {router_confidence:.2f}) → Routed as: {routing_intent}")
            else:
                logger.info(f"🎯 [LANGGRAPH] Intent: {router_intent} (confidence: {router_confidence:.2f})")
            
            logger.info(f"📅 [LANGGRAPH] Processing: '{message[:50]}...'")
            
            # Try loading persisted graph state from in-memory cache or Redis (Issue #24/#50)
            persisted_state = None
            try:
                saved_session = await self.session_manager.get_session(session_id) or {}
                persisted_state = saved_session.get("booking_state")
            except Exception as e:
                logger.debug(f"(debug) Unable to load persisted booking_state from Redis: {e}")

            # Check if we have existing state for this session
            if session_id in BookingAgentLangGraph._conversation_states:
                # Continue from previous state
                initial_state = BookingAgentLangGraph._conversation_states[session_id].copy()
                current_step = initial_state.get('step')
                logger.info(f"📦 [LANGGRAPH] Resuming from step: {current_step}")
                
                # CRITICAL: Inject latest conversation history on resume
                conversation_history = (context or {}).get("conversation_history", [])
                if conversation_history:
                    initial_state["conversation_history"] = conversation_history
                    logger.info(f"📚 [LANGGRAPH] Updated history with {len(conversation_history)} messages on resume")
                
                # CRITICAL FIX: Update patient_data from router on EVERY resume!
                # Router re-checks patient on each request, we must use latest data
                patient_data = (context or {}).get('patient_data')
                if patient_data:
                    initial_state["patient_data"] = patient_data
                    logger.info(f"✅ [RESUME] (Memory path) Updated patient_data from router")
                    
                    # If patient is registered, mark as verified
                    if patient_data.get('already_registered'):
                        initial_state["patient_verified"] = True
                        initial_state["patient_id"] = patient_data.get('id')
                        initial_state["name"] = patient_data.get('name')
                        initial_state["national_id"] = patient_data.get('national_id')
                        initial_state["gender"] = patient_data.get('gender', 'male')
                        logger.info(f"✅ [RESUME] (Memory path) Patient verified from router: {patient_data.get('name')} (ID: {patient_data.get('id')})")
                
                # Update with current message
                initial_state["current_message"] = message
                initial_state["messages"].append({"role": "user", "content": message})
                initial_state["router_intent"] = router_intent
                initial_state["router_confidence"] = router_confidence
                
                # CRITICAL: Pass confirmed data AND service selection from router session
                # If router confirmed name/ID or selected service, pass to LangGraph
                try:
                    saved_session = await self.session_manager.get_session(session_id) or {}
                    logger.info(f"🔍 [DEBUG] Saved session keys: {list(saved_session.keys())}")
                    
                    if saved_session.get("confirmed_name"):
                        initial_state["name"] = saved_session["confirmed_name"]
                        logger.info(f"✅ [RESUME] Set state['name'] = '{saved_session['confirmed_name']}'")
                        logger.info(f"🔍 [DEBUG] state now has name: {initial_state.get('name')}")
                    else:
                        logger.warning(f"⚠️ [DEBUG] No confirmed_name in saved_session")
                        
                    if saved_session.get("confirmed_national_id"):
                        initial_state["national_id"] = saved_session["confirmed_national_id"]
                        logger.info(f"✅ [RESUME] Using confirmed ID from session: {saved_session['confirmed_national_id']}")
                    
                    # CRITICAL FIX (Bug 28): Load service selection on resume (in-memory path)
                    if saved_session.get("selected_service_type_id"):
                        initial_state["selected_service_type_id"] = saved_session["selected_service_type_id"]
                        logger.info(f"✅ [RESUME] Loaded service_type_id={saved_session['selected_service_type_id']}")
                    
                    if saved_session.get("selected_service_id"):
                        initial_state["selected_service_id"] = saved_session["selected_service_id"]
                        logger.info(f"✅ [RESUME] Loaded selected_service_id={saved_session['selected_service_id']}")
                    
                    if saved_session.get("selected_service_name"):
                        initial_state["selected_service_name"] = saved_session["selected_service_name"]
                        logger.info(f"✅ [RESUME] Loaded selected_service_name={saved_session['selected_service_name']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error loading confirmed data from session: {e}")
                
                # Mark as resuming to skip verify_patient and go to process_user_input
                initial_state["_resuming"] = True
                initial_state["_previous_step"] = current_step
            elif persisted_state:
                # Resume from Redis-persisted state (Issue #24/#50)
                initial_state = persisted_state.copy()
                current_step = initial_state.get('step')
                logger.info(f"📦 [LANGGRAPH] Resuming (Redis) from step: {current_step}")
                
                # CRITICAL: Inject latest conversation history on resume
                conversation_history = (context or {}).get("conversation_history", [])
                if conversation_history:
                    initial_state["conversation_history"] = conversation_history
                    logger.info(f"📚 [LANGGRAPH] (Redis) Updated history with {len(conversation_history)} messages on resume")
                
                # CRITICAL FIX: Update patient_data from router on EVERY resume!
                # Router re-checks patient on each request, we must use latest data
                patient_data = (context or {}).get('patient_data')
                if patient_data:
                    initial_state["patient_data"] = patient_data
                    logger.info(f"✅ [RESUME] (Redis path) Updated patient_data from router")
                    
                    # If patient is registered, mark as verified
                    if patient_data.get('already_registered'):
                        initial_state["patient_verified"] = True
                        initial_state["patient_id"] = patient_data.get('id')
                        initial_state["name"] = patient_data.get('name')
                        initial_state["national_id"] = patient_data.get('national_id')
                        initial_state["gender"] = patient_data.get('gender', 'male')
                        logger.info(f"✅ [RESUME] (Redis path) Patient verified from router: {patient_data.get('name')} (ID: {patient_data.get('id')})")
                
                # Update with current message and context
                initial_state["current_message"] = message
                initial_state.setdefault("messages", []).append({"role": "user", "content": message})
                initial_state["router_intent"] = router_intent
                initial_state["router_confidence"] = router_confidence
                
                # CRITICAL: Pass confirmed data AND service selection from router session
                try:
                    saved_session = await self.session_manager.get_session(session_id) or {}
                    logger.info(f"🔍 [DEBUG] (Redis path) Saved session keys: {list(saved_session.keys())}")
                    
                    if saved_session.get("confirmed_name"):
                        initial_state["name"] = saved_session["confirmed_name"]
                        logger.info(f"✅ [RESUME] (Redis path) Set state['name'] = '{saved_session['confirmed_name']}'")
                    else:
                        logger.warning(f"⚠️ [DEBUG] (Redis path) No confirmed_name in saved_session")
                        
                    if saved_session.get("confirmed_national_id"):
                        initial_state["national_id"] = saved_session["confirmed_national_id"]
                        logger.info(f"✅ [RESUME] (Redis path) Using confirmed ID from session: {saved_session['confirmed_national_id']}")
                    
                    # CRITICAL FIX (Bug 28): Load selected_service_type_id from session on resume
                    # Without this, error_no_service_type happens in loop!
                    if saved_session.get("selected_service_type_id"):
                        initial_state["selected_service_type_id"] = saved_session["selected_service_type_id"]
                        logger.info(f"✅ [RESUME] (Redis path) Loaded service_type_id={saved_session['selected_service_type_id']} from session")
                    
                    if saved_session.get("selected_service_id"):
                        initial_state["selected_service_id"] = saved_session["selected_service_id"]
                        logger.info(f"✅ [RESUME] (Redis path) Loaded selected_service_id={saved_session['selected_service_id']}")
                    
                    if saved_session.get("selected_service_name"):
                        initial_state["selected_service_name"] = saved_session["selected_service_name"]
                        logger.info(f"✅ [RESUME] (Redis path) Loaded selected_service_name={saved_session['selected_service_name']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error loading confirmed data from session (Redis path): {e}")
                
                initial_state["_resuming"] = True
                initial_state["_previous_step"] = current_step
            else:
                # Build initial state from payload (first message)
                logger.info(f"🆕 [LANGGRAPH] Starting new conversation")
                
                # CRITICAL: Get conversation history from context for context-aware responses
                conversation_history = (context or {}).get("conversation_history", [])
                logger.info(f"📚 [LANGGRAPH] Injecting {len(conversation_history)} messages from conversation history")
                
                # CRITICAL: If patient is already registered, pre-fill their data
                initial_name = None
                initial_national_id = None
                initial_gender = None
                patient_verified = False
                
                if patient_data and patient_data.get('already_registered'):
                    initial_name = patient_data.get('name')
                    initial_national_id = patient_data.get('national_id')
                    initial_gender = patient_data.get('gender', 'male')
                    patient_verified = True
                    logger.info(f"✅ [LANGGRAPH] Pre-filling registered patient data: {initial_name}")
                
                # CRITICAL: Extract selected service from context (State Sync Fix - Issue: Amnesia Bug)
                # If user selected a service in previous turn (via conversational agent),
                # pass it to LangGraph so it doesn't ask again
                selected_service_id = (context or {}).get('selected_service_id')
                selected_service_name = (context or {}).get('selected_service_name')
                selected_service_type_id = (context or {}).get('selected_service_type_id')  # ✅ NEW: Bug 24 fix
                
                if selected_service_name:
                    logger.info(f"✅ [LANGGRAPH] Pre-filling selected service from Router: {selected_service_name} (ID: {selected_service_id}, Type ID: {selected_service_type_id})")
                
                initial_state: BookingState = {
                    "session_id": session_id,
                    "phone_number": phone_number,
                    "sender_name": sender_name,
                    "arabic_name": arabic_name,
                    "name": initial_name,  # Pre-filled if registered
                    "national_id": initial_national_id,  # Pre-filled if registered
                    "gender": initial_gender,  # Pre-filled if registered
                    "patient_verified": patient_verified,  # Skip verification if already registered
                    "patient_id": patient_data.get('id') if patient_data else None,  # Store patient ID
                    "patient_data": patient_data,  # CRITICAL: Pass full patient data for verification node to check
                    "selected_service_id": selected_service_id,  # CRITICAL: Pre-fill selected service
                    "selected_service_name": selected_service_name,  # CRITICAL: Pre-fill selected service
                    "selected_service_type_id": selected_service_type_id,  # ✅ NEW: Bug 24 fix - Prevents error_no_service_type
                    "current_message": message,
                    "is_pure_intent": is_pure_intent,
                    "messages": [{"role": "user", "content": message}],
                    "conversation_history": conversation_history,  # CRITICAL: Pass full history
                    "step": "start",
                    "started": True,
                    "awaiting_confirmation": False,
                    "critical_failures": 0,
                    "message_repeat_count": 0,
                    "last_user_message": "",
                    "router_intent": router_intent,
                    "router_confidence": router_confidence
                }
            
            # Configuration for LangGraph (thread_id = session_id for persistence)
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": "booking"
                },
                "recursion_limit": 50  # Prevent infinite loops
            }
            
            # Invoke graph with automatic state restoration from checkpoint
            logger.info(f"🔄 [LANGGRAPH] Invoking graph for session {session_id}")
            
            # Use ainvoke for async execution
            result = await self.graph.ainvoke(initial_state, config=config)
            
            previous_step = initial_state.get('_previous_step', 'start')
            new_step = result.get('step')
            
            # Check if we made actual progress or regressed (Issue #4: Fix catastrophic marked as progress)
            progress_indicators = ['registration_id', 'registration_complete', 'patient_verified', 'awaiting_service', 'service_selected', 'time_slot_selected']
            regression_indicators = ['awaiting_registration_confirmation', 'awaiting_name', 'error_recovery', 'catastrophic_failure']
            error_states = ['error_recovery', 'catastrophic_failure', 'patient_verification_error', 'registration_error']
            
            # Check for error progression (error → worse error = still error, not progress!)
            is_error_progression = (previous_step in error_states or 'error' in previous_step) and (new_step in error_states or 'error' in new_step)
            
            made_progress = new_step in progress_indicators or (new_step != previous_step and new_step not in regression_indicators and not is_error_progression)
            regressed = (previous_step in progress_indicators and new_step in regression_indicators) or is_error_progression
            
            if regressed:
                logger.error(f"❌ [LANGGRAPH] REGRESSION DETECTED: {previous_step} → {new_step} (moving backwards!)")
            elif made_progress:
                logger.info(f"✅ [LANGGRAPH] Progress made: {previous_step} → {new_step}")
            else:
                logger.warning(f"⚠️ [LANGGRAPH] No progress: {previous_step} → {new_step}")
            
            # Save state for next message (in-memory persistence)
            BookingAgentLangGraph._conversation_states[session_id] = result.copy()
            logger.info(f"💾 [LANGGRAPH] State saved for session {session_id}")
            
            # CRITICAL: Add state version for conflict detection (Issue: State sync mismatch)
            import time
            result["_state_version"] = int(time.time() * 1000)  # Millisecond timestamp
            result["_last_updated_by"] = "langgraph"
            
            # Persist booking state in Redis session for cross-process continuity (Issues #24/#50)
            try:
                await self.session_manager.update_session(session_id, {"booking_state": result})
                logger.info(f"💾 [LANGGRAPH] State persisted via SessionManager (version={result['_state_version']}) for {session_id}")
            except Exception as e:
                logger.error(f"❌ Failed to persist booking_state via SessionManager: {e}")
            
            # Extract response from final state
            response = self._build_response_from_state(result)
            
            # Cache response (PRESERVE FROM ORIGINAL)
            self._cache_request(phone_number, message, response)
            
            # Reset failure counter on success (PRESERVE FROM ORIGINAL)
            self._reset_failure_counter(self.session_id)
            
            return response
            
        except Exception as exc:
            # PRESERVE EXACT ERROR HANDLING FROM ORIGINAL
            logger.error(f"🚨 LangGraph booking error: {exc}", exc_info=True)
            
            # Record failure
            failure_count = self._record_failure(self.session_id)
            
            # Check if we should trigger recovery
            if self._should_trigger_recovery(self.session_id):
                logger.error(f"🚨 CATASTROPHIC FAILURE: triggering recovery")
                return self._trigger_recovery(self.session_id, arabic_name)
            
            # Return error response (Issue #7: Honest, helpful error messages)
            language = detect_language(message)
            if language == "arabic":
                response = f"عذراً يا {arabic_name} 🙏\n\nنظام الحجز مشغول حالياً\n\n**ممكن:**\n• اتصل على: 920033304 📞\n• أو جرب بعد دقيقة"
            else:
                response = "Sorry, booking system is busy 🙏\n\nPlease:\n• Call: 920033304 📞\n• Or try again in a minute"
            
            return {
                "response": response,
                "intent": "booking",
                "status": "error",
                "error": str(exc),
                "failure_count": failure_count
            }
    
    def _build_response_from_state(self, state: BookingState) -> dict:
        """
        Convert final LangGraph state to response dict.
        
        Maintains compatibility with existing response format.
        """
        current_step = state.get("step", "unknown")
        
        # Check if conversational agent should handle response
        if state.get("needs_conversational_response"):
            logger.info(f"🤖 [LANGGRAPH] Response will be generated by conversational agent (context: {state.get('context_for_agent', {}).get('action')})")
            # Return minimal response - router will use conversational agent
            return {
                "response": "",  # Empty - will be filled by conversational agent
                "intent": "booking",
                "status": "needs_conversational_response",
                "booking_state": dict(state),
                "use_conversational_agent": True
            }
        
        # Extract last assistant message
        messages = state.get("messages", [])
        last_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_message = msg.get("content", "")
                break
        
        # Fallback if no message
        if not last_message:
            arabic_name = state.get("arabic_name", "حبيبنا")
            last_message = f"تمام يا {arabic_name}! 👌\nخليني أساعدك"
        
        # Map step to status
        status_map = {
            "awaiting_service_type": "awaiting_service_type",
            "awaiting_service": "awaiting_service",
            "awaiting_doctor": "awaiting_doctor",
            "awaiting_specialist": "awaiting_specialist",
            "awaiting_device": "awaiting_device",
            "awaiting_time_slot": "showing_time_slots",
            "awaiting_confirmation": "awaiting_confirmation",
            "completed": "completed",
            "cancelled": "cancelled"
        }
        
        status = status_map.get(current_step, "in_progress")
        
        # Check for errors
        if current_step.endswith("_error") or current_step.startswith("error_"):
            status = "error"
        
        # CRITICAL FIX: Extract displayed_services for smart selection (Issue: Smart selection broken)
        displayed_services = state.get("displayed_services") or state.get("services")
        
        result = {
            "response": last_message,
            "intent": "booking",
            "status": status,
            "booking_state": dict(state)  # Include full state for debugging
        }
        
        # Add displayed_services if available (enables smart selection feature)
        if displayed_services:
            result["displayed_services"] = displayed_services
            logger.debug(f"📋 [LANGGRAPH] Returning {len(displayed_services)} displayed_services for smart selection")
        
        return result
    
    # ==========================================
    # PRESERVE ALL HELPER METHODS FROM ORIGINAL
    # ==========================================
    
    async def _check_request_duplicate(self, phone_number: str, message: str, session_id: str) -> Optional[dict]:
        """
        Check for duplicate requests with smart caching.
        
        CRITICAL: Don't cache failed attempts or confirmation steps!
        If state didn't progress, user is likely retrying due to error.
        """
        cache_key = hashlib.md5(f"{phone_number}:{message.lower().strip()}".encode()).hexdigest()
        
        if cache_key in BookingAgentLangGraph._request_cache:
            cached = BookingAgentLangGraph._request_cache[cache_key]
            time_since = time.time() - cached["timestamp"]
            
            # Check if we're at a confirmation step - don't use cache
            # User might be retrying because of previous misclassification
            try:
                current_session = await self.session_manager.get_session(session_id) or {}
                current_step = (current_session.get("booking_state") or {}).get("step", "")
                
                # Don't cache confirmation steps - LLM needs to understand each time
                if "confirmation" in current_step:
                    logger.info(f"⏭️ Skipping duplicate check (at confirmation step: {current_step})")
                    return None
            except:
                pass
            
            if time_since < 30:
                logger.warning(f"🔁 DUPLICATE REQUEST detected: '{message}' ({time_since:.1f}s ago)")
                return cached["response"]
        
        return None
    
    def _cache_request(self, phone_number: str, message: str, response: dict):
        """PRESERVE EXACT METHOD FROM ORIGINAL BookingAgent"""
        cache_key = hashlib.md5(f"{phone_number}:{message.lower().strip()}".encode()).hexdigest()
        BookingAgentLangGraph._request_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Clean old cache entries
        current_time = time.time()
        keys_to_delete = [
            key for key, value in BookingAgentLangGraph._request_cache.items()
            if current_time - value["timestamp"] > 60
        ]
        for key in keys_to_delete:
            del BookingAgentLangGraph._request_cache[key]
    
    def _record_failure(self, session_id: str) -> int:
        """PRESERVE EXACT METHOD FROM ORIGINAL BookingAgent"""
        if session_id not in BookingAgentLangGraph._failure_counters:
            BookingAgentLangGraph._failure_counters[session_id] = {"count": 0, "last_failure": 0}
        
        BookingAgentLangGraph._failure_counters[session_id]["count"] += 1
        BookingAgentLangGraph._failure_counters[session_id]["last_failure"] = time.time()
        
        failure_count = BookingAgentLangGraph._failure_counters[session_id]["count"]
        logger.warning(f"⚠️ FAILURE TRACKER: Session {session_id} has {failure_count} failures")
        
        return failure_count
    
    def _reset_failure_counter(self, session_id: str):
        """PRESERVE EXACT METHOD FROM ORIGINAL BookingAgent"""
        if session_id in BookingAgentLangGraph._failure_counters:
            logger.info(f"✅ FAILURE TRACKER: Session {session_id} reset")
            del BookingAgentLangGraph._failure_counters[session_id]
    
    def _should_trigger_recovery(self, session_id: str) -> bool:
        """PRESERVE EXACT METHOD FROM ORIGINAL BookingAgent"""
        if session_id not in BookingAgentLangGraph._failure_counters:
            return False
        
        failure_data = BookingAgentLangGraph._failure_counters[session_id]
        failure_count = failure_data["count"]
        time_since_last = time.time() - failure_data["last_failure"]
        
        if failure_count >= 3 and time_since_last < 300:
            logger.error(f"🚨 RECOVERY MODE: Session {session_id} has {failure_count} failures")
            return True
        
        return False
    
    def _trigger_recovery(self, session_id: str, arabic_name: str) -> dict:
        """PRESERVE EXACT METHOD FROM ORIGINAL BookingAgent"""
        logger.error(f"🔧 RECOVERY TRIGGERED for session {session_id}")
        
        # Reset failure counter
        self._reset_failure_counter(session_id)
        
        # Clear LangGraph checkpoint for this session
        try:
            # Delete checkpoint from Redis to force fresh start
            checkpoint_key = f"langgraph:checkpoint:{session_id}"
            self.redis_client.delete(checkpoint_key)
            logger.info("✅ Cleared LangGraph checkpoint")
        except Exception as e:
            logger.error(f"Error clearing checkpoint: {e}")
        
        # Clear instance from memory
        if session_id in BookingAgentLangGraph._instances:
            del BookingAgentLangGraph._instances[session_id]
            logger.info("✅ Cleared agent instance from memory")
        
        if session_id in BookingAgentLangGraph._initialized:
            del BookingAgentLangGraph._initialized[session_id]
            logger.info("✅ Reset initialization flag")
        
        # Return recovery message (PRESERVE EXACT TEMPLATE)
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
