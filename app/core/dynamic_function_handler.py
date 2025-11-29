"""
Dynamic Function Handler with State Awareness
==============================================
Executes functions with full awareness of booking state.

Each function does ONE thing and updates state accordingly.
Reem calls these functions when needed - not automatically!
"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..models.conversation_context import ConversationContext, BookingState
from ..models.workflow_result import WorkflowResult, WorkflowStatus
from ..workflows.workflow_executor import WorkflowExecutor
from ..api.agent_api import AgentApiClient
from ..monitoring.hybrid_metrics import get_metrics
from .semantic_matcher import get_semantic_matcher


@dataclass
class FunctionResult:
    """Result from function execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    function_name: str = ""
    booking_state: Optional[BookingState] = None
    needs_wrapping: bool = True  # Does Reem need to wrap this result?
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "data": self.data or {},
            "message": self.message,
            "function_name": self.function_name,
            "booking_state": self.booking_state.to_dict() if self.booking_state else None,
            "needs_wrapping": self.needs_wrapping
        }


class DynamicFunctionHandler:
    """
    Executes functions with full awareness of booking state.
    
    This is the "executor" that Reem delegates to when she needs
    to perform technical operations. Each function is granular and
    interruptible by design.
    """
    
    @staticmethod
    def parse_doctor_selection(user_message: str, available_doctors: list) -> Optional[int]:
        """
        Parse doctor selection from user message - SMART like a human!
        
        Examples:
        - "1" → returns available_doctors[0]['id']
        - "الأول" → returns available_doctors[0]['id']
        - "هبة" → matches "Heba Omar" → returns that doctor's id
        - "د. أحمد" → matches "Ahmad" → returns that doctor's id
        - "batoul" → matches "Batoul Fehan" → returns that doctor's id
        
        Args:
            user_message: User's message
            available_doctors: List of doctor dicts with 'id' and 'name' fields
            
        Returns:
            doctor_id if found, None otherwise
        """
        if not available_doctors:
            return None
        
        msg = user_message.strip().lower()
        
        # Check for digit selection
        if msg.isdigit():
            index = int(msg) - 1  # Convert to 0-based index
            if 0 <= index < len(available_doctors):
                return available_doctors[index].get('id')
        
        # Check for Arabic ordinal numbers
        ordinals = {
            'الأول': 0, 'الاول': 0, 'اول': 0, 'اولى': 0,
            'الثاني': 1, 'الثانى': 1, 'ثاني': 1, 'ثانية': 1,
            'الثالث': 2, 'ثالث': 2, 'ثالثة': 2,
            'الرابع': 3, 'رابع': 3, 'رابعة': 3,
            'الخامس': 4, 'خامس': 4, 'خامسة': 4
        }
        
        for ordinal, index in ordinals.items():
            if ordinal in msg:
                if index < len(available_doctors):
                    return available_doctors[index].get('id')
        
        # 🚨 CRITICAL: SMART NAME MATCHING (like a human receptionist!)
        # If user says "هبة" or "Heba", match it to "Heba Omar"
        # Remove common prefixes
        clean_msg = msg.replace('د.', '').replace('دكتور', '').replace('دكتورة', '').replace('dr.', '').replace('dr', '').strip()
        
        for i, doctor in enumerate(available_doctors):
            doctor_name = doctor.get('name', '').lower()
            
            # Check if user's message is in doctor's name
            if clean_msg in doctor_name:
                logger.info(f"🎯 SMART MATCH: User said '{user_message}' → Matched doctor '{doctor.get('name')}' (ID: {doctor.get('id')})")
                return doctor.get('id')
            
            # Check if doctor's name parts are in user's message
            name_parts = doctor_name.split()
            for part in name_parts:
                if len(part) >= 3 and part in clean_msg:  # At least 3 chars to avoid false matches
                    logger.info(f"🎯 SMART MATCH: User said '{user_message}' → Matched doctor '{doctor.get('name')}' (ID: {doctor.get('id')})")
                    return doctor.get('id')
        
        return None
    
    def __init__(self):
        self.workflow = WorkflowExecutor()
        self.api_client = AgentApiClient()
        self.metrics = get_metrics()
        self.function_failures = {}  # Track failures per function
        logger.info("✅ DynamicFunctionHandler initialized with metrics")
    
    async def execute(
        self, 
        function_name: str, 
        arguments: dict,
        context: ConversationContext
    ) -> FunctionResult:
        """
        Execute function and update state accordingly.
        
        Args:
            function_name: Name of function to execute
            arguments: Function arguments from Reem
            context: Full conversation context
            
        Returns:
            FunctionResult with success status and data
        """
        
        logger.info(f"🔧 Executing function: {function_name}")
        logger.debug(f"   Arguments: {arguments}")
        
        start_time = time.time()
        
        try:
            # Route to appropriate handler
            if function_name == "execute_booking_step":
                return await self._handle_booking_step(
                    step_name=arguments["step_name"],
                    data=arguments.get("data", {}),
                    context=context
                )
            
            elif function_name == "pause_booking":
                return await self._pause_booking(
                    reason=arguments.get("reason", "user_request"),
                    context=context
                )
            
            elif function_name == "resume_booking":
                return await self._resume_booking(context)
            
            elif function_name == "cancel_booking":
                return await self._cancel_booking(
                    reason=arguments.get("reason", "user_request"),
                    context=context
                )
            
            elif function_name == "get_service_details":
                return await self._get_service_details(
                    service_id=arguments["service_id"],
                    context=context
                )
            
            elif function_name == "get_pricing":
                return await self._get_pricing(
                    service_id=arguments["service_id"],
                    detailed=arguments.get("detailed", False),
                    context=context
                )
            
            elif function_name == "check_availability":
                return await self._check_availability(
                    service_id=arguments["service_id"],
                    date=arguments.get("date"),
                    context=context
                )
            
            elif function_name == "search_services":
                return await self._search_services(
                    query=arguments["query"],
                    context=context
                )
            
            elif function_name == "get_all_services":
                return await self._get_all_services(
                    category=arguments.get("category", ""),
                    context=context
                )
            
            elif function_name == "view_my_bookings":
                return await self._view_my_bookings(
                    show_past=arguments.get("show_past", False),
                    context=context
                )
            
            else:
                logger.error(f"❌ Unknown function: {function_name}")
                # 🚨 CRITICAL: ALWAYS wrap errors!
                result = FunctionResult(
                    success=False,
                    message="unknown_function",  # Error code for wrapping
                    function_name=function_name,
                    needs_wrapping=True,  # ✅ Must wrap!
                    data={"function_name": function_name}
                )
                
                # Record metrics
                duration = time.time() - start_time
                self.metrics.record_function_execution(
                    function_name=function_name,
                    duration=duration,
                    success=False,
                    session_id=context.session_id,
                    error="Unknown function"
                )
                
                return result
        
        except Exception as e:
            logger.error(f"❌ Function execution failed: {e}", exc_info=True)
            
            # Track failures
            if function_name not in self.function_failures:
                self.function_failures[function_name] = 0
            self.function_failures[function_name] += 1
            
            # Warn if function failing repeatedly
            if self.function_failures[function_name] >= 3:
                logger.warning(
                    f"⚠️ Function {function_name} has failed {self.function_failures[function_name]} times. "
                    f"Consider using fallback responses."
                )
            
            # Record metrics for failed execution
            duration = time.time() - start_time
            self.metrics.record_function_execution(
                function_name=function_name,
                duration=duration,
                success=False,
                session_id=context.session_id,
                error=str(e)
            )
            
            # 🚨 CRITICAL: ALWAYS wrap errors!
            return FunctionResult(
                success=False,
                message="function_execution_error",  # Error code
                function_name=function_name,
                needs_wrapping=True,  # ✅ Must wrap!
                data={
                    "error": str(e),
                    "function_name": function_name
                }
            )
    
    # ==================== BOOKING WORKFLOW FUNCTIONS ====================
    
    async def _handle_booking_step(
        self,
        step_name: str,
        data: dict,
        context: ConversationContext
    ) -> FunctionResult:
        """
        Execute ONE booking step, not entire workflow.
        
        This is the key to granular control - each step is separate!
        """
        
        booking_state = context.booking_state
        
        # Mark booking as active if not already
        if booking_state.status == "idle":
            old_status = booking_state.status
            booking_state.status = "active"
            booking_state.started_at = time.time()
            logger.info(f"📅 Starting booking workflow")
            
            # Record state transition
            self.metrics.record_state_transition(
                from_state=old_status,
                to_state="active",
                session_id=context.session_id,
                metadata={"step": step_name}
            )
        
        logger.info(f"🔄 Executing booking step: {step_name}")
        
        # Execute the specific step using workflow executor
        try:
            if step_name == "collect_service":
                result = await self._collect_service_step(data, context)
            
            elif step_name == "check_availability":
                result = await self._check_availability_step(data, context)
            
            elif step_name == "collect_datetime" or step_name == "select_slot":  # select_slot is legacy name
                # 🚨 CRITICAL VALIDATION: Don't collect datetime without service!
                # This prevents infinite loops where user provides service but system skips it
                if not booking_state.collected_data.get("service_id"):
                    logger.warning(f"⚠️ Attempted collect_datetime WITHOUT service_id!")
                    logger.warning(f"   🧠 SMART APPROACH: Using intelligent service extraction instead of keywords")
                    
                    # 🧠 SMART APPROACH: Get the user's original message
                    # Let the sophisticated collect_service logic handle extraction
                    user_message = None
                    
                    # Priority 1: Get from function data
                    user_message = data.get("user_message")
                    
                    # Priority 2: Get from conversation history (last user message)
                    if not user_message and context.conversation_history:
                        for msg in reversed(context.conversation_history):
                            if (hasattr(msg, 'role') and msg.role == "user") or (isinstance(msg, dict) and msg.get("role") == "user"):
                                user_message = msg.content if hasattr(msg, 'content') else msg.get("content", "")
                                break
                    
                    # Priority 3: Check if explicitly provided service_name or context
                    explicit_service = data.get("service_name") or context.last_discussed_service
                    
                    if user_message or explicit_service:
                        service_query = explicit_service or user_message
                        logger.info(f"🧠 SMART REDIRECT: Passing to collect_service for intelligent matching")
                        logger.info(f"   Query: '{service_query[:100]}...'")
                        logger.info(f"   Will use: Exact Match → Fuzzy → Semantic Search → LLM")
                        
                        # 🎯 Let collect_service do its magic with cascading architecture:
                        # Level 1: Exact Match (fastest)
                        # Level 2: Keyword Mapping/Synonyms
                        # Level 3: Fuzzy Matching (typos)
                        # Level 4: Semantic Search (understands meaning)
                        # Level 5: Graceful Failure (asks user)
                        result = await self._collect_service_step(
                            {"service_name": service_query}, 
                            context
                        )
                        
                        # If service collection succeeded, also save the date/time user provided
                        if result.get("success") and result.get("data"):
                            result_data = result["data"]
                            # Preserve the date/time from original request
                            if data.get("date"):
                                result_data["date"] = data.get("date")
                            if data.get("time"):
                                result_data["time"] = data.get("time")
                            logger.info(f"✅ Saved date/time from original message: {data.get('date')} {data.get('time')}")
                    else:
                        # No service info at all - ask user
                        logger.warning(f"❌ No service information found - asking user directly")
                        return FunctionResult(
                            success=False,
                            message="service_required_before_datetime",
                            data={
                                "reason": "datetime_without_service",
                                "date": data.get("date"),
                                "time": data.get("time")
                            },
                            function_name="execute_booking_step",
                            booking_state=booking_state,
                            needs_wrapping=True
                        )
                else:
                    # Service already collected - proceed with datetime
                    result = await self._collect_datetime_step(data, context)
            
            elif step_name == "confirm_booking":
                result = await self._confirm_booking_step(data, context)
            
            else:
                return FunctionResult(
                    success=False,
                    message=f"Unknown step: {step_name}",
                    function_name="execute_booking_step"
                )
            
            # Update progress if successful
            if result["success"]:
                booking_state.progress[step_name] = True
                # CRITICAL FIX: Update with RESULT data (has service_id), not INPUT data!
                if "data" in result and result["data"]:
                    result_data = result["data"].copy()
                    
                    # 🎯 CRITICAL FIX: Extract service from 'services' array if present
                    # Problem: collect_service returns {'services': [{'id': 30, ...}]}
                    # But confirm_booking expects {'service_id': 30, 'service_name': '...'}
                    if "services" in result_data and isinstance(result_data["services"], list):
                        services_list = result_data["services"]
                        if len(services_list) > 0:
                            # Extract first service (user's selection)
                            selected_service = services_list[0]
                            logger.info(f"🔧 EXTRACTING service from array: {selected_service.get('name')} (ID: {selected_service.get('id')})")
                            
                            # Flatten service data to booking_state format
                            result_data["service_id"] = selected_service.get("id")
                            result_data["service_name"] = selected_service.get("name")
                            result_data["service_price"] = selected_service.get("price", 0)
                            result_data["duration_minutes"] = selected_service.get("duration_minutes", 60)
                            result_data["requires_doctor"] = selected_service.get("requires_doctor", False)
                            result_data["requires_specialist"] = selected_service.get("requires_specialist", False)
                            result_data["requires_device"] = selected_service.get("requires_device", False)
                            
                            # Remove the array (we've flattened it)
                            del result_data["services"]
                            logger.info(f"✅ FLATTENED: service_id={result_data['service_id']}, service_name={result_data['service_name']}")
                    
                    booking_state.collected_data.update(result_data)
                    logger.info(f"📥 Updated booking_state with result data: {list(result_data.keys())}")
                logger.info(f"✅ Completed booking step: {step_name}")
            
            return FunctionResult(
                success=result["success"],
                data=result.get("data", {}),
                message=result.get("message", ""),
                function_name="execute_booking_step",
                booking_state=booking_state,
                needs_wrapping=True
            )
        
        except Exception as e:
            logger.error(f"❌ Booking step failed: {e}", exc_info=True)
            return FunctionResult(
                success=False,
                message=f"Step failed: {str(e)}",
                function_name="execute_booking_step",
                booking_state=booking_state
            )
    
    async def _collect_service_step(
        self,
        data: dict,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Collect service information"""
        
        service_id = data.get("service_id")
        service_name = data.get("service_name")
        
        # 🚨 CRITICAL: Check booking_state FIRST (user already selected!)
        booking_state = context.booking_state
        if booking_state and booking_state.collected_data.get("service_id"):
            saved_service_id = booking_state.collected_data["service_id"]
            saved_service_name = booking_state.collected_data.get("service_name", "selected service")
            logger.info(f"✅ FOUND SAVED SERVICE in booking_state: ID={saved_service_id}, Name={saved_service_name}")
            
            # 🚨 CRITICAL: Check if user is selecting a DIFFERENT service!
            # User might say "ابتسامة لثوية" after seeing "ميزو بوتوكس"
            if service_name and service_name.strip():
                # Normalize for comparison
                new_service_normalized = service_name.lower().strip()
                saved_service_normalized = saved_service_name.lower().strip()
                
                # Check if the new service is significantly different
                # Allow partial matches (e.g., "ميزو" matches "ميزو بوتوكس")
                if (new_service_normalized not in saved_service_normalized and 
                    saved_service_normalized not in new_service_normalized):
                    logger.info(f"🔄 USER CHANGED SERVICE: '{saved_service_name}' → '{service_name}'")
                    logger.info(f"🔍 Clearing old service and OLD BOOKING DATA, then searching for new service...")
                    
                    # 🚨 CRITICAL: Clear ALL old booking data, not just service!
                    # This includes booking_id from previous booking!
                    old_booking_id = booking_state.collected_data.get("booking_id")
                    if old_booking_id:
                        logger.info(f"   🗑️ Clearing old booking_id={old_booking_id} (different service)")
                    
                    booking_state.collected_data.pop("service_id", None)
                    booking_state.collected_data.pop("service_name", None)
                    booking_state.collected_data.pop("service_price", None)
                    booking_state.collected_data.pop("booking_id", None)  # Clear old booking!
                    booking_state.collected_data.pop("booked_service_id", None)  # Clear service tracking!
                    booking_state.collected_data.pop("doctor_id", None)  # Clear doctor selection!
                    booking_state.collected_data.pop("doctor_name", None)
                    # Keep date/time if user specified them (might want same slot for different service)
                else:
                    logger.info(f"🎯 Same service confirmed - SKIP search, proceed to next step!")
                    # Return the already-saved service
                    return {
                        "success": True,
                        "message": "service_already_selected",
                        "data": {
                            "service_id": saved_service_id,
                            "service_name": saved_service_name,
                            "status": "confirmed",
                            "next_step": "check_availability"
                        }
                    }
            else:
                logger.info(f"🎯 User already confirmed this service - SKIP search, proceed to next step!")
                # No new service name provided, use saved one
                return {
                    "success": True,
                    "message": "service_already_selected",
                    "data": {
                        "service_id": saved_service_id,
                        "service_name": saved_service_name,
                        "status": "confirmed",
                        "next_step": "check_availability"
                    }
                }
        
        # CRITICAL: Check arguments FIRST, context is FALLBACK only!
        if not service_id and not service_name:
            # Fallback to context only if nothing provided
            if context.last_discussed_service:
                logger.info(f"💬 FALLBACK: Using service from context: {context.last_discussed_service}")
                service_name = context.last_discussed_service
            else:
                # No service in data or context
                return {
                    "success": False,
                    "message": "service_selection_needed",
                    "data": {
                        "reason": "no_service_provided",
                        "context_available": False
                    }
                }
        
        # Log what we're using
        if service_name:
            logger.info(f"🔍 Collecting service: '{service_name}'")
        
        # If only name provided, search for service using CASCADING ARCHITECTURE
        if service_name and not service_id:
            """
            🎯 PROFESSIONAL CASCADING SEARCH ARCHITECTURE:
            ==================================================
            Each level is progressively smarter but slower. We try fast methods first!
            
            LEVEL 1: Exact Match (fastest, 100% accurate)
                    ↓ (if no match)
            LEVEL 2: Keyword Mapping/Synonyms (fast, handles known variations)
                    ↓ (if no match)  
            LEVEL 3: Fuzzy Matching (handles typos, ~90% accurate)
                    ↓ (if no match)
            LEVEL 4: Semantic Search with Cosine Similarity (slowest, understands meaning)
                    ↓ (if no match)
            LEVEL 5: Graceful Failure (ask user to clarify)
            
            Examples:
            - Level 1: "ليزر بوكسر" → exact match
            - Level 2: "تحت الحزام" → synonym maps to "بوكسر"
            - Level 3: "بوكسرر" (typo) → fuzzy matches "بوكسر"
            - Level 4: "إزالة الشعر" → semantic similarity to "ليزر"
            - Level 5: No match found → ask user to rephrase
            """
            
            search_term = service_name.lower().strip()
            
            # 🎯 LEVEL 2: SMART SYNONYM MAPPING
            # Understand common customer phrases
            # Maps what customers SAY → what services are CALLED in database
            synonym_map = {
                "تحت الحزام": "بوكسر",
                "تحت الحزام رجال": "ليزر بوكسر رجال",
                "تحت الحزام نساء": "ليزر بيكيني نساء",
                "منطقة الحساسة": "بيكيني",
                "منطقة حساسة": "بيكيني",
                "المنطقة الحساسة": "بيكيني",
                "كامل الجسم": "فل بدي",
                "الجسم كامل": "فل بدي",
                "كل الجسم": "فل بدي",
                "جسم كامل": "فل بدي",
                "full body": "فل بدي",
                "بكيني": "بيكيني",
                "باكيني": "بيكيني",
            }
            
            # Apply synonym mapping (check longest phrases first)
            original_search = search_term
            for phrase, replacement in sorted(synonym_map.items(), key=lambda x: len(x[0]), reverse=True):
                if phrase in search_term:
                    search_term = search_term.replace(phrase, replacement)
                    logger.info(f"🎯 SMART MATCH: '{original_search}' → '{search_term}' (synonym mapping)")
                    service_name = search_term  # Update service_name for display
                    break
            
            logger.info(f"🔍 TWO-LEVEL SEARCH: Looking for '{search_term}' (following intelligent_booking_agent.py pattern)")
            
            # Smart keyword mapping for common services
            # Maps user search terms to category keywords
            category_hints = {
                "فل بدي": ["ليزر", "laser"],
                "فول بدي": ["ليزر", "laser"],
                "بوكسر": ["ليزر", "laser"],
                "boxer": ["ليزر", "laser"],
                "بيكيني": ["ليزر", "laser"],
                "bikini": ["ليزر", "laser"],
                "بوتوكس": ["بوتوكس", "botox"],
                "فيلر": ["فيلر", "filler"],
                "ميزو": ["ميزو", "meso", "نضارة"],
                "تنظيف": ["تنظيف", "clean"],
                "تقشير": ["تقشير", "peel"],
                "خيوط": ["خيوط", "thread"],
                "بلازما": ["نضارة", "plasma"],
            }
            
            # LEVEL 1: Get categories using RAW API call (like intelligent_booking_agent)
            try:
                categories_result = await self.api_client.get("/services", params={"limit": 50})
                categories = categories_result.get("results") or categories_result.get("data") or []
                logger.info(f"📚 LEVEL 1: Retrieved {len(categories)} categories from API")
            except Exception as e:
                logger.error(f"❌ Failed to fetch categories: {e}")
                return {
                    "success": False,
                    "message": "api_error",
                    "data": {"error": str(e)}
                }
            
            # Filter out test services (following intelligent_booking_agent pattern)
            test_keywords = ["test", "zzzz", "dummy", "sample", "xxxx", "temp"]
            valid_categories = []
            for cat in categories:
                if not isinstance(cat, dict):
                    continue
                
                cat_name = (cat.get("name") or cat.get("name_ar") or cat.get("nameAr") or "").lower()
                
                # Skip test services
                if any(keyword in cat_name for keyword in test_keywords):
                    logger.debug(f"  ⏭️ Skipping test category: {cat_name}")
                    continue
                
                # Must have name and id
                if cat_name and ("id" in cat or "service_type_id" in cat):
                    valid_categories.append(cat)
            
            logger.info(f"📚 LEVEL 1: {len(valid_categories)} valid categories (after filtering test services)")
            
            # LEVEL 2: Deep search in subservices
            all_subservices = []
            matching = []
            categories_to_search = []
            
            # 🎯 GENDER-BASED FILTERING: Check patient gender for smart filtering
            patient_gender = None
            if context.patient and context.patient.gender:
                patient_gender = context.patient.gender.lower()
                logger.info(f"👤 Patient gender: {patient_gender}")
            
            # Determine which categories to search based on hints
            for category in valid_categories:
                category_name = (category.get("name") or category.get("name_ar") or category.get("nameAr") or "").lower().strip()
                should_search = False
                
                # 1. Direct match with category name
                if search_term in category_name or category_name in search_term:
                    should_search = True
                    logger.info(f"📁 Direct match: '{category.get('name')}' contains '{service_name}'")
                
                # 2. Check keyword hints
                else:
                    for hint_key, hint_keywords in category_hints.items():
                        if hint_key in search_term:
                            # Check if this category matches any hint keyword
                            if any(kw in category_name for kw in hint_keywords):
                                should_search = True
                                logger.info(f"💡 Hint match: '{search_term}' → category '{category.get('name')}'")
                                break
                
                # 🎯 CRITICAL: Gender-based filtering for laser services
                if should_search and patient_gender:
                    # Check if this is a laser service
                    is_laser = "ليزر" in search_term or "laser" in search_term
                    
                    if is_laser:
                        # Filter based on gender
                        if patient_gender == "male" and "نساء" in category_name:
                            should_search = False
                            logger.info(f"⚠️ Skipping women's laser for male patient: '{category.get('name')}'")
                        elif patient_gender == "female" and "رجال" in category_name:
                            should_search = False
                            logger.info(f"⚠️ Skipping men's laser for female patient: '{category.get('name')}'")
                        elif patient_gender == "male" and "رجال" in category_name:
                            logger.info(f"✅ Gender match: '{category.get('name')}' for male patient")
                        elif patient_gender == "female" and "نساء" in category_name:
                            logger.info(f"✅ Gender match: '{category.get('name')}' for female patient")
                
                if should_search:
                    categories_to_search.append(category)
            
            # Fallback: If no hints matched, search ALL categories
            if not categories_to_search:
                logger.warning(f"⚠️ No category hints matched for '{service_name}', will search ALL {len(valid_categories)} categories")
                categories_to_search = valid_categories
            
            # Fetch and search subservices for selected categories
            for category in categories_to_search:
                category_id = category.get("service_type_id") or category.get("id")
                category_name = category.get("name") or category.get("name_ar") or "Unknown"
                
                if not category_id:
                    logger.warning(f"  ⚠️ Category '{category_name}' has no ID, skipping")
                    continue
                
                logger.info(f"📁 Fetching subservices for: '{category_name}' (ID: {category_id})")
                
                try:
                    # CRITICAL: Use RAW API call with service_type_id (following intelligent_booking_agent)
                    subservices_result = await self.api_client.get("/services/", params={
                        "service_type_id": category_id,
                        "limit": 100
                    })
                    subservices = subservices_result.get("results") or subservices_result.get("data") or []
                    
                    logger.info(f"  └─ Retrieved {len(subservices)} subservices from API")
                    
                    # Filter and validate subservices
                    for sub in subservices:
                        if not isinstance(sub, dict):
                            continue
                        
                        sub_name = (sub.get("name") or sub.get("name_ar") or sub.get("nameAr") or "").lower()
                        
                        # Skip test services
                        if any(keyword in sub_name for keyword in test_keywords):
                            continue
                        
                        # Must have name and id
                        if not sub_name or "id" not in sub:
                            continue
                        
                        all_subservices.append(sub)
                        
                        # Token-based flexible matching (intelligent_booking_agent pattern)
                        import re
                        
                        # Extract search tokens (ignore small words)
                        search_tokens = [
                            token.strip().lower() 
                            for token in re.split(r'[\s/\-,،]+', search_term) 
                            if len(token.strip()) > 1  # Ignore single chars
                        ]
                        
                        # Extract service tokens
                        service_tokens = [
                            token.strip().lower()
                            for token in re.split(r'[\s/\-,،]+', sub_name)
                            if len(token.strip()) > 1
                        ]
                        
                        # 🎯 SMART FUZZY MATCHING: Check for partial matches too
                        # Helps match "بوكسر" with "ليزر بوكسر للرجال"
                        matches_found = 0
                        for search_token in search_tokens:
                            # Exact match in service tokens
                            if search_token in service_tokens:
                                matches_found += 1
                            # Or partial match (search token is part of any service token)
                            elif any(search_token in service_token for service_token in service_tokens):
                                matches_found += 0.7  # Partial match worth 70%
                            # Or vice versa (service token is part of search token)
                            elif any(service_token in search_token for service_token in service_tokens):
                                matches_found += 0.5  # Worth 50%
                        
                        match_ratio = matches_found / len(search_tokens) if search_tokens else 0
                        
                        # 🚨 CRITICAL FIX: Show ALL services from matched category
                        # A human receptionist shows all available options, not just keyword matches
                        # Match if ≥30% of search tokens found (lowered from 50% to be inclusive)
                        if match_ratio >= 0.3 or len(search_tokens) <= 2:
                            # 🎯 CRITICAL: Store match_score for sorting later!
                            sub["match_score"] = int(match_ratio * 100)  # Convert to percentage (0-100)
                            matching.append(sub)
                            logger.info(f"  ✅ MATCH: '{sub.get('name')}' (ID: {sub['id']}, match: {match_ratio:.0%})")
                
                except Exception as e:
                    logger.error(f"  ❌ Failed to fetch subservices for '{category_name}': {e}")
                    continue
            
            # If no matches found in subservices, try semantic search as FINAL fallback
            if not matching:
                logger.warning(f"⚠️ No fuzzy matches found for: '{service_name}'")
                logger.warning(f"   - Searched {len(valid_categories)} categories")
                logger.warning(f"   - Searched {len(all_subservices)} total subservices")
                
                # 🎯 Special handling for laser services without gender info
                is_laser_query = "ليزر" in search_term or "laser" in search_term
                if is_laser_query and not patient_gender:
                    logger.warning(f"⚠️ Laser query without gender info - showing both categories")
                    return {
                        "success": False,
                        "message": "gender_needed_for_laser",
                        "data": {
                            "requested_service": service_name,
                            "reason": "gender_info_needed",
                            "categories_available": ["ليزر رجال", "ليزر نساء"]
                        }
                    }
                
                # 🧠 LEVEL 4: SEMANTIC SEARCH (Final Fallback)
                # Uses AI embeddings and cosine similarity to understand meaning
                # Example: "إزالة الشعر" → matches "ليزر" (hair removal → laser)
                logger.info("🧠 LEVEL 4: Attempting SEMANTIC SEARCH with AI embeddings...")
                
                try:
                    semantic_matcher = get_semantic_matcher()
                    semantic_matches = semantic_matcher.find_semantic_matches(
                        query=service_name,
                        services=all_subservices,
                        threshold_high=0.75,  # 75%+ = high confidence
                        threshold_medium=0.60,  # 60-75% = medium confidence
                        threshold_low=0.50,  # 50-60% = show with caution
                        max_results=5
                    )
                    
                    if semantic_matches:
                        logger.info(f"✅ SEMANTIC SEARCH SUCCESS: Found {len(semantic_matches)} matches!")
                        
                        # Convert semantic matches to regular format
                        matching = [match.service for match in semantic_matches]
                        
                        # Add match scores for sorting
                        for idx, match in enumerate(semantic_matches):
                            matching[idx]["match_score"] = int(match.similarity_score * 100)
                            matching[idx]["match_type"] = match.match_type
                        
                        logger.info(f"🎯 Using semantic matches: {[s.get('name') for s in matching]}")
                    else:
                        logger.warning("⚠️ Semantic search found no matches")
                
                except Exception as e:
                    logger.error(f"❌ Semantic search failed: {e}", exc_info=True)
                
                # If STILL no matches after semantic search, graceful failure
                if not matching:
                    logger.warning(f"🚫 LEVEL 5: GRACEFUL FAILURE - No matches found after all attempts")
                    return {
                        "success": False,
                        "message": "service_not_found",
                        "data": {
                            "requested_service": service_name,
                            "reason": "no_matching_services_all_methods",
                            "searched_categories": len(valid_categories),
                            "searched_subservices": len(all_subservices),
                            "methods_tried": ["exact_match", "synonyms", "fuzzy_matching", "semantic_search"]
                        }
                    }
            
            # CRITICAL: If multiple matches, need to disambiguate! (Following intelligent_booking_agent pattern)
            if len(matching) > 1:
                logger.warning(f"⚠️ Found {len(matching)} matching services for '{service_name}':")
                for idx, match in enumerate(matching, 1):
                    logger.warning(f"   {idx}. '{match.get('name')}' (ID: {match['id']})")
                
                # SMARTNESS: Try to find best match based on user's specific keywords
                # Check for session count keywords: "4 جلسات", "جلسة واحدة", etc.
                best_match = None
                user_input_lower = service_name.lower()
                
                # Extract session count from user input
                session_keywords = {
                    "4 جلسات": ["4 جلسات", "اربع جلسات", "أربع جلسات", "٤ جلسات"],
                    "6 جلسات": ["6 جلسات", "ست جلسات", "٦ جلسات"],
                    "8 جلسات": ["8 جلسات", "ثمان جلسات", "٨ جلسات"],
                    "جلسة واحدة": ["جلسة واحدة", "جلسة", "١ جلسة"],
                }
                
                user_wants_sessions = None
                for session_type, keywords in session_keywords.items():
                    if any(kw in user_input_lower for kw in keywords):
                        user_wants_sessions = session_type
                        logger.info(f"🎯 User specifically wants: {session_type}")
                        break
                
                # Try to match based on session count
                if user_wants_sessions:
                    # Extract the number from user's request (e.g., "4" from "4 جلسات")
                    import re
                    session_number_match = re.search(r'(\d+|اربع|أربع|ست|ثمان|٤|٦|٨)', user_input_lower)
                    
                    # Map Arabic words to numbers
                    arabic_to_num = {
                        'اربع': '4', 'أربع': '4', '٤': '4',
                        'ست': '6', '٦': '6',
                        'ثمان': '8', '٨': '8'
                    }
                    
                    target_number = None
                    if session_number_match:
                        num_str = session_number_match.group(1)
                        target_number = arabic_to_num.get(num_str, num_str)
                    
                    # Filter matches to ONLY those with the target session count
                    if target_number:
                        filtered_matches = []
                        for match in matching:
                            match_name = match.get("name", "")
                            # Check if service name contains the target number followed by "جلسات" or "جلسة"
                            if re.search(rf'(عدد\s*)?{target_number}\s*(جلسات|جلسة)', match_name):
                                filtered_matches.append(match)
                                logger.info(f"✅ Session count match: '{match.get('name')}' (ID: {match['id']}) - has {target_number} sessions")
                        
                        # If we found filtered matches, use them
                        if filtered_matches:
                            if len(filtered_matches) == 1:
                                best_match = filtered_matches[0]
                                logger.info(f"✅ BEST MATCH by session count: '{best_match.get('name')}' (ID: {best_match['id']})")
                            else:
                                # Multiple matches with same session count - narrow down the list
                                logger.info(f"🎯 Narrowed down to {len(filtered_matches)} services with {target_number} sessions")
                                matching = filtered_matches  # Replace matching list with filtered one
                
                # 🚨 CRITICAL CHANGE: If can't disambiguate, return ALL matches for user to choose
                if not best_match:
                    # DON'T ask clarifying questions - show numbered list directly
                    # A human receptionist shows options, doesn't ask abstract questions
                    
                    logger.warning(f"⚠️ Found {len(matching)} matching services - showing numbered list for user selection")
                    
                    # 🎯 HUMAN BEHAVIOR: Filter by relevance and limit to top 5-7
                    # Real receptionist shows BEST options, not everything!
                    
                    # Step 1: Filter out low-quality matches (< 50% relevance)
                    high_quality = [m for m in matching if m.get("match_score", 0) >= 50]
                    
                    # Step 2: If we have good matches, use them. Otherwise use all.
                    filtered_matches = high_quality if high_quality else matching
                    
                    # Step 3: Sort by match score (100% first, then descending)
                    sorted_matches = sorted(filtered_matches, key=lambda m: m.get("match_score", 0), reverse=True)
                    
                    # Step 4: Limit to top 7 (human-readable number)
                    top_matches = sorted_matches[:7]
                    
                    logger.info(f"📋 Filtered {len(matching)} → {len(filtered_matches)} relevant → showing top {len(top_matches)}")
                    
                    # 🎯 SMART AUTO-CONFIRM: If only 1 result with 100% match, auto-save it!
                    # Human receptionist would say "Perfect! I'll book the Xeomin we just discussed"
                    # Not "Please select from: 1. Xeomin" (that's stupid!)
                    if len(top_matches) == 1 and top_matches[0].get("match_score", 0) == 100:
                        perfect_match = top_matches[0]
                        logger.info(f"🎯 SINGLE PERFECT MATCH (100%) - AUTO-CONFIRMING: '{perfect_match.get('name')}' (ID: {perfect_match['id']})")
                        logger.info(f"💡 SMART BEHAVIOR: Not asking user to 'select from 1 option' - that's robotic!")
                        
                        # 🚨 CRITICAL: Extract price from service name if not in price field
                        # Many services have price in name like "بوتوكس ابتسامة لثوية  350"
                        import re
                        price = perfect_match.get("price", 0)
                        if not price or price == 0:
                            service_name = perfect_match.get("name", "")
                            # Match number at end of name (e.g., "Service Name 350" → 350)
                            price_match = re.search(r'(\d+)\s*$', service_name)
                            if price_match:
                                price = int(price_match.group(1))
                                logger.info(f"💰 EXTRACTED PRICE from name: {price} SAR")
                            else:
                                logger.warning(f"⚠️ NO PRICE found in API or name: {service_name}")
                        else:
                            logger.info(f"💰 PRICE from API: {price} SAR")
                        
                        # Save to booking_state immediately so it won't search again
                        booking_state.collected_data["service_id"] = perfect_match["id"]
                        booking_state.collected_data["service_name"] = perfect_match.get("name", "Unknown")
                        booking_state.collected_data["service_price"] = price  # Use extracted price!
                        logger.info(f"✅ SAVED TO BOOKING STATE (price={price}) - Next time won't search again!")
                        
                        # Return as single match (not multiple)
                        return {
                            "success": True,
                            "data": {
                                "service_id": perfect_match["id"],
                                "service_name": perfect_match.get("name", "Unknown"),
                                "service_price": price,  # Use extracted price!
                                "duration_minutes": perfect_match.get("duration_minutes", 60),
                                "requires_doctor": bool(perfect_match.get("doctor_id") or perfect_match.get("requires_doctor")),
                                "requires_specialist": bool(perfect_match.get("specialist_id") or perfect_match.get("requires_specialist")),
                                "requires_device": bool(perfect_match.get("device_id") or perfect_match.get("requires_device")),
                                "status": "auto_confirmed",
                                "next_step": "collect_datetime"  # Proceed directly to date/time
                            },
                            "message": f"Found perfect match: {perfect_match.get('name')}"
                        }
                    
                    # Format services for user selection
                    formatted_services = []
                    for match in top_matches:
                        # Extract price from name if not in price field (e.g., "جلسة ليزر 350" → price=350)
                        import re
                        price = match.get("price", 0)
                        if not price or price == 0:
                            service_name = match.get("name", "")
                            price_match = re.search(r'(\d+)\s*$', service_name)  # Find number at end of name
                            if price_match:
                                price = int(price_match.group(1))
                        
                        formatted_services.append({
                            "id": match["id"],
                            "name": match.get("name", "Unknown"),
                            "price": price,
                            "duration_minutes": match.get("duration_minutes", 60),
                            "requires_doctor": bool(match.get("doctor_id") or match.get("requires_doctor")),
                            "requires_specialist": bool(match.get("specialist_id") or match.get("requires_specialist")),
                            "requires_device": bool(match.get("device_id") or match.get("requires_device"))
                        })
                    
                    # Save list to context for later selection tracking
                    context.metadata["last_displayed_list"] = formatted_services
                    context.metadata["last_list_type"] = "services"
                    context.metadata["last_service_query"] = service_name  # Remember what they searched for
                    
                    return {
                        "success": True,
                        "multiple_matches": True,
                        "data": {
                            "services": formatted_services,
                            "count": len(formatted_services),
                            "query": service_name,
                            "needs_user_selection": True,
                            "next_step": "awaiting_service_selection"
                        },
                        "message": f"Found {len(formatted_services)} matching services - user needs to choose"
                    }
                
                service_id = best_match["id"]
                service_name_found = best_match.get("name", "Unknown")
                
                # 🎯 Extract session count if present (for human-like response)
                session_count_match = re.search(r'عدد\s*(\d+)\s*جلسات?', service_name_found)
                session_count = session_count_match.group(1) if session_count_match else None
                
                logger.info(f"✅ AUTO-SELECTED SERVICE: '{service_name_found}' (ID: {service_id}) based on keyword matching")
                if session_count:
                    logger.info(f"🎯 Session count detected: {session_count} sessions")
            else:
                # Single match - perfect!
                first_match = matching[0]
                service_id = first_match["id"]
                service_name_found = first_match.get("name", "Unknown")
                logger.info(f"✅ FOUND SINGLE SERVICE: '{service_name_found}' (ID: {service_id})")
        
        # Get full service details
        service = await self.api_client.get_service(service_id)
        
        if not service:
            return {
                "success": False,
                "message": f"Service {service_id} not found"
            }
        
        # 🎯 Extract session count from service name for human-like responses
        import re
        service_name_for_response = service.get("name", "Unknown")
        session_count_match = re.search(r'عدد\s*(\d+)\s*جلسات?', service_name_for_response)
        session_count = session_count_match.group(1) if session_count_match else None
        
        return {
            "success": True,
            "data": {
                "service": service,
                "service_id": service_id,
                "service_name": service.get("name"),
                "session_count": session_count,  # 🎯 Add session count for LLM to mention
                "next_step": "check_availability"
            },
            "message": "Service collected successfully"
        }
    
    async def _check_availability_step(
        self,
        data: dict,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Check available slots"""
        
        from datetime import datetime, timedelta
        from app.utils.date_parser import extract_date_from_context
        
        # 🔥 CRITICAL: HUMAN-LIKE MEMORY - Trust booking_state FIRST, not LLM hallucinations!
        # A human receptionist remembers what was selected, even if they mishear the next request
        service_id = (
            context.booking_state.collected_data.get("service_id") or  # ✅ MEMORY FIRST!
            context.booking_state.collected_data.get("service", {}).get("id") or
            data.get("service_id")  # ⚠️ LLM parameter LAST (can hallucinate)
        )
        
        # Debug: Show what sources we checked
        if service_id:
            if context.booking_state.collected_data.get("service_id") == service_id:
                logger.info(f"✅ Using service_id={service_id} from booking_state (HUMAN MEMORY)")
            elif data.get("service_id") == service_id:
                logger.warning(f"⚠️ Using service_id={service_id} from LLM parameter (booking_state was empty!)")
        else:
            logger.warning(f"❌ No service_id found in booking_state OR data parameter")
        
        # SMARTNESS: If no service_id, try to get from last discussed service
        if not service_id and context.last_discussed_service:
            logger.warning(f"⚠️ No service_id provided, but last_discussed_service='{context.last_discussed_service}'")
            logger.warning("💡 Agent should have tracked service_id in booking_state!")
            logger.warning("💡 HUMAN BEHAVIOR: A human would remember the service we just discussed!")
        
        if not service_id:
            return {
                "success": False,
                "message": "Service not selected yet",
                "data": {
                    "reason": "no_service_id",
                    "last_discussed": context.last_discussed_service,
                    "booking_state_data": list(context.booking_state.collected_data.keys())
                }
            }
        
        # SMARTNESS: Extract date from conversation if not provided (HUMAN-LIKE)
        # CRITICAL: intelligent_booking_agent uses 'date' NOT 'date_from'/'date_to'!
        logger.info(f"🗓️ Date extraction: data.get('date')={data.get('date')}")
        date = data.get("date")
        if not date:
            # Try to extract from recent conversation
            recent_messages = context.conversation_history[-3:] if context.conversation_history else []
            current_msg = data.get("user_message", "")
            logger.info(f"🧠 SMARTNESS: Attempting to extract date from conversation (current_msg='{current_msg}')")
            
            try:
                extracted_date = extract_date_from_context(current_msg, recent_messages)
                if extracted_date:
                    date = extracted_date
                    logger.info(f"✅ SMARTNESS: Auto-extracted date from conversation: {date}")
                else:
                    # Default to tomorrow (following intelligent_booking_agent pattern)
                    date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info(f"⚠️ No date extracted, using tomorrow: {date}")
            except Exception as e:
                logger.warning(f"⚠️ Date extraction failed: {e}, using tomorrow")
                date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Validate date (following intelligent_booking_agent pattern)
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            today = datetime.now().date()
            date_only = date_obj.date()
            
            # 🔥 HUMAN-LIKE BEHAVIOR: Auto-correct past dates to tomorrow
            # A human receptionist would understand "yesterday" is a mistake and correct it
            if date_only < today:
                logger.warning(f"⚠️ Date is in the PAST: {date} - Auto-correcting to tomorrow")
                date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"✅ Auto-corrected date to: {date}")
        except ValueError:
            logger.error(f"🚨 Invalid date format: {date}")
            return {
                "success": False,
                "message": "Invalid date format",
                "data": {"date": date}
            }
        
        # Get service details for resource requirements (following intelligent_booking_agent pattern)
        logger.info(f"🔍 Fetching service details for service_id={service_id}")
        device_id = None
        specialist_id = None
        doctor_id = None
        
        try:
            service = await self.api_client.get_service(service_id)
            service_name = service.get("name", "").lower()
            requires_doctor = service.get("doctor_id") or service.get("requires_doctor")
            requires_specialist = service.get("specialist_id") or service.get("requires_specialist")
            requires_device = service.get("device_id") or service.get("requires_device")
            
            logger.info(f"📋 Service requirements: doctor={requires_doctor}, specialist={requires_specialist}, device={requires_device}")
            
            # 🧠 SMART: Save requirements to booking_state for later use (e.g., in confirmation message)
            context.booking_state.collected_data["requires_doctor"] = bool(requires_doctor)
            context.booking_state.collected_data["requires_specialist"] = bool(requires_specialist)
            context.booking_state.collected_data["requires_device"] = bool(requires_device)
            logger.info(f"✅ Saved service requirements to booking_state")
            
            # 🚨 CRITICAL: Auto-select device if required (intelligent_booking_agent pattern)
            if requires_device:
                logger.info(f"✅ Service requires device - smart auto-selecting")
                
                # Check if service has a preferred device_id in database
                device_id_field = service.get("device_id")
                logger.info(f"🔍 Service device_id field value: {device_id_field} (type: {type(device_id_field)})")
                
                preferred_device_id = device_id_field if isinstance(device_id_field, int) else None
                
                if preferred_device_id:
                    logger.info(f"✅ Service has preferred device_id={preferred_device_id} in database - using it")
                    device_id = preferred_device_id
                else:
                    # No preferred device - do smart matching
                    try:
                        devices_result = await self.api_client.get_devices(limit=50)
                        devices = devices_result.get("results", []) or devices_result.get("data", [])
                        
                        if not devices or len(devices) == 0:
                            logger.error("❌ No devices available in system")
                            return {
                                "success": False,
                                "message": "No devices available",
                                "data": {"service_id": service_id}
                            }
                        
                        logger.info(f"🔍 Checking {len(devices)} devices for match with service '{service_name}'")
                        
                        # Smart device matching with token-based fuzzy logic (intelligent_booking_agent pattern)
                        selected_device = None
                        best_match_score = 0
                        
                        for device in devices:
                            device_name_original = device.get("name", "")
                            device_name = device_name_original.lower()
                            
                            # Strategy 1: Exact substring match (fastest)
                            if device_name and device_name in service_name:
                                selected_device = device
                                logger.info(f"✅ EXACT match: '{device_name_original}' found in service name")
                                break
                            
                            # Strategy 2: Token-based matching (handles variations)
                            import re
                            device_tokens = [
                                token.strip() 
                                for token in re.split(r'[/\-,،\s]+', device_name) 
                                if token.strip() and len(token.strip()) > 2 and not token.strip().isascii()
                            ]
                            
                            # Count how many device tokens appear in service name
                            matches = sum(1 for token in device_tokens if token in service_name)
                            match_score = matches / len(device_tokens) if device_tokens else 0
                            
                            if match_score > best_match_score:
                                best_match_score = match_score
                                selected_device = device
                                logger.info(f"🔍 TOKEN match: '{device_name_original}' score={match_score:.2f} ({matches}/{len(device_tokens)} tokens)")
                    
                        if selected_device and best_match_score > 0:
                            logger.info(f"✅ AUTO-SELECTED device: {selected_device.get('name')} (score={best_match_score:.2f})")
                        elif selected_device:
                            logger.info(f"✅ AUTO-SELECTED device: {selected_device.get('name')} (exact match)")
                        else:
                            # No match at all - use first
                            selected_device = devices[0]
                            logger.warning(f"⚠️ No device match - using first available: {selected_device.get('name')}")
                        
                        device_id = selected_device.get("id")
                        logger.info(f"✅ Auto-selected device_id={device_id} ({selected_device.get('name')})")
                        
                    except Exception as device_error:
                        logger.error(f"❌ Failed to auto-select device: {device_error}")
                        # Continue without device - API will tell us if it's required
            
            # 🚨 CRITICAL: DON'T auto-select doctor - let user choose!
            # Human receptionist shows available doctors and asks which one customer prefers
            elif requires_doctor:
                # Check if doctor already selected by user
                if not collected.get("doctor_id"):
                    logger.warning(f"⚠️ Service requires doctor but user hasn't selected one yet!")
                    logger.info(f"🔄 This should have been handled in confirm_booking step - continuing for now")
                    # Note: Doctor selection should happen in confirm_booking step
                    # where we show list and user chooses
                    # For now, continue - confirm_booking will handle it
                else:
                    doctor_id = collected.get("doctor_id")
                    logger.info(f"✅ Using user-selected doctor_id={doctor_id}")
            
            # 🚨 CRITICAL: Auto-select specialist if required (intelligent_booking_agent pattern)
            elif requires_specialist:
                logger.info(f"✅ Service requires specialist - auto-selecting first available")
                try:
                    specialists_result = await self.api_client.get_specialists(limit=1)
                    specialists = specialists_result.get("results", []) or specialists_result.get("data", [])
                    if specialists and len(specialists) > 0:
                        specialist_id = specialists[0].get("id")
                        logger.info(f"✅ Auto-selected specialist_id={specialist_id}")
                    else:
                        logger.error("❌ No specialists available in system")
                except Exception as specialist_error:
                    logger.warning(f"⚠️ Failed to auto-select specialist: {specialist_error}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch service details: {e}")
        
        # 🔥 CRITICAL: Get patient_id from context (HUMAN-LIKE BEHAVIOR)
        # A human receptionist would check who the patient is before checking slots
        patient_id = None
        if context.patient and context.patient.id:
            patient_id = context.patient.id
            logger.info(f"✅ Using patient_id={patient_id} from context")
        else:
            logger.warning(f"⚠️ No patient_id in context - slots API might fail!")
        
        # Build params with auto-selected resources + patient_id
        params = {
            "service_id": service_id,
            "date": date
        }
        
        # Add patient_id if available (CRITICAL for some backend validations)
        if patient_id:
            params["patient_id"] = patient_id
        
        if device_id:
            params["device_id"] = device_id
        if specialist_id:
            params["specialist_id"] = specialist_id
        if doctor_id:
            params["doctor_id"] = doctor_id
        
        logger.info(f"🔍 Getting slots with params: {params}")
        
        # 🚨 REACTIVE ERROR RECOVERY (intelligent_booking_agent pattern)
        try:
            # TRY 1: Call API with current params
            slots = await self.api_client.get_available_slots(**params)
        except Exception as api_error:
            error_msg = str(api_error).lower()
            logger.warning(f"⚠️ Slots API error: {error_msg[:200]}")
            
            # REACTIVE: Check what resource is REQUIRED and auto-select
            requires_device = "requires a device_id" in error_msg or "requires device_id" in error_msg
            requires_specialist = "requires a specialist_id" in error_msg or "requires specialist_id" in error_msg
            requires_doctor = "requires a doctor_id" in error_msg or "requires doctor_id" in error_msg
            
            if requires_device and not device_id:
                # REACTIVE device selection
                logger.warning(f"🔧 REACTIVE: Service requires device - auto-selecting")
                try:
                    devices_result = await self.api_client.get_devices(limit=50)
                    devices = devices_result.get("results", []) or devices_result.get("data", [])
                    if devices and len(devices) > 0:
                        device_id = devices[0].get("id")
                        logger.warning(f"✅ REACTIVE: Auto-selected device_id={device_id}")
                        # Retry with device_id
                        params["device_id"] = device_id
                        slots = await self.api_client.get_available_slots(**params)
                    else:
                        raise Exception("No devices available")
                except Exception as retry_error:
                    logger.error(f"❌ REACTIVE retry failed: {retry_error}")
                    raise api_error  # Re-raise original error
            
            elif requires_specialist and not specialist_id:
                # REACTIVE specialist selection
                logger.warning(f"🔧 REACTIVE: Service requires specialist - auto-selecting")
                try:
                    specialists_result = await self.api_client.get_specialists(limit=1)
                    specialists = specialists_result.get("results", []) or specialists_result.get("data", [])
                    if specialists and len(specialists) > 0:
                        specialist_id = specialists[0].get("id")
                        logger.warning(f"✅ REACTIVE: Auto-selected specialist_id={specialist_id}")
                        # Retry with specialist_id
                        params["specialist_id"] = specialist_id
                        slots = await self.api_client.get_available_slots(**params)
                    else:
                        raise Exception("No specialists available")
                except Exception as retry_error:
                    logger.error(f"❌ REACTIVE retry failed: {retry_error}")
                    raise api_error
            
            elif requires_doctor:
                # Can't auto-select doctor - need user selection
                logger.error(f"🚨 Service requires doctor - cannot auto-select, need user choice")
                return {
                    "success": False,
                    "message": "This service requires a doctor. Please select a doctor first.",
                    "data": {
                        "service_id": service_id,
                        "required_resource": "doctor_id",
                        "next_action": "show_doctors"
                    }
                }
            else:
                # Unknown error - re-raise
                raise api_error
        
        return {
            "success": True,
            "data": {
                "available_times": slots,  # Just for reference, NOT for selection!
                "date": date,
                "service_id": service_id,
                "time_count": len(slots),
                "next_step": "collect_datetime"  # Directly ask for date+time, NO slot selection
            },
            "message": f"Found {len(slots)} available slots for {date}"
        }
    
    async def _collect_datetime_step(
        self,
        data: dict,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """SMART date/time collection - validates availability like a human receptionist!
        
        Human-like behavior:
        1. If user provides date only → Check available times and suggest
        2. If user provides date+time → Validate availability, suggest alternatives if not available
        3. Always be proactive and helpful
        
        Example:
        User: "بكرة" (tomorrow)
        Bot: Checks available times
        Bot: "عندنا مواعيد متاحة بكرة: الساعة 10 صباحاً، 2 العصر، 5 مساءً. أي وقت يناسبك؟"
        """
        
        appointment_date = data.get("date")
        appointment_time = data.get("time")
        service_id = context.booking_state.collected_data.get("service_id")
        
        # 🚨 HUMAN BEHAVIOR: Ask what's missing, don't just fail!
        if not appointment_date and not appointment_time:
            return {
                "success": False,
                "message": "need_date_and_time",
                "data": {
                    "service_name": context.booking_state.collected_data.get("service_name"),
                    "reason": "both_missing",
                    "question": "متى تبي الموعد؟ (مثلاً: بكرة الساعة 3 العصر)"
                }
            }
        elif not appointment_date:
            return {
                "success": False,
                "message": "missing_booking_date",
                "data": {
                    "time": appointment_time,
                    "service_name": context.booking_state.collected_data.get("service_name"),
                    "reason": "have_time_need_date",
                    "question": f"تمام الساعة {appointment_time}. أي يوم تبي؟"
                }
            }
        
        # Normalize time format to HH:MM (remove seconds if present)
        if appointment_time and len(appointment_time) > 5:
            appointment_time = appointment_time[:5]
        
        # Validate time format and handle ambiguity
        if appointment_time and ":" not in appointment_time:
            logger.warning(f"⚠️ Time format issue: {appointment_time} - attempting to fix")
            try:
                hour = int(appointment_time)
                if 1 <= hour <= 12:
                    # Ambiguous - need AM/PM clarification
                    return {
                        "success": False,
                        "message": "need_time_clarification",
                        "data": {
                            "hour": hour,
                            "date": appointment_date,
                            "question": f"الساعة {hour} صباحاً ولا مساءً؟"
                        }
                    }
                elif 13 <= hour <= 23:
                    appointment_time = f"{hour:02d}:00"
                    logger.info(f"✅ Converted hour {hour} to time: {appointment_time}")
            except:
                pass
        
        # 🎯 SMART BEHAVIOR: Check availability before confirming!
        # A human receptionist would NEVER confirm a time without checking availability
        if not appointment_time:
            # User provided date only - check available times and suggest
            logger.info(f"🔍 User provided date only ({appointment_date}) - checking available times to suggest...")
            
            if service_id:
                try:
                    # Check what times are available on this date
                    availability_result = await self._check_availability_step(
                        {"service_id": service_id, "date": appointment_date},
                        context
                    )
                    
                    if availability_result.get("success") and availability_result.get("data", {}).get("available_times"):
                        available_times = availability_result["data"]["available_times"]
                        
                        # Format times nicely for display
                        time_suggestions = []
                        for slot in available_times[:5]:  # Show up to 5 times
                            # Handle both dict and string slots
                            time_str = slot.get("time", "") if isinstance(slot, dict) else str(slot)
                            if time_str:
                                time_suggestions.append(time_str)
                        
                        if time_suggestions:
                            logger.info(f"✅ Found {len(time_suggestions)} available times for {appointment_date}")
                            return {
                                "success": False,
                                "message": "suggest_available_times",
                                "data": {
                                    "date": appointment_date,
                                    "available_times": time_suggestions,
                                    "service_name": context.booking_state.collected_data.get("service_name"),
                                    "question": f"عندنا مواعيد متاحة يوم {appointment_date}"
                                }
                            }
                except Exception as e:
                    logger.warning(f"⚠️ Could not check availability: {e}")
            
            # Fallback: ask for time without suggestions
            return {
                "success": False,
                "message": "missing_booking_time",
                "data": {
                    "date": appointment_date,
                    "service_name": context.booking_state.collected_data.get("service_name"),
                    "reason": "have_date_need_time",
                    "question": f"تمام يوم {appointment_date}. أي ساعة تناسبك؟"
                }
            }
        
        # User provided both date AND time - validate availability!
        logger.info(f"🔍 Validating availability for {appointment_date} at {appointment_time}...")
        
        if service_id:
            try:
                # Check if this specific time is available
                availability_result = await self._check_availability_step(
                    {"service_id": service_id, "date": appointment_date},
                    context
                )
                
                if availability_result.get("success"):
                    available_times = availability_result.get("data", {}).get("available_times", [])
                    
                    # 🚨 CRITICAL: Ensure available_times is actually a list
                    if not isinstance(available_times, list):
                        logger.warning(f"⚠️ available_times is not a list: {type(available_times)}")
                        available_times = []
                    
                    # Check if user's requested time is in the available list
                    # Handle both dict slots and string slots
                    time_available = False
                    if available_times:
                        try:
                            time_available = any(
                                (slot.get("time", "") if isinstance(slot, dict) else str(slot)).startswith(appointment_time[:5])
                                for slot in available_times
                            )
                        except Exception as e:
                            logger.error(f"❌ Error checking time availability: {e}")
                            time_available = False
                    
                    if not time_available and available_times:
                        # Time NOT available - suggest alternatives like a human!
                        logger.warning(f"⚠️ Requested time {appointment_time} not available on {appointment_date}")
                        
                        # Get alternative times - handle both dict and string slots
                        alternative_times = []
                        for slot in available_times[:5]:
                            if isinstance(slot, dict):
                                time_val = slot.get("time")
                                if time_val:
                                    alternative_times.append(time_val)
                            elif slot:  # String or other type
                                alternative_times.append(str(slot))
                        
                        return {
                            "success": False,
                            "message": "time_not_available_suggest_alternatives",
                            "data": {
                                "requested_date": appointment_date,
                                "requested_time": appointment_time,
                                "available_times": alternative_times,
                                "service_name": context.booking_state.collected_data.get("service_name"),
                                "reason": "requested_time_unavailable"
                            }
                        }
                    
                    logger.info(f"✅ Time {appointment_time} is available on {appointment_date}")
            
            except Exception as e:
                logger.warning(f"⚠️ Could not validate availability: {e}")
                # Continue anyway - let booking API validate
        
        # All good - proceed to confirmation
        logger.info(f"✅ Date and time collected: {appointment_date} at {appointment_time}")
        
        # 🧠 SMART: Include service data from booking_state in response
        # This prevents service_name from being lost when wrapping result
        collected = context.booking_state.collected_data
        result_data = {
            "date": appointment_date,
            "time": appointment_time,
            "next_step": "confirm_booking"
        }
        
        # Include service data if available
        if collected.get("service_id"):
            result_data["service_id"] = collected.get("service_id")
        if collected.get("service_name"):
            result_data["service_name"] = collected.get("service_name")
            logger.info(f"✅ Including service_name in result: {collected.get('service_name')}")
        if collected.get("service_price"):
            result_data["service_price"] = collected.get("service_price")
        
        # Include service requirements to determine what to show
        if collected.get("requires_doctor") is not None:
            result_data["requires_doctor"] = collected.get("requires_doctor")
        if collected.get("requires_specialist") is not None:
            result_data["requires_specialist"] = collected.get("requires_specialist")
        if collected.get("requires_device") is not None:
            result_data["requires_device"] = collected.get("requires_device")
        
        # Only include doctor/specialist/device if service actually requires them
        if collected.get("requires_doctor") and collected.get("doctor_id"):
            result_data["doctor_id"] = collected.get("doctor_id")
            if collected.get("doctor_name"):
                result_data["doctor_name"] = collected.get("doctor_name")
        elif collected.get("requires_specialist") and collected.get("specialist_id"):
            result_data["specialist_id"] = collected.get("specialist_id")
            if collected.get("specialist_name"):
                result_data["specialist_name"] = collected.get("specialist_name")
        elif collected.get("requires_device") and collected.get("device_id"):
            result_data["device_id"] = collected.get("device_id")
            if collected.get("device_name"):
                result_data["device_name"] = collected.get("device_name")
        
        return {
            "success": True,
            "data": result_data,
            "message": f"Date and time validated: {appointment_date} at {appointment_time}"
        }
    
    async def _check_existing_bookings(
        self,
        patient_id: int,
        service_id: Optional[int] = None,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🧠 SMART: Check if patient has existing bookings to prevent duplicates.
        
        Returns:
            {
                "has_bookings": bool,
                "bookings": list,
                "conflict": dict or None  # If duplicate service+date found
            }
        """
        try:
            logger.info(f"🔍 Checking existing bookings for patient_id={patient_id}")
            
            # Query API for patient's bookings
            # Filter by date if provided (check bookings on same date)
            params = {"patient_id": patient_id, "limit": 50}
            if date:
                params["date_from"] = date
                params["date_to"] = date
            
            bookings_response = await self.api_client.get_bookings(**params)
            bookings = bookings_response.get("results", []) or bookings_response.get("data", [])
            
            logger.info(f"✅ Found {len(bookings)} existing bookings for patient")
            
            # Check for conflicts (same service, same date)
            conflict = None
            if service_id and date:
                for booking in bookings:
                    booking_service_id = booking.get("service_id") or booking.get("service", {}).get("id")
                    booking_date = booking.get("start_date") or booking.get("appointment_date")
                    
                    if booking_service_id == service_id and booking_date == date:
                        logger.warning(f"⚠️ CONFLICT: Found existing booking for same service+date!")
                        logger.warning(f"   Booking ID: {booking.get('id')}, Service: {booking_service_id}, Date: {booking_date}")
                        conflict = booking
                        break
            
            return {
                "has_bookings": len(bookings) > 0,
                "bookings": bookings,
                "conflict": conflict
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to check existing bookings: {e}")
            return {
                "has_bookings": False,
                "bookings": [],
                "conflict": None
            }
    
    async def _confirm_booking_step(
        self,
        data: dict,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Final booking confirmation"""
        
        collected = context.booking_state.collected_data
        
        # 🚨 CRITICAL: Check if booking ALREADY created FOR THIS SERVICE!
        # Don't create duplicate bookings, but allow booking DIFFERENT services!
        existing_booking_id = collected.get("booking_id")
        booked_service_id = collected.get("booked_service_id")  # Service that was actually booked
        current_service_id = collected.get("service_id")  # Service user wants to book now
        
        if existing_booking_id and booked_service_id:
            # Check if it's the SAME service or a DIFFERENT one
            if booked_service_id == current_service_id:
                # Same service - prevent duplicate booking
                logger.warning(f"⚠️ BOOKING ALREADY EXISTS FOR THIS SERVICE! ID={existing_booking_id}, Service={booked_service_id}")
                logger.warning(f"🚫 User trying to confirm same service again - SKIP creation, just confirm it's done!")
                
                # Get doctor name if available
                doctor_name = collected.get("doctor_name")
                if not doctor_name and collected.get("doctor_id"):
                    try:
                        doctor_data = await self.api_client.get_doctor(collected.get("doctor_id"))
                        doctor_name = doctor_data.get("name", "سيتم تحديده")
                    except:
                        doctor_name = "سيتم تحديده"
                
                return {
                    "success": True,
                    "data": {
                        "booking_id": existing_booking_id,
                        "status": "already_confirmed",
                        "message": "booking_already_confirmed",
                        "appointment_date": collected.get("date") or collected.get("appointment_date"),
                        "appointment_time": collected.get("time") or collected.get("appointment_time"),
                        "service_name": collected.get("service_name"),
                        "service_id": collected.get("service_id"),
                        "service_price": collected.get("service_price", 0),
                        "doctor_name": doctor_name or "سيتم تحديده",
                        "doctor_id": collected.get("doctor_id"),
                        "patient_id": context.patient.id if context.patient else collected.get("patient_id"),
                        # Include requirements so confirmation knows what to display
                        "requires_doctor": collected.get("requires_doctor", False),
                        "requires_specialist": collected.get("requires_specialist", False),
                        "requires_device": collected.get("requires_device", False)
                    },
                    "message": "Booking already confirmed - returning existing booking details"
                }
            else:
                # Different service - user wants to book ANOTHER service!
                logger.info(f"🔄 DIFFERENT SERVICE BOOKING DETECTED!")
                logger.info(f"   Previous booking: ID={existing_booking_id}, Service={booked_service_id}")
                logger.info(f"   New booking request: Service={current_service_id}")
                logger.info(f"   ✅ Allowing new booking - clearing previous booking context")
                
                # Clear previous booking data (keep only new service data)
                collected.pop("booking_id", None)
                collected.pop("booked_service_id", None)
                # Note: We keep service_id, date, time, etc. for the NEW booking
                
                logger.info(f"   ✅ Proceeding to create NEW booking for service {current_service_id}")
        elif existing_booking_id and not booked_service_id:
            # 🚨 CRITICAL BUG FIX: Legacy case WITHOUT booked_service_id tracking
            # DON'T ASSUME - FETCH from API and COMPARE with current service!
            logger.warning(f"⚠️ Found booking_id={existing_booking_id} but no booked_service_id tracked")
            logger.warning(f"   🔍 Fetching from API to check if same service or different...")
            
            # Get current service user wants to book
            current_service_id = collected.get("service_id")
            current_service_name = collected.get("service_name")
            
            # Fetch the ACTUAL booking from database
            booking_service_id = None
            booking_service_name = None
            
            try:
                booking_details = await self.api_client.get_booking(existing_booking_id)
                if booking_details:
                    # Extract service info from ACTUAL booking
                    booking_service_name = booking_details.get("service_name") or booking_details.get("service", {}).get("name")
                    booking_service_id = booking_details.get("service_id") or booking_details.get("service", {}).get("id")
                    logger.info(f"✅ Fetched booking #{existing_booking_id} from API: {booking_service_name} (ID: {booking_service_id})")
            except Exception as e:
                logger.error(f"❌ Could not fetch booking details: {e}")
                # Can't determine - assume different service to be safe
                booking_service_id = None
            
            # 🧠 SMART COMPARISON: Check if SAME service or DIFFERENT
            if booking_service_id and current_service_id and booking_service_id == current_service_id:
                # SAME SERVICE - Return existing booking (duplicate prevention)
                logger.warning(f"✅ SAME SERVICE: booking #{existing_booking_id} is for service_id={booking_service_id}")
                logger.warning(f"🚫 User confirming same service again - returning existing booking")
                
                # Get doctor name if available
                doctor_name = collected.get("doctor_name")
                if not doctor_name and collected.get("doctor_id"):
                    try:
                        doctor_data = await self.api_client.get_doctor(collected.get("doctor_id"))
                        doctor_name = doctor_data.get("name", "سيتم تحديده")
                    except:
                        doctor_name = "سيتم تحديده"
                
                return {
                    "success": True,
                    "data": {
                        "booking_id": existing_booking_id,
                        "status": "already_confirmed",
                        "message": "booking_already_confirmed",
                        "appointment_date": collected.get("date") or collected.get("appointment_date"),
                        "appointment_time": collected.get("time") or collected.get("appointment_time"),
                        "service_name": booking_service_name or current_service_name,
                        "service_id": booking_service_id,
                        "service_price": collected.get("service_price", 0),
                        "doctor_name": doctor_name or "سيتم تحديده",
                        "doctor_id": collected.get("doctor_id"),
                        "patient_id": context.patient.id if context.patient else collected.get("patient_id"),
                        # Include requirements so confirmation knows what to display
                        "requires_doctor": collected.get("requires_doctor", False),
                        "requires_specialist": collected.get("requires_specialist", False),
                        "requires_device": collected.get("requires_device", False)
                    },
                    "message": "Booking already confirmed - same service"
                }
            
            else:
                # DIFFERENT SERVICE - User wants to book another service!
                logger.info(f"🔄 DIFFERENT SERVICE DETECTED!")
                logger.info(f"   Existing booking #{existing_booking_id}: service_id={booking_service_id}")
                logger.info(f"   User wants to book: service_id={current_service_id}")
                logger.info(f"   ✅ Clearing old booking context and proceeding with NEW booking")
                
                # Clear previous booking data (keep only new service data)
                collected.pop("booking_id", None)
                collected.pop("booked_service_id", None)
                # Note: We keep current service_id, date, time, etc. for the NEW booking
                
                logger.info(f"   ✅ Proceeding to create NEW booking for service {current_service_id}")
                # Fall through to create new booking below
        
        # 🔍 DEBUG: Log what we have
        logger.info(f"📋 Confirming booking with collected data: {collected}")
        logger.info(f"   - service_id: {collected.get('service_id')}")
        logger.info(f"   - service_name: {collected.get('service_name')}")
        logger.info(f"   - date: {collected.get('date')}")
        logger.info(f"   - time: {collected.get('time')}")
        
        # 🚨 VALIDATION: Check for required fields early
        required_fields = {
            "service_id": "معرف الخدمة",
            "date": "تاريخ الموعد",
            "time": "وقت الموعد"
        }
        
        missing_fields = []
        for field, display_name in required_fields.items():
            if not collected.get(field):
                missing_fields.append({"field": field, "display_name": display_name})
        
        if missing_fields:
            logger.error(f"❌ Missing required fields for booking: {[f['field'] for f in missing_fields]}")
            return {
                "success": False,
                "message": "missing_required_fields",
                "data": {
                    "missing_fields": missing_fields,
                    "collected_fields": list(collected.keys()),
                    "service_name": collected.get("service_name")
                }
            }
        
        # 🚨 CRITICAL FALLBACK: Extract service from 'services' array if service_id missing
        service_id = collected.get("service_id")
        if not service_id and "services" in collected and isinstance(collected["services"], list):
            services_list = collected["services"]
            if len(services_list) > 0:
                selected_service = services_list[0]
                logger.warning(f"⚠️ FALLBACK EXTRACTION: service_id was None, extracting from services array")
                logger.info(f"🔧 Extracted: {selected_service.get('name')} (ID: {selected_service.get('id')})")
                
                # Update collected data with flattened service
                collected["service_id"] = selected_service.get("id")
                collected["service_name"] = selected_service.get("name")
                collected["service_price"] = selected_service.get("price", 0)
                service_id = collected["service_id"]
                
                # Save back to booking_state
                context.booking_state.collected_data.update(collected)
                logger.info(f"✅ FALLBACK SUCCESS: service_id={service_id}")
        
        # Validate all required data
        patient_id = context.patient.id if context.patient else None
        
        if not all([service_id, patient_id]):
            logger.error(f"❌ Missing required booking data: service_id={service_id}, patient_id={patient_id}")
            logger.error(f"❌ Full collected data: {collected}")
            return {
                "success": False,
                "message": "Missing required booking data"
            }
        
        # Create booking - use time+date instead of deprecated slot_choice_id
        start_time = collected.get("time")
        start_date = collected.get("date")
        
        # 🚨 HUMAN BEHAVIOR: Ask for missing info instead of failing!
        if not start_date:
            return {
                "success": False,
                "message": "missing_booking_date",
                "data": {
                    "service_id": service_id,
                    "service_name": collected.get("service_name"),
                    "reason": "need_date"
                }
            }
        
        if not start_time:
            return {
                "success": False,
                "message": "missing_booking_time",
                "data": {
                    "service_id": service_id,
                    "service_name": collected.get("service_name"),
                    "date": start_date,
                    "reason": "need_time"
                }
            }
        
        # 🧠 SMART API CHECK: Before creating booking, check for conflicts
        logger.info(f"🧠 SMART CHECK: Querying API for existing bookings...")
        existing_check = await self._check_existing_bookings(
            patient_id=patient_id,
            service_id=service_id,
            date=start_date
        )
        
        if existing_check.get("conflict"):
            # Found existing booking for same service+date!
            conflict_booking = existing_check["conflict"]
            conflict_id = conflict_booking.get("id")
            logger.warning(f"🚫 API CONFLICT: Patient already has booking ID={conflict_id} for this service on {start_date}")
            
            # Return the existing booking details
            return {
                "success": True,
                "data": {
                    "booking_id": conflict_id,
                    "status": "already_confirmed",
                    "message": "booking_already_confirmed",
                    "appointment_date": start_date,
                    "appointment_time": start_time,
                    "service_name": collected.get("service_name"),
                    "service_id": service_id,
                    "service_price": collected.get("service_price", 0),
                    "patient_id": patient_id,
                    "source": "api_check"  # Indicates this came from API validation
                },
                "message": "Booking already exists in database - returned from API check"
            }
        
        # 🎯 DIRECT BOOKING CREATION - NO LANGGRAPH!
        # Simple, reliable, fast - just call the API directly
        
        # Get service details for resource requirements
        try:
            service_details = await self.api_client.get_service(service_id)
            requires_doctor = service_details.get("requires_doctor", False)
            requires_specialist = service_details.get("requires_specialist", False)
            requires_device = service_details.get("requires_device", False)
            duration_minutes = service_details.get("duration_minutes", 60)
            
            logger.info(f"📋 Service {service_id} requirements: doctor={requires_doctor}, specialist={requires_specialist}, device={requires_device}")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch service details: {e}")
            requires_doctor = False
            requires_specialist = False
            requires_device = True  # Default to device
            duration_minutes = 60
        
        # Auto-select required resource (ONE only!)
        doctor_id = None
        specialist_id = None
        device_id = None
        
        if requires_device:
            # Most common case - auto-select first device
            try:
                devices = await self.api_client.get_devices(limit=1)
                device_list = devices.get("results", []) or devices.get("data", [])
                if device_list:
                    device_id = device_list[0].get("id")
                    logger.info(f"✅ Auto-selected device_id={device_id}")
            except Exception as e:
                logger.error(f"❌ Could not get device: {e}")
        
        elif requires_specialist:
            # Auto-select specialist
            try:
                specialists = await self.api_client.get_specialists(limit=1)
                specialist_list = specialists.get("results", []) or specialists.get("data", [])
                if specialist_list:
                    specialist_id = specialist_list[0].get("id")
                    logger.info(f"✅ Auto-selected specialist_id={specialist_id}")
            except Exception as e:
                logger.error(f"❌ Could not get specialist: {e}")
        
        elif requires_doctor:
            # 🎯 SMART BEHAVIOR: Check if doctor provided in THIS call (from user selection)
            doctor_id_from_data = data.get("doctor_id")  # User just selected!
            doctor_id_from_collected = collected.get("doctor_id")  # Previously saved
            
            # If user just selected doctor (from_data), save it and use it
            if doctor_id_from_data:
                logger.info(f"✅ User just selected doctor_id={doctor_id_from_data} - SAVING IT!")
                collected["doctor_id"] = doctor_id_from_data
                doctor_id = doctor_id_from_data
            elif doctor_id_from_collected:
                logger.info(f"✅ Doctor already in booking_state: doctor_id={doctor_id_from_collected}")
                doctor_id = doctor_id_from_collected
            else:
                # No doctor selected yet - show list
                # 🧑‍⚕️ Fetch available doctors and ask user to choose
                logger.info(f"👨‍⚕️ Service requires doctor - fetching available doctors...")
                try:
                    doctors_response = await self.api_client.get_doctors(limit=10)
                    doctors_list = doctors_response.get("results", []) or doctors_response.get("data", [])
                    
                    if not doctors_list:
                        logger.error(f"❌ No doctors available!")
                        return {
                            "success": False,
                            "message": "no_doctors_available",
                            "data": {
                                "service_id": service_id,
                                "service_name": collected.get("service_name")
                            }
                        }
                    
                    # Format doctors for user selection
                    formatted_doctors = []
                    for doc in doctors_list:
                        formatted_doctors.append({
                            "id": doc.get("id"),
                            "name": doc.get("name", "Unknown"),
                            "specialty": doc.get("specialty", ""),
                            "experience": doc.get("experience_years", "")
                        })
                    
                    logger.info(f"✅ Found {len(formatted_doctors)} doctors available")
                    
                    # Save to context for selection tracking
                    context.metadata["available_doctors"] = formatted_doctors
                    context.metadata["pending_doctor_selection"] = True
                    
                    return {
                        "success": False,  # Not complete yet - need doctor selection
                        "message": "doctor_selection_needed",
                        "data": {
                            "service_id": service_id,
                            "service_name": collected.get("service_name"),
                            "doctors": formatted_doctors,
                            "count": len(formatted_doctors),
                            "needs_doctor_selection": True,
                            "next_step": "awaiting_doctor_selection"
                        }
                    }
                except Exception as e:
                    logger.error(f"❌ Failed to fetch doctors: {e}")
                    return {
                        "success": False,
                        "message": "doctor_fetch_failed",
                        "data": {
                            "service_id": service_id,
                            "service_name": collected.get("service_name"),
                            "error": str(e)
                        }
                    }
        
        # 🎯 Create booking directly via API
        try:
            logger.info(f"📞 Creating booking: patient={patient_id}, service={service_id}, date={start_date}, time={start_time}")
            logger.info(f"   Resources: device={device_id}, specialist={specialist_id}, doctor={doctor_id}")
            
            # 🚨 VALIDATION: If service requires doctor, doctor_id must be set
            if requires_doctor and not doctor_id:
                logger.error(f"❌ Service requires doctor but doctor_id is None!")
                return {
                    "success": False,
                    "message": "doctor_selection_needed",
                    "data": {
                        "service_id": service_id,
                        "service_name": collected.get("service_name"),
                        "reason": "doctor_required_but_not_selected"
                    }
                }
            
            booking_result = await self.api_client.create_booking(
                patient_id=patient_id,
                service_id=service_id,
                start_date=start_date,
                start_time=start_time,
                doctor_id=doctor_id,
                specialist_id=specialist_id,
                device_id=device_id,
                duration_minutes=duration_minutes
            )
            
            booking_id = booking_result.get("id") or booking_result.get("booking_id")
            
            if booking_id:
                # ✅ Success!
                context.booking_state.status = "completed"
                logger.info(f"✅ Booking created successfully: ID={booking_id} for Service={service_id}")
                
                # 🚨 CRITICAL: Save booking_id AND booked_service_id to prevent duplicate creations
                # This allows the system to track which service was booked and allow booking OTHER services
                collected["booking_id"] = booking_id
                collected["booked_service_id"] = service_id  # Track which service this booking is for
                context.booking_state.collected_data.update(collected)
                logger.info(f"   ✅ Saved: booking_id={booking_id}, booked_service_id={service_id}")
                
                # Get doctor name if available
                doctor_name = collected.get("doctor_name")
                if not doctor_name and doctor_id:
                    try:
                        doctor_data = await self.api_client.get_doctor(doctor_id)
                        doctor_name = doctor_data.get("name", "سيتم تحديده")
                        collected["doctor_name"] = doctor_name
                    except:
                        doctor_name = "سيتم تحديده"
                
                return {
                    "success": True,
                    "data": {
                        "booking_id": booking_id,
                        "status": "booking_created",
                        "message": "booking_successfully_created",
                        "appointment_date": start_date,
                        "appointment_time": start_time,
                        "service_name": collected.get("service_name"),
                        "service_id": service_id,
                        "service_price": collected.get("service_price", 0),
                        "doctor_name": doctor_name or "سيتم تحديده",
                        "doctor_id": doctor_id,
                        "specialist_id": specialist_id,
                        "device_id": device_id,
                        "patient_id": patient_id,
                        "duration_minutes": duration_minutes,
                        # Include requirements so confirmation knows what to display
                        "requires_doctor": collected.get("requires_doctor", False),
                        "requires_specialist": collected.get("requires_specialist", False),
                        "requires_device": collected.get("requires_device", False)
                    },
                    "message": "Booking created successfully"
                }
            else:
                # API didn't return booking_id
                logger.error(f"❌ Booking created but no ID returned: {booking_result}")
                return {
                    "success": False,
                    "message": "booking_no_id_returned",
                    "data": {"api_response": str(booking_result)[:200]}
                }
        
        except Exception as e:
            error_str = str(e)
            logger.error(f"❌ Booking creation failed: {e}", exc_info=True)
            
            # 🎯 SMART HUMAN BEHAVIOR: If time not available, suggest alternatives!
            # Like a receptionist: "Sorry that's taken, but I have 2pm, 5pm, or 6pm - which works?"
            if "validation_error" in error_str.lower() or "no longer available" in error_str.lower() or "غير متاح" in error_str:
                logger.warning(f"⚠️ Booking failed: Time not available. Fetching alternatives like a human would...")
                
                try:
                    # Get available times for same date with doctor if required
                    check_data = {"service_id": service_id, "date": start_date}
                    if doctor_id:
                        check_data["doctor_id"] = doctor_id
                    if specialist_id:
                        check_data["specialist_id"] = specialist_id
                    if device_id:
                        check_data["device_id"] = device_id
                    
                    availability_result = await self._check_availability_step(check_data, context)
                    
                    if availability_result.get("success"):
                        available_times = availability_result.get("data", {}).get("available_times", [])
                        
                        # Handle both dict and string slots (fixed bug)
                        alternative_times = []
                        for slot in available_times[:5]:  # Show up to 5 alternatives
                            if isinstance(slot, dict):
                                time_str = slot.get("time", "")
                            else:
                                time_str = str(slot)
                            if time_str and time_str != start_time:  # Don't suggest same time that failed!
                                alternative_times.append(time_str)
                        
                        if alternative_times:
                            logger.info(f"✅ Found {len(alternative_times)} alternative times for {start_date}")
                            return {
                                "success": False,
                                "message": "time_not_available_suggest_alternatives",
                                "data": {
                                    "requested_date": start_date,
                                    "requested_time": start_time,
                                    "available_times": alternative_times,
                                    "service_name": collected.get("service_name"),
                                    "doctor_id": doctor_id,
                                    "reason": "api_rejected_already_booked"
                                }
                            }
                        else:
                            logger.warning(f"⚠️ No alternative times found for {start_date}")
                except Exception as check_error:
                    logger.warning(f"⚠️ Could not check alternatives: {check_error}")
            
            # Generic error
            return {
                "success": False,
                "message": "booking_api_error",
                "data": {"error": str(e)[:200]}
            }
    
    async def _pause_booking(
        self,
        reason: str,
        context: ConversationContext
    ) -> FunctionResult:
        """
        Pause booking - save state for later.
        
        This is what enables dynamic conversation flow!
        """
        
        booking_state = context.booking_state
        
        if booking_state.status != "active":
            logger.warning(f"⚠️ Attempted to pause non-active booking (status: {booking_state.status})")
            # 🚨 CRITICAL: ALWAYS wrap errors!
            return FunctionResult(
                success=False,
                message="no_booking_to_pause",  # Error code
                function_name="pause_booking",
                needs_wrapping=True,  # ✅ Must wrap!
                data={"current_status": booking_state.status}
            )
        
        old_status = booking_state.status
        booking_state.status = "paused"
        booking_state.paused_at_step = booking_state.get_next_step()
        booking_state.paused_at = time.time()
        
        logger.info(f"⏸️ Booking paused at: {booking_state.paused_at_step} (reason: {reason})")
        
        # Record state transition
        self.metrics.record_state_transition(
            from_state=old_status,
            to_state="paused",
            session_id=context.session_id,
            metadata={"reason": reason, "paused_at_step": booking_state.paused_at_step}
        )
        
        return FunctionResult(
            success=True,
            data={
                "paused_at": booking_state.paused_at_step,
                "reason": reason
            },
            message="booking_paused",  # Code for wrapping
            function_name="pause_booking",
            booking_state=booking_state,
            needs_wrapping=True  # ✅ Wrap for natural response
        )
    
    async def _resume_booking(
        self,
        context: ConversationContext
    ) -> FunctionResult:
        """
        Resume paused booking from exact point.
        """
        
        booking_state = context.booking_state
        
        if not booking_state.can_resume():
            logger.warning(f"⚠️ Attempted to resume non-pausable booking (status: {booking_state.status})")
            
            # 🚨 CRITICAL: ALWAYS wrap errors! Never send raw messages to users!
            return FunctionResult(
                success=False,
                message="no_booking_to_resume",  # Error code for wrapping
                function_name="resume_booking",
                needs_wrapping=True,  # ✅ MUST wrap in human language!
                data={
                    "current_status": booking_state.status,
                    "reason": "booking_not_paused"
                }
            )
        
        old_status = booking_state.status
        booking_state.status = "active"
        resume_from = booking_state.paused_at_step
        booking_state.paused_at_step = None
        
        logger.info(f"▶️ Booking resumed from: {resume_from}")
        
        # Record state transition
        self.metrics.record_state_transition(
            from_state=old_status,
            to_state="active",
            session_id=context.session_id,
            metadata={"resumed_from": resume_from}
        )
        
        return FunctionResult(
            success=True,
            data={
                "resume_from": resume_from,
                "collected_so_far": booking_state.collected_data,
                "next_step": booking_state.get_next_step(),
                "progress": booking_state.progress
            },
            message=f"Resuming booking from: {resume_from}",
            function_name="resume_booking",
            booking_state=booking_state,
            needs_wrapping=True  # Reem should wrap with natural language
        )
    
    async def _cancel_booking(
        self,
        reason: str,
        context: ConversationContext
    ) -> FunctionResult:
        """
        Cancel booking completely - clear all state.
        """
        
        booking_state = context.booking_state
        
        previous_status = booking_state.status
        previous_data = booking_state.collected_data.copy()
        
        booking_state.status = "cancelled"
        booking_state.progress = {}
        booking_state.collected_data = {}
        booking_state.paused_at_step = None
        
        logger.info(f"❌ Booking cancelled (was: {previous_status}, reason: {reason})")
        
        # Record state transition
        self.metrics.record_state_transition(
            from_state=previous_status,
            to_state="cancelled",
            session_id=context.session_id,
            metadata={"reason": reason, "had_data": bool(previous_data)}
        )
        
        return FunctionResult(
            success=True,
            data={
                "previous_status": previous_status,
                "previous_data": previous_data,
                "reason": reason
            },
            message="booking_cancelled",  # Code for wrapping
            function_name="cancel_booking",
            booking_state=booking_state,
            needs_wrapping=True  # ✅ Wrap for natural response
        )
    
    # ==================== INFORMATION FUNCTIONS ====================
    
    async def _get_service_details(
        self,
        service_id: int,
        context: ConversationContext
    ) -> FunctionResult:
        """Get detailed service information for user questions"""
        
        logger.info(f"📋 Fetching service details: {service_id}")
        
        service = await self.api_client.get_service(service_id)
        
        if not service:
            return FunctionResult(
                success=False,
                message=f"Service {service_id} not found",
                function_name="get_service_details"
            )
        
        return FunctionResult(
            success=True,
            data={"service": service},
            message="Service details retrieved",
            function_name="get_service_details",
            needs_wrapping=True
        )
    
    async def _get_pricing(
        self,
        service_id: int,
        detailed: bool,
        context: ConversationContext
    ) -> FunctionResult:
        """Get pricing information"""
        
        logger.info(f"💰 Fetching pricing: service={service_id}, detailed={detailed}")
        
        service = await self.api_client.get_service(service_id)
        
        if not service:
            return FunctionResult(
                success=False,
                message=f"Service {service_id} not found",
                function_name="get_pricing"
            )
        
        pricing_data = {
            "service_name": service.get("name"),
            "base_price": service.get("price"),
            "currency": "SAR",
            "detailed": detailed
        }
        
        return FunctionResult(
            success=True,
            data=pricing_data,
            message="Pricing retrieved",
            function_name="get_pricing",
            needs_wrapping=True
        )
    
    async def _check_availability(
        self,
        service_id: int,
        date: Optional[str],
        context: ConversationContext
    ) -> FunctionResult:
        """Check available slots - called when user asks about availability"""
        
        from datetime import datetime, timedelta
        
        # 🔥 CRITICAL: HUMAN-LIKE MEMORY - Check booking_state FIRST, not LLM parameter!
        # The LLM can hallucinate service_id=1, but booking_state has the CORRECT user selection
        actual_service_id = (
            context.booking_state.collected_data.get("service_id") or  # ✅ User's selection
            context.booking_state.collected_data.get("service", {}).get("id") or
            service_id  # ⚠️ LLM parameter as fallback
        )
        
        if actual_service_id != service_id:
            logger.warning(f"⚠️ LLM hallucinated service_id={service_id}, using booking_state={actual_service_id} instead")
        else:
            logger.info(f"✅ LLM provided correct service_id={service_id} (matches booking_state)")
        
        if not date:
            date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")  # Default to tomorrow
        
        logger.info(f"📅 Checking availability: service={actual_service_id}, date={date}")
        
        # Get service details for resource requirements
        service = await self.api_client.get_service(actual_service_id)
        requires_device = service.get("device_id") or service.get("requires_device")
        
        # Call the step method which has all the logic for device auto-selection
        result = await self._check_availability_step(
            data={"service_id": actual_service_id, "date": date},  # ✅ Use corrected ID
            context=context
        )
        
        # Convert dict result to FunctionResult
        if result.get("success"):
            return FunctionResult(
                success=True,
                data=result.get("data", {}),
                message=result.get("message", "Availability checked"),
                function_name="check_availability",
                needs_wrapping=True
            )
        else:
            return FunctionResult(
                success=False,
                message=result.get("message", "Failed to check availability"),
                function_name="check_availability"
            )
    
    async def _search_services(
        self,
        query: str,
        context: ConversationContext
    ) -> FunctionResult:
        """
        Search services by keyword.
        
        This uses the same two-level search logic as _collect_service_step
        but returns results for Reem to present, rather than auto-selecting.
        """
        
        logger.info(f"🔍 Searching services: '{query}'")
        
        # Use the same two-level search logic from _collect_service_step
        # This searches categories first, then gets subservices
        try:
            # Call _collect_service_step to do the search
            result = await self._collect_service_step(
                data={"service_name": query},
                context=context
            )
            
            # If multiple matches, return them all
            if result.get("multiple_matches"):
                services = result.get("data", {}).get("services", [])
                return FunctionResult(
                    success=True,
                    data={
                        "services": services,
                        "query": query,
                        "count": len(services),
                        "needs_user_selection": True,
                        "multiple_matches": True  # Put it in data dict, not as parameter
                    },
                    message=f"Found {len(services)} matching services",
                    function_name="search_services",
                    needs_wrapping=True
                )
            
            # If single match or error, return as is
            elif result.get("success"):
                # Single service found - return it
                # 🔥 CRITICAL: Pass service_id through so conversation_orchestrator can save it!
                service_data = result.get("data", {}).get("service", {})
                service_id = result.get("data", {}).get("service_id")  # Extract service_id from collect result
                service_name = result.get("data", {}).get("service_name")
                
                return FunctionResult(
                    success=True,
                    data={
                        "services": [service_data] if service_data else [],
                        "service_id": service_id,  # 🔥 PASS IT THROUGH!
                        "service_name": service_name,
                        "service": service_data,  # Also include service object for backward compatibility
                        "query": query,
                        "count": 1,
                        "auto_selected": True  # Flag that this was auto-selected
                    },
                    message="Found 1 service",
                    function_name="search_services",
                    needs_wrapping=True
                )
            else:
                # Search failed
                return FunctionResult(
                    success=False,
                    message=result.get("message", "No services found"),
                    function_name="search_services"
                )
                
        except Exception as e:
            logger.error(f"❌ Service search failed: {e}")
            return FunctionResult(
                success=False,
                message=f"Search error: {str(e)}",
                function_name="search_services"
            )
    
    async def _get_all_services(
        self,
        category: str,
        context: ConversationContext
    ) -> FunctionResult:
        """Get ALL services with full subcategory expansion - showing EVERYTHING"""
        
        logger.info(f"📚 Fetching ALL services with subcategories (category: {category or 'all'})")
        
        # Get categories (level 1)
        categories = await self.api_client.get_services(limit=100)
        
        # 🚨 CRITICAL: Get conversation history to determine user intent
        turn = context.turn
        history_summary = ""
        if turn > 1:
            # User asked before and is asking again
            history_summary = "user_asking_repeatedly"
            logger.info(f"🔄 REPEATED REQUEST: User asked {turn} times - they want MORE details!")
        
        # Filter categories by gender
        patient_gender = context.get_patient_gender()
        filtered_categories = []
        for cat in categories:
            cat_name = cat.get("name", "").lower()
            
            # Skip test services
            if "test" in cat_name:
                continue
            
            # Gender filtering
            if patient_gender == "male" and "نساء" in cat_name:
                continue
            if patient_gender == "female" and "رجال" in cat_name:
                continue
            
            # Category filtering if requested
            if category:
                if category.lower() not in cat_name and category.lower() not in cat.get("description", "").lower():
                    continue
            
            filtered_categories.append(cat)
        
        logger.info(f"📁 Found {len(filtered_categories)} categories (after filtering)")
        
        # 🚨 EXPAND TO SUBCATEGORIES: Fetch ALL subcategories from ALL categories
        all_subcategories = []
        for cat in filtered_categories:
            cat_id = cat.get("id")
            cat_name = cat.get("name")
            
            try:
                # Fetch subcategories for this category
                subservices = await self.api_client.get_services(service_type_id=cat_id, limit=100)
                logger.info(f"   📂 Category '{cat_name}': {len(subservices)} subcategories")
                
                # Add category info to each subcategory
                for sub in subservices:
                    sub["category_name"] = cat_name
                    sub["category_id"] = cat_id
                    all_subcategories.append(sub)
            
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch subcategories for {cat_name}: {e}")
                # Add category itself as fallback
                cat["category_name"] = cat_name
                cat["is_category"] = True
                all_subcategories.append(cat)
        
        logger.info(f"✅ Total subcategories: {len(all_subcategories)}")
        
        # Organize by category for better presentation
        services_by_category = {}
        for sub in all_subcategories:
            cat_name = sub.get("category_name", "أخرى")
            if cat_name not in services_by_category:
                services_by_category[cat_name] = []
            services_by_category[cat_name].append(sub)
        
        return FunctionResult(
            success=True,
            data={
                "services": all_subcategories,
                "services_by_category": services_by_category,
                "categories": [cat.get("name") for cat in filtered_categories],
                "category_filter": category or "all",
                "total_count": len(all_subcategories),
                "category_count": len(filtered_categories),
                "repeated_request": history_summary,  # Signal that user is asking again
                "patient_gender": patient_gender
            },
            message=f"Retrieved {len(all_subcategories)} services from {len(filtered_categories)} categories",
            function_name="get_all_services",
            needs_wrapping=True
        )
    
    async def _view_my_bookings(
        self,
        show_past: bool,
        context: ConversationContext
    ) -> FunctionResult:
        """
        🧠 SMART: View patient's bookings from API database.
        
        This provides REAL-TIME data, not cached session data.
        """
        
        logger.info(f"📅 Fetching bookings for patient (show_past={show_past})")
        
        # Get patient ID from context
        if not context.patient or not context.patient.id:
            logger.warning(f"⚠️ No patient in context - cannot fetch bookings")
            return FunctionResult(
                success=False,
                message="patient_not_found",
                data={"reason": "no_patient_in_context"},
                function_name="view_my_bookings",
                needs_wrapping=True
            )
        
        patient_id = context.patient.id
        patient_name = context.get_patient_name()
        
        try:
            # 🧠 SMART API CALL: Get patient's bookings from database
            from datetime import datetime, timedelta
            today = datetime.now().date()
            
            # Prepare filters
            params = {"patient_id": patient_id, "limit": 100}
            
            if not show_past:
                # Only upcoming bookings (from today onwards)
                params["date_from"] = today.isoformat()
                logger.info(f"📅 Fetching upcoming bookings from {today}")
            else:
                # Include past bookings (last 30 days + future)
                past_date = (today - timedelta(days=30)).isoformat()
                params["date_from"] = past_date
                logger.info(f"📅 Fetching all bookings from {past_date}")
            
            # Call API
            bookings_response = await self.api_client.get_bookings(**params)
            bookings = bookings_response.get("results", []) or bookings_response.get("data", [])
            
            logger.info(f"✅ Found {len(bookings)} bookings for patient {patient_id}")
            
            # Organize bookings
            upcoming = []
            past = []
            
            for booking in bookings:
                booking_date_str = booking.get("start_date") or booking.get("appointment_date")
                if booking_date_str:
                    try:
                        booking_date = datetime.fromisoformat(booking_date_str.replace("Z", "+00:00")).date()
                        if booking_date >= today:
                            upcoming.append(booking)
                        else:
                            past.append(booking)
                    except:
                        # If date parsing fails, assume upcoming
                        upcoming.append(booking)
                else:
                    upcoming.append(booking)
            
            logger.info(f"📊 Organized: {len(upcoming)} upcoming, {len(past)} past")
            
            return FunctionResult(
                success=True,
                data={
                    "bookings": bookings,
                    "upcoming": upcoming,
                    "past": past,
                    "total_count": len(bookings),
                    "upcoming_count": len(upcoming),
                    "past_count": len(past),
                    "patient_name": patient_name,
                    "show_past": show_past
                },
                message=f"Found {len(bookings)} bookings ({len(upcoming)} upcoming, {len(past)} past)",
                function_name="view_my_bookings",
                needs_wrapping=True
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to fetch bookings: {e}", exc_info=True)
            return FunctionResult(
                success=False,
                message="bookings_fetch_failed",
                data={"error": str(e), "patient_id": patient_id},
                function_name="view_my_bookings",
                needs_wrapping=True
            )
