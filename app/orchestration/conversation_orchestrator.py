"""
Conversation Orchestrator
=========================
Single entry point for all conversation logic.

This replaces the complex Router with a simpler coordinator that:
1. Loads context
2. Passes to Reem
3. Executes functions if Reem requests them
4. Returns result to Reem for natural wrapping
5. Saves context

NO intent classification, NO routing logic - just context management!
"""
from typing import Dict, Any, Optional
from loguru import logger
import time

from ..agents.reem_agent import ReemAgent
from ..workflows.workflow_executor import WorkflowExecutor
from ..core.dynamic_function_handler import DynamicFunctionHandler
from ..core.selection_handler import SelectionHandler
from ..models.conversation_context import ConversationContext, SessionMetrics
from ..models.agent_response import Message, ResponseType
from ..models.workflow_result import PatientInfo, WorkflowStatus
from ..memory.session_manager import SessionManager
from ..api.agent_api import AgentApiClient
from ..api.wasender_client import WaSenderClient


class ConversationOrchestrator:
    """
    Single orchestrator that delegates ALL conversations to Reem.
    
    Architecture:
        User Message
            ↓
        Load Context
            ↓
        Reem (decides everything)
            ↓
        Execute Function (if needed)
            ↓
        Reem Wraps Result
    """
    
    def __init__(self):
        self.reem = ReemAgent()
        self.function_handler = DynamicFunctionHandler()  # Enhanced with intelligent_booking_agent logic patterns
        self.session_manager = SessionManager()
        self.api_client = AgentApiClient()
        self.wasender = WaSenderClient()
        logger.info("✅ ConversationOrchestrator initialized with enhanced DynamicFunctionHandler")
    
    async def handle_message(
        self,
        message: str,
        session_id: str,
        user_phone: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main entry point for all messages.
        
        This is the ONLY method called from webhook handler!
        
        Args:
            message: User's message text
            session_id: Session identifier (e.g., "whatsapp:966123456789")
            user_phone: User's phone number
            metadata: Optional metadata (request_id, IP, etc.)
        
        Returns:
            Natural language response (ready to send to user)
        
        Flow:
            1. Load conversation context
            2. Add patient info if registered
            3. Call Reem
            4. Handle function calls (if any)
            5. Update context
            6. Return response
        """
        
        start_time = time.time()
        logger.info(f"🔵 [ORCHESTRATOR] Processing message from {user_phone}: '{message[:50]}...'")
        
        try:
            # 1. Load context
            context = await self._load_context(session_id, user_phone)
            
            # 2. Add patient info if registered
            if not context.is_registered():
                patient_info = await self._get_patient_info(user_phone)
                if patient_info:
                    context.patient = patient_info
                    logger.info(f"✅ Loaded registered patient: {patient_info.name} (ID: {patient_info.id})")
                    # 🔥 CRITICAL: Log patient_id for debugging slots API
                    logger.info(f"📋 Patient context ready: patient.id={patient_info.id}, patient.name={patient_info.name}")
            
            # 🔥 CRITICAL: Always log patient status for debugging
            if context.patient:
                logger.info(f"✅ Patient in context: ID={context.patient.id}, Name={context.patient.name}")
            else:
                logger.warning(f"⚠️ NO PATIENT in context - slots API will not have patient_id!")
            
            # 3. Add user message to history
            user_message = Message(role="user", content=message)
            context.add_message(user_message)
            
            # 3.5. Check if we're in middle of a workflow
            in_workflow = context.metadata.get("in_booking_flow", False)
            workflow_pending = context.metadata.get("workflow_pending", False)
            
            # Detect if user wants to exit workflow
            exit_signals = ["لا", "خلاص", "مو الحين", "بعدين", "ألغي", "cancel", "stop"]
            wants_to_exit = any(signal in message.lower() for signal in exit_signals)
            
            if in_workflow and wants_to_exit:
                logger.info(f"🚪 User wants to exit workflow - returning control to Reem")
                context.metadata["in_booking_flow"] = False
                context.metadata["workflow_pending"] = False
                # Reem will handle the exit gracefully
            
            # Detect if user is asking questions mid-workflow (needs Reem, not LangGraph)
            question_signals = ["وش", "كم", "ليش", "كيف", "متى", "وين", "what", "how", "why", "when"]
            is_question = any(signal in message.lower() for signal in question_signals)
            
            if in_workflow and workflow_pending and is_question:
                logger.info(f"❓ User asking question mid-workflow - Reem will handle, then resume workflow")
                # Let Reem answer the question, but keep workflow active
            
            # 3.6. Check if user is making a numbered selection (following intelligent_booking_agent pattern)
            if SelectionHandler.is_number_selection(message):
                selection_number = int(message.strip())
                logger.info(f"🔢 Detected number selection: {selection_number}")
                
                # Handle the selection
                selection_result = await SelectionHandler.handle_selection(
                    number=selection_number,
                    context=context
                )
                
                if selection_result["success"]:
                    # Selection successful - proceed with next step
                    logger.info(f"✅ Selection handled: {selection_result.get('selection_type')}")
                    
                    # Determine next action based on selection type
                    next_step = selection_result.get("next_step")
                    
                    if next_step == "check_availability":
                        # User selected a service - now check availability
                        logger.info(f"➡️ Next step: check_availability for service_id={selection_result['data'].get('service_id')}")
                        
                        # Simulate calling check_availability step
                        final_response = f"ممتاز عزيزي! اخترت {selection_result['data'].get('service_name')} 👍\n\n"
                        final_response += f"💰 السعر: {selection_result['data'].get('service_price')} ريال\n"
                        final_response += f"⏱️ مدة الجلسة: {selection_result['data'].get('duration_minutes')} دقيقة\n\n"
                        final_response += "متى تبي الموعد؟ (مثلاً: باكر، يوم السبت، 10 نوفمبر)"
                    
                    elif next_step == "confirm_booking":
                        # User selected a slot - confirm booking
                        final_response = f"تمام! اخترت موعد يوم {selection_result['data'].get('slot_date')} الساعة {selection_result['data'].get('slot_time')} ✅\n\n"
                        final_response += "تأكيد الحجز؟"
                    
                    else:
                        # Default response
                        final_response = f"تمام! {selection_result.get('message')} ✅"
                else:
                    # Selection failed - inform user
                    final_response = selection_result.get("message", "معليش! فيه مشكلة في الاختيار.")
                
                # Add assistant message to history
                assistant_message = Message(role="assistant", content=final_response)
                context.add_message(assistant_message)
                
                # Update context
                context.turn += 1
                await self._save_context(context)
                
                # Update metrics
                elapsed = time.time() - start_time
                await self._update_metrics(session_id, elapsed, {})
                
                logger.info(f"✅ [ORCHESTRATOR] Selection response generated in {elapsed:.2f}s: {len(final_response)} chars")
                
                return final_response
            
            # 4. Call Reem (she decides everything!)
            reem_response = await self.reem.chat(
                message=message,
                context=context
            )
            
            # 5. Handle function calls (if Reem needs technical action)
            if reem_response.has_function_call():
                function_name = reem_response.function_call.name
                logger.info(f"🔧 Reem requested function: {function_name}")
                
                # Check if this is a workflow that needs continuous interaction
                if function_name == "execute_workflow":
                    workflow_name = reem_response.function_call.arguments.get("workflow")
                    
                    # For booking workflow, check if it needs multi-turn interaction
                    if workflow_name == "booking":
                        logger.info(f"📋 Starting booking workflow - may require multiple turns")
                        context.metadata["in_booking_flow"] = True
                        context.metadata["booking_initiated_at"] = time.time()
                
                final_response = await self._handle_function_call(
                    reem_response.function_call,
                    context,
                    message
                )
                
                # Check if workflow is still pending (needs more user input)
                if context.metadata.get("workflow_pending"):
                    logger.info(f"⏳ Workflow pending - will continue on next message")
            else:
                # Direct response from Reem
                final_response = reem_response.content
                
                # Clear workflow flags if Reem responded directly
                if context.metadata.get("in_booking_flow"):
                    logger.info(f"💬 Reem handling conversation directly - not using workflow yet")
                    # Don't clear the flag - user might still want to book later
            
            # 6. Add assistant message to history
            assistant_message = Message(role="assistant", content=final_response)
            context.add_message(assistant_message)
            
            # 6.5. Track discussed services for context awareness
            self._track_discussed_services(final_response, context)
            
            # 7. Update context with any changes from Reem
            context.update_from_dict(reem_response.context_updates)
            context.turn += 1
            
            # 8. Save context
            await self._save_context(context)
            
            # 9. Update metrics
            elapsed = time.time() - start_time
            await self._update_metrics(session_id, elapsed, reem_response.metadata)
            
            logger.info(f"✅ [ORCHESTRATOR] Response generated in {elapsed:.2f}s: {len(final_response)} chars")
            
            return final_response
        
        except Exception as e:
            logger.error(f"❌ [ORCHESTRATOR] Error: {e}", exc_info=True)
            
            # Friendly error message
            return "آسف عزيزي! حصل خطأ تقني بسيط 😅\nممكن تعيد رسالتك مرة ثانية؟"
    
    async def _handle_function_call(
        self,
        function_call,
        context: ConversationContext,
        original_message: str
    ) -> str:
        """
        Execute function call and return natural language response.
        
        NEW APPROACH: Using DynamicFunctionHandler for granular control!
        
        Flow:
            1. Call DynamicFunctionHandler.execute()
            2. Update context with any state changes
            3. If result needs wrapping, let Reem wrap it
            4. Return natural response
        """
        
        function_name = function_call.name
        arguments = function_call.arguments
        
        logger.info(f"🔧 Executing function: {function_name} with args: {arguments}")
        logger.debug(f"   Current booking state: {context.booking_state.status}")
        
        try:
            # 🚨 CRITICAL: Prevent hallucinated service_id calls
            if function_name in ["get_service_details", "get_pricing"]:
                logger.error(f"🚨 HALLUCINATION DETECTED: LLM called {function_name} - these functions are deprecated!")
                logger.error(f"   Arguments: {arguments}")
                logger.error(f"   Redirecting to search_services instead...")
                
                # Extract service name from context or return error
                return f"""معذرة! دعني أبحث لك عن هذه الخدمة بطريقة صحيحة.
                
ممكن تعيد طلبك؟ مثلاً: "عطني تفاصيل عن ليزر كربوني" أو "كم سعر البوتوكس؟" """
            
            # Route to DynamicFunctionHandler for advanced functions
            advanced_functions = [
                "execute_booking_step",
                "pause_booking",
                "resume_booking", 
                "cancel_booking",
                # "get_service_details",  # ❌ DEPRECATED - causes hallucination
                # "get_pricing",           # ❌ DEPRECATED - causes hallucination
                "check_availability",
                "search_services",
                "get_all_services",
                "view_my_bookings"  # 📅 View patient's bookings
            ]
            
            if function_name in advanced_functions:
                logger.info(f"🎯 Routing to DynamicFunctionHandler: {function_name}")
                
                # Execute via DynamicFunctionHandler
                # SMARTNESS: Pass original message for date extraction and context awareness
                if "data" not in arguments:
                    arguments["data"] = {}
                if isinstance(arguments.get("data"), dict):
                    arguments["data"]["user_message"] = original_message
                
                function_result = await self.function_handler.execute(
                    function_name=function_name,
                    arguments=arguments,
                    context=context
                )
                
                # Update context with new booking state if changed
                if function_result.booking_state:
                    context.booking_state = function_result.booking_state
                    logger.info(f"📊 Booking state updated: {context.booking_state.status}")
                
                # 🔥 CRITICAL FIX: If search_services auto-selected a service, save to booking_state!
                # This prevents LLM hallucination when user provides date next
                if function_name == "search_services" and function_result.success:
                    # Check if this was an auto-selection (single match)
                    if function_result.data and "service_id" in function_result.data and function_result.data.get("auto_selected"):
                        # Service was auto-selected - save it!
                        service_id = function_result.data["service_id"]
                        service_name = function_result.data.get("service_name", "")
                        service_price = function_result.data.get("service", {}).get("price", 0)
                        
                        context.booking_state.collected_data["service_id"] = service_id
                        context.booking_state.collected_data["service_name"] = service_name
                        context.booking_state.collected_data["service_price"] = service_price
                        context.booking_state.status = "active"  # Start booking flow
                        context.booking_state.progress["collect_service"] = True  # Mark step complete
                        
                        logger.info(f"💾 AUTO-SAVE: service_id={service_id} ({service_name}) saved to booking_state")
                        logger.info(f"📊 Booking state: status=active, service_id={service_id}, price={service_price}")
                        logger.info(f"✅ This prevents LLM hallucination - service_id now in booking_state for next turn!")
                
                # Check if result needs wrapping by Reem
                if function_result.needs_wrapping:
                    if function_result.success:
                        # Convert to WorkflowResult format for Reem's wrap_result method
                        from ..models.workflow_result import WorkflowResult
                        workflow_result = WorkflowResult(
                            success=function_result.success,
                            status=WorkflowStatus.SUCCESS,
                            data=function_result.data,
                            metadata={"function": function_name}
                        )
                        
                        wrapped_response = await self.reem.wrap_result(
                            workflow_result=workflow_result,
                            context=context,
                            original_message=original_message
                        )
                        
                        # 🚨 CRITICAL: Save booking status to context.metadata for smart follow-ups
                        if function_result.data:
                            status = function_result.data.get("status")
                            message = function_result.data.get("message")
                            if status:
                                context.metadata["last_status"] = status
                                logger.info(f"💾 Saved last_status to context: {status}")
                            if message:
                                context.metadata["last_message"] = message
                        
                        return wrapped_response
                    else:
                        # CRITICAL: Function failed - wrap error in human-friendly language!
                        error_code = function_result.message or "unknown_error"
                        error_data = function_result.data
                        
                        logger.warning(f"⚠️ Function failed with code: {error_code}")
                        
                        wrapped_error = await self.reem.wrap_error(
                            error_code=error_code,
                            context=context,
                            error_data=error_data
                        )
                        
                        return wrapped_error
                else:
                    # Function doesn't need wrapping (already has response)
                    return function_result.message or "تمام!"
            
            # Handle legacy workflow execution (backward compatibility)
            elif function_name == "execute_workflow":
                workflow_name = arguments.get("workflow")
                params = arguments.get("params", {})
                
                # Add phone to context for workflows that need it
                workflow_context = {
                    "phone": context.phone_number,
                    "session_id": context.session_id
                }
                
                # Execute workflow
                workflow_result = await self.workflow_executor.execute(
                    workflow_name=workflow_name,
                    params=params,
                    context=workflow_context
                )
                
                # Update context if workflow succeeded
                if workflow_result.success:
                    # Update patient info if registration succeeded
                    if workflow_name == "registration" and workflow_result.data.get("patient_id"):
                        context.patient = PatientInfo(
                            id=workflow_result.data["patient_id"],
                            name=workflow_result.data["name"],
                            phone=workflow_result.data["phone"],
                            national_id=workflow_result.data.get("national_id"),
                            gender=workflow_result.data.get("gender"),
                            city=workflow_result.data.get("city"),
                            country_code=workflow_result.data.get("country_code"),
                            already_registered=True
                        )
                        logger.info(f"✅ Patient registered: {context.patient.name} (ID: {context.patient.id})")
                    
                    # Clear workflow pending flag if completed
                    context.metadata["workflow_pending"] = False
                    if workflow_name == "booking":
                        context.metadata["in_booking_flow"] = False
                        logger.info(f"✅ Booking workflow completed successfully")
                
                # Mark workflow as pending if not complete
                elif workflow_result.status == WorkflowStatus.PENDING:
                    context.metadata["workflow_pending"] = True
                    context.metadata["workflow_next_step"] = workflow_result.next_step
                    logger.info(f"⏳ Workflow pending: {workflow_result.next_step}")
                
                # Let Reem wrap the result naturally
                wrapped_response = await self.reem.wrap_result(
                    workflow_result=workflow_result,
                    context=context,
                    original_message=original_message
                )
                
                return wrapped_response
            
            # Legacy: get_service_info (old name, redirect to get_service_details)
            elif function_name == "get_service_info":
                logger.info(f"🔄 Redirecting legacy 'get_service_info' to 'get_service_details'")
                # Redirect to new handler
                function_result = await self.function_handler.execute(
                    function_name="get_service_details",
                    arguments=arguments,
                    context=context
                )
                
                if function_result.needs_wrapping and function_result.success:
                    from ..models.workflow_result import WorkflowResult
                    workflow_result = WorkflowResult(
                        success=function_result.success,
                        status=WorkflowStatus.SUCCESS,
                        data=function_result.data,
                        metadata={"function": "get_service_info"}
                    )
                    
                    wrapped_response = await self.reem.wrap_result(
                        workflow_result=workflow_result,
                        context=context,
                        original_message=original_message
                    )
                    
                    return wrapped_response
                else:
                    return function_result.message or "تمام!"
            
            else:
                logger.error(f"❌ Unknown function: {function_name}")
                # Intelligent fallback for unknown functions
                return """معليش! ما قدرت أنفذ الطلب بالضبط 😅

بس أقدر أساعدك بطريقة ثانية:
• تبي تعرف عن خدماتنا؟
• تبي تحجز موعد؟
• ولا تفضل تكلمنا مباشرة؟ 📞 920033304

قول لي وش تحتاج وأنا هنا! 🌟"""
        
        except Exception as e:
            logger.error(f"❌ Function execution failed: {e}", exc_info=True)
            
            # Intelligent fallback based on function type
            if "service" in function_name.lower():
                return """معليش! النظام بطيء شوي 😅

بس أقدر أقول لك عن خدماتنا المميزة:
• البوتوكس - من 750 ريال
• الفيلر - من 800 ريال
• ليزر إزالة الشعر
• تنظيف البشرة
• ميزوثيرابي

أي وحدة تهمك؟ أو تبي رقمنا: 📞 920033304"""
            
            elif "booking" in function_name.lower() or "availability" in function_name.lower():
                return """معليش! حصلت مشكلة في فحص المواعيد 😅

تبي أكلمك مباشرة لتأكيد الموعد؟
أو تفضل تكلم فريقنا: 📞 920033304

احنا متواجدين 10ص-10م (السبت-الخميس)"""
            
            else:
                return """معليش! حصلت مشكلة بسيطة 😅

بس ما تزعل، أقدر أساعدك:
• تبي تعرف عن خدماتنا؟
• تبي تحجز موعد؟
• ولا تكلمنا مباشرة؟ 📞 920033304

أنا هنا أساعدك! 🌟"""
    
    async def _load_context(
        self,
        session_id: str,
        user_phone: str
    ) -> ConversationContext:
        """
        Load conversation context from session storage.
        
        If no context exists, creates a new one.
        """
        
        # Try to load from Redis
        session_data = await self.session_manager.get_session(session_id)
        
        if session_data and session_data.get("context"):
            # Restore context from saved data
            try:
                context = ConversationContext.from_dict(session_data["context"])
                logger.info(f"✅ Loaded context: Turn {context.turn}, History: {len(context.conversation_history)} msgs")
                return context
            except Exception as e:
                logger.warning(f"⚠️ Failed to restore context: {e}")
        
        # Create new context
        logger.info(f"🆕 Creating new context for {session_id}")
        context = ConversationContext(
            session_id=session_id,
            phone_number=user_phone,
            turn=0,
            language="arabic"
        )
        
        return context
    
    async def _save_context(self, context: ConversationContext):
        """
        Save conversation context to session storage.
        """
        
        try:
            session_data = await self.session_manager.get_session(context.session_id) or {}
            session_data["context"] = context.to_dict()
            session_data["last_updated"] = time.time()
            
            await self.session_manager.put_session(context.session_id, session_data)
            
            logger.debug(f"💾 Context saved: {context.session_id}")
        
        except Exception as e:
            logger.error(f"❌ Failed to save context: {e}")
    
    async def _get_patient_info(self, phone: str) -> Optional[PatientInfo]:
        """
        Get patient info from database if registered.
        
        Tries multiple phone formats (with/without country code).
        """
        
        try:
            logger.info(f"🔍 Looking up patient for phone: {phone}")
            
            # Try to find patient by phone
            patient = await self.api_client.search_patient(phone)
            
            if patient:
                patient_id = patient.get("id")
                patient_name = patient.get("name")
                
                logger.info(f"✅ Patient found in database: ID={patient_id}, Name={patient_name}")
                
                # Extract first name for personalization
                first_name = patient_name.split()[0] if patient_name else None
                
                patient_info = PatientInfo(
                    id=patient_id,
                    name=patient_name,
                    phone=patient.get("patient_phone") or patient.get("phone"),
                    national_id=patient.get("identification_id") or patient.get("national_id"),
                    gender=patient.get("gender"),
                    email=patient.get("email"),
                    city=patient.get("city"),
                    country_code=patient.get("country_code"),
                    already_registered=True
                )
                
                logger.info(f"📋 Patient info created: first_name={first_name}")
                return patient_info
            else:
                logger.warning(f"❌ Patient NOT found in database for phone: {phone}")
                return None
        
        except Exception as e:
            logger.error(f"⚠️ Patient lookup error for {phone}: {e}")
            return None
    
    def _track_discussed_services(
        self,
        response: str,
        context: ConversationContext
    ):
        """
        Track which services were discussed to maintain context awareness.
        
        This is CRITICAL for handling follow-up questions like "give me details"
        or "how much does it cost?" - we need to know what "it" refers to!
        """
        
        # Service keywords to track (Arabic names)
        # Order matters: Check longer/specific terms first to avoid false matches
        service_keywords = {
            "فل بدي": ("فل بدي", "فول بدي", "full body", "fullbody"),
            "ميزوثيرابي": ("ميزوثيرابي", "ميزو", "mesotherapy", "meso"),
            "بوتوكس": ("بوتوكس", "botox"),
            "فيلر": ("فيلر", "filler", "fillers"),
            "ليزر": ("ليزر", "laser"),
            "تنظيف": ("تنظيف", "cleansing", "تنضيف"),
            "خيوط": ("خيوط", "thread", "threads"),
            "تقشير": ("تقشير", "peel", "peeling"),
            "بلازما": ("بلازما", "plasma", "بلازم"),
            "ديرما": ("ديرما", "derma", "ديرمابن"),
        }
        
        response_lower = response.lower()
        
        # Check which services were mentioned
        # Check in order (longer terms first to avoid substring issues)
        for service_name, keywords in service_keywords.items():
            if any(kw.lower() in response_lower for kw in keywords):
                # Update last discussed service
                context.last_discussed_service = service_name
                
                # Add to conversation topics if not already there
                if service_name not in context.conversation_topics:
                    context.conversation_topics.append(service_name)
                
                logger.info(f"💬 Tracked discussed service: {service_name}")
                break  # Only track the first/main service mentioned
    
    async def _update_metrics(
        self,
        session_id: str,
        elapsed_time: float,
        response_metadata: Dict[str, Any]
    ):
        """
        Update session metrics for monitoring.
        """
        
        try:
            # This could save to database or metrics service
            # For now, just log
            tokens = response_metadata.get("tokens", 0)
            logger.debug(f"📊 Metrics: {session_id} | Time: {elapsed_time:.2f}s | Tokens: {tokens}")
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to update metrics: {e}")
