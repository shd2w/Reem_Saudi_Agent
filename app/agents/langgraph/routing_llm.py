"""
LLM-Driven Routing - Single Brain Architecture

Instead of rigid if/else rules, LLM makes ALL routing decisions based on conversation context.
This creates a single intelligent brain that understands the full conversation flow.
"""
from typing import Optional
from loguru import logger
from .booking_state import BookingState
from langgraph.graph import END


class LLMRouter:
    """
    LLM-driven router that replaces rigid rules with intelligent decisions.
    
    The LLM acts as the single "brain" that:
    1. Reads the full conversation context
    2. Understands the current state
    3. Decides the next logical step
    4. Routes accordingly
    
    No more hard-coded if/else logic!
    """
    
    def __init__(self, llm_reasoner):
        """
        Initialize LLM router with reasoning capability.
        
        Args:
            llm_reasoner: LLMReasoner instance for making intelligent decisions
        """
        self.llm = llm_reasoner
        logger.info("🧠 LLMRouter initialized - Single brain architecture enabled")
    
    async def route_patient_flow(self, state: BookingState) -> str:
        """
        LLM decides: What should happen after patient verification?
        
        Instead of:
            if step == "patient_verified":
                if selected_service:
                    return "fetch_resources"  # Rigid!
                    
        LLM considers:
        - Has user already mentioned a service?
        - Does user seem ready to proceed?
        - What would a human receptionist do?
        """
        step = state.get("step", "")
        
        # Quick checks for obvious states (optimization)
        if step == "needs_registration":
            logger.info("🧠 [LLM_ROUTER] Obvious case: needs_registration → confirm_registration")
            return "confirm_registration"
        
        if step == "patient_verification_error":
            logger.info("🧠 [LLM_ROUTER] Obvious case: error → handle_error")
            return "handle_error"
        
        # For patient_verified, let LLM decide the best path
        if step == "patient_verified":
            context = self._build_patient_context(state)
            
            available_steps = [
                "fetch_service_types - Ask user which service category they want",
                "fetch_resources - User already selected service, get doctors/devices",
                "await_user_input - Need more information from user"
            ]
            
            decision = await self._ask_llm_decision(
                context=context,
                available_steps=available_steps,
                question="Patient is verified. What should we do next?"
            )
            
            logger.info(f"🧠 [LLM_ROUTER:patient] {step} → {decision} (intelligent decision)")
            return decision
        
        # Default
        logger.info(f"🧠 [LLM_ROUTER:patient] {step} → await_user_input (default)")
        return "await_user_input"
    
    async def route_service_flow(self, state: BookingState) -> str:
        """
        LLM decides: What should happen during service selection?
        
        Considers:
        - Did user select a specific service?
        - Is user browsing options?
        - Does user need help?
        """
        step = state.get("step", "")
        
        # Obvious cases
        if step == "awaiting_service_type":
            return "await_user_input"
        
        if step == "awaiting_service":
            return "await_user_input"
        
        if step.endswith("_error"):
            return "handle_error"
        
        # LLM decides for selection cases
        if step in ["service_type_selected", "service_selected"]:
            context = self._build_service_context(state)
            
            available_steps = [
                "fetch_services - Show service variants/options",
                "fetch_resources - Service selected, get doctors/specialists",
                "await_user_input - Wait for user to choose"
            ]
            
            decision = await self._ask_llm_decision(
                context=context,
                available_steps=available_steps,
                question="User is selecting service. What's next?"
            )
            
            logger.info(f"🧠 [LLM_ROUTER:service] {step} → {decision}")
            return decision
        
        return "await_user_input"
    
    async def route_confirmation_flow(self, state: BookingState) -> str:
        """
        LLM decides: Did user confirm? Should we create the booking?
        
        Instead of checking keywords like "نعم", LLM understands:
        - "يلا" = yes
        - "بينا" = yes
        - "ماشي" = yes
        - "لا شكراً" = no
        - Context matters!
        """
        step = state.get("step", "")
        
        if step == "awaiting_confirmation":
            last_message = state.get("current_message", "")
            
            context = f"""
Recent conversation:
{self._format_history(state, last_n=3)}

Current situation:
- Bot asked user to confirm booking
- User responded: "{last_message}"

Available actions:
- create_booking - User confirmed (yes, يلا, تمام, etc.)
- cancel_booking - User declined (no, cancel, لا, etc.)
- await_user_input - Not clear, need clarification

Question: What action should we take based on user's response?
"""
            
            decision = await self._ask_llm_simple(
                context=context,
                valid_options=["create_booking", "cancel_booking", "await_user_input"]
            )
            
            logger.info(f"🧠 [LLM_ROUTER:confirmation] '{last_message}' → {decision}")
            return decision
        
        elif step == "booking_created":
            return "send_confirmation"
        elif step == "completed":
            return END
        elif step.endswith("_error"):
            return "handle_error"
        else:
            return "await_user_input"
    
    async def route_next_step(self, state: BookingState) -> str:
        """
        General routing - LLM looks at current step and decides next action.
        
        This is the main router that handles any state.
        """
        step = state.get("step", "")
        is_resuming = state.get("_resuming", False)
        
        # CRITICAL FIX: If resuming, don't route to verify_patient (creates loop!)
        # When resuming, the state is already set, we should just continue
        if is_resuming and step == "start":
            logger.info(f"🧠 [LLM_ROUTER:general] Resuming session with step=start → fetch_service_types")
            return "fetch_service_types"
        
        # CRITICAL FIX: Handle awaiting_registration_confirmation
        # If patient_data exists and already_registered, skip to patient_verified
        # Otherwise, stay at confirm_registration to re-ask
        if step == "awaiting_registration_confirmation":
            patient_data = state.get("patient_data")
            if patient_data and patient_data.get("already_registered"):
                logger.info(f"🧠 [LLM_ROUTER:general] {step} + patient registered → confirm_registration (will skip to verified)")
                return "confirm_registration"
            else:
                # User needs to respond to confirmation question
                router_intent = state.get("router_intent", "").lower()
                if router_intent in ["confirmation", "registration"]:
                    logger.info(f"🧠 [LLM_ROUTER:general] {step} + user confirmed → start_registration")
                    return "start_registration"
                else:
                    logger.info(f"🧠 [LLM_ROUTER:general] {step} → await_user_input (waiting for response)")
                    return "await_user_input"
        
        # Let LLM handle complex cases
        if step in ["start", "awaiting_service_selection"]:
            context = self._build_general_context(state)
            
            available_steps = [
                "verify_patient - Check patient info/registration",
                "fetch_service_types - Ask what service they want",
                "await_user_input - Wait for user response"
            ]
            
            decision = await self._ask_llm_decision(
                context=context,
                available_steps=available_steps,
                question="What should we do next in the booking flow?"
            )
            
            logger.info(f"🧠 [LLM_ROUTER:general] {step} → {decision}")
            return decision
        
        # Handle loop detection
        if step == "loop_detected":
            logger.info(f"🧠 [LLM_ROUTER:general] {step} → handle_loop")
            return "handle_loop"
        
        # Handle error states
        if step.startswith("error_") or step.endswith("_error"):
            logger.info(f"🧠 [LLM_ROUTER:general] {step} → handle_error")
            return "handle_error"
        
        # Default: Wait for user input (CRITICAL FIX: Don't route back to process_user_input!)
        # process_user_input node cannot route to itself - it's not in the edges list!
        logger.info(f"🧠 [LLM_ROUTER:general] {step} → await_user_input (default)")
        return "await_user_input"
    
    # Helper methods
    
    def _build_patient_context(self, state: BookingState) -> str:
        """Build context for patient flow decisions"""
        return f"""
Recent conversation:
{self._format_history(state, last_n=5)}

Current state:
- Step: {state.get('step')}
- Patient: {state.get('name') or 'Unknown'} (verified: {state.get('patient_verified')})
- Service mentioned: {state.get('selected_service_name') or state.get('last_discussed_service') or 'None'}
- User's last message: "{state.get('current_message', '')}"

Question: Patient is verified. What's the most natural next step?
"""
    
    def _build_service_context(self, state: BookingState) -> str:
        """Build context for service flow decisions"""
        return f"""
Recent conversation:
{self._format_history(state, last_n=5)}

Current state:
- Step: {state.get('step')}
- Service type selected: {state.get('selected_service_type_name') or 'None'}
- Service selected: {state.get('selected_service_name') or 'None'}
- Available services: {len(state.get('services', []))} options
- User's last message: "{state.get('current_message', '')}"

Question: User is selecting a service. What should happen next?
"""
    
    def _build_general_context(self, state: BookingState) -> str:
        """Build general context for any decision"""
        return f"""
Recent conversation:
{self._format_history(state, last_n=5)}

Current state:
- Step: {state.get('step')}
- Patient: {state.get('name') or 'Unknown'} (verified: {state.get('patient_verified')})
- Service: {state.get('selected_service_name') or 'Not selected'}
- Doctor: {state.get('doctor_name') or 'Not selected'}
- User's last message: "{state.get('current_message', '')}"

Context: We're in a clinic booking conversation.
"""
    
    def _format_history(self, state: BookingState, last_n: int = 5) -> str:
        """Format conversation history for LLM"""
        history = state.get("conversation_history", [])
        
        if not history:
            return "No previous conversation"
        
        recent = history[-last_n:] if len(history) > last_n else history
        
        formatted = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Bot"
            content = msg.get("content", "")[:100]  # Truncate long messages
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    async def _ask_llm_decision(
        self,
        context: str,
        available_steps: list,
        question: str
    ) -> str:
        """
        Ask LLM to make a routing decision.
        
        Returns the step name (e.g., "fetch_resources")
        """
        steps_text = "\n".join(f"- {s}" for s in available_steps)
        
        prompt = f"""
{context}

Available next steps:
{steps_text}

{question}

CRITICAL INSTRUCTIONS:
- Respond with ONLY the step name (e.g., "fetch_resources")
- No explanation, no punctuation, no extra text
- Choose the most natural step a human receptionist would take
- Consider the conversation flow and user's intent
- Be smart about what the user needs next

Response format: just the step name
"""
        
        try:
            # Call LLM
            response = await self.llm.chat_completion([
                {
                    "role": "system",
                    "content": "You are a routing assistant. Make intelligent decisions based on conversation context. Respond with only the step name."
                },
                {"role": "user", "content": prompt}
            ])
            
            # FIX: chat_completion returns dict {"content": "...", "function_call": None}
            if isinstance(response, dict):
                response = response.get("content", "")
            
            # Clean response
            step = str(response).strip().lower()
            
            # Remove common prefixes
            for prefix in ["next step:", "step:", "action:", "response:"]:
                if step.startswith(prefix):
                    step = step[len(prefix):].strip()
            
            # Remove quotes
            step = step.replace('"', '').replace("'", "")
            
            # Validate it's one of the available steps
            step_names = [s.split(" - ")[0].strip() for s in available_steps]
            
            if step not in step_names:
                logger.warning(f"⚠️ LLM returned invalid step '{step}', using first available: {step_names[0]}")
                step = step_names[0]
            
            return step
            
        except Exception as e:
            logger.error(f"❌ LLM routing decision failed: {e}")
            # Fallback to first available step
            fallback = available_steps[0].split(" - ")[0].strip()
            logger.warning(f"⚠️ Using fallback step: {fallback}")
            return fallback
    
    async def _ask_llm_simple(
        self,
        context: str,
        valid_options: list
    ) -> str:
        """Simpler LLM decision for binary/simple choices"""
        
        options_text = ", ".join(valid_options)
        
        prompt = f"""
{context}

Valid options: {options_text}

Respond with ONLY one of these options (exactly as written).
"""
        
        try:
            response = await self.llm.chat_completion([
                {
                    "role": "system",
                    "content": "You make simple routing decisions. Respond with only the option name."
                },
                {"role": "user", "content": prompt}
            ])
            
            # FIX: chat_completion returns dict {"content": "...", "function_call": None}
            if isinstance(response, dict):
                response = response.get("content", "")
            
            step = str(response).strip().lower()
            
            # Clean and validate
            for option in valid_options:
                if option.lower() in step or step in option.lower():
                    return option
            
            # Fallback
            logger.warning(f"⚠️ LLM returned '{step}', using fallback: {valid_options[0]}")
            return valid_options[0]
            
        except Exception as e:
            logger.error(f"❌ LLM simple decision failed: {e}")
            return valid_options[0]


# Singleton instance
_llm_router: Optional[LLMRouter] = None


def get_llm_router(llm_reasoner) -> LLMRouter:
    """Get or create LLM router singleton"""
    global _llm_router
    
    if _llm_router is None:
        _llm_router = LLMRouter(llm_reasoner)
    
    return _llm_router
