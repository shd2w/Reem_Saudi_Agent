"""
Patient verification and registration nodes.

Extracted from BookingAgent._verify_patient_registration()
"""
from loguru import logger
from typing import Optional
from ..booking_state import BookingState
from ....utils.phone_parser import extract_saudi_phone  # Issue #45: phone fallback


async def verify_patient_node(
    state: BookingState,
    api_client,
    session_manager
) -> BookingState:
    """
    Verify if patient exists in the system, or start registration.
    
    CRITICAL FIX (Issue #28): Check if service was discussed FIRST
    - If no service selected/discussed → ask for service first
    - If service known → proceed with patient verification
    """
    
    # CRITICAL: Skip verification if resuming from saved state
    # Otherwise we'll overwrite the step and break the flow!
    if state.get("_resuming"):
        logger.info(f"⏭️ [NODE:verify_patient] Skipping verification - resuming from saved state (step={state.get('step')})")
        return state
    
    # ═══════════════════════════════════════════════════════════════
    # ISSUE #28 FIX: Service-First Flow
    # ═══════════════════════════════════════════════════════════════
    
    # 🚨 CRITICAL: Check patient registration status FIRST
    patient_data = state.get("patient_data", {})
    is_registered = patient_data.get("already_registered", False) or state.get("patient_verified", False)
    
    # Check if user has selected or discussed a service
    selected_service = state.get("selected_service_id")
    selected_service_name = state.get("selected_service_name")  # From Router state sync
    discussed_service = state.get("last_discussed_service")
    
    # 🚨 REGISTRATION-FIRST FLOW: If patient NOT registered, ask for registration FIRST
    # Service-first flow ONLY applies to registered patients
    if not is_registered:
        logger.info(f"🚨 [NODE:verify_patient] Patient NOT registered - proceeding to registration (skipping service-first flow)")
        # Fall through to registration logic below
    elif not selected_service and not selected_service_name and not discussed_service:
        # CRITICAL: Check conversation history FIRST - maybe service was already discussed!
        conversation_history = state.get("conversation_history", [])
        service_mentioned_in_history = None
        
        if conversation_history:
            # Check last few messages for service mentions
            service_keywords = ["بوتوكس", "ليزر", "فيلر", "تنظيف", "ميزو", "خيوط", "تقشير", "botox", "laser", "filler"]
            for msg in reversed(conversation_history[-5:]):  # Check last 5 messages
                msg_content = msg.get("content", "").lower()
                for keyword in service_keywords:
                    if keyword in msg_content:
                        service_mentioned_in_history = keyword
                        logger.info(f"✅ [NODE:verify_patient] Found service in history: '{keyword}' - using as context")
                        break
                if service_mentioned_in_history:
                    break
        
        if service_mentioned_in_history:
            # Service WAS discussed - save it and continue
            logger.info(f"🎯 [NODE:verify_patient] Service '{service_mentioned_in_history}' found in conversation history - continuing with patient verification")
            state["last_discussed_service"] = service_mentioned_in_history
        else:
            # No service discussed yet - ask for service FIRST before registration
            logger.warning(
                f"⚠️ [NODE:verify_patient] BLOCKED: No service in history! "
                f"Asking for service before patient verification (Issue #28)"
            )
            
            sender_name = state.get("arabic_name") or state.get("sender_name") or "حبيبنا"
            
            state["step"] = "awaiting_service_selection"
            
            # CRITICAL FIX (Bug 25): Fetch REAL services from API, not hardcoded list!
            from app.api.agent_api import AgentApiClient
            from app.cache.service_cache import service_cache
            
            api_client = AgentApiClient()
            
            try:
                # Fetch actual service types from API
                services_list = await service_cache.get_services(api_client)
                
                if services_list:
                    # Extract service names for response
                    service_names = []
                    for svc in services_list[:10]:  # Limit to first 10
                        name = svc.get("name_ar") or svc.get("name") or svc.get("nameAr")
                        if name:
                            service_names.append(name)
                    
                    logger.info(f"✅ [NODE:verify_patient] Fetched {len(service_names)} services from API")
                else:
                    # Fallback to common services
                    service_names = ["ليزر", "فيلر", "بوتكس", "تنظيف البشرة"]
                    logger.warning(f"⚠️ [NODE:verify_patient] API returned no services, using fallback")
            except Exception as e:
                logger.error(f"❌ [NODE:verify_patient] Failed to fetch services: {e}")
                service_names = ["ليزر", "فيلر", "بوتكس", "تنظيف البشرة"]
            
            # Generate response with REAL service names
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            
            response = await response_gen.ask_for_service(
                user_name=sender_name,
                services=service_names  # ✅ Pass real services!
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            return state
    
    # Service was discussed - proceed with normal patient verification
    logger.info(
        f"✅ [NODE:verify_patient] Service context exists "
        f"(selected_id={selected_service}, selected_name={selected_service_name}, discussed={discussed_service}) - proceeding"
    )
    
    try:
        phone_number = state.get("phone_number", "").strip()
        
        # CRITICAL FIX (Issue #17, #21): Robust sender_name extraction with multiple fallback sources
        sender_name = (
            state.get("sender_name") or                    # From state (primary)
            state.get("arabic_name") or                    # From transliteration
            state.get("payload", {}).get("sender_name") or # From original payload
            "Friend"                                       # Safe fallback
        )
        
        # Validate sender_name is not empty or invalid
        if not sender_name or sender_name.strip() in ["", "null", "undefined", "None"]:
            sender_name = "Friend"
            logger.debug(f"⚠️ [NODE:verify_patient] sender_name empty/invalid - using fallback: '{sender_name}'")
        
        logger.info(f"🔍 [NODE:verify_patient] Checking patient {phone_number} (sender: {sender_name})")
        
        # CRITICAL: Validate phone number is not empty (Issue #1)
        if not phone_number or phone_number.strip() == "":
            # Attempt to recover from current message (Issue #45)
            maybe_msg = state.get("current_message", "")
            recovered = extract_saudi_phone(maybe_msg) if maybe_msg else None
            if recovered:
                phone_number = recovered
                state["phone_number"] = recovered
                logger.info(f"☎️ [NODE:verify_patient] Recovered phone from message: {recovered}")
            else:
                logger.error(f"❌ [NODE:verify_patient] CRITICAL: Phone number is EMPTY! State: {list(state.keys())}")
                state["patient_verified"] = False
                state["step"] = "patient_verification_error"
                state["last_error"] = {"message": "Phone number missing from state", "node": "verify_patient"}
                return state
        
        # CRITICAL FIX: Check if patient data already loaded by router!
        # Router does comprehensive lookup with 7 phone format attempts
        # If router found patient, we should USE that data instead of re-searching
        existing_patient_data = state.get("patient_data")
        if existing_patient_data and existing_patient_data.get("already_registered"):
            patient_id = existing_patient_data.get("id")
            patient_name = existing_patient_data.get("name")
            
            logger.info(f"✅ [NODE:verify_patient] USING PATIENT DATA FROM ROUTER: {patient_name} (ID: {patient_id})")
            logger.info(f"✅ [NODE:verify_patient] Skipping API call - patient already verified by router")
            
            # Mark as verified and populate state
            state["patient_id"] = patient_id
            state["patient_verified"] = True
            state["step"] = "patient_verified"
            
            # Update names if available
            if patient_name:
                state["arabic_name"] = patient_name
                state["sender_name"] = patient_name
            
            # Keep the existing patient_data (it's already populated)
            logger.info(f"📝 [NODE:verify_patient] Patient verification complete using router data")
            return state
        
        # If no existing patient data, search for patient by phone
        # Normalize phone: remove country code (966 for Saudi, 20 for Egypt)
        from ....utils.phone_parser import remove_country_code
        normalized_phone = remove_country_code(phone_number)
        
        logger.info(f"📡 [NODE:verify_patient] No router data - performing API search: GET /patients/search?phone={normalized_phone} (original: {phone_number})")
        try:
            search_result = await api_client.get(f"/patients/search?phone={normalized_phone}")
        except Exception as api_error:
            # 404 means patient not found - this is EXPECTED during first-time booking (Issue #17)
            if "404" in str(api_error):
                logger.info(f"ℹ️ [NODE:verify_patient] Patient not found (404) - will start registration")
                search_result = None
            else:
                # Real error - reraise
                raise
        
        if search_result and search_result.get("data"):
            # Patient found - continue with booking
            patient = search_result["data"][0] if isinstance(search_result["data"], list) else search_result["data"]
            
            # Store complete patient data for use in conversation
            state["patient_id"] = patient.get("id")
            state["patient_verified"] = True
            state["patient_data"] = {
                "id": patient.get("id"),
                "name": patient.get("name"),
                "national_id": patient.get("national_id"),
                "phone": patient.get("phone"),
                "gender": patient.get("gender"),
                "already_registered": True  # Flag to prevent re-registration prompts
            }
            
            # Update arabic_name if we have patient's actual name
            if patient.get("name"):
                state["arabic_name"] = patient.get("name")
                state["sender_name"] = patient.get("name")
            
            state["step"] = "patient_verified"
            logger.info(f"✅ [NODE:verify_patient] Patient found and data loaded: {patient.get('name')} (ID: {patient.get('id')})")
            logger.info(f"📝 [NODE:verify_patient] Patient data available for conversation: name={patient.get('name')}, phone={patient.get('phone')}")
        else:
            # Patient not found - start registration (Issue #15: removed duplicate log)
            state["patient_verified"] = False
            state["step"] = "needs_registration"
            state["registration"] = {
                "phone": phone_number,
                "sender_name": sender_name
            }
        
        return state
        
    except Exception as e:
        # Enhanced error context for debugging (Issue #23)
        logger.error(
            f"❌ [NODE:verify_patient] Error: {e}",
            exc_info=True,
            extra={
                "error_type": type(e).__name__,
                "state_keys": list(state.keys()),
                "phone_number": state.get("phone_number", "MISSING"),
                "sender_name": sender_name if 'sender_name' in locals() else "NOT_EXTRACTED",
                "resuming": state.get("_resuming", False),
                "current_step": state.get("step", "NONE")
            }
        )
        state["patient_verified"] = False
        state["step"] = "patient_verification_error"
        state["last_error"] = {
            "message": str(e),
            "node": "verify_patient",
            "error_type": type(e).__name__,
            "context": {
                "phone": state.get("phone_number", "missing"),
                "resuming": state.get("_resuming", False)
            }
        }
        return state


async def confirm_registration_node(state: BookingState) -> BookingState:
    """Ask user for confirmation before starting registration (Issue #30)."""
    
    # CRITICAL FIX: Check if patient already verified by router!
    # Router does comprehensive patient lookup, if found we should skip confirmation
    patient_data = state.get("patient_data")
    if patient_data and patient_data.get("already_registered"):
        patient_id = patient_data.get("id")
        patient_name = patient_data.get("name")
        logger.info(f"✅ [NODE:confirm_registration] Patient already registered (router data): {patient_name} (ID: {patient_id})")
        logger.info(f"✅ [NODE:confirm_registration] Skipping confirmation - marking as verified")
        
        # Mark as verified and skip to next step
        state["patient_id"] = patient_id
        state["patient_verified"] = True
        state["name"] = patient_name
        state["national_id"] = patient_data.get("national_id")
        state["gender"] = patient_data.get("gender", "male")
        state["step"] = "patient_verified"
        
        # CRITICAL: Generate proper welcome message for registered patient
        # Use LLM to generate natural response
        from app.services.llm_response_generator import get_llm_response_generator
        response_gen = get_llm_response_generator()
        
        # Get service context if available
        service_mentioned = state.get("last_discussed_service") or state.get("selected_service_name")
        
        # Generate contextual welcome (await since we're already in async function)
        welcome_message = await response_gen.generate_welcome_back(
            patient_name=patient_name,
            service=service_mentioned
        )
        
        # Add welcome message to conversation
        state["messages"].append({
            "role": "assistant",
            "content": welcome_message
        })
        
        logger.info(f"✅ [NODE:confirm_registration] Generated welcome message for registered patient")
        
        return state
    
    # Patient not found, ask for confirmation using LLM
    arabic_name = state.get("arabic_name", "حبيبنا")
    logger.info(f"🤔 [NODE:confirm_registration] Patient not found - asking user to confirm registration")

    # CRITICAL: Use LLM to generate natural confirmation request
    from app.core.response_generator import get_response_generator
    response_gen = get_response_generator()
    
    # Get service context if available
    service_mentioned = state.get("last_discussed_service") or state.get("selected_service_name")
    
    # LLM generates natural confirmation message
    message_content = await response_gen.ask_registration_confirmation(
        user_name=arabic_name,
        service=service_mentioned
    )

    state["messages"].append({
        "role": "assistant",
        "content": message_content
    })
    state["step"] = "awaiting_registration_confirmation"
    return state


async def start_registration_node(state: BookingState) -> BookingState:
    """Start patient registration flow (Issue #20: Check router context)"""
    arabic_name = state.get("arabic_name", "حبيبنا")
    phone_number = state.get("phone_number", "")
    router_intent = state.get("router_intent", "")
    current_step = state.get("step", "")
    
    logger.info(f"📋 [NODE:start_registration] Starting registration for {phone_number} (current_step: {current_step})")
    logger.debug(f"📋 [DEBUG] arabic_name='{arabic_name}', phone_number='{phone_number}', router_intent='{router_intent}'")
    
    # CRITICAL FIX: If already at awaiting_name, don't ask again (Issue #40)
    if current_step == "awaiting_name":
        logger.info(f"⏭️ [NODE:start_registration] Already at awaiting_name - skipping duplicate prompt")
        return state  # Don't ask for name again!
    
    # Format phone number for display (Issue #18: Ensure phone is populated)
    if not phone_number or phone_number.strip() == "":
        logger.error(f"❌ [NODE:start_registration] CRITICAL: Phone number is EMPTY at registration!")
        phone_display = "[رقم غير متوفر]"
    else:
        phone_display = phone_number
    
    state["step"] = "awaiting_name"
    
    # CRITICAL: Use LLM to generate natural name request (NO TEMPLATES!)
    from app.core.response_generator import get_response_generator
    response_gen = get_response_generator()
    
    # LLM generates contextual message with phone confirmation
    message_content = await response_gen.ask_for_name(
        user_name=arabic_name,
        phone_display=phone_display
    )
    
    logger.debug(f"📋 [DEBUG] LLM-generated registration message:\n{message_content}")
    
    state["messages"].append({
        "role": "assistant",
        "content": message_content
    })
    
    return state


async def handle_registration_name_node(state: BookingState) -> BookingState:
    """Process registration name input - handles multi-line structured input (Issue #4)"""
    message = state["current_message"]
    sender_name = state.get("sender_name", "")
    
    logger.info(f"📋 [NODE:registration_name] Got input: {message[:50]}...")
    
    # CRITICAL: Check if name already confirmed by router (Issue: Router confirms but node doesn't check)
    if not state.get("name") and not state.get("registration", {}).get("name"):
        logger.info(f"📝 [NODE:registration_name] Starting name collection for {sender_name}")
        state["step"] = "registration_name"
        
        # LLM-FIRST: Generate natural response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        response = await response_gen.ask_for_name(user_name=sender_name)
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state
    
    # VALIDATION: Check for cancellation keywords
    cancel_keywords = ["إلغاء", "cancel", "stop", "back", "رجوع"]
    if any(keyword in message.lower() for keyword in cancel_keywords):
        logger.info(f"🚫 [NODE:registration_name] User wants to cancel registration")
        state["step"] = "registration_cancelled"
        
        # LLM-FIRST: Generate natural cancellation response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        response = await response_gen.handle_cancellation(user_name=sender_name)
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state
    
    # VALIDATION: Check if input is actually a name
    # Names should have at least 2 words and be mostly Arabic letters
    import re
    arabic_ratio = len(re.findall(r'[\u0600-\u06FF]', message)) / max(len(message), 1)
    words = message.strip().split()
    
    if len(message.strip()) < 2 or len(message.strip()) > 100:
        logger.warning(f"⚠️ [NODE:registration_name] Invalid name length: {len(message)} chars")
        
        # LLM-FIRST: Generate natural error response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        response = await response_gen.handle_invalid_name(
            user_name=sender_name,
            reason="الاسم قصير جداً أو طويل جداً"
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state
    
    if len(words) < 2:
        logger.warning(f"⚠️ [NODE:registration_name] Name needs first and last: '{message}'")
        
        # LLM-FIRST: Generate natural error response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        response = await response_gen.handle_invalid_name(
            user_name=sender_name,
            reason="نحتاج الاسم الكامل (الاسم الأول والعائلة)"
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state
    
    if arabic_ratio < 0.5:
        logger.warning(f"⚠️ [NODE:registration_name] Name should be in Arabic: '{message}'")
        
        # LLM-FIRST: Generate natural error response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        response = await response_gen.handle_invalid_name(
            user_name=sender_name,
            reason="نحتاج الاسم بالعربي"
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state
    
    # Issue #13: Validate name consistency if both Arabic and English provided
    if sender_name and sender_name != "Friend":
        logger.debug(f"📋 [DEBUG] Name validation: sender_name='{sender_name}', user_input='{message[:30]}...'")
    
    if not state.get("registration"):
        state["registration"] = {}
    
    # CRITICAL FIX: Handle multi-line structured input (Issue #4)
    # If user sends "name\nphone\nconfirmation" in one message, parse it
    lines = [line.strip() for line in message.split('\n') if line.strip()]
    
    if len(lines) >= 2:
        # Multi-line input detected - extract name and ID
        logger.info(f"📋 [NODE:registration_name] Multi-line input detected: {len(lines)} lines")
        
        name = lines[0]
        national_id = lines[1]
        
        state["registration"]["name"] = name
        state["registration"]["national_id"] = national_id
        
        # Skip to completion directly
        logger.info(f"✅ [NODE:registration_name] Parsed: name='{name}', id='{national_id}'")
        state["step"] = "registration_complete_multiline"
        
        return state
    else:
        # Single line - traditional flow (one-by-one, no confirmation)
        captured_name = message.strip()
        state["registration"]["name"] = captured_name
        state["name"] = captured_name  # Also store in state for easy access
        state["step"] = "registration_id"
        
        # LLM-FIRST: Generate natural response for asking national ID
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        response = await response_gen.ask_for_national_id(user_name=captured_name)
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        
        return state

async def handle_registration_id_node(state: BookingState, api_client) -> BookingState:
    """Process registration ID and create patient"""
    message = state["current_message"]
    registration = state.get("registration", {})
    phone_number = state.get("phone_number", "")
    
    # CRITICAL: Auto-detect gender from name using LLM (never ask user!)
    if not state.get("gender"):
        from app.utils.gender_detector import detect_gender_from_name
        
        # Get name from state
        patient_name = state.get("name") or registration.get("name") or ""
        
        if patient_name:
            # Auto-detect gender using LLM
            detected_gender = await detect_gender_from_name(patient_name)
            state["gender"] = detected_gender
            logger.info(f"🤖 [NODE:registration_id] LLM detected gender: {detected_gender} from name '{patient_name}'")
        else:
            # No name available, default to male
            state["gender"] = "male"
            logger.warning(f"⚠️ [NODE:registration_id] No name available for gender detection, defaulting to male")
    
    # OLD CODE: Manual gender asking (REMOVED - now auto-detect with LLM!)
    if False:  # Disabled - we use LLM now!
        # Check if current message is a gender response
        message_lower = message.lower().strip()
        if message_lower in ["ذكر", "male", "رجل", "m", "1", "١"]:  # Added "1" option
            state["gender"] = "male"
            logger.info(f"✅ [NODE:registration_id] Gender set to: male")
        elif message_lower in ["أنثى", "female", "امرأة", "f", "بنت", "2", "٢"]:  # Added "2" option
            state["gender"] = "female"
            logger.info(f"✅ [NODE:registration_id] Gender set to: female")
        else:
            # CRITICAL FIX: Validate Saudi National ID format
            import re
            
            # Clean input (remove spaces/special chars)
            clean_id = re.sub(r'\D', '', message.strip())
            
            # Validation: Saudi National ID rules
            if len(clean_id) != 10:
                # Wrong length - CRITICAL: Count digits from original message, not just clean_id
                # Issue: User typed "حجز" but system said "0 digits" (wrong!)
                original_digits = len(clean_id)  # Actual digit count from input
                
                logger.warning(f"⚠️ [NODE:registration_id] Invalid ID length: {original_digits} digits (need 10)")
                state["step"] = "registration_id"
                
                # Use LLM to generate natural error message (NO TEMPLATES!)
                from app.core.response_generator import get_response_generator
                response_gen = get_response_generator()
                user_name = state.get("name") or "حبيبنا"
                
                reason = "رقم الهوية يجب أن يكون 10 أرقام بالضبط"
                if original_digits > 0:
                    reason += f" (أنت أرسلت {original_digits} رقم)"
                
                response = await response_gen.handle_invalid_id(
                    user_name=user_name,
                    provided_value=message,
                    reason=reason
                )
                
                state["messages"].append({
                    "role": "assistant",
                    "content": response
                })
                return state
            
            elif clean_id[0] not in ['1', '2']:
                # Wrong starting digit (must be 1=citizen or 2=resident)
                logger.warning(f"⚠️ [NODE:registration_id] Invalid ID prefix: starts with '{clean_id[0]}' (need 1 or 2)")
                state["step"] = "registration_id"
                
                # LLM-FIRST: Generate natural error response
                from app.core.response_generator import get_response_generator
                response_gen = get_response_generator()
                user_name = state.get("name") or "حبيبنا"
                response = await response_gen.handle_invalid_id(
                    user_name=user_name,
                    provided_value=clean_id,
                    reason=f"رقم الهوية يجب أن يبدأ بـ 1 (مواطن) أو 2 (مقيم)، رقمك يبدأ بـ {clean_id[0]}"
                )
                
                state["messages"].append({
                    "role": "assistant",
                    "content": response
                })
                return state
            
            elif not clean_id.isdigit():
                # Contains non-digits
                logger.warning(f"⚠️ [NODE:registration_id] Invalid ID format: contains non-digits")
                state["step"] = "registration_id"
                
                # LLM-FIRST: Generate natural error response
                from app.core.response_generator import get_response_generator
                response_gen = get_response_generator()
                user_name = state.get("name") or "حبيبنا"
                response = await response_gen.handle_invalid_id(
                    user_name=user_name,
                    provided_value=message,
                    reason="رقم الهوية يجب أن يكون أرقام فقط"
                )
                
                state["messages"].append({
                    "role": "assistant",
                    "content": response
                })
                return state
            
            else:
                # ✅ Valid Saudi National ID - proceed directly to patient creation
                state["national_id"] = clean_id
                logger.info(f"💾 [NODE:registration_id] Saved valid Saudi National ID: {clean_id} (type: {'مواطن' if clean_id[0] == '1' else 'مقيم'})")
                
                # Gender is already auto-detected above, no need to ask!
                # Continue to patient creation below...
    
    # CRITICAL FIX (Issue #28): Properly extract national_id from current message
    # The current message at this point SHOULD be the national ID (not the name!)
    
    # Clean the message to extract only digits
    import re
    clean_id = re.sub(r'\D', '', message.strip())
    
    # Validate it's a proper 10-digit Saudi ID
    if len(clean_id) != 10:
        logger.error(f"❌ [NODE:registration_id] CRITICAL: Expected 10-digit ID, got: '{message}' (cleaned: '{clean_id}')")
        state["step"] = "registration_id"
        
        # LLM-FIRST: Generate natural error response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        user_name = state.get("name") or state.get("sender_name") or "حبيبنا"
        response = await response_gen.handle_invalid_id(
            user_name=user_name,
            provided_value=message,
            reason=f"رقم الهوية يجب أن يكون 10 أرقام بالضبط (أنت أرسلت {len(clean_id)} رقم)"
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state
    
    # Use the cleaned 10-digit ID
    national_id_to_use = clean_id
    state["national_id"] = clean_id  # Save to state
    logger.info(f"✅ [NODE:registration_id] Extracted valid ID: {national_id_to_use}")
    
    try:
        # CRITICAL: Remove country code from phone number
        # WhatsApp session: "201098424043" (with country code 20)
        # API expects: "1098424043" (without country code)
        local_phone = phone_number
        if phone_number.startswith("966"):  # Saudi Arabia country code
            local_phone = phone_number[3:]  # Remove "966"
        elif phone_number.startswith("20"):  # Egypt country code
            local_phone = phone_number[2:]  # Remove "20"
        
        # CRITICAL: Extract birth date from Saudi national ID
        from app.utils.national_id_parser import get_birth_date_from_national_id
        birth_date = get_birth_date_from_national_id(
            national_id_to_use,
            fallback="1990-01-01"  # Fallback if ID parsing fails
        )
        
        # CRITICAL: Validate name before API call (Issue #11: 400 Bad Request)
        patient_name = registration.get("name") or state.get("name") or ""
        invalid_names = [".", "..", "سجل الثاني", "null", "undefined", "", "Unknown"]
        
        if not patient_name or patient_name in invalid_names or len(patient_name.strip()) < 3:
            logger.error(f"❌ Cannot create patient: Invalid name '{patient_name}'")
            state["step"] = "registration_name"
            
            # LLM-FIRST: Generate natural error response
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            response = await response_gen.handle_invalid_name(
                user_name="حبيبنا",
                reason="لم نستطع الحصول على الاسم بشكل صحيح"
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
            return state
        
        # CRITICAL: Ensure phone has country code (Issue #11: 400 Bad Request)
        if not local_phone.startswith("966"):
            patient_phone = f"966{local_phone}"
        else:
            patient_phone = local_phone
        
        logger.info(f"📱 Formatted patient phone: {patient_phone} (from: {local_phone})")
        
        # Create patient via API - MATCH EXACT BACKEND STRUCTURE
        # Required fields: name, identification_id, gender, patient_phone, birth_date
        # Optional fields: Pass as empty strings (backend expects all fields)
        patient_data = {
            "name": patient_name,  # REQUIRED: Validated above
            "identification_id": national_id_to_use,  # REQUIRED: National ID
            "gender": state.get("gender", "male"),  # REQUIRED: Auto-detected or default
            "patient_phone": patient_phone,  # REQUIRED: With country code
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
        
        logger.info(f"📡 [NODE:registration_id] Creating patient: {patient_data}")
        # CRITICAL: Use /customer/create endpoint, not /patients (405 error fix)
        result = await api_client.post("/customer/create", json_body=patient_data)
        
        if result and result.get("id"):
            patient_id = result.get("id")
            state["patient_id"] = patient_id
            state["patient_verified"] = True
            state["registration"] = None
            state["step"] = "registration_complete"
            
            logger.info(f"✅ [NODE:registration_id] Patient created: ID={patient_id}")
            
            # LLM-FIRST: Generate natural success response
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            user_name = state.get("name") or "حبيبنا"
            service = state.get("last_discussed_service") or state.get("selected_service_name")
            response = await response_gen.confirm_registration(user_name=user_name, service=service)
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
        else:
            # API returned but no patient ID - registration failed
            logger.error(f"❌ [NODE:registration_id] API returned but no patient ID: {result}")
            state["step"] = "registration_error"
            state["last_error"] = {"message": "Failed to create patient - no ID returned", "node": "registration_id"}
            
            # LLM-FIRST: Generate natural error response
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            user_name = state.get("name") or "حبيبنا"
            response = await response_gen.handle_error(
                user_name=user_name,
                error_type="فشل في إنشاء الحساب",
                can_retry=True
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
        
        return state
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ [NODE:registration_id] Error creating patient: {error_message}")
        
        # CRITICAL: Smart error recovery - detect known validation errors and auto-fix
        retry_count = state.get("registration_retry_count", 0)
        
        # Check if this is a known fixable error
        is_fixable = False
        if "reference_by" in error_message or "whatsapp_bot" in error_message:
            is_fixable = True
            logger.warning(f"⚠️ [NODE:registration_id] Detected fixable validation error: reference_by field")
        
        # Auto-retry once with fixed data
        if is_fixable and retry_count == 0:
            logger.info(f"🔄 [NODE:registration_id] Attempting auto-retry with corrected data...")
            state["registration_retry_count"] = 1
            
            # The corrected reference_by value is already in the code now ("whatsapp")
            # Just need to trigger a retry by keeping step at current position
            # But for users with old state, we need to clear the error and retry
            state["step"] = "awaiting_gender"  # Go back to trigger registration again
            state["last_error"] = None
            
            logger.info(f"✅ [NODE:registration_id] Auto-retry scheduled - will use corrected payload")
            # Don't show error to user yet, let retry happen silently
            return state
        
        # Retry failed or not fixable - escalate to error state
        logger.error(f"❌ [NODE:registration_id] Registration failed after retry/unfixable error")
        state["step"] = "registration_error"
        state["last_error"] = {
            "message": error_message,
            "node": "registration_id",
            "retry_count": retry_count,
            "fixable": is_fixable
        }
        
        # CRITICAL: Add to manual review queue for staff follow-up
        registration_data = {
            "name": state.get("name"),
            "phone": state.get("phone_number"),
            "national_id": state.get("national_id"),
            "gender": state.get("gender"),
            "error": error_message,
            "timestamp": str(__import__("datetime").datetime.now())
        }
        
        # Queue for manual processing
        try:
            from app.queue.manual_review_queue import get_manual_review_queue
            queue = get_manual_review_queue()
            session_key = state.get("session_key", "unknown")
            entry_id = queue.add_failed_registration(
                session_key=session_key,
                registration_data=registration_data,
                error_message=error_message,
                retry_count=retry_count
            )
            
            if entry_id:
                logger.info(f"✅ [NODE:registration_id] Added to manual review queue: {entry_id}")
                state["manual_review_id"] = entry_id
            
        except Exception as queue_error:
            logger.error(f"⚠️ [NODE:registration_id] Failed to add to manual queue: {queue_error}")
        
        # Store in state as backup
        state["failed_registration_data"] = registration_data
        
        # LLM-FIRST: Generate natural error response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        user_name = state.get("name") or "حبيبنا"
        response = await response_gen.handle_error(
            user_name=user_name,
            error_type="خطأ في التسجيل (سيتم مراجعته يدوياً)",
            can_retry=False
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state


async def handle_multiline_registration_node(state: BookingState, api_client) -> BookingState:
    """Handle multiline registration completion (Issue #4)"""
    registration = state.get("registration", {})
    phone_number = state.get("phone_number", "")
    
    name = registration.get("name")
    national_id = registration.get("national_id")
    
    logger.info(f"📋 [NODE:multiline_registration] Creating patient from multiline input")
    
    try:
        # CRITICAL: Remove country code from phone number
        local_phone = phone_number
        if phone_number.startswith("966"):  # Saudi Arabia
            local_phone = phone_number[3:]
        elif phone_number.startswith("20"):  # Egypt
            local_phone = phone_number[2:]
        
        # CRITICAL: Extract birth date from Saudi national ID
        from app.utils.national_id_parser import get_birth_date_from_national_id
        birth_date = get_birth_date_from_national_id(
            national_id,
            fallback="1990-01-01"  # Fallback if ID parsing fails
        )
        
        # Create patient via API - MATCH EXACT BACKEND STRUCTURE
        # Required fields: name, identification_id, gender, patient_phone, birth_date
        # Optional fields: Pass as empty strings (backend expects all fields)
        patient_data = {
            "name": name,  # REQUIRED: Parsed from multiline input
            "identification_id": national_id,  # REQUIRED: National ID
            "gender": state.get("gender", "male"),  # REQUIRED: Collected (default: male)
            "patient_phone": local_phone,  # REQUIRED: From WhatsApp (without country code)
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
        
        logger.info(f"📡 [NODE:multiline_registration] API call: POST /customer/create - {patient_data}")
        # CRITICAL: Use /customer/create endpoint, not /patients (405 error fix)
        result = await api_client.post("/customer/create", json_body=patient_data)
        
        if result and result.get("id"):
            patient_id = result.get("id")
            state["patient_id"] = patient_id
            state["patient_verified"] = True
            state["registration"] = None
            state["step"] = "registration_complete"
            
            logger.info(f"✅ [NODE:multiline_registration] Patient created: ID={patient_id}")
            
            # LLM-FIRST: Generate natural success response
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            service = state.get("last_discussed_service") or state.get("selected_service_name")
            response = await response_gen.confirm_registration(user_name=name, service=service)
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
        else:
            # API returned but no patient ID - registration failed
            logger.error(f"❌ [NODE:multiline_registration] API returned but no patient ID: {result}")
            state["step"] = "registration_error"
            state["last_error"] = {"message": "Failed to create patient - no ID returned", "node": "multiline_registration"}
            # LLM-FIRST: Generate natural error response
            from app.core.response_generator import get_response_generator
            response_gen = get_response_generator()
            response = await response_gen.handle_error(
                user_name=name,
                error_type="فشل في إنشاء الحساب",
                can_retry=True
            )
            
            state["messages"].append({
                "role": "assistant",
                "content": response
            })
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [NODE:multiline_registration] Error: {e}")
        state["step"] = "registration_error"
        state["last_error"] = {"message": str(e), "node": "multiline_registration"}
        # LLM-FIRST: Generate natural error response
        from app.core.response_generator import get_response_generator
        response_gen = get_response_generator()
        user_name = registration.get("name") or "حبيبنا"
        response = await response_gen.handle_error(
            user_name=user_name,
            error_type="خطأ في التسجيل",
            can_retry=True
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response
        })
        return state
