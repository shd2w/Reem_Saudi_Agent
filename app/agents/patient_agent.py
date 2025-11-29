"""
Patient Agent - Patient registration and profile management
============================================================
Handles patient-related operations with backend API integration.

Features:
- Patient registration
- Profile lookup and verification
- Profile updates
- Patient information retrieval
- Natural conversation flow

Author: Agent Orchestrator Team
Version: 1.0.0
"""

from typing import Dict, Any, Optional
from loguru import logger

from ..api.agent_api import AgentApiClient
from ..memory.session_manager import SessionManager
from ..utils.language_detector import detect_language
from ..utils.api_normalizer import normalize_api_response
from ..services.llm_response_generator import get_llm_response_generator


class PatientAgent:
    """
    Professional patient agent for registration and profile management.
    
    Handles:
    - New patient registration
    - Patient profile lookup
    - Profile updates
    - Patient verification
    """
    
    def __init__(self, session_key: str):
        self.session_key = session_key
        self.api_client = AgentApiClient()
        self.session_manager = SessionManager()
        self.llm_generator = get_llm_response_generator()
    
    async def handle(self, payload: dict, context: dict = None) -> dict:
        """
        Handle patient-related requests with conversation context.
        
        Args:
            payload: Message payload with user message and context
            context: Conversation context (history, sender info, etc.)
            
        Returns:
            Response dictionary with patient operation result
        """
        try:
            message = payload.get("message", "").lower()
            phone_number = payload.get("phone_number")
            sender_name = payload.get("sender_name", "")
            
            logger.info(f"👤 Patient agent processing: '{message[:50]}...'")
            
            # Determine patient action (Arabic + English keywords)
            if any(word in message for word in ["register", "sign up", "new patient", "create account", "سجل", "تسجيل", "جديد"]):
                return await self._handle_registration(phone_number, sender_name, payload, context)
            
            elif any(word in message for word in ["update", "change", "modify", "edit", "تحديث", "غير", "عدل"]):
                return await self._handle_update(phone_number, payload)
            
            elif any(word in message for word in ["check", "view", "show", "my profile", "my info", "ملفي", "بياناتي"]):
                return await self._handle_view_profile(phone_number)
            
            else:
                # General patient info request - natural Saudi response
                language = detect_language(message)
                if language == "arabic":
                    return {
                        "response": f"أهلاً {sender_name}! 🙋‍♂️\n\nتبي تسجل معنا كمريض جديد؟ ولا عندك حساب وتبي تعدل بياناتك؟\n\nقول لي وأنا أساعدك! ✨",
                        "intent": "patient",
                        "status": "inquiry"
                    }
                else:
                    return {
                        "response": f"Hello {sender_name}! 🙋‍♂️\n\nWould you like to register as a new patient, or update your existing information?\n\nLet me know how I can help! ✨",
                        "intent": "patient",
                        "status": "inquiry"
                    }
                
        except Exception as exc:
            logger.error(f"Patient agent error: {exc}", exc_info=True)
            # Arabic error message
            language = detect_language(message)
            if language == "arabic":
                response = "عذراً يا حبيبي، صار عندي مشكلة بسيطة 😅\nجرب مرة ثانية بعد شوي"
            else:
                response = "Oops! Had a small issue 😅\nTry again in a moment"
            
            return {
                "response": response,
                "intent": "patient",
                "status": "error",
                "error": str(exc)
            }
    
    async def _handle_registration(self, phone_number: str, sender_name: str, payload: dict, context: dict) -> dict:
        """Handle patient registration with WhatsApp number confirmation"""
        try:
            from ..core.llm_reasoner import get_llm_reasoner
            llm = get_llm_reasoner()
            
            # Get or initialize registration state
            session_data = self.session_manager.get(self.session_key) or {}
            reg_state = session_data.get("registration_state", {})
            
            message = payload.get("message", "").lower()
            
            # Step 1: Confirm WhatsApp number or ask for alternative
            if not reg_state or not reg_state.get("phone_confirmed"):
                # Format phone for display
                formatted_phone = phone_number
                
                llm_context = {
                    "sender_name": sender_name,
                    "intent": "patient_registration_start",
                    "registration_step": "phone_confirmation",
                    "whatsapp_number": formatted_phone,
                    "message": "Ask user if they want to use their WhatsApp number for registration, or provide a different number"
                }
                
                response_text = llm.generate_reply(
                    user_id=phone_number,
                    user_message=message,
                    context=llm_context
                )
                
                # Initialize registration state
                reg_state = {
                    "started": True,
                    "step": "phone_confirmation",
                    "whatsapp_number": phone_number,
                    "phone_confirmed": False
                }
                session_data["registration_state"] = reg_state
                self.session_manager.put(self.session_key, session_data, ttl_minutes=30)
                
                return {
                    "response": response_text,
                    "intent": "patient",
                    "status": "registration_phone_confirmation"
                }
            
            # Step 2: User confirmed phone or provided alternative
            if reg_state.get("step") == "phone_confirmation":
                # Check if user said yes/confirmed
                if any(word in message for word in ["yes", "نعم", "أيه", "تمام", "أوكي", "زين", "صح", "أبشر"]):
                    # Use WhatsApp number
                    reg_state["confirmed_phone"] = phone_number
                    reg_state["phone_confirmed"] = True
                    reg_state["step"] = "collect_name"
                else:
                    # Extract phone number from message
                    import re
                    phone_pattern = r'(\+?\d{10,15})'
                    phone_match = re.search(phone_pattern, message)
                    
                    if phone_match:
                        reg_state["confirmed_phone"] = phone_match.group(1)
                        reg_state["phone_confirmed"] = True
                        reg_state["step"] = "collect_name"
                    else:
                        # Ask again for phone
                        llm_context = {
                            "sender_name": sender_name,
                            "intent": "patient_registration_phone_retry",
                            "registration_step": "phone_confirmation_retry",
                            "message": "User didn't confirm or provide valid number, ask again politely"
                        }
                        
                        response_text = llm.generate_reply(
                            user_id=phone_number,
                            user_message=message,
                            context=llm_context
                        )
                        
                        return {
                            "response": response_text,
                            "intent": "patient",
                            "status": "registration_phone_retry"
                        }
                
                # Ask for name
                llm_context = {
                    "sender_name": sender_name,
                    "intent": "patient_registration_ask_name",
                    "registration_step": "collect_name",
                    "confirmed_phone": reg_state["confirmed_phone"]
                }
                
                response_text = llm.generate_reply(
                    user_id=phone_number,
                    user_message=message,
                    context=llm_context
                )
                
                session_data["registration_state"] = reg_state
                self.session_manager.put(self.session_key, session_data, ttl_minutes=30)
                
                return {
                    "response": response_text,
                    "intent": "patient",
                    "status": "registration_collecting_name"
                }
            
            # Step 3: Collect name
            if reg_state.get("step") == "collect_name":
                # Extract name from message
                reg_state["patient_name"] = message.strip()
                reg_state["step"] = "collect_gender"
                
                llm_context = {
                    "sender_name": sender_name,
                    "intent": "patient_registration_ask_gender",
                    "registration_step": "collect_gender",
                    "collected_name": reg_state["patient_name"]
                }
                
                response_text = llm.generate_reply(
                    user_id=phone_number,
                    user_message=message,
                    context=llm_context
                )
                
                session_data["registration_state"] = reg_state
                self.session_manager.put(self.session_key, session_data, ttl_minutes=30)
                
                return {
                    "response": response_text,
                    "intent": "patient",
                    "status": "registration_collecting_gender"
                }
            
            # Step 4: Collect gender
            if reg_state.get("step") == "collect_gender":
                # Extract gender
                if any(word in message for word in ["male", "ذكر", "رجال", "شاب"]):
                    reg_state["gender"] = "male"
                elif any(word in message for word in ["female", "أنثى", "بنات", "حريم", "بنت"]):
                    reg_state["gender"] = "female"
                else:
                    reg_state["gender"] = "unknown"
                
                reg_state["step"] = "complete"
                
                # Create patient record via API
                # CRITICAL: API expects specific field names (400 error fix)
                patient_data = {
                    "name": reg_state["patient_name"],
                    "patient_phone": reg_state["confirmed_phone"],  # API expects "patient_phone"
                    "identification_id": reg_state.get("national_id", ""),  # API expects "identification_id"
                    "gender": reg_state["gender"],
                    "registration_source": "whatsapp"
                }
                
                try:
                    # CRITICAL: Use /customer/create endpoint, not /patients (405 error fix)
                    result = await self.api_client.post("/customer/create", data=patient_data)
                    
                    if result and result.get("success"):
                        # Registration successful!
                        llm_context = {
                            "sender_name": sender_name,
                            "intent": "patient_registration_completed",
                            "registration_step": "completed",
                            "patient_data": patient_data,
                            "patient_id": result.get("id")
                        }
                        
                        response_text = llm.generate_reply(
                            user_id=phone_number,
                            user_message=message,
                            context=llm_context
                        )
                        
                        # Clear registration state
                        session_data["registration_state"] = {}
                        self.session_manager.put(self.session_key, session_data)
                        
                        logger.info(f"✅ Patient registered successfully: {patient_data['name']}")
                        
                        return {
                            "response": response_text,
                            "intent": "patient",
                            "status": "registration_completed",
                            "patient_id": result.get("id")
                        }
                    else:
                        # Registration failed
                        error_msg = result.get("error", "Unknown error")
                        logger.error(f"Patient registration API failed: {error_msg}")
                        
                        llm_context = {
                            "sender_name": sender_name,
                            "intent": "patient_registration_failed",
                            "registration_step": "error",
                            "error_message": error_msg
                        }
                        
                        response_text = llm.generate_reply(
                            user_id=phone_number,
                            user_message=message,
                            context=llm_context
                        )
                        
                        return {
                            "response": response_text,
                            "intent": "patient",
                            "status": "registration_error"
                        }
                        
                except Exception as api_error:
                    logger.error(f"Patient registration API error: {api_error}", exc_info=True)
                    
                    language = detect_language(message)
                    if language == "arabic":
                        response = "عذراً يا حبيبي، صار عندي خلل بالتسجيل. ممكن تجرب مرة ثانية؟ 🙏"
                    else:
                        response = "Sorry, I had an issue with registration. Could you try again? 🙏"
                    
                    return {
                        "response": response,
                        "intent": "patient",
                        "status": "registration_error"
                    }
            
            # Fallback
            language = detect_language(message)
            if language == "arabic":
                response = "تمام، خلني أساعدك بالتسجيل! 🙌"
            else:
                response = "Great! Let me help you register! 🙌"
            
            return {
                "response": response,
                "intent": "patient",
                "status": "registration_in_progress"
            }
            
        except Exception as exc:
            logger.error(f"Registration error: {exc}", exc_info=True)
            
            language = detect_language(payload.get("message", ""))
            if language == "arabic":
                response = "عذراً، صار عندي خلل بسيط. ممكن نبدأ من جديد؟ 🙏"
            else:
                response = "Sorry, had a small issue. Can we start over? 🙏"
            
            return {
                "response": response,
                "intent": "patient",
                "status": "error"
            }
    
    async def _handle_lookup(self, phone_number: str, sender_name: str) -> dict:
        """Handle patient lookup"""
        try:
            patient = await self._find_patient(phone_number)
            
            if patient:
                name = patient.get("name", sender_name)
                return {
                    "response": f"Hello {name}! I found your patient record. How can I help you today? You can:\n• Book an appointment\n• View your bookings\n• Update your profile",
                    "intent": "patient",
                    "status": "found",
                    "patient": patient
                }
            else:
                return {
                    "response": f"I couldn't find your patient record. Would you like to register? Reply with 'register' to get started.",
                    "intent": "patient",
                    "status": "not_found"
                }
                
        except Exception as exc:
            logger.error(f"Lookup error: {exc}")
            return {
                "response": "I'm having trouble accessing patient records. Please try again.",
                "intent": "patient",
                "status": "error"
            }
    
    async def _handle_view_profile(self, phone_number: str) -> dict:
        """Handle view profile request"""
        try:
            patient = await self._find_patient(phone_number)
            
            if not patient:
                return {
                    "response": "I couldn't find your patient record. Would you like to register?",
                    "intent": "patient",
                    "status": "not_found"
                }
            
            # Format patient info
            name = patient.get("name", "Unknown")
            gender = patient.get("gender", "Not specified")
            dob = patient.get("date_of_birth", "Not specified")
            patient_id = patient.get("id", "N/A")
            
            profile_text = f"""📋 Your Profile:

👤 Name: {name}
🆔 Patient ID: {patient_id}
⚧ Gender: {gender}
🎂 Date of Birth: {dob}
📱 Phone: {phone_number}

Need to update any information? Reply with 'update profile'."""
            
            return {
                "response": profile_text,
                "intent": "patient",
                "status": "profile_displayed",
                "patient": patient
            }
            
        except Exception as exc:
            logger.error(f"View profile error: {exc}")
            return {
                "response": "Sorry, I couldn't retrieve your profile. Please try again.",
                "intent": "patient",
                "status": "error"
            }
    
    async def _handle_update(self, phone_number: str, payload: dict) -> dict:
        """Handle profile update request"""
        try:
            patient = await self._find_patient(phone_number)
            
            if not patient:
                return {
                    "response": "I couldn't find your patient record. Please register first.",
                    "intent": "patient",
                    "status": "not_found"
                }
            
            return {
                "response": "To update your profile, please contact our support team or visit our clinic. For security reasons, profile updates require verification.",
                "intent": "patient",
                "status": "update_info"
            }
            
        except Exception as exc:
            logger.error(f"Update error: {exc}")
            return {
                "response": "Sorry, I couldn't process your update request. Please try again.",
                "intent": "patient",
                "status": "error"
            }
    
    async def _find_patient(self, phone_number: str) -> Optional[dict]:
        """Find patient by phone number"""
        try:
            result = await self.api_client.get("/patients", params={"q": phone_number})
            
            if result and result.get("data"):
                patients = result["data"]
                if patients:
                    logger.info(f"✓ Patient found: {patients[0].get('name')}")
                    return patients[0]
            
            logger.info(f"Patient not found for phone: {phone_number}")
            return None
            
        except Exception as exc:
            logger.error(f"Find patient error: {exc}")
            return None
    
    def _save_registration_state(self, registration_state: dict) -> None:
        """Save registration state to session"""
        try:
            session_data = self.session_manager.get(self.session_key) or {}
            session_data["registration_state"] = registration_state
            self.session_manager.put(self.session_key, session_data, ttl_minutes=120)
        except Exception as exc:
            logger.error(f"Save registration state error: {exc}")
    
    def _clear_registration_state(self) -> None:
        """Clear registration state from session"""
        try:
            session_data = self.session_manager.get(self.session_key) or {}
            session_data["registration_state"] = {}
            self.session_manager.put(self.session_key, session_data, ttl_minutes=120)
        except Exception as exc:
            logger.error(f"Clear registration state error: {exc}")
