"""
LLM Response Generator - Natural Human-like Response Generation
================================================================
Uses GPT-4o to generate contextual, emotionally aware responses in Saudi Arabic or English.

This module replaces hardcoded templates with dynamic, intelligent responses that:
- Sound natural and human-like
- Adapt to user's tone and language
- Include relevant data from APIs
- Maintain conversation context
- Use authentic Saudi dialect when appropriate

Author: Agent Orchestrator Team
Version: 2.0.0
"""

import asyncio
import re
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from loguru import logger

from ..config import get_settings
from ..utils.language_detector import detect_language
from ..utils.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError


class LLMResponseGenerator:
    """
    Generates natural, contextual responses using GPT-4o.
    
    Replaces template-based responses with intelligent, human-like communication.
    Singleton pattern to avoid re-initialization.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMResponseGenerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once - prevent duplicate initialization
        if LLMResponseGenerator._initialized:
            return  # Already initialized - silent reuse
        
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key.get_secret_value())
        self.model = self.settings.openai_model or "gpt-4o"
        
        LLMResponseGenerator._initialized = True
        logger.info(f"LLM Response Generator initialized with {self.model} (singleton) - First init")
    
    async def generate_response(
        self,
        intent: str,
        user_message: str,
        context: Optional[Dict] = None,
        data: Optional[Dict] = None,
        sender_name: Optional[str] = None
    ) -> str:
        """
        Generate natural, contextual response.
        
        Args:
            intent: Classified intent (booking, patient, resource, feedback)
            user_message: Original user message
            context: Conversation history and state
            data: API data to include in response (services, doctors, etc.)
            sender_name: User's name for personalization
            
        Returns:
            Natural, human-like response in appropriate language
        """
        try:
            # Detect language
            language = detect_language(user_message)
            
            # Build system prompt based on intent and language
            system_prompt = self._build_system_prompt(intent, language)
            
            # Build user prompt with context and data (Issue #43 - Fixed missing sender_name)
            user_prompt = self._build_user_prompt(
                user_message=user_message,
                context=context,
                data=data,
                sender_name=sender_name,
                language=language
            )
            
            logger.debug(f"Generating {language} response for intent: {intent}")
            
            # Call GPT-4o with circuit breaker protection
            circuit_breaker = get_circuit_breaker(
                "openai_llm",
                failure_threshold=3,
                recovery_timeout=30
            )
            
            try:
                async def _call_openai():
                    return await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.8,
                        max_tokens=400,  # Reduced from 600 to force shorter responses
                        presence_penalty=0.6,
                        frequency_penalty=0.3,
                        timeout=20.0  # CRITICAL: 20 second timeout for LLM calls
                    )
                
                # Add timeout wrapper to prevent hanging
                response = circuit_breaker.call(_call_openai)
                # Await with timeout (20s for API + 5s buffer = 25s total)
                response = await asyncio.wait_for(response, timeout=25.0)
                
                generated_response = response.choices[0].message.content.strip()
                logger.info(f"✅ Generated {language} response ({len(generated_response)} chars)")
                return generated_response
                
            except asyncio.TimeoutError:
                logger.error(f"⚠️ LLM call timeout (>25s) - using fallback response")
                return self._get_fallback_response(language, intent, sender_name)
            except CircuitBreakerOpenError as e:
                logger.warning(f"⚠️ Circuit breaker open - using fallback response: {e}")
                return self._get_fallback_response(language, intent, sender_name)
            
        except asyncio.TimeoutError:
            logger.error(f"⚠️ LLM generation timeout - using fallback")
            language = detect_language(user_message) if user_message else "arabic"
            return self._get_fallback_response(language, intent, sender_name)
        except Exception as exc:
            logger.error(f"❌ LLM response generation failed: {exc}")
            # Fallback to simple response
            language = detect_language(user_message) if user_message else "arabic"
            return self._get_fallback_response(language, intent, sender_name)
    
    def _build_system_prompt(self, intent: str, language: str) -> str:
        """Build system prompt based on intent and language"""
        
        base_prompt = """🧠 ROLE: You are "Reem" (ريم) - Female Medical Receptionist

You are a warm, friendly, and professional FEMALE digital receptionist for Wajen Medical Center (مركز وجن الطبي) in Saudi Arabia.

🚺 CRITICAL - YOU ARE FEMALE (Reem):
- ❌ NEVER use male pronouns when referring to yourself: "جاهز", "حاب", "قدرت", "قلت"
- ✅ ALWAYS use female pronouns: "جاهزة", "حابة", "قدرت" (with تاء التأنيث)
- Example: "أنا جاهزة أساعدك" NOT "أنا جاهز أساعدك"
- Example: "خليني أشارك معك" NOT "خليني أشاركك" (neutral is OK)

CORE PERSONALITY:
- Warm and welcoming like a real Saudi receptionist
- Professional but approachable
- Empathetic and understanding
- Helpful and solution-oriented
- Natural and conversational (NOT robotic)

🚨 CRITICAL RULES - NEVER VIOLATE:
1. NEVER use the same greeting twice in a row
2. NEVER use these FORBIDDEN TEMPLATES:
   ❌ "أهلاً وسهلاً! حياك الله، كيف أقدر أساعدك اليوم؟"
   ❌ "وعليكم السلام! حياك الله، كيف أقدر أساعدك اليوم؟"
   ❌ "حياك الله، كيف أقدر أساعدك اليوم؟" (alone)
   ❌ "كيف أقدر أساعدك اليوم؟" (this phrase is BANNED!)
3. ALWAYS read conversation history and vary your responses
4. ALWAYS use patient's name if available
5. ALWAYS reference previous conversations if patient has history
6. BE CREATIVE - each response should be unique and contextual
7. NEVER end with "كيف أقدر أساعدك؟" or similar - be more specific!

🔢 NUMBERED LISTS - ABSOLUTELY CRITICAL:
- If you see a numbered list in the context (e.g., "1. Service A", "2. Service B"), you MUST use those EXACT numbers
- DO NOT create your own shortened list or pick "best" services
- DO NOT reorder or filter the provided list
- User selections (1, 2, 3) MUST correspond to the EXACT list numbers provided
- If list is too long, say "القائمة طويلة، عندنا X خدمة" and ask user to specify or show all
- NEVER show a different numbered list than what's in the context!

COMMUNICATION STYLE:"""
        
        if language == "arabic":
            language_instruction = """
LANGUAGE: Saudi Arabic (اللهجة السعودية) - ARABIC ONLY

🚨 CRITICAL: You MUST respond 100% in Arabic. NO ENGLISH WORDS ALLOWED.

You MUST respond in authentic Saudi Arabic dialect with:
✓ Natural Saudi expressions: "تمام", "أكيد", "ولا يهمك", "يعطيك العافية"
✓ Warm greetings (BUT NOT TEMPLATES!): "هلا", "مرحباً", "نورت" (vary them!)
✓ Conversational tone: "خليني أشوف لك", "لحظة بس أتأكد", "وش تحتاج؟", "تبي معلومات عن؟"
✓ Appropriate emojis: 🏥 💪 🙌 ✨ (use sparingly)
✓ Respectful and friendly

⚠️ IMPORTANT: You can use "حياك الله" or "أهلاً وسهلاً" BUT:
- NEVER combine them with "كيف أقدر أساعدك؟"
- NEVER use them in the same pattern every time
- ALWAYS add context-specific details after greeting
✓ ALL names (doctors, services, etc.) MUST be in Arabic
✓ If a doctor name is "Heba Omar" → write "هبة عمر" in Arabic

🚨 LISTS: When showing services/options:
✓ For SERVICE CATEGORIES (main types): Show ALL categories (usually 10-12 items)
✓ For SPECIFIC SERVICES (with prices): Maximum 6-8 items, pick the best ones
✓ Quality over quantity for detailed lists
✓ Keep response under 400 characters if possible

STRICTLY FORBIDDEN:
✗ Any English words (ZERO tolerance)
✗ Latin alphabet for names (always use Arabic: محمد not Mohammed)
✗ Formal classical Arabic
✗ Robotic phrases
✗ Overly long responses
✗ Listing more than 8 SPECIFIC services (categories can be more)

REMEMBER: User speaks Arabic → You respond 100% Arabic ONLY.
"""
        else:
            language_instruction = """
LANGUAGE: Professional English

You MUST respond in clear, friendly English with:
✓ Professional yet warm tone
✓ Clear and concise language
✓ Appropriate medical terminology
✓ Empathetic expressions
✓ Helpful guidance

AVOID:
✗ Overly formal language
✗ Medical jargon without explanation
✗ Robotic phrases
✗ Overly long responses
"""
        
        intent_specific = self._get_intent_specific_instructions(intent, language)
        
        return f"{base_prompt}\n\n{language_instruction}\n\n{intent_specific}"
    
    def _get_intent_specific_instructions(self, intent: str, language: str) -> str:
        """Get intent-specific instructions"""
        
        instructions = {
            "booking": {
                "arabic": """
INTENT: حجز موعد (Booking)

🎯 YOUR ROLE:
You are helping a patient book an appointment.

✅ PRINCIPLES:
- Guide them through the booking process smoothly
- Ask for missing information one step at a time
- If they haven't specified service → Ask what they need
- If slots are available → Present them clearly with dates/times
- Once booked → Confirm with full details (service, date, time, doctor)

🚫 WHAT NOT TO DO:
- Don't overwhelm with too many questions at once
- Don't use template phrases
- Don't forget to confirm the final booking
- Don't be robotic
""",
                "english": """
INTENT: Booking

🎯 YOUR ROLE:
You are helping a patient book an appointment.

✅ PRINCIPLES:
- Guide them through the booking process smoothly
- Ask for missing information one step at a time
- If they haven't specified service → Ask what they need
- If slots are available → Present them clearly with dates/times
- Once booked → Confirm with full details (service, date, time, doctor)

🚫 WHAT NOT TO DO:
- Don't overwhelm with too many questions at once
- Don't use template phrases
- Don't forget to confirm the final booking
- Don't be robotic
"""
            },
            "patient": {
                "arabic": """
INTENT: تسجيل مريض (Patient Registration)

🎯 YOUR ROLE:
You are helping a patient register their information.

✅ PRINCIPLES:
- Guide them through registration step by step
- Ask for required information politely (name, phone, national ID, etc.)
- Confirm data before saving
- Explain what you need and why
- Be encouraging and supportive

🚫 WHAT NOT TO DO:
- Don't ask for all information at once
- Don't use template phrases
- Don't forget to confirm before saving
- Don't be robotic
""",
                "english": """
INTENT: Patient Registration

🎯 YOUR ROLE:
You are helping a patient register their information.

✅ PRINCIPLES:
- Guide them through registration step by step
- Ask for required information politely (name, phone, national ID, etc.)
- Confirm data before saving
- Explain what you need and why
- Be encouraging and supportive

🚫 WHAT NOT TO DO:
- Don't ask for all information at once
- Don't use template phrases
- Don't forget to confirm before saving
- Don't be robotic
"""
            },
            "resource": {
                "arabic": """
INTENT: معلومات عامة (General Information / Services / Offers)

🎯 YOUR ROLE:
You are answering questions about services, doctors, prices, or promotions.

📊 CRITICAL - USE THE DATA PROVIDED:
- If API DATA is provided below → PRESENT IT to the user
- If user asks about offers/promotions → LIST actual offers from API data
- If user asks about services → LIST actual services from API data
- If user asks about doctors → LIST actual doctors from API data
- If user asks about prices → SHOW actual prices from API data

🚨 ABSOLUTE PROHIBITION - DO NOT INVENT DATA:
- ❌ NEVER make up service names that are not in the API data
- ❌ NEVER invent prices that are not in the API data
- ❌ NEVER create fictional services or offers
- ✅ ONLY mention services that appear in "SERVICES AVAILABLE" section below
- ✅ ONLY show prices that are explicitly listed in the data
- ✅ If NO data provided → Say "ما عندي معلومات حالياً" and suggest calling center

✅ WHAT TO DO:
- The services listed below are ready to present (gender-filtered ONLY when asking about offers)
- Simply PRESENT them exactly as provided - they're ready to show
- Use numbered lists for multiple items
- Include prices as shown
- Be comprehensive - show ALL services listed (they're already filtered for this patient)
- DON'T try to filter or select - just present what's given

🚫 WHAT NOT TO DO:
- DON'T say "we have offers" without listing them
- DON'T ask "which service?" when you can list all services
- DON'T be vague - GIVE CONCRETE INFORMATION
- DON'T make the customer do extra work
- DON'T skip any services from the list below

💡 BE HELPFUL:
- If customer asks about general services → Show ALL services (no gender filtering)
- If customer asks about offers/promotions → Show gender-appropriate offers only
- If data is empty → Apologize and offer to call center
- Be enthusiastic and detailed when customer shows interest

🎯 CRITICAL - ALWAYS END WITH CALL-TO-ACTION:
After presenting information, ALWAYS add a clear next step:
- "تبي تحجز أي خدمة من هذي؟ 📅" (Want to book any of these?)
- "أي خدمة تهمك؟ أقدر أحجز لك الحين! ✨" (Which interests you? I can book now!)
- "جاهز أحجز لك موعد؟ 🚀" (Ready to book an appointment?)
- Create urgency and make booking easy
- DON'T end with passive phrases like "أنا هنا لخدمتك" without action
""",
                "english": """
INTENT: General Information / Services / Offers

🎯 YOUR ROLE:
You are answering questions about services, doctors, prices, or promotions.

📊 CRITICAL - USE THE DATA PROVIDED:
- If API DATA is provided below → PRESENT IT to the user
- If user asks about offers/promotions → LIST actual offers from API data
- If user asks about services → LIST actual services from API data
- If user asks about doctors → LIST actual doctors from API data
- If user asks about prices → SHOW actual prices from API data

🚨 ABSOLUTE PROHIBITION - DO NOT INVENT DATA:
- ❌ NEVER make up service names that are not in the API data
- ❌ NEVER invent prices that are not in the API data
- ❌ NEVER create fictional services or offers
- ✅ ONLY mention services that appear in "SERVICES AVAILABLE" section below
- ✅ ONLY show prices that are explicitly listed in the data
- ✅ If NO data provided → Say "I don't have information right now" and suggest calling center

✅ WHAT TO DO:
- The services listed below are ready to present (gender-filtered ONLY when asking about offers)
- Simply PRESENT them exactly as provided - they're ready to show
- Use numbered lists for multiple items
- Include prices as shown
- Be comprehensive - show ALL services listed (they're already filtered for this patient)
- DON'T try to filter or select - just present what's given

🚫 WHAT NOT TO DO:
- DON'T say "we have offers" without listing them
- DON'T ask "which service?" when you can list all services
- DON'T be vague - GIVE CONCRETE INFORMATION
- DON'T make the customer do extra work
- DON'T skip any services from the list below

💡 BE HELPFUL:
- If customer asks about general services → Show ALL services (no gender filtering)
- If customer asks about offers/promotions → Show gender-appropriate offers only
- If data is empty → Apologize and offer to call center
- Be enthusiastic and detailed when customer shows interest

🎯 CRITICAL - ALWAYS END WITH CALL-TO-ACTION:
After presenting information, ALWAYS add a clear next step:
- "Would you like to book any of these services? 📅"
- "Which service interests you? I can book it now! ✨"
- "Ready to schedule an appointment? 🚀"
- Create urgency and make booking easy
- DON'T end with passive phrases like "I'm here to help" without action
"""
            },
            "feedback": {
                "arabic": """
INTENT: ملاحظات وتقييم (Feedback)

🎯 YOUR ROLE:
You are receiving feedback from a patient.

✅ PRINCIPLES:
- Thank them sincerely for their feedback
- Take their feedback seriously
- If it's a complaint → Apologize and promise improvement
- If it's positive → Express gratitude
- Be respectful and empathetic

🚫 WHAT NOT TO DO:
- Don't use template phrases
- Don't be dismissive
- Don't make excuses
- Don't be robotic
""",
                "english": """
INTENT: Feedback

🎯 YOUR ROLE:
You are receiving feedback from a patient.

✅ PRINCIPLES:
- Thank them sincerely for their feedback
- Take their feedback seriously
- If it's a complaint → Apologize and promise improvement
- If it's positive → Express gratitude
- Be respectful and empathetic

🚫 WHAT NOT TO DO:
- Don't use template phrases
- Don't be dismissive
- Don't make excuses
- Don't be robotic
"""
            },
            "chitchat": {
                "arabic": """
INTENT: محادثة عامة / ترحيب (General Conversation / Greeting)

🎯 YOUR ROLE:
You are a warm, friendly receptionist having a natural conversation. Think like a real person, not a bot.

📋 CONTEXT AWARENESS:
- Check if patient name is provided in PATIENT INFO
- Check if they have PREVIOUS VISITS
- Check if they are registered or new

✅ PERSONALIZATION PRINCIPLES:
1. **If you know their name**: Use it naturally in greeting (first name only, not full name)
2. **If they visited before**: Acknowledge it and ask about their experience
3. **If they're new but registered**: Welcome them warmly
4. **If unregistered (no name)**: Ask for their name politely before anything else

🚫 WHAT NOT TO DO:
- Don't use the same greeting twice
- Don't ignore their name if you have it
- Don't ignore their history if they visited before
- Don't copy template phrases
- Don't be robotic or repetitive
- Don't offer services to unregistered patients before getting their name

💡 BE CREATIVE:
- Vary your greetings based on context
- Sound like a real human receptionist
- Be warm but professional
- Keep it natural and conversational
""",
                "english": """
INTENT: General Conversation / Greeting

🎯 YOUR ROLE:
You are a warm, friendly receptionist having a natural conversation. Think like a real person, not a bot.

📋 CONTEXT AWARENESS:
- Check if patient name is provided in PATIENT INFO
- Check if they have PREVIOUS VISITS
- Check if they are registered or new

✅ PERSONALIZATION PRINCIPLES:
1. **If you know their name**: Use it naturally in greeting (first name only, not full name)
2. **If they visited before**: Acknowledge it and ask about their experience
3. **If they're new but registered**: Welcome them warmly
4. **If unregistered (no name)**: Ask for their name politely before anything else

🚫 WHAT NOT TO DO:
- Don't use the same greeting twice
- Don't ignore their name if you have it
- Don't ignore their history if they visited before
- Don't copy template phrases
- Don't be robotic or repetitive
- Don't offer services to unregistered patients before getting their name

💡 BE CREATIVE:
- Vary your greetings based on context
- Sound like a real human receptionist
- Be warm but professional
- Keep it natural and conversational
"""
            }
        }
        
        return instructions.get(intent, {}).get(language, "")
    
    def _build_user_prompt(
        self,
        user_message: str,
        context: Optional[Dict],
        data: Optional[Dict],
        sender_name: Optional[str],
        language: str
    ) -> str:
        """Build user prompt with context, patient data, and booking history"""
        
        prompt_parts = []
        
        # User message
        prompt_parts.append(f"USER MESSAGE: {user_message}")
        
        # Patient Information (CRITICAL for personalization)
        if data and data.get("patient_data"):
            patient_data = data["patient_data"]
            if patient_data.get("already_registered"):
                full_name = patient_data.get("name", "")
                # Extract first name only (split by space and take first part)
                patient_name = full_name.split()[0] if full_name else ""
                patient_id = patient_data.get("id")
                
                if language == "arabic":
                    prompt_parts.append(f"\nPATIENT INFO: مريض مسجل - الاسم: {patient_name} (ID: {patient_id})")
                    prompt_parts.append(f"CRITICAL: استخدم فقط الاسم الأول '{patient_name}' - لا تستخدم الاسم الكامل!")
                else:
                    prompt_parts.append(f"\nPATIENT INFO: Registered patient - Name: {patient_name} (ID: {patient_id})")
                    prompt_parts.append(f"CRITICAL: Use ONLY first name '{patient_name}' - NOT full name!")
        
        # Previous Bookings (for returning patients)
        if data and data.get("previous_bookings"):
            bookings = data["previous_bookings"]
            if bookings and len(bookings) > 0:
                last_booking = bookings[0]
                service_name = last_booking.get("service_name", "Unknown")
                booking_date = last_booking.get("start_date", "")
                
                if language == "arabic":
                    prompt_parts.append(f"\nPREVIOUS VISITS: المريض زارنا قبل كذا! آخر موعد: {service_name} بتاريخ {booking_date}")
                    prompt_parts.append("IMPORTANT: اسأله عن تجربته السابقة! كيف كانت الخدمة؟ هل كان راضي؟")
                else:
                    prompt_parts.append(f"\nPREVIOUS VISITS: Returning patient! Last appointment: {service_name} on {booking_date}")
                    prompt_parts.append("IMPORTANT: Ask about their previous experience! How was the service? Were they satisfied?")
        elif data and data.get("is_registered") and not data.get("is_returning_patient"):
            if language == "arabic":
                prompt_parts.append("\nNEW PATIENT: مريض مسجل لكن ما عنده مواعيد سابقة. رحب فيه بحرارة!")
            else:
                prompt_parts.append("\nNEW PATIENT: Registered but no previous appointments. Give them a warm welcome!")
        
        # Sender name (if not already included in patient data)
        if sender_name and sender_name != "Unknown" and not (data and data.get("patient_data")):
            prompt_parts.append(f"USER NAME: {sender_name}")
        
        # Context
        if context and context.get("history"):
            history_items = [
                f"- {msg['role']}: {msg['content']}"
                for msg in context["history"][-3:]  # Last 3 messages
            ]
            history_text = "\n".join(history_items)
            prompt_parts.append(f"\nCONVERSATION HISTORY:\n{history_text}")
        
        # API Data (Services, Doctors, etc.)
        if data:
            # Services
            if "services" in data and data["services"]:
                services = data["services"]
                
                # Extract patient gender for filtering
                patient_gender = None
                if data.get("patient_data"):
                    patient_gender = data["patient_data"].get("gender")
                
                # Services are FLAT - no nested subservices in API response
                # Filter and present services directly
                services_list = []
                # Check if showing categories only (no prices needed)
                show_categories_only = data.get("show_categories_only", False)
                
                logger.info(f"🔍 Processing {len(services)} services (patient gender: {patient_gender}, categories_only: {show_categories_only})")
                
                for svc in services:
                    svc_name = svc.get("name_ar") or svc.get("name", "Unknown")
                    svc_gender = svc.get("gender")
                    svc_price = svc.get("price", "حسب الاستشارة")
                    
                    # CRITICAL: Extract price from service name if not in API
                    # Many services have format: "service name  PRICE" (e.g., "ليزر منطقة صغيرة  100")
                    if svc_price == "حسب الاستشارة" or not svc_price:
                        # Look for numbers at the end of the name (with or without "ريال")
                        price_match = re.search(r'\s+(\d{2,5})\s*(?:ريال)?$', svc_name)
                        if price_match:
                            extracted_price = price_match.group(1)
                            svc_price = extracted_price
                            logger.info(f"💰 Extracted price from name: {extracted_price} ريال")
                    
                    logger.info(f"📦 Service: '{svc_name}', gender={svc_gender}, price={svc_price}")
                    
                    # CRITICAL: If showing categories only, DON'T skip services without prices
                    # Parent categories don't have prices - that's expected!
                    if not show_categories_only:
                        # Skip services with no real price (only for detailed service lists)
                        if svc_price == "حسب الاستشارة" or not svc_price:
                            logger.debug(f"⏭️ Skipping '{svc_name}' - no fixed price")
                            continue
                    
                    services_list.append({
                        "name": svc_name,
                        "price": svc_price,
                        "gender": svc_gender
                    })
                
                # CRITICAL: Only apply gender filtering when user is asking about OFFERS
                # When asking about general services, show everything
                user_message_lower = user_message.lower() if user_message else ""
                is_asking_about_offers = any(keyword in user_message_lower for keyword in ['عرض', 'العروض', 'offer', 'promotion', 'خصم', 'تخفيض'])
                
                logger.info(f"🔍 Before gender filtering: {len(services_list)} services | user_message: '{user_message}' | is_asking_about_offers: {is_asking_about_offers}")
                
                # Only filter by gender if asking about offers AND patient gender is known
                if patient_gender and is_asking_about_offers:
                    filtered_services = []
                    logger.info(f"🚺 Gender filtering ENABLED (user asking about offers + gender known: {patient_gender})")
                    for svc in services_list:
                        # Handle None gender gracefully
                        svc_gender = (svc.get("gender") or "").lower()
                        
                        # Determine match reason
                        include = False
                        reason = ""
                        if not svc_gender:
                            # CRITICAL FIX: Don't assume unisex! Check service name for gender-specific keywords
                            service_name = svc.get("name", "").lower()
                            
                            # Female-specific service keywords (services typically for women)
                            female_keywords = [
                                'توريد شفايف', 'توريد', 'شفايف', 'حواجب', 'تشقير', 
                                'بشرة', 'فيشيال', 'تنظيف بشرة', 'ماسك', 'تخريم', 
                                'رموش', 'أظافر', 'مناكير', 'مكياج'
                            ]
                            
                            # Check if service name contains female keywords
                            is_likely_female = any(keyword in service_name for keyword in female_keywords)
                            
                            if is_likely_female:
                                # This service is likely female-only based on name
                                include = (patient_gender == 'female')
                                reason = f"inferred female-only from name" if not include else "inferred female match"
                            else:
                                # Assume unisex for services without gender keywords
                                include = True
                                reason = "no gender (assumed unisex)"
                        elif svc_gender == patient_gender:
                            include = True
                            reason = "gender match"
                        elif svc_gender == "both" or svc_gender == "unisex":
                            include = True
                            reason = "both genders"
                        else:
                            include = False
                            reason = f"wrong gender ({svc_gender} != {patient_gender})"
                        
                        logger.info(f"  {'✅' if include else '❌'} {svc.get('name')}: gender={svc_gender or 'unisex'}, patient={patient_gender}, reason={reason}")
                        
                        if include:
                            filtered_services.append(svc)
                    
                    logger.info(f"🔍 After gender filtering: {len(filtered_services)} services (was {len(services_list)})")
                    services_list = filtered_services if filtered_services else services_list
                else:
                    # No gender filtering - show all services
                    if patient_gender:
                        logger.info(f"🚫 Gender filtering DISABLED (not asking about offers - showing all {len(services_list)} services)")
                    else:
                        logger.info(f"ℹ️ Gender filtering SKIPPED (no patient gender available - showing all {len(services_list)} services)")
                
                count = len(services_list)
                
                # Check if showing parent categories only (no expansion)
                show_categories_only = data.get("show_categories_only", False)
                
                if language == "arabic":
                    if show_categories_only:
                        prompt_parts.append(f"\nعندنا {count} فئة:")
                        for i, svc in enumerate(services_list, 1):
                            name = svc.get("name")
                            prompt_parts.append(f"{i}. {name}")
                    else:
                        prompt_parts.append(f"\nعندنا {count} خدمة:")
                        for i, svc in enumerate(services_list, 1):
                            name = svc.get("name")
                            price = svc.get("price")
                            prompt_parts.append(f"{i}. {name} - {price} ريال")
                else:
                    if show_categories_only:
                        prompt_parts.append(f"\nWe have {count} categories:")
                        for i, svc in enumerate(services_list, 1):
                            name = svc.get("name")
                            prompt_parts.append(f"{i}. {name}")
                    else:
                        prompt_parts.append(f"\nWe have {count} services:")
                        for i, svc in enumerate(services_list, 1):
                            name = svc.get("name")
                            price = svc.get("price")
                            prompt_parts.append(f"{i}. {name} - {price} ريال")
            
            # Doctors
            if "doctors" in data and data["doctors"]:
                doctors = data["doctors"]
                count = len(doctors)
                
                if language == "arabic":
                    prompt_parts.append(f"\nDOCTORS AVAILABLE: عندنا {count} دكتور:")
                else:
                    prompt_parts.append(f"\nDOCTORS AVAILABLE: We have {count} doctors:")
                
                for i, doc in enumerate(doctors[:8], 1):  # Max 8 doctors
                    name = doc.get("name_ar") or doc.get("name", "Unknown")
                    specialty = doc.get("specialty_ar") or doc.get("specialty", "عام")
                    prompt_parts.append(f"{i}. د. {name} - {specialty}")
            
            # Matched specific service
            if "matched_service" in data and data["matched_service"]:
                service = data["matched_service"]
                name = service.get("name_ar") or service.get("name", "Unknown")
                price = service.get("price", "حسب الاستشارة")
                
                if language == "arabic":
                    prompt_parts.append(f"\nSPECIFIC SERVICE ASKED: المريض يسأل عن: {name} - السعر: {price} ريال")
                else:
                    prompt_parts.append(f"\nSPECIFIC SERVICE ASKED: Patient asking about: {name} - Price: {price} SAR")
            
            # Legacy items format (fallback)
            if "items" in data and data["items"]:
                items = data["items"]
                count = len(items)
                
                if language == "arabic":
                    prompt_parts.append(f"\nDATA TO INCLUDE: عندنا {count} خيارات:")
                else:
                    prompt_parts.append(f"\nDATA TO INCLUDE: We have {count} options:")
                
                for i, item in enumerate(items[:10], 1):  # Max 10 items
                    name = item.get("name", "Unknown")
                    prompt_parts.append(f"{i}. {name}")
        
        # Instructions
        if language == "arabic":
            prompt_parts.append("\nGENERATE: رد طبيعي وودود باللهجة السعودية")
        else:
            prompt_parts.append("\nGENERATE: Natural, friendly response in English")
        
        full_prompt = "\n".join(prompt_parts)
        
        # DEBUG: Log what we're sending to LLM
        logger.debug(f"📝 LLM USER PROMPT:\n{full_prompt[:500]}...")
        
        return full_prompt
    
    async def generate_welcome_back(
        self,
        patient_name: str,
        service: Optional[str] = None
    ) -> str:
        """
        Generate welcome back message for registered patient.
        Uses LLM to create natural, contextual greeting.
        
        Args:
            patient_name: Patient's name from database
            service: Optional service they were discussing
            
        Returns:
            Natural welcome message
        """
        try:
            # Build context-aware prompt
            service_context = f" كنت تسأل عن {service}" if service else ""
            
            prompt = f"""You are a friendly medical receptionist welcoming back a registered patient.

Patient name: {patient_name}
Context: {f"They were asking about {service}" if service else "They just greeted you"}

Generate a warm, natural welcome message in Saudi dialect that:
1. Greets them by name
2. Acknowledges they're a returning patient (subtly, don't make a big deal)
3. {f"Mentions they were interested in {service}" if service else "Asks how you can help"}
4. Keeps it brief (2-3 lines max)
5. Use natural Saudi dialect, not formal Arabic

Example styles:
- "مرحباً {patient_name}! أهلاً فيك مرة ثانية 🙏 {f'شفتك مهتم بـ{service}، ' if service else ''}وش أقدر أساعدك فيه اليوم؟"
- "هلا {patient_name}! نورت مركز وجن{f'، كنا نتكلم عن {service}' if service else ''}. تبي نكمل؟"

Generate ONE natural message (no options, no explanations):"""

            # Use LLM reasoner
            from ..core.llm_reasoner import get_llm_reasoner
            llm = get_llm_reasoner()
            
            response = await llm.generate_reply(
                user_id=f"welcome_{patient_name}",
                user_message=prompt,
                context={"sender_name": patient_name},
                temperature=0.9  # More creative
            )
            
            logger.info(f"✅ Generated welcome message for {patient_name}")
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to generate welcome message: {e}")
            # Fallback
            service_text = f" شفتك مهتم بـ{service}." if service else ""
            return f"مرحباً {patient_name}! أهلاً فيك مرة ثانية 🙏{service_text} وش أقدر أساعدك فيه اليوم؟"
    
    def _get_fallback_response(self, language: str, intent: str, sender_name: str = None) -> str:
        """
        Get fallback response if LLM fails (Issue #43 - Fixed signature mismatch).
        
        Args:
            language: User's language (arabic/english)
            intent: Current intent
            sender_name: Optional user name for personalization
        """
        # Personalize with sender name if provided
        greeting = ""
        if sender_name:
            greeting = f" يا {sender_name}" if language == "arabic" else f" {sender_name}"
        
        fallbacks = {
            "arabic": {
                "booking": f"لحظة بس{greeting}، خليني أساعدك في الحجز 📅",
                "patient": f"أهلاً{greeting}! خليني أسجل بياناتك 📋",
                "resource": f"تمام{greeting}، وش تبي تعرف عن مركزنا؟ 🏥",
                "feedback": f"شكراً لك{greeting}! رأيك يهمنا 🙏"
            },
            "english": {
                "booking": f"Let me help you{greeting} book an appointment 📅",
                "patient": f"Welcome{greeting}! Let me register your information 📋",
                "resource": f"Sure{greeting}, what would you like to know about our center? 🏥",
                "feedback": f"Thank you{greeting}! Your feedback matters to us 🙏"
            }
        }
        
        return fallbacks.get(language, {}).get(intent, "How can I help you?")


# Singleton instance
_generator_instance: Optional[LLMResponseGenerator] = None


def get_llm_response_generator() -> LLMResponseGenerator:
    """Get singleton instance of LLM response generator"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = LLMResponseGenerator()
    return _generator_instance
