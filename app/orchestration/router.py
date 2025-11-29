# -*- coding: utf-8 -*-
"""
Intent Router with WhatsApp Response Flow
==========================================
Professional orchestration layer that routes messages to specialized agents
and sends responses back to WhatsApp via WaSender.

Features:
- Intent classification and routing
- Agent orchestration
- WhatsApp response sending
- Error handling and recovery
- Session management
- Retry logic for failed sends

Author: Agent Orchestrator Team
Version: 1.0.0
"""

from typing import Dict, Any, Optional
from loguru import logger
from ..config import settings
import time 

from ..agents.booking_agent import BookingAgent
from ..agents.booking_agent_langgraph import BookingAgentLangGraph
from ..agents.booking_agent_factory import BookingAgentFactory 
from ..agents.patient_agent import PatientAgent
from ..agents.feedback_agent import FeedbackAgent
from ..agents.resource_agent import ResourceAgent
from ..api.wasender_client import WaSenderClient
from ..memory.session_manager import SessionManager
from ..memory.history_cache import get_history_cache  # Redis history caching
from ..core.llm_reasoner import get_llm_reasoner
from ..core.intent_classifier import classify_intent
from ..utils.entity_extractor import get_entity_extractor
from ..utils.phone_parser import extract_generic_phone, normalize_phone_digits  # Issue #37
from ..utils.adaptive_confidence import get_confidence_manager  # Issue #10


class IntentRouter:
    """
    Professional intent router with complete WhatsApp response flow.
    Singleton pattern to avoid re-initialization.
    
    Features:
    - Intent classification and routing
    - Agent orchestration
    - WhatsApp response sending
    - Error handling and recovery
    - Session management
    - Retry logic for failed sends
    
    Confidence Score Calibration (Issue #16):
    - 0.95-1.00: Explicit keyword match or LLM high confidence
    - 0.85-0.94: Strong pattern match or LLM medium confidence
    - 0.70-0.84: Contextual inference or LLM low confidence
    - 0.50-0.69: Weak signals, needs LLM validation
    """
    
    _instance = None
    _initialized = False
    _routing_failures = {}  # Track routing failures per session for progressive error messages
    _error_patterns = {}  # Track repeated error patterns per session (Issue #17)
    _agent_circuit_breakers = {}  # Track agent failures for circuit breaker (Issue #24)
    _intent_cache = {}  # Per-session intent cache for repeated messages (Issue #36)
    
    # RETRY STRATEGY CONFIGURATION (Issue #33)
    RETRY_MAX_ATTEMPTS = 2  # Max retries (total attempts = 3)
    RETRY_BACKOFF_MS = [100, 500]  # Backoff delays in milliseconds
    RETRY_ON_ERRORS = ["ConnectionError", "TimeoutError", "TemporaryFailure"]  # Retryable error types
    
    def __new__(cls, session_key: str = None):
        if cls._instance is None:
            cls._instance = super(IntentRouter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, session_key: str = None):
        # Only initialize once
        if IntentRouter._initialized:
            return  # Silent reuse - no logging noise
        
        self.session_key = session_key or "default_session"
        self.wasender_client = WaSenderClient()
        self.session_manager = SessionManager()
        self.llm_reasoner = get_llm_reasoner()
        
        # Redis history cache (lazy initialization)
        self.history_cache = None
        
        IntentRouter._initialized = True
        logger.info("✅ IntentRouter initialized with LLM reasoning layer (singleton, history cache: lazy)")
    
    def _get_history_cache(self):
        """Lazy initialize Redis history cache"""
        if self.history_cache is None:
            self.history_cache = get_history_cache(self.session_manager.redis)
            logger.info("✅ HistoryCache initialized (lazy loading)")
        return self.history_cache
    
    def _track_error_pattern(self, session_key: str, error_signature: str) -> int:
        """
        Track repeated error patterns for duplicate detection (Issue #17).
        
        Args:
            session_key: Session identifier
            error_signature: Unique signature of the error (type + message hash)
            
        Returns:
            Number of times this error pattern has occurred
        """
        import hashlib
        
        current_time = time.time()
        
        if session_key not in IntentRouter._error_patterns:
            IntentRouter._error_patterns[session_key] = {}
        
        session_errors = IntentRouter._error_patterns[session_key]
        
        # Clean old errors (older than 5 minutes)
        expired_patterns = [
            sig for sig, data in session_errors.items()
            if current_time - data["last_occurrence"] > 300
        ]
        for sig in expired_patterns:
            del session_errors[sig]
        
        # Track this error
        if error_signature not in session_errors:
            session_errors[error_signature] = {
                "count": 0,
                "first_occurrence": current_time,
                "last_occurrence": current_time
            }
        
        session_errors[error_signature]["count"] += 1
        session_errors[error_signature]["last_occurrence"] = current_time
        
        error_count = session_errors[error_signature]["count"]
        time_span = current_time - session_errors[error_signature]["first_occurrence"]
        
        logger.warning(f"🔁 ERROR PATTERN TRACKER: {error_signature[:30]}... occurred {error_count} times in {time_span:.0f}s")
        
        return error_count
    
    def _should_escalate_repeated_error(self, session_key: str, error_signature: str) -> bool:
        """
        Check if repeated error should trigger escalation (Issue #17).
        
        Returns:
            True if 3+ same errors within 2 minutes
        """
        if session_key not in IntentRouter._error_patterns:
            return False
        
        session_errors = IntentRouter._error_patterns[session_key]
        if error_signature not in session_errors:
            return False
        
        error_data = session_errors[error_signature]
        time_span = time.time() - error_data["first_occurrence"]
        
        # Escalate if 3+ occurrences within 2 minutes
        if error_data["count"] >= 3 and time_span < 120:
            logger.error(f"🚨 REPEATED ERROR ESCALATION: {error_signature[:30]}... occurred {error_data['count']} times in {time_span:.0f}s")
            return True
        
        return False
    
    def _track_agent_failure(self, agent_name: str) -> int:
        """
        Track agent failures for circuit breaker (Issue #24).
        
        Args:
            agent_name: Name of the failing agent
            
        Returns:
            Number of consecutive failures
        """
        if agent_name not in IntentRouter._agent_circuit_breakers:
            IntentRouter._agent_circuit_breakers[agent_name] = {
                "failures": 0,
                "last_failure": 0,
                "circuit_open": False,
                "open_until": 0
            }
        
        breaker = IntentRouter._agent_circuit_breakers[agent_name]
        breaker["failures"] += 1
        breaker["last_failure"] = time.time()
        
        # Open circuit after 3 failures
        if breaker["failures"] >= 3:
            breaker["circuit_open"] = True
            breaker["open_until"] = time.time() + 300  # 5 minutes
            logger.error(f"🚨 CIRCUIT BREAKER OPENED: {agent_name} disabled for 5 minutes after {breaker['failures']} failures")
        
        return breaker["failures"]
    
    def _reset_agent_circuit_breaker(self, agent_name: str):
        """Reset circuit breaker after successful agent call"""
        if agent_name in IntentRouter._agent_circuit_breakers:
            logger.info(f"✅ CIRCUIT BREAKER RESET: {agent_name} back to normal")
            del IntentRouter._agent_circuit_breakers[agent_name]
    
    def _is_agent_circuit_open(self, agent_name: str) -> tuple[bool, str]:
        """
        Check if agent circuit breaker is open (Issue #24).
        
        Returns:
            (is_open, reason)
        """
        if agent_name not in IntentRouter._agent_circuit_breakers:
            return False, ""
        
        breaker = IntentRouter._agent_circuit_breakers[agent_name]
        
        # Check if circuit should be closed (timeout expired)
        if breaker["circuit_open"] and time.time() > breaker["open_until"]:
            logger.info(f"🔄 CIRCUIT BREAKER CLOSING: {agent_name} timeout expired, allowing retry")
            del IntentRouter._agent_circuit_breakers[agent_name]
            return False, ""
        
        if breaker["circuit_open"]:
            time_remaining = int(breaker["open_until"] - time.time())
            return True, f"{agent_name} temporarily disabled ({time_remaining}s remaining)"
        
        return False, ""
    
    def _build_router_error_response(self, session_key: str, sender_name: str = "", error_type: str = "routing", language: str = None) -> str:
        """
        Build progressive error response based on failure count.
        Provides safe alternatives that don't suggest currently failing actions.
        Uses detected language for localization (Issue #25).
        Names are sanitized before use (Issue #40).
        
        Args:
            session_key: Session identifier
            sender_name: User's name for personalization (will be sanitized)
            error_type: Type of error (routing, agent, timeout)
            language: User's preferred language (arabic/english), auto-detected if None
        
        Returns:
            Error message with progressive guidance
        """
        # CRITICAL: Sanitize username to prevent broken error messages (Issue #40)
        clean_name = self._sanitize_username(sender_name)
        if not clean_name:
            # If sanitization fails, use empty (LLM will handle)
            clean_name = ""
            logger.debug(f"Sanitization failed for '{sender_name}' - LLM will handle naturally")
        sender_name = clean_name  # Use sanitized name
        
        # CRITICAL: Get user's language preference (Issue #25)
        if language is None:
            language = "arabic"  # Default fallback
        
        # Track failure
        if session_key not in IntentRouter._routing_failures:
            IntentRouter._routing_failures[session_key] = {"count": 0, "last_failure": 0, "error_type": error_type}
        
        IntentRouter._routing_failures[session_key]["count"] += 1
        IntentRouter._routing_failures[session_key]["last_failure"] = time.time()
        IntentRouter._routing_failures[session_key]["error_type"] = error_type
        
        failure_count = IntentRouter._routing_failures[session_key]["count"]
        logger.warning(f"⚠️ ROUTER FAILURE TRACKER: Session {session_key} has {failure_count} consecutive failures (type={error_type}, lang={language})")
        
        # Progressive error messages - localized by language (Issue #25)
        if language == "english":
            # English error messages
            if failure_count == 1:
                return f"""Hi {sender_name} 😅

A small error occurred in the system

**Try these alternatives:**
• Type 'services' to view our services
• Type 'my bookings' to check your appointments
• Or call us directly: 920033304 📞

Please check your internet connection ✓"""
            
            elif failure_count == 2:
                return f"""{sender_name}, sorry for the issue 😞

The system has a temporary problem

**Best options:**
• Call directly: 920033304 📞
• Or try again in 5 minutes

Please don't repeat the same request - try something different 🔄

We apologize for the inconvenience 🙏"""
            
            else:
                return f"""Dear {sender_name} 😔

There's an ongoing technical issue

**Best solution right now:**
📞 **Call: 920033304**

Our team is ready to help you directly
From 9 AM to 9 PM

Please try again after an hour

Sorry for the inconvenience """

        else:
            # Arabic error messages (default) - Avoid double يا (Issue #45)
            # If sender_name already contains يا, don't add another
            name_prefix = "" if sender_name.startswith("يا") else "يا "
            
            if failure_count == 1:
                # Issue #7: Better error message - don't blame user, provide alternatives
                return f"""يا عيني {name_prefix}{sender_name} 🙏

عذراً، نظام الحجز مشغول حالياً

**ممكن تجرب:**
• اتصل على: 920033304 📞
• أو جرب مرة ثانية بعد دقيقة

نعتذر عن الإزعاج"""

            elif failure_count == 2:
                return f"""{name_prefix}{sender_name}، آسف على المشكلة 

النظام عنده مشكلة مؤقتة

**أفضل الخيارات:**
• اتصل مباشرة: 920033304 
• أو جرب بعد 5 دقائق

لا تعيد نفس الطلب - جرب خيار ثاني 🔄

نعتذر عن الإزعاج 🙏"""
            
            else:
                return f"""عزيزي {sender_name} 😔

في مشكلة تقنية مستمرة

**الحل الأفضل الحين:**
📞 **اتصل: 920033304**

فريقنا جاهز يساعدك مباشرة
من ٩ صباحاً إلى ٩ مساءً

حاول مرة ثانية بعد ساعة

معذرة على الإزعاج 💔"""
    
    def _get_user_name(self, payload: dict, session_data: dict, entities: dict, message_text: str = "") -> str:
        """
        Get user name with proper precedence and sanitization (Issue #26).
        1. Patient name from database (registered patient)
        2. Confirmed Arabic name (verified by user in Arabic context)
        3. Extracted Arabic name (from current message if user is speaking Arabic)
        4. Contact pushName (only if no Arabic alternative)
        
        All names are sanitized to remove diacritics and validate.
        
        Args:
            payload: Message payload
            session_data: Session data
            entities: Extracted entities
            message_text: Current message text for language detection
            
        Returns:
            User name to use (sanitized)
        """
        # Detect if user is speaking Arabic
        message_language = self._detect_message_language(message_text)
        
        # Priority 1: Patient name from database (HIGHEST PRIORITY)
        patient_data = session_data.get('patient_data')
        if patient_data and patient_data.get('already_registered') and patient_data.get('name'):
            db_name = patient_data.get('name', '').strip()
            # Use first name only for natural conversation (more friendly than full name)
            first_name = db_name.split()[0] if db_name else ''
            if first_name:
                sanitized_db_name = self._sanitize_username(first_name)
                if sanitized_db_name:
                    logger.debug(f"Using patient FIRST NAME from database: {sanitized_db_name}")
                    return sanitized_db_name
        
        # Priority 2: Confirmed name in session
        if session_data.get('confirmed_name'):
            confirmed = self._sanitize_username(session_data['confirmed_name'])
            if confirmed and self._is_arabic_text(confirmed):
                logger.debug(f"Using confirmed Arabic name: {confirmed}")
                return confirmed
        
        # Priority 2: Extracted name (if user is speaking Arabic and name is Arabic)
        if entities.get('name') and message_language == 'ar':
            extracted = self._sanitize_username(entities['name'])
            if extracted and self._is_arabic_text(extracted):
                logger.debug(f"Using extracted Arabic name: {extracted}")
                return extracted
        
        # Priority 3: Contact pushName (only if Latin is acceptable OR no Arabic name available)
        push_name = payload.get('sender_name') or payload.get('pushName')
        if push_name and push_name not in ['Unknown', '', None]:
            sanitized_push = self._sanitize_username(push_name)
            
            # Skip if sanitization failed
            if not sanitized_push:
                logger.debug(f"Invalid pushName '{push_name}' - LLM will handle naturally")
            # If user is speaking Arabic, don't use Latin pushName
            elif message_language == 'ar' and not self._is_arabic_text(sanitized_push):
                logger.debug(f"Skipping Latin pushName '{sanitized_push}' in Arabic context")
            else:
                logger.debug(f"Using sanitized pushName: {sanitized_push}")
                return sanitized_push
        
        # No valid name - return empty, let LLM handle it naturally
        logger.debug(f"No valid name available - LLM will handle naturally")
        return ""
    
    def _detect_message_language(self, text: str) -> str:
        """Detect if message is Arabic or English"""
        if not text:
            return 'ar'
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars = sum(1 for c in text if c.isalpha())
        if total_chars == 0:
            return 'ar'
        return 'ar' if (arabic_chars / total_chars) > 0.3 else 'en'
    
    def _is_arabic_text(self, text: str) -> bool:
        """Check if text contains primarily Arabic characters"""
        if not text:
            return False
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        latin_chars = sum(1 for c in text if c.isalpha() and 'a' <= c.lower() <= 'z')
        return arabic_chars > latin_chars
    
    def _format_price(self, price) -> str:
        """
        Format service price intelligently (Issue #47).
        
        Only shows "حسب الاستشارة" if explicitly set by API.
        Otherwise shows the actual price regardless of format.
        
        Args:
            price: Price value (can be int, float, string, or None)
            
        Returns:
            Formatted price string
        """
        if price is None or price == "" or price == "null":
            # No price provided - don't show anything
            return ""
        
        # If API explicitly says "حسب الاستشارة" or similar, show it
        price_str = str(price).strip()
        if "حسب" in price_str or "استشارة" in price_str or "consult" in price_str.lower():
            return f" - {price_str}"
        
        # Try to parse as number
        try:
            # Handle string numbers like "500" or "500.0"
            price_num = float(price_str.replace(',', ''))
            if price_num > 0:
                # Format as integer if it's a whole number
                if price_num == int(price_num):
                    return f" - {int(price_num)} ريال"
                else:
                    return f" - {price_num:.2f} ريال"
            else:
                # Zero or negative - don't show price
                return ""
        except (ValueError, AttributeError):
            # Not a number - show as-is if it looks like a price
            if price_str and len(price_str) < 50:  # Reasonable length
                return f" - {price_str}"
            return ""
    
    def _sanitize_username(self, name: str) -> str:
        """
        Sanitize username to ensure it's displayable and meaningful (Issue #26).
        
        Handles:
        - Arabic diacritics-only names
        - Empty/whitespace names
        - Single character names
        - Encoding issues
        
        Args:
            name: Raw username from WhatsApp/input
            
        Returns:
            Sanitized username or empty string if invalid
        """
        if not name or not isinstance(name, str):
            return ""
        
        # Remove Arabic diacritics (combining marks)
        # These include: َ ِ ُ ً ٍ ٌ ّ ْ ٓ ٰ ٔ ٕ
        import re
        # Arabic diacritics range: U+064B-U+065F, U+0610-U+061A, U+06D6-U+06ED
        diacritics_pattern = r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]'
        clean_name = re.sub(diacritics_pattern, '', name)
        
        # Remove extra whitespace
        clean_name = ' '.join(clean_name.split())
        
        # Validate: Must have at least 2 characters after cleaning
        if len(clean_name) < 2:
            logger.warning(f"⚠️ Invalid username after sanitization: '{name}' → '{clean_name}' (too short)")
            return ""
        
        # Validate: Must have at least one letter (not just numbers/symbols)
        if not any(c.isalpha() for c in clean_name):
            logger.warning(f"⚠️ Invalid username after sanitization: '{name}' → '{clean_name}' (no letters)")
            return ""
        
        # Check for common invalid patterns
        invalid_patterns = ['unknown', 'null', 'undefined', 'none', '###', '***']
        if clean_name.lower() in invalid_patterns:
            logger.warning(f"⚠️ Invalid username pattern: '{clean_name}'")
            return ""
        
        logger.debug(f"✅ Username sanitized: '{name}' → '{clean_name}'")
        return clean_name
    
    def _sanitize_input(self, message: str) -> str:
        """
        Sanitize user input to prevent injection attacks and handle malformed input (Issue #14).
        
        Args:
            message: Raw user message
            
        Returns:
            Sanitized message safe for processing
        """
        if not message or not isinstance(message, str):
            return ""
        
        # Step 1: Length check (prevent extremely long inputs)
        MAX_MESSAGE_LENGTH = 2000
        if len(message) > MAX_MESSAGE_LENGTH:
            logger.warning(f"⚠️ [SANITIZE] Message too long ({len(message)} chars) - truncating to {MAX_MESSAGE_LENGTH}")
            message = message[:MAX_MESSAGE_LENGTH]
        
        # Step 2: Strip control characters (except newlines and tabs for structured input)
        import re
        # Remove null bytes and other dangerous control chars
        message = message.replace('\x00', '').replace('\r', '')
        
        # Step 3: Normalize whitespace (preserve Arabic spaces)
        message = message.strip()
        
        # Step 4: Remove suspicious patterns (but preserve legitimate Arabic and numbers)
        # Don't remove Arabic diacritics - they're legitimate
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                 # JavaScript protocol
            r'on\w+\s*=',                  # Event handlers (onclick, onload, etc.)
        ]
        
        original = message
        for pattern in suspicious_patterns:
            message = re.sub(pattern, '', message, flags=re.IGNORECASE | re.DOTALL)
        
        if message != original:
            logger.warning(f"⚠️ [SANITIZE] Suspicious pattern removed from input")
        
        # Step 5: Final safety check - ensure it's not empty after sanitization
        if not message.strip():
            logger.warning(f"⚠️ [SANITIZE] Input became empty after sanitization")
            return ""
        
        # Log sanitization only if something changed
        if message != original.strip():
            logger.info(f"🧹 [SANITIZE] Input cleaned: {len(original)} → {len(message)} chars")
        
        return message.strip()
    
    def _extract_service_type(self, service_name: str) -> str:
        """
        Extract high-level service type from variant name.
        
        Examples:
        - "عدد 3 جلسات بوكسر + الابط مع الرتوش 594" → "ليزر"
        - "بوتوكس فورهيد 50 وحدة" → "بوتوكس"
        - "فيلر ريستلين 1 مل" → "فيلر"
        - "ليزر رجال" → "ليزر"
        
        Args:
            service_name: Full service/variant name from API
            
        Returns:
            High-level service type (category)
        """
        if not service_name:
            return "خدمة"
        
        service_lower = service_name.lower()
        
        # Define service type keywords (most specific first)
        service_types = [
            (["بوتوكس", "botox"], "بوتوكس"),
            (["فيلر", "filler", "ريستلين", "جوفيديرم"], "فيلر"),
            (["ليزر", "laser", "بوكسر"], "ليزر"),
            (["تنظيف", "cleaning", "نظافة"], "تنظيف البشرة"),
            (["ميزو", "meso", "ميزوثيرابي"], "ميزوثيرابي"),
            (["خيوط", "threads", "شد"], "خيوط الشد"),
            (["تقشير", "peeling", "peel"], "تقشير"),
            (["بلازما", "plasma"], "بلازما"),
            (["ابر", "needles", "حقن"], "الإبر"),
            (["تبييض", "whitening"], "تبييض"),
            (["علاج", "therapy", "استشارة"], "علاج طبي"),
        ]
        
        # Match keywords
        for keywords, service_type in service_types:
            if any(keyword in service_lower for keyword in keywords):
                logger.debug(f"🔍 Extracted service type: '{service_name}' → '{service_type}'")
                return service_type
        
        # Fallback: Return first 2-3 words (remove prices, numbers, etc.)
        import re
        # Remove prices like "594 ريال" or "500"
        clean_name = re.sub(r'\d+\s*(ريال|رس|sr)?', '', service_name)
        # Remove special chars and extra whitespace
        clean_name = re.sub(r'[+\-،,]', ' ', clean_name)
        clean_name = ' '.join(clean_name.split()[:3])  # First 3 words max
        
        if clean_name and clean_name.strip():
            logger.debug(f"🔍 Extracted service type (fallback): '{service_name}' → '{clean_name.strip()}'")
            return clean_name.strip()
        
        # Ultimate fallback
        return "خدمة"
    
    def _get_user_language(self, session_key: str) -> str:
        """
        Get user's preferred language (Issue #25).
        
        Returns:
            'english' or 'arabic'
        """
        try:
            session_data = self.session_manager.get(session_key) or {}
            return session_data.get("preferred_language", "arabic")
        except:
            return "arabic"  # Safe fallback
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry function with exponential backoff (Issue #33).
        
        Args:
            func: Async function to retry
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        import asyncio
        
        last_exception = None
        for attempt in range(self.RETRY_MAX_ATTEMPTS + 1):  # 0, 1, 2 (3 total attempts)
            try:
                if attempt > 0:
                    # Backoff delay
                    delay_ms = self.RETRY_BACKOFF_MS[min(attempt - 1, len(self.RETRY_BACKOFF_MS) - 1)]
                    logger.info(f"🔄 Retry attempt {attempt}/{self.RETRY_MAX_ATTEMPTS} after {delay_ms}ms backoff")
                    await asyncio.sleep(delay_ms / 1000.0)
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"✅ Retry succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Check if error is retryable
                is_retryable = any(retryable in error_type for retryable in self.RETRY_ON_ERRORS)
                
                if not is_retryable or attempt >= self.RETRY_MAX_ATTEMPTS:
                    if not is_retryable:
                        logger.warning(f"❌ Error not retryable: {error_type}")
                    else:
                        logger.error(f"❌ All retry attempts exhausted ({attempt + 1} attempts)")
                    raise
                
                logger.warning(f"⚠️ Attempt {attempt + 1} failed with {error_type}: {str(e)[:100]}")
        
        # Should never reach here, but just in case
        raise last_exception
    
    async def _should_use_llm_only(self, message: str, intent: str) -> bool:
        """
        Determine if we should use LLM-only response or route to agent.
        
        Args:
            message: User message
            intent: Classified intent
            
        Returns:
            True if LLM-only, False if agent needed
        """
        # Use LLM-only for:
        # - Greetings
        # - General questions
        # - Feedback/thanks
        # - Casual conversation
        
        llm_only_intents = ["greeting", "feedback", "unknown"]
        
        if intent in llm_only_intents:
            return True
        
        # For resource intent, use LLM with API data (conversational)
        if intent == "resource":
            return True
        
        # For booking/patient intents, ALWAYS route to agent
        # These require structured flows and API calls
        if intent in ["booking", "patient"]:
            return False
        
        # Default: use LLM for unknown cases
        return True

    async def route(self, payload: dict) -> dict:
        """
        Route incoming message to appropriate handler with timeout protection.
        
        Args:
            payload: Parsed message payload with phone_number, message, sender_name, etc.
            
        Returns:
            Dictionary with processing result and status
            
        Note: Wraps _route_internal with timeout to prevent hanging requests
        """
        import asyncio
        
        # Request timeout: 120 seconds max (safety net against slow API/LLM calls)
        # Increased from 45s due to: API calls (30s) + LLM calls (20s) + processing (10s) = ~60s
        timeout_seconds = 120
        
        try:
            return await asyncio.wait_for(
                self._route_internal(payload),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            request_id = payload.get("request_id", "unknown")
            phone = payload.get("phone_number", "unknown")
            logger.error(f"🚨 [REQ:{request_id}] REQUEST TIMEOUT after {timeout_seconds}s for {phone}")
            logger.error(f"🚨 This usually indicates Redis issues or network problems")
            
            return {
                "response": "عذراً، النظام مشغول حالياً 🙏\n\nيرجى المحاولة مرة أخرى أو الاتصال على: 920033304",
                "status": "timeout",
                "error": f"Request timeout after {timeout_seconds}s"
            }
    
    async def _route_internal(self, payload: dict) -> dict:
        """
        Internal routing logic (called by route() with timeout protection).
        
        Args:
            payload: Parsed message payload
            
        Returns:
            Processing result
        """
        # CRITICAL: Use inbound request_id if provided by middleware; otherwise generate
        request_id = str(payload.get("request_id") or "")[:32]
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())[:8]  # Short ID for readability
        
        phone_number = None
        session_data = {}  # Initialize early to avoid UnboundLocalError
        
        try:
            # CRITICAL: Sanitize input (Issue #14)
            raw_message = payload.get("message", "")
            message_text = self._sanitize_input(raw_message) or ""  # Ensure never None
            
            phone_number = payload.get("phone_number") or payload.get("phone")
            session_key = f"whatsapp:{phone_number}" if phone_number else "unknown"
            # DEBUG: Full session id for correlation (privacy-safe at DEBUG level)
            logger.debug(f"SESSION_ID(full)={session_key}")
            
            # Load session data EARLY for context-aware processing
            session_data = await self.session_manager.get_session(session_key) or {}

            # Phone correlation (Issue #37): compare last 10 digits of session vs message phone
            # BUT: Skip extraction if we're asking for other numeric data (Issue: False positives)
            session_digits = normalize_phone_digits(phone_number or "")
            
            # Context-aware extraction: Don't extract phone if asking for ID/numbers
            booking_step = (session_data.get("booking_state") or {}).get("step", "")
            asking_for_numeric_data = booking_step in [
                "awaiting_registration_id",
                "awaiting_national_id", 
                "awaiting_time_slot"
            ]
            
            msg_phone_candidate = None if asking_for_numeric_data else extract_generic_phone(message_text)
            message_digits = normalize_phone_digits(msg_phone_candidate) if msg_phone_candidate else None
            phone_mismatch = bool(session_digits and message_digits and session_digits != message_digits)
            
            if phone_mismatch:
                logger.warning(f"📱 Phone mismatch detected: session={session_digits}, message={message_digits}")
            elif asking_for_numeric_data and message_text.strip().isdigit():
                logger.debug(f"⏭️ Skipped phone extraction (asking for {booking_step})")
            
            # CRITICAL: Extract real client IP from headers (Issue #36)
            # Check for proxy headers first
            real_ip = payload.get("x-forwarded-for") or payload.get("x-real-ip") or payload.get("remote_addr", "unknown")
            if "," in str(real_ip):
                # X-Forwarded-For can have multiple IPs, first is the real client
                real_ip = real_ip.split(",")[0].strip()
            
            # Log request start with ID and real IP (Issue #23, #36)
            logger.info(f"🔵 [REQ:{request_id}] Incoming from {phone_number} (IP: {real_ip}): '{message_text[:50]}...'")
            
            # ⏱️ Start performance timer
            request_start_time = time.time()
            timing_checkpoints = {"start": request_start_time}
            
            # Add request ID to payload for downstream tracking
            payload["request_id"] = request_id
            payload["real_client_ip"] = real_ip  # Store for rate limiting
            
            message_text = str(payload.get("message", ""))
            
            # Get session data early to determine proper name
            session_data = self.session_manager.get(session_key) or {}
            
            if not phone_number:
                logger.error("No phone number in payload - cannot send response")
                return {
                    "intent": "error",
                    "status": "error",
                    "error": "missing_phone_number"
                }
            
            # Cleanup stale session before processing
            await self.session_manager.cleanup_stale_session(session_key)
            
            # Reload session after cleanup (cleanup might have modified it)
            session_data = await self.session_manager.get_session(session_key) or {}
            booking_state = session_data.get("booking_state", {})
            
            # 🚨 CRITICAL: Store phone_number in session for agents to access
            session_data["phone_number"] = phone_number
            
            # ═══════════════════════════════════════════════════════════════
            # CRITICAL: Auto-lookup patient data from database (EARLY)
            # This runs for ALL intents on first message to load patient info
            # ═══════════════════════════════════════════════════════════════
            patient_data = session_data.get("patient_data")
            
            # Lookup if: no patient_data OR patient_data exists but not registered yet
            should_lookup = (not patient_data or not patient_data.get("already_registered")) and phone_number
            
            # OPTIMIZATION: Only log if we're actually doing a lookup (reduce noise)
            if should_lookup:
                logger.info(f"🔍 PATIENT LOOKUP: Checking database for {phone_number}")
            else:
                logger.debug(f"⏭️ PATIENT LOOKUP: Skipped - already loaded (registered={patient_data.get('already_registered') if patient_data else False})")
            
            if should_lookup:
                # First message from user - automatically check if patient exists
                try:
                    from ..api.agent_api import AgentApiClient
                    from ..utils.phone_parser import remove_country_code, is_valid_phone
                    
                    # CRITICAL: Validate phone number BEFORE any API calls!
                    is_valid, invalid_reason = is_valid_phone(phone_number)
                    
                    if not is_valid:
                        logger.warning(f"⚠️ INVALID PHONE NUMBER: {phone_number} - Reason: {invalid_reason}")
                        logger.warning(f"⚠️ Skipping patient lookup - phone number is clearly invalid")
                        # Mark as not found so system proceeds with registration
                        patient_data = {"already_registered": False}
                    else:
                        logger.info(f"✅ Phone number validated: {phone_number} ({len(phone_number)} digits)")
                        
                        lookup_client = AgentApiClient()
                        
                        # Normalize phone: remove country code (966 for Saudi, 20 for Egypt)
                        normalized_phone = remove_country_code(phone_number)
                        
                        # 🚨 SMART: Detect country code to avoid trying wrong formats
                        country_code = None
                        original_number = phone_number.lstrip("+")
                        if original_number.startswith("966"):
                            country_code = "966"  # Saudi Arabia
                            logger.info(f"🇸🇦 Detected Saudi number: {phone_number}")
                        elif original_number.startswith("20"):
                            country_code = "20"  # Egypt
                            logger.info(f"🇪🇬 Detected Egyptian number: {phone_number}")
                        elif original_number.startswith("971"):
                            country_code = "971"  # UAE
                            logger.info(f"🇦🇪 Detected UAE number: {phone_number}")
                        else:
                            # Default to Saudi (business location)
                            country_code = "966"
                            logger.info(f"🔍 No country code detected, defaulting to Saudi")
                        
                        logger.info(f"🔍 AUTO-LOOKUP: Checking patient database for {phone_number} (normalized: {normalized_phone})")
                        
                        # Build smart phone format list based on detected country
                        search_result = None
                        phone_formats_to_try = [
                            normalized_phone,  # Local format
                            f"0{normalized_phone}",  # With leading zero
                            phone_number,  # Original format
                        ]
                        
                        # Add country-specific formats ONLY
                        if country_code == "20":
                            phone_formats_to_try.extend([
                                f"20{normalized_phone}",
                                f"+20{normalized_phone}",
                            ])
                        elif country_code == "966":
                            phone_formats_to_try.extend([
                                f"966{normalized_phone}",
                                f"+966{normalized_phone}",
                            ])
                        elif country_code == "971":
                            phone_formats_to_try.extend([
                                f"971{normalized_phone}",
                                f"+971{normalized_phone}",
                            ])
                        
                        logger.info(f"📞 Will try {len(phone_formats_to_try)} phone formats for country +{country_code}")
                        
                        # 🚀 OPTIMIZATION: Try all formats in parallel instead of sequential
                        import asyncio
                        
                        async def try_phone_format(phone_format: str):
                            """Try a single phone format"""
                            try:
                                # Try search endpoint first
                                result = await lookup_client.get(f"/patients/search?phone={phone_format}")
                                if result and result.get("results") and len(result["results"]) > 0:
                                    return ("search", phone_format, result)
                                
                                # If search fails, try direct patient endpoint
                                try:
                                    direct_result = await lookup_client.get(f"/patients/{phone_format}")
                                    if direct_result and direct_result.get("patient"):
                                        # Convert direct result to search result format
                                        result = {
                                            "results": [direct_result["patient"]],
                                            "count": 1
                                        }
                                        return ("direct", phone_format, result)
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.debug(f"Format {phone_format} failed: {e}")
                            return (None, phone_format, None)
                        
                        # Execute all lookups in parallel
                        logger.info(f"🚀 Trying all {len(phone_formats_to_try)} formats in parallel...")
                        start_time = time.time()
                        results = await asyncio.gather(*[try_phone_format(fmt) for fmt in phone_formats_to_try], return_exceptions=True)
                        lookup_time = time.time() - start_time
                        logger.info(f"⏱️ Parallel lookup completed in {lookup_time:.2f}s")
                        timing_checkpoints["patient_lookup"] = time.time()
                        
                        # Find first successful result
                        search_result = None
                        for method, phone_format, result in results:
                            if result and not isinstance(result, Exception):
                                search_result = result
                                logger.info(f"✅ Found patient with phone format: {phone_format} (via {method})")
                                break
                        
                        # API returns {"count": X, "results": [...]}
                        if search_result and search_result.get("results") and len(search_result["results"]) > 0:
                            patient = search_result["results"][0]
                            patient_id = patient.get("id")
                            
                            # Fetch FULL patient details (search only returns basic fields)
                            logger.info(f"📋 Fetching full patient details for ID: {patient_id}")
                            try:
                                full_patient = await lookup_client.get(f"/patients/{patient_id}")
                                
                                # Extract patient info from nested structure
                                if full_patient and full_patient.get("patient"):
                                    patient_details = full_patient["patient"]
                                    
                                    # API uses different field names: patient_phone or phone
                                    patient_phone = (patient_details.get("patient_phone") or 
                                                   patient_details.get("full_phone") or 
                                                   patient_details.get("phone"))
                                    
                                    patient_data = {
                                        "id": patient_details.get("id"),
                                        "name": patient_details.get("name"),
                                        "national_id": patient_details.get("identification_id") or patient_details.get("national_id"),
                                        "phone": patient_phone,
                                        "gender": patient_details.get("gender"),
                                        "email": patient_details.get("email"),
                                        "city": patient_details.get("city"),
                                        "birth_date": patient_details.get("birth_date"),
                                        "age": patient_details.get("age"),
                                        "patient_code": patient_details.get("patient_code"),
                                        "already_registered": True
                                    }
                                    
                                    logger.info(f"✅ Full patient details loaded: {patient_data.get('name')} - National ID: {patient_data.get('national_id')}")
                                else:
                                    # Fallback to basic info from search
                                    patient_phone = patient.get("patient_phone") or patient.get("full_phone") or patient.get("phone")
                                    patient_data = {
                                        "id": patient.get("id"),
                                        "name": patient.get("name"),
                                        "phone": patient_phone,
                                        "gender": patient.get("gender"),
                                        "already_registered": True
                                    }
                                    logger.warning(f"⚠️ Could not fetch full details, using basic info")
                                    
                            except Exception as detail_error:
                                logger.warning(f"⚠️ Failed to fetch full patient details: {detail_error}")
                                # Fallback to basic info from search
                                patient_phone = patient.get("patient_phone") or patient.get("full_phone") or patient.get("phone")
                                patient_data = {
                                    "id": patient.get("id"),
                                    "name": patient.get("name"),
                                    "phone": patient_phone,
                                    "gender": patient.get("gender"),
                                    "already_registered": True
                                }
                            
                            # Save to session for future use
                            session_data["patient_data"] = patient_data
                            if patient_data.get("name"):
                                session_data["sender_name"] = patient_data.get("name")
                            
                            await self.session_manager.put_session(session_key, session_data)
                            logger.info(f"✅ PATIENT FOUND: {patient_data.get('name')} (ID: {patient_data.get('id')}) - Data loaded to session")
                        else:
                            logger.info(f"ℹ️ PATIENT NOT FOUND: {normalized_phone} - New user, will need registration")
                            patient_data = {"already_registered": False}
                        session_data["patient_data"] = patient_data
                        await self.session_manager.put_session(session_key, session_data)
                except Exception as lookup_error:
                    logger.warning(f"⚠️ Patient lookup failed: {lookup_error}")
                    patient_data = None
            
            # CRITICAL: Check state version for conflict detection (Issue: State sync mismatch)
            state_version = booking_state.get("_state_version")
            last_updated_by = booking_state.get("_last_updated_by")
            if state_version:
                logger.debug(f"📊 State loaded: version={state_version}, updated_by={last_updated_by}, step={booking_state.get('step')}")
            
            logger.debug(f"📥 LOADED SESSION: topic={session_data.get('current_topic')}, last_intent={session_data.get('last_intent')}, phase={session_data.get('journey_phase')}")
            
            # CRITICAL FIX (Issue #15): Reset catastrophic failure state on new attempt
            if booking_state.get("step") == "catastrophic_failure":
                logger.warning(f"🔄 [RECOVERY] Catastrophic failure detected - resetting booking state for fresh start")
                session_data["booking_state"] = {}
                session_data["registration_expected"] = None
                session_data["current_topic"] = None
                session_data["journey_phase"] = "discovery"
                session_data["error_turn_count"] = 0  # Reset error counter
                await self.session_manager.put_session(session_key, session_data)
                booking_state = {}  # Use fresh state
                logger.info(f"✅ [RECOVERY] Session reset - user can try again")
            
            # CRITICAL: Increment turn counter immediately (Issue #13)
            # This ensures it's saved in ALL code paths, not just success path
            current_turn = session_data.get("conversation_turn", 0)
            new_turn = current_turn + 1
            session_data["conversation_turn"] = new_turn
            logger.info(f"📊 Turn counter incremented: {current_turn} → {new_turn}")
            
            # CRITICAL FIX: Save turn counter IMMEDIATELY to prevent loss on early returns
            await self.session_manager.put_session(session_key, session_data)
            logger.debug(f"💾 Turn counter persisted to session: {new_turn}")
            
            # CRITICAL: Detect new conversation vs continuation
            # If turn counter was 0 and we have old intent/booking data, this is likely a NEW conversation
            # Clean up stale data from previous sessions
            if current_turn == 0 and (session_data.get("last_intent") or session_data.get("booking_state")):
                logger.warning(f"🧹 NEW CONVERSATION detected (turn 0) but old data exists - cleaning up")
                logger.warning(f"   Old data: last_intent={session_data.get('last_intent')}, booking_step={session_data.get('booking_state', {}).get('step')}")
                
                # Clear ephemeral state for fresh start
                session_data["last_intent"] = None
                session_data["booking_state"] = {}
                session_data["current_topic"] = None
                session_data["journey_phase"] = "discovery"
                session_data["show_booking_offer"] = False
                
                await self.session_manager.put_session(session_key, session_data)
                booking_state = {}
                logger.info(f"✅ Stale session data cleaned - starting fresh conversation")
            
            # Determine sender name using smart name precedence (respects Arabic context)
            entities = {}  # Will be populated later
            sender_name = self._get_user_name(payload, session_data, entities, message_text)
            
            # Update last activity timestamp
            session_data['last_activity_timestamp'] = time.time()
            
            # CRITICAL: Extract last bot message BEFORE filter (so filter can use it)
            # Get from Redis history cache
            history_cache = self._get_history_cache()
            last_bot_msg = await history_cache.get_last_message(session_key, role="assistant")
            last_bot_message = last_bot_msg["content"] if last_bot_msg else None
            session_data["last_bot_message"] = last_bot_message or ""
            
            # FILTER: Context-aware single character handling (Issue #13 & #14)
            # ⚠️ ONLY filter if NOT using intelligent agent - let LLM brain handle everything!
            use_intelligent_agent = getattr(settings, 'use_intelligent_agent', False)
            
            if use_intelligent_agent and len(message_text.strip()) <= 1:
                logger.info(f"🧠 [INTELLIGENT AGENT] Passing single character '{message_text}' to LLM - let AI brain decide!")
            
            if not use_intelligent_agent and len(message_text.strip()) <= 1:
                char = message_text.strip()
                
                # ALLOW: Numbers if in selection context (Issue #14 fix)
                # Check multiple indicators:
                # 1. Services explicitly saved to session
                has_services_shown = bool(session_data.get("last_shown_services"))
                
                # 2. Last bot message contained a numbered list (CRITICAL FIX!)
                last_bot_msg = session_data.get("last_bot_message", "")
                has_numbered_list = False
                if last_bot_msg:
                    # Check for numbered list patterns: "1.", "1-", "١.", etc.
                    import re
                    numbered_patterns = [r'\n\s*\d+[\.\-\)]\s+', r'\n\s*[١-٩]+[\.\-\)]\s+']
                    for pattern in numbered_patterns:
                        if re.search(pattern, last_bot_msg):
                            has_numbered_list = True
                            break
                
                # 3. Awaiting selection flag (set by intelligent agent when showing lists)
                awaiting_selection = bool(session_data.get("awaiting_selection"))
                if awaiting_selection:
                    # Clear flag after 5 minutes to avoid stale selections
                    list_timestamp = session_data.get("last_list_timestamp", 0)
                    if time.time() - list_timestamp > 300:  # 5 minutes
                        session_data["awaiting_selection"] = False
                        awaiting_selection = False
                        logger.info(f"⏰ [FILTER] Cleared stale awaiting_selection flag (> 5 minutes old)")
                
                # 4. Booking flow active
                booking_active = booking_state.get("started")
                
                in_selection_context = booking_active or has_services_shown or has_numbered_list or awaiting_selection
                
                if char.isdigit() and in_selection_context:
                    logger.info(f"✅ Allowing single digit '{char}' - user in selection context (booking={booking_active}, services={has_services_shown}, numbered_list={has_numbered_list}, awaiting={awaiting_selection})")
                    # Clear awaiting_selection flag since user made their selection
                    if awaiting_selection:
                        session_data["awaiting_selection"] = False
                        logger.info(f"🔄 [FILTER] Cleared awaiting_selection flag - user made selection")
                    # Continue processing - this is a valid selection (including #9!)
                
                # ALLOW: Question marks and exclamation (legitimate queries)
                elif char in ['?', '!', '؟']:  # Include Arabic question mark
                    logger.info(f"✅ Allowing punctuation '{char}' - legitimate query")
                
                # BLOCK: Everything else (typing artifacts)
                else:
                    logger.info(f"⏭️ Ignoring single character: '{char}' (Reason: Typing artifact, not in selection context - booking={booking_active}, services={has_services_shown}, numbered_list={has_numbered_list}, awaiting={awaiting_selection})")
                    return {
                        "intent": "ignored",
                        "status": "filtered",
                        "message": "Single character ignored"
                    }
            
            # FILTER: Ignore pure whitespace or empty messages
            if not message_text or message_text.isspace():
                logger.info(f"⏭️ Ignoring empty/whitespace message")
                return {
                    "intent": "ignored",
                    "status": "filtered",
                    "message": "Empty message ignored"
                }
            
            # DEBOUNCING: Check if messages are coming too fast (< 2 seconds apart)
            last_msg_time = session_data.get('last_message_timestamp', 0)
            current_time = time.time()
            time_since_last = current_time - last_msg_time
            
            if time_since_last < 2.0 and last_msg_time > 0:
                # User is typing rapidly - ignore intermediate messages
                logger.info(f"⏭️ Debouncing: Message too fast ({time_since_last:.2f}s since last). Waiting for user to finish typing...")
                session_data['last_message_timestamp'] = current_time
                await self.session_manager.put_session(session_key, session_data)
                return {
                    "intent": "debounced",
                    "status": "filtered",
                    "message": "Message debounced - user still typing"
                }
            
            # Update last message timestamp
            session_data['last_message_timestamp'] = current_time
            
            # Build context FIRST (needed by both fast path and LLM path)
            # ENHANCED: Track conversation flow and last bot question
            # Get conversation history from Redis cache
            history_cache = self._get_history_cache()
            conversation_history = await history_cache.get_recent_context(session_key, last_n=30)
            logger.info(f"📖 Retrieved {len(conversation_history)} messages from Redis history cache")
            
            # DEBUG: Check if tool_calls exist in history
            tool_call_count = sum(1 for msg in conversation_history if msg.get("tool_calls"))
            logger.info(f"🔧 History contains {tool_call_count} messages with tool_calls")
            
            # Detect current message language with safe fallback (Issue #35)
            msg_lang_code = self._detect_message_language(message_text)  # 'ar' or 'en'
            msg_lang = "arabic" if msg_lang_code == "ar" else "english"

            # Extract current topic from session
            current_topic = session_data.get("current_topic", None)  # e.g., "physical_therapy"
            last_discussed_service = session_data.get("last_discussed_service", None)
            
            # TURN COUNTER: Already incremented at beginning (Issue #13)
            # Use the NEW turn value that was set earlier
            # (new_turn variable already exists from line 811)
            
            # ENHANCED CONTEXT: Include comprehensive session state (Issue #34)
            context = {
                "last_intent": session_data.get("last_intent"),
                "last_message": session_data.get("last_message"),
                "last_bot_question": last_bot_message,  # What bot just asked
                "current_topic": current_topic,  # What we're discussing now
                "last_discussed_service": last_discussed_service,  # Service user asked about
                "booking_active": booking_state.get("started", False),
                "booking_step": booking_state.get("step"),
                "conversation_history": conversation_history,
                "conversation_turn": new_turn,  # Use NEW turn value after increment
                "preferred_language": session_data.get("preferred_language", "arabic"),  # User's language preference (fallback)
                "current_message_language": msg_lang,  # Canonical 'arabic'/'english' with fallback
                # Phone correlation hints (Issue #37)
                "phone_mismatch": phone_mismatch,
                "session_phone_tail": (session_digits[-4:] if session_digits else None),
                "message_phone_tail": (message_digits[-4:] if message_digits else None),
                # ENHANCED STATE (Issue #34):
                "error_count": session_data.get("error_turn_count", 0),  # Number of errors encountered
                "last_error_time": session_data.get("last_error_timestamp"),  # When last error occurred
                "journey_phase": session_data.get("journey_phase", "discovery"),  # User's journey stage
                "previous_topic": session_data.get("previous_topic"),  # Topic before context switch
                "conversation_summary": f"Turn {new_turn}, Phase: {session_data.get('journey_phase', 'discovery')}, Topic: {current_topic or 'none'}",
                # Registration context (expected field during registration flow)
                "registration_expected": session_data.get("registration_expected"),
                # CRITICAL: Patient data for personalization
                "patient_data": session_data.get("patient_data"),
                "sender_name": sender_name
            }
            
            # LOG CONVERSATION CONTEXT for debugging
            router_step = context.get("booking_step")  # Step from router context
            state_step = booking_state.get('step')      # Step from booking state
            logger.info(f"💬 CONTEXT: topic={current_topic}, step=[router={router_step}, state={state_step}], turn={context['conversation_turn']}, lang={session_data.get('preferred_language', 'arabic')}, last_bot='{last_bot_message[:40] if last_bot_message else 'none'}...'")
            
            # LLM-ONLY: All messages processed by LLM
            intent = None
            confidence = 0.0
            use_llm = True  # Always true - no fast path

            # Intent cache lookup (Issue #36) - short TTL cache per session
            try:
                normalized_msg = message_text.strip().lower()
                cache_map = IntentRouter._intent_cache.setdefault(session_key, {})
                entry = cache_map.get(normalized_msg)
                if entry and (time.time() - entry.get("ts", 0)) < 120:
                    intent = entry.get("intent")
                    confidence = entry.get("confidence", 0.7)
                    use_llm = False
                    logger.info(f"⚡ Intent cache hit → {intent} (conf={confidence:.2f})")
            except Exception:
                pass
            
            # ═══════════════════════════════════════════════════════════════
            # LLM-FIRST APPROACH: All keyword patterns removed!
            # Let the LLM handle all intent detection using natural language understanding
            # This makes the bot more intelligent and adaptable
            # ═══════════════════════════════════════════════════════════
            
            msg_lower = message_text.lower()
            
            # 🚨 FAST PATH: Numbered response detection (BEFORE LLM)
            # If bot showed numbered list and user responds with a number → selection
            import re
            intent_classification = None
            confidence_from_llm = 0.0
            
            # CRITICAL: Define these variables early so they're available throughout
            last_bot_msg = context.get('last_bot_question') or ''  # Ensure not None
            numbered_patterns = [r'\n\s*\d+[\.\-\)]\s+', r'\n\s*[١-٩]+[\.\-\)]\s+']
            has_numbered_list = any(re.search(pattern, last_bot_msg) for pattern in numbered_patterns)
            
            if message_text.strip().isdigit() or re.match(r'^\d+[\s\-\.،]*$', message_text.strip()):
                
                if has_numbered_list:
                    logger.info(f"⚡ FAST PATH: Numbered response '{message_text}' after numbered list → intent=selection (GUARANTEED)")
                    intent = "selection"
                    intent_classification = "selection"
                    confidence = 0.95
                    confidence_from_llm = 0.95
                    use_llm = False  # Skip LLM - we're certain!
                else:
                    use_llm = True
            else:
                use_llm = True
            
            # ═══════════════════════════════════════════════════════════════
            # LLM-FIRST: Always use LLM for intelligent intent classification
            # No shortcuts, no keywords, pure natural language understanding
            # ═══════════════════════════════════════════════════════════════
            
            use_intelligent_agent = getattr(settings, 'use_intelligent_agent', False)
            
            if use_intelligent_agent:
                # Skip redundant intent classification - intelligent agent does it all in ONE call!
                logger.info(f"🚀 [OPTIMIZATION] Skipping intent classification LLM call - intelligent agent will handle it")
                intent = "intelligent_routing"
                intent_classification = "intelligent_routing"
                confidence = 1.0
                confidence_from_llm = 1.0
                use_llm = False  # CRITICAL: Don't waste LLM call!
            elif not intent_classification:  # Only use LLM if fast path didn't match
                # ALWAYS use LLM for intelligent intent classification
                logger.info(f"🤖 LLM-FIRST: Using LLM for intelligent intent classification (message: '{message_text[:50]}...')")
                use_llm = True
            
            # Use LLM reasoner for intent classification with full context
            from ..core.llm_reasoner import get_llm_reasoner
            llm = get_llm_reasoner()
            
            # Only build LLM prompt if using LLM
            if use_llm:
                # Build strict JSON prompt for intent classification
                # Get conversation history for FULL context
                conv_history = context.get('conversation_history', [])
                
                # Format history for LLM to see ACTUAL messages
                history_text = ""
                if conv_history:
                    recent_history = conv_history[-3:]  # Last 3 messages for context
                    for msg in recent_history:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:300]  # Increased to see full numbered lists
                        history_text += f"\n{role}: {content}"
                else:
                    history_text = "\n(No previous messages - first interaction)"
                
                # Get last bot message for context
                last_bot_msg = context.get('last_bot_question', 'none')
                # CRITICAL: Don't truncate - need full context to see numbered lists!
                last_bot_preview = last_bot_msg[:500] if last_bot_msg else 'none'
                
                classification_prompt = f"""You are an intent classifier. Analyze the message and return ONLY a JSON object.

User message: "{message_text}"

Last Bot Message:{last_bot_preview}

Recent Conversation History:{history_text}

Current Context:
- Active booking: {booking_state.get('started', False)}
- Current step: {booking_state.get('step', 'none')}
- Last intent: {session_data.get('last_intent', 'none')}
- Conversation turn: {context.get('conversation_turn', 0)}
- Current topic: {current_topic or 'none'}

🎯 CONTEXT PRIORITIZATION (CRITICAL):
**IMMEDIATE conversation context (last 1-3 messages) is MORE IMPORTANT than historical data**
- What bot JUST said/showed > Patient history
- What user is responding to NOW > Their booking count
- If user responded to numbered list → That's a selection (ignore everything else!)
- If user is answering bot's question → That's the intent (not chitchat!)

🚨 REJECTION + NEW TOPIC DETECTION (CRITICAL):
- If user starts with "لا" (no) + question word → NEW topic, NOT continuation!
- "لا وش العروض" = User REJECTED previous topic + asking about offers → intent=question
- "لا عادي" = User declining → intent=confirmation (negative)
- "لا شكراً" = User declining → intent=confirmation (negative)
- "لا" at START of message = User is changing direction/rejecting previous context
- After "لا", classify the REST of the message independently!

🚨 CRITICAL NUMBERED LIST DETECTION:
- If last bot message contains a numbered list (1., 2., 3., etc.) → User's number response = selection
- If user message is JUST a number (e.g., "5") → intent=selection (choosing from list)
- If user message is a number + context (e.g., "5 please") → intent=selection
- Check last bot message for patterns like "1.", "2.", "3." to confirm list was shown

IMPORTANT CONTEXT AWARENESS:
- If bot just asked "which doctor?" and user says "هبة" → intent=selection (answering question)
- If bot just showed services and user says "2" → intent=selection (choosing from list)
- If user is in booking_step and provides info → intent=selection or booking (continuing flow)
- Consider what bot JUST asked to understand user's response

CRITICAL REGISTRATION CONTEXT:
- If current_step is "registration_id" or "awaiting_name" → user is in REGISTRATION flow
- During registration, numeric input (national ID) → intent=registration or booking, NOT selection
- During registration, name input → intent=registration or booking, NOT selection
- Registration is part of the booking process, not service selection
- Only classify as "selection" if user is CHOOSING from a menu/list, not providing registration data

Possible intents:
- booking: User wants to book/continue booking (NEW request only, e.g., "احجز", "موعد", "أبي حجز")
- registration: User is providing registration data (name, national ID) during account creation
- selection: User is selecting/expressing interest in a service (e.g., "العلاج الطبيعي", "بوتوكس", "ليزر")
- follow_up: User wants more details about CURRENT topic (e.g., "تفاصيل", "المزيد", "details", "more info")
- question: Asking about services, prices, doctors, clinic info, offers (e.g., "وش خدماتكم؟", "ايش عندكم؟", "وش الأسعار؟", "وش العروض؟", "what services?", "عندكم بوتوكس؟", "what offers?")
- chitchat: ONLY greetings, small talk, checking if human (e.g., "مرحبا", "كيفك؟", "هلا", "hello", "تمام")
- complaint: User is upset or sarcastic (e.g., "جد؟", "Really?", "Seriously?" when expressing surprise/frustration)
- confirmation: Yes/no/okay responses (in booking context)
- cancel: User explicitly wants to STOP/CANCEL current process (only "cancel", "stop", "إلغاء", "وقف")

🚨 CRITICAL: "وش خدماتكم" = question (NOT chitchat!)
🚨 CRITICAL: Any "وش", "ايش", "what", "عندكم", "العروض" = question (asking for information)
🚨 CRITICAL: "جد؟" (Really?) = complaint/question (user surprised/confused, NOT chitchat!)

CRITICAL DISTINCTION - REJECTION + NEW TOPIC:
- "لا وش العروض" = question (user said NO to previous, asking about offers) ✅
- "لا تمام" = confirmation (user declining politely)
- "مرة ثانية" or "وش العروض" alone = question (asking about offers)

CRITICAL DISTINCTION - CONTEXT-BASED:
- "احجزلي" (book for me) = booking (NEW request)
- "639404829" during registration_id step = registration (providing ID)
- "شادي سالم" during awaiting_name step = registration (providing name)
- "هبة" (Heba - doctor name) when choosing doctor = selection (answering "which doctor?")
- "2" or "الثاني" when choosing from service menu = selection (choosing from list)
- User answering a registration question = registration, NOT selection
- User choosing from a menu/list = selection, NOT registration

CRITICAL: Output ONLY valid JSON in this exact format, nothing else:
{{"intent": "booking", "confidence": 0.95, "reason": "User requested appointment"}}

DO NOT include conversational text, emojis, or explanations outside the JSON."""

                llm_context = {
                    "sender_name": sender_name,
                    "intent": "intent_classification_json",
                    "booking_active": booking_state.get("started", False),
                    "current_step": booking_state.get("step"),
                    "message": classification_prompt
                }
                
                # Get LLM classification with JSON parsing
                import json
                import re
                
                # CRITICAL: Use classify_intent() to avoid polluting conversational memory!
                llm_response = llm.classify_intent(
                    user_id=phone_number,
                    user_message=classification_prompt,
                    context=llm_context
                ).strip()
                
                # Extract JSON from response (handle cases where LLM adds text around it)
                json_match = re.search(r'\{[^}]+\}', llm_response)
                if json_match:
                    try:
                        intent_data = json.loads(json_match.group(0))
                        intent_classification = intent_data.get("intent", "chitchat").lower()
                        confidence_from_llm = intent_data.get("confidence", 0.6)
                        reason = intent_data.get("reason", "")
                        logger.info(f"🧠 LLM reasoning: Intent={intent_classification}, Confidence={confidence_from_llm}, Reason={reason}")
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from LLM: {llm_response}")
                        intent_classification = "chitchat"
                        confidence_from_llm = 0.5
                else:
                    # Fallback: extract intent from text
                    logger.warning(f"No JSON found in LLM response: {llm_response}")
                    intent_classification = llm_response.lower()
                    confidence_from_llm = 0.6
                
                logger.info(f"🧠 LLM reasoning output: Intent = {intent_classification}, Confidence = {confidence_from_llm}")
                
                # 🤔 HUMAN-LIKE UNCERTAINTY HANDLING: Ask for clarification when unsure
                # Humans don't guess with low confidence - they ask!
                LOW_CONFIDENCE_THRESHOLD = 0.65
                if confidence_from_llm < LOW_CONFIDENCE_THRESHOLD:
                    logger.warning(f"❓ LOW CONFIDENCE ({confidence_from_llm:.2f}) - checking if clarification needed")
                    
                    # 🧠 INTELLIGENT OVERRIDE: Don't ask for clarification if message is CLEAR
                    # Keywords that indicate clear intent
                    question_keywords = ["وش", "ايش", "what", "عندكم", "فيه", "شو", "كم", "how", "متى", "when", "وين", "where"]
                    greeting_keywords = ["هلا", "هلو", "مرحبا", "السلام", "سلام", "صباح", "مساء", "hello", "hi", "hey"]
                    
                    is_clear_question = any(keyword in message_text.lower() for keyword in question_keywords)
                    is_clear_greeting = any(keyword in message_text.lower() for keyword in greeting_keywords)
                    
                    if is_clear_question:
                        logger.info(f"✅ LOW CONFIDENCE but message contains question keywords → treating as 'question' intent")
                        intent_classification = "question"
                        confidence_from_llm = 0.7  # Boost confidence
                        # Continue with normal processing below
                    elif is_clear_greeting:
                        logger.info(f"✅ LOW CONFIDENCE but message contains greeting keywords → treating as 'chitchat' intent")
                        intent_classification = "chitchat"
                        confidence_from_llm = 0.8  # High confidence for greetings
                        # Continue with normal processing below
                    else:
                        # Truly ambiguous - ask for clarification
                        logger.warning(f"❓ Asking for clarification (ambiguous message, low confidence)")
                        
                        # Build clarification message based on context
                        from ..utils.language_detector import detect_language
                        language = detect_language(message_text)
                        
                        if language == "ar":
                            clarification_msg = f"عذراً {sender_name if sender_name != 'Unknown' else 'عزيزي'}، مش متأكد فهمت قصدك. ممكن توضح أكتر؟ 😊"
                            
                            # Add helpful context if available
                            if has_numbered_list:
                                clarification_msg = f"عذراً {sender_name if sender_name != 'Unknown' else 'عزيزي'}، قصدك تختار من القائمة؟ ممكن تكتب رقم الخيار أو اسم الخدمة؟ 📋"
                            elif current_topic:
                                clarification_msg = f"عذراً، قصدك عن {current_topic}؟ ممكن توضح أكتر؟ 🤔"
                        else:
                            clarification_msg = f"Sorry {sender_name if sender_name != 'Unknown' else ''}, I'm not sure I understood. Could you clarify? 😊"
                            
                            if has_numbered_list:
                                clarification_msg = f"Sorry {sender_name if sender_name != 'Unknown' else ''}, did you mean to select from the list? You can type the number or service name. 📋"
                            elif current_topic:
                                clarification_msg = f"Sorry, do you mean about {current_topic}? Could you clarify? 🤔"
                        
                        # Send clarification message to user
                        await self._send_whatsapp_response(
                            phone_number=phone_number,
                            message=clarification_msg
                        )
                        
                        # Save to history
                        await self._update_session(
                            session_key=session_key,
                            payload=payload,
                            response=clarification_msg,
                            intent="clarification_needed",
                            confidence=confidence_from_llm
                        )
                        
                        return {
                            "intent": "clarification_needed",
                            "status": "clarification",
                            "response": clarification_msg,
                            "confidence": confidence_from_llm,
                            "original_intent": intent_classification
                        }
                
                # Parse intent from LLM response - TRUST THE LLM, don't inflate confidence
                # CRITICAL: Check for ALL intent types including selection and follow_up
                if "booking" in intent_classification or "حجز" in intent_classification:
                    intent = "booking"
                    confidence = confidence_from_llm
                elif "selection" in intent_classification or "choosing" in intent_classification or "اختيار" in intent_classification:
                    intent = "selection"
                    confidence = confidence_from_llm
                    logger.info(f"✅ LLM classified as SELECTION with confidence {confidence_from_llm}")
                elif "follow_up" in intent_classification or "follow-up" in intent_classification or "متابعة" in intent_classification or "details" in intent_classification:
                    intent = "follow_up"
                    confidence = confidence_from_llm
                    logger.info(f"✅ LLM classified as FOLLOW-UP with confidence {confidence_from_llm}")
                elif "chitchat" in intent_classification or "small" in intent_classification or "casual" in intent_classification:
                    intent = "chitchat"
                    confidence = confidence_from_llm
                elif "question" in intent_classification or "سؤال" in intent_classification or "asking" in intent_classification:
                    intent = "question"
                    confidence = confidence_from_llm
                elif "complaint" in intent_classification or "sarcasm" in intent_classification:
                    intent = "chitchat"  # Handle as friendly conversation
                    confidence = confidence_from_llm
                elif "confirmation" in intent_classification or "تأكيد" in intent_classification:
                    intent = "confirmation"
                    confidence = confidence_from_llm
                elif "cancel" in intent_classification or "إلغاء" in intent_classification:
                    intent = "cancel"
                    confidence = confidence_from_llm
                elif "registration" in intent_classification or "تسجيل" in intent_classification:
                    intent = "registration"
                    confidence = confidence_from_llm
                    logger.info(f"✅ LLM classified as REGISTRATION with confidence {confidence_from_llm}")
                else:
                    # CRITICAL FIX: Don't default to chitchat - trust LLM's classification
                    logger.warning(f"⚠️ Unknown intent from LLM: '{intent_classification}' - using as-is")
                    intent = intent_classification  # Use LLM's exact classification
                    confidence = confidence_from_llm
            
            # REMOVED: Low confidence override - trust LLM instead
            # The LLM is trained to understand context, don't second-guess it
            
            # INTENT NORMALIZATION: Fix common typos from LLM (Issue #44)
            intent_corrections = {
                "confirmattion": "confirmation",  # Double 't' typo
                "confirmaton": "confirmation",    # Missing 'i'
                "confimation": "confirmation",    # Missing 'r'
                "bookng": "booking",              # Missing 'i'
                "questin": "question",            # Missing 'o'
                "selecton": "selection",          # Missing 'i'
            }
            
            if intent and intent in intent_corrections:
                corrected = intent_corrections[intent]
                logger.warning(f"⚠️ Auto-corrected intent typo: '{intent}' → '{corrected}' (Issue #44)")
                intent = corrected
            
            # ═══════════════════════════════════════════════════════════════
            # LLM-FIRST: No keyword-based continuation detection!
            # Trust the LLM's intent classification completely
            # The LLM understands context better than keyword matching
            # ═══════════════════════════════════════════════════════════
            
            # TRUST THE LLM - No overrides, no keyword checks
            logger.info(f"✅ Trusting LLM classification: {intent} (confidence={confidence:.2f})")
            
            # CRITICAL: Extract entities EVEN on fast path (Issue #19)
            # Fast path saves time on LLM but still needs entity extraction
            extractor = get_entity_extractor()
            
            # Determine context for entity extraction
            extraction_context = None
            if intent == "registration" or session_data.get("awaiting_registration"):
                extraction_context = "registration"
            elif booking_state.get("started"):
                extraction_context = "booking"
            
            # ALWAYS extract if in appropriate context (Issue #19)
            entities = {}
            if extraction_context or intent in ["registration", "booking"]:
                # CRITICAL: Pass current step to prevent false positives (Issue: ID extracted as phone)
                current_step = booking_state.get("step") or session_data.get("current_step")
                
                entities = extractor.extract_name_and_phone(
                    message_text, 
                    phone_number,
                    context=extraction_context,
                    current_step=current_step
                )
                
                # Also extract date/time if in booking context (NOT during registration)
                # CRITICAL: Skip during registration to avoid extracting hour from national ID
                if (booking_state.get("started") or intent == "booking") and current_step not in ['registration_id', 'awaiting_id', 'awaiting_national_id', 'awaiting_name']:
                    date_time = extractor.extract_date_time(message_text)
                    entities.update(date_time)
                
                # LLM-FIRST: Service extraction should be done by LLM, not keywords
                # The LLM will understand service mentions in context
                logger.debug(f"✅ LLM-FIRST: Service extraction handled by entity extractor")
            else:
                logger.debug(f"⏭️ Skipping entity extraction (context: {intent})")
            
            # CRITICAL: Smart topic management on intent switch (Issue #18)
            # Clear or update topic when user switches context
            previous_intent = session_data.get("last_intent")
            current_topic = session_data.get("current_topic")
            
            # Define intents that should clear topic
            # CRITICAL: DO NOT clear topic for chitchat/questions - users greet mid-conversation!
            topic_clearing_intents = ["patient", "feedback", "resource"]
            # Define intents that represent context switch from booking
            context_switch_intents = {
                "booking": ["view_appointments", "cancel"]  # These are still booking but different sub-intents
            }
            
            # ═══════════════════════════════════════════════════════════════
            # LLM-FIRST: Context switching based on LLM intent, not keywords
            # ═══════════════════════════════════════════════════════════
            
            # Check for explicit context switches based on LLM-detected intent
            if current_topic and intent != previous_intent:
                # User switches to completely different intent (LLM detected)
                if intent in topic_clearing_intents:
                    logger.info(f"🔄 TOPIC CLEARED: User switched from '{previous_intent}' to '{intent}' (was discussing '{current_topic}')")
                    session_data["previous_topic"] = current_topic
                    session_data["current_topic"] = None
                    current_topic = None
                elif intent in ["chitchat", "question"]:
                    # Preserve topic for greetings/questions - natural conversation flow
                    logger.info(f"💬 {intent.upper()} detected - preserving topic '{current_topic}' (natural mid-conversation)")
            
            # Update session with managed topic
            if "current_topic" in session_data:
                await self.session_manager.put_session(session_key, session_data)
            
            # Enhanced telemetry: Log entity extraction
            if entities:
                logger.info(f"📊 slot_extraction: session={session_key}, slots_found={list(entities.keys())}, values={entities}")
            
            # NAME EXTRACTION: Store directly without confirmation (user wants one-by-one flow)
            # Don't ask "هل اسمك فعلاً...?" - just accept and move to next step
            if entities.get('name'):
                old_name = session_data.get('confirmed_name')
                new_name = entities.get('name')
                
                # Audit trail
                await self.session_manager.audit_slot_change(
                    session_key,
                    'confirmed_name',
                    old_name,
                    new_name,
                    'extract',
                    entities.get('name_confidence')
                )
                
                session_data['confirmed_name'] = new_name
                await self.session_manager.put_session(session_key, session_data)
                logger.info(f"✅ Name extracted and stored: {new_name}")
            
            logger.info(f"📊 Intent classified: {intent} (confidence: {confidence:.2f})")
            
            # INTENT DRIFT DETECTION (Issue #38)
            # Track consecutive intent changes to detect user confusion/frustration
            previous_intent = session_data.get('last_intent')
            intent_history = session_data.get('intent_history', [])
            
            if previous_intent and previous_intent != intent:
                # Intent changed - track it
                intent_history.append(intent)
                # Keep only last 5 intents
                intent_history = intent_history[-5:]
                session_data['intent_history'] = intent_history
                
                # Check for natural conversation flow first (Issue #9: Expanded natural flows)
                # These are EXPECTED conversation progressions, NOT drift
                natural_flows = [
                    # Booking flow
                    ('chitchat', 'booking'),        # Greeting → "I want to book"
                    ('chitchat', 'registration'),   # Greeting → New patient registration (ADDED)
                    ('chitchat', 'question'),       # Greeting → "What services?"
                    ('chitchat', 'selection'),      # Greeting → Direct selection
                    ('booking', 'registration'),    # Booking → Needs to register first (ADDED)
                    ('booking', 'question'),        # "I want to book" → "What services?"
                    ('booking', 'selection'),       # "I want to book" → Selects service
                    ('question', 'selection'),      # Asked about service → Selects it
                    ('question', 'follow_up'),      # User asks → Wants more details
                    ('question', 'booking'),        # Asked about services → Decides to book
                    ('selection', 'booking'),       # Selection → Proceeds to booking
                    ('selection', 'confirmation'),  # User selects service → Confirms booking
                    ('selection', 'follow_up'),     # Selected service → Asks more about it
                    ('follow_up', 'selection'),     # Asked details → Makes selection
                    ('follow_up', 'booking'),       # Asked details → Decides to book
                    ('registration', 'booking'),    # Completed registration → Proceed to booking (ADDED)
                ]
                
                is_natural_flow = (previous_intent, intent) in natural_flows
                
                # CRITICAL: Log major intent changes with reasoning (Issue: Intent drift visibility)
                major_intent_changes = [
                    ('chitchat', 'registration'),
                    ('chitchat', 'booking'),
                    ('question', 'booking'),
                    ('booking', 'registration')
                ]
                
                if (previous_intent, intent) in major_intent_changes:
                    # Log why this change happened
                    step = booking_state.get('step')
                    if step in ['awaiting_name', 'registration_name', 'needs_registration']:
                        logger.info(f"🔄 Intent change {previous_intent} → {intent}: System-initiated (patient not found, starting registration)")
                    else:
                        logger.info(f"🔄 Intent change {previous_intent} → {intent}: LLM detected from context (confidence={confidence:.2f}, msg='{message_text[:30]}...')")
                
                # Check for PROBLEMATIC intent bouncing (Issue #9 fix: Stricter criteria)
                # Exclude confirmation/cancel as they're expected follow-ups
                non_followup_intents = [i for i in intent_history if i not in ['confirmation', 'cancel']]
                unique_intents = len(set(non_followup_intents))
                
                # Real drift indicators:
                # 1. Ping-ponging: Same 2 intents alternating (e.g., question → booking → question → booking)
                # 2. Random jumping: 6+ unique intents in 5 turns (truly erratic behavior)
                # Note: 4-5 intents is normal (chitchat → registration → selection → confirmation)
                has_ping_pong = False
                if len(non_followup_intents) >= 4:
                    # Check for alternating pattern
                    for i in range(len(non_followup_intents) - 3):
                        if (non_followup_intents[i] == non_followup_intents[i+2] and 
                            non_followup_intents[i+1] == non_followup_intents[i+3] and
                            non_followup_intents[i] != non_followup_intents[i+1]):
                            has_ping_pong = True
                            break
                
                # Only flag drift if showing ACTUAL problematic patterns
                # FIXED: 4 intents is normal flow (chitchat → registration → selection → confirmation)
                # Only flag 6+ as truly erratic
                is_truly_erratic = unique_intents >= 6  # 6+ different intents = random jumping
                is_problematic = (has_ping_pong or is_truly_erratic) and not is_natural_flow
                
                if is_problematic:
                    if has_ping_pong:
                        logger.warning(
                            f"⚠️ INTENT PING-PONG DETECTED: User alternating between intents: {non_followup_intents}"
                        )
                    else:
                        logger.warning(
                            f"⚠️ INTENT DRIFT DETECTED: {unique_intents} unique intents in {len(non_followup_intents)} turns: {sorted(set(non_followup_intents))}"
                        )
                    logger.warning("⚠️ Possible user confusion or frustration — consider offering help")
                    session_data['intent_drift_detected'] = True
                elif is_natural_flow:
                    logger.debug(f"✅ Natural flow: {previous_intent} → {intent} (expected progression)")
                    session_data['intent_drift_detected'] = False
                else:
                    session_data['intent_drift_detected'] = False
            
            # Enhanced telemetry: Log intent classification details
            logger.info(f"📊 intent_classification: session={session_key}, intent={intent}, confidence={confidence:.2f}, prev_intent={previous_intent or 'none'}, drift={session_data.get('intent_drift_detected', False)}")
            
            # CRITICAL: Detect and store user's language preference
            from ..utils.language_detector import detect_language
            detected_language = detect_language(message_text)
            
            # Check for explicit language preference statements
            message_lower = message_text.lower()
            explicit_english = any(phrase in message_lower for phrase in [
                "speak english", "in english", "i need to speak in english",
                "can you speak english", "english please", "talk in english"
            ])
            explicit_arabic = any(phrase in message_lower for phrase in [
                "بالعربي", "تكلم عربي", "أريد العربية", "عربي من فضلك"
            ])
            
            # Store language preference
            if explicit_english:
                session_data["preferred_language"] = "english"
                session_data["language_explicitly_requested"] = True
                logger.info(f"🌐 EXPLICIT LANGUAGE REQUEST: User wants English")
            elif explicit_arabic:
                session_data["preferred_language"] = "arabic"
                session_data["language_explicitly_requested"] = True
                logger.info(f"🌐 EXPLICIT LANGUAGE REQUEST: User wants Arabic")
            elif not session_data.get("preferred_language"):
                # Auto-detect on first message
                session_data["preferred_language"] = detected_language
                logger.info(f"🌐 LANGUAGE DETECTED: {detected_language}")
            
            # Update detected language for current message
            session_data["current_message_language"] = detected_language
            
            # Special handling for explicit language requests
            if explicit_english or explicit_arabic:
                preferred_lang = session_data["preferred_language"]
                
                if preferred_lang == "english":
                    language_response = "Sure! I'll communicate with you in English from now on. 😊\nHow can I help you today?"
                else:
                    language_response = "تمام! بتكلم معاك بالعربي من الحين 😊\nوش تبغى اليوم؟"
                
                logger.info(f"✅ LANGUAGE PREFERENCE SET: {preferred_lang}")
                
                # Save session with language preference
                await self.session_manager.put_session(session_key, session_data)
                
                # Send immediate response
                try:
                    await self.wasender_client.send_message(phone_number, language_response)
                    await self.session_manager.add_to_history(session_key, "assistant", language_response)
                except Exception as e:
                    logger.error(f"Failed to send language confirmation: {e}")
                
                return {
                    "response": language_response,
                    "intent": "language_preference",
                    "status": "language_set",
                    "language": preferred_lang
                }
            
            # Add user message to history (keep last 30 messages for better context)
            await self.session_manager.add_to_history(session_key, "user", message_text, max_history=30)
            
            # 🐛 FIX: Refresh session_data after adding to history
            # Previously, session_data was fetched at line 736 and never refreshed,
            # causing history to always show 0 messages even though it was saved to Redis
            session_data = await self.session_manager.get_session(session_key) or {}
            logger.debug(f"📚 Refreshed session_data - history now has {len(session_data.get('history', []))} messages")
            
            # CRITICAL: Save current intent as prev_intent for next message
            session_data["last_intent"] = intent
            session_data["last_message"] = message_text
            session_data["last_activity_timestamp"] = time.time()
            
            # SERVICE SELECTION: Check if user is selecting a service by number or name
            selected_service = None
            last_shown_services = session_data.get("last_shown_services", [])
            
            if last_shown_services and intent in ["selection", "follow_up"]:
                message_stripped = message_text.strip()
                
                # Try matching by number (e.g., "1", "٢", "15", "٢٥")
                # Convert Arabic numerals to Western first
                arabic_numerals = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}
                converted_num = "".join(arabic_numerals.get(char, char) for char in message_stripped)
                
                if converted_num.isdigit():
                    service_num = int(converted_num) - 1  # 0-indexed
                    
                    if 0 <= service_num < len(last_shown_services):
                        selected_service = last_shown_services[service_num]
                        logger.info(f"🎯 SERVICE SELECTED BY NUMBER: {service_num + 1} -> {selected_service['name']}")
                        
                        # CRITICAL: Override message_text with service name so LLM understands
                        original_message = message_text
                        message_text = selected_service['name']
                        payload["message"] = selected_service['name']
                        logger.info(f"✅ MESSAGE TRANSFORMED: '{original_message}' → '{message_text}' for LLM clarity")
                else:
                    # Try matching by name (Arabic or English)
                    # CRITICAL FIX (Bug 26): Don't auto-select if user just said category name!
                    # User: "ليزر" → Show variants, don't pick first one
                    # User: "ليزر منطقة صغيرة 100" → Select specific service
                    message_lower = message_text.lower().strip()
                    message_words = len(message_lower.split())
                    
                    # Only select if message is specific enough (3+ words or exact match)
                    # Short messages like "ليزر" are likely category names, not selections
                    for svc in last_shown_services:
                        svc_name_ar = svc.get("name", "").lower()
                        svc_name_en = svc.get("name_en", "").lower()
                        
                        # EXACT match (user typed full service name)
                        is_exact_match = (svc_name_ar == message_lower) or (svc_name_en == message_lower)
                        
                        # CONTAINS match BUT only if message is specific (3+ words)
                        is_specific_partial = (
                            message_words >= 3 and (
                                (svc_name_ar and message_lower in svc_name_ar) or
                                (svc_name_en and message_lower in svc_name_en)
                            )
                        )
                        
                        if is_exact_match or is_specific_partial:
                            selected_service = svc
                            logger.info(f"🎯 SERVICE SELECTED BY NAME: '{message_text}' -> {svc['name']} (exact={is_exact_match}, specific={is_specific_partial})")
                            break
                        else:
                            # Partial match but too short - likely category name
                            if message_lower in svc_name_ar or svc_name_ar in message_lower:
                                logger.debug(f"⏭️ SKIPPING partial match: '{message_text}' in '{svc['name']}' (too short - likely category, not selection)")
                
                # Save selected service to session (Issue #11 & #22: Detect changes)
                if selected_service:
                    previous_selection = session_data.get("selected_service_name")
                    previous_topic = session_data.get("current_topic")
                    is_changing_selection = previous_selection and previous_selection != selected_service.get("name")
                    
                    # CRITICAL FIX: Extract service TYPE (high-level) instead of variant name
                    # Variant: "عدد 3 جلسات بوكسر + الابط مع الرتوش 594"
                    # Type: "ليزر" or "بوتوكس" etc.
                    service_full_name = selected_service.get("name", "")
                    service_type = self._extract_service_type(service_full_name)
                    
                    # Issue #22: Detect topic changes
                    is_changing_topic = (
                        previous_topic and 
                        previous_topic != service_type and
                        previous_topic != previous_selection  # Not just selection/topic sync
                    )
                    
                    if is_changing_selection:
                        logger.warning(f"🔄 SELECTION CHANGED: {previous_selection} → {service_full_name}")
                        session_data["selection_changed"] = True
                        session_data["previous_selection"] = previous_selection
                    elif is_changing_topic:
                        # Topic change without selection change (Issue #22)
                        logger.warning(f"🔄 TOPIC CHANGED: {previous_topic} → {service_type}")
                        session_data["topic_changed"] = True
                        session_data["previous_topic_before_change"] = previous_topic
                        session_data["selection_changed"] = False
                        session_data["previous_selection"] = None
                    else:
                        session_data["selection_changed"] = False
                        session_data["previous_selection"] = None
                        session_data["topic_changed"] = False
                        session_data["previous_topic_before_change"] = None
                    
                    # CRITICAL FIX (Bug 24): Extract service_type_id for LangGraph
                    # LangGraph needs this to fetch services when user confirms booking
                    service_type_id = selected_service.get("service_type_id")
                    
                    session_data["selected_service_id"] = selected_service.get("id")
                    session_data["selected_service_name"] = service_full_name  # Full variant name
                    session_data["selected_service_price"] = selected_service.get("price")
                    session_data["selected_service_type_id"] = service_type_id  # ✅ NEW: For LangGraph
                    session_data["current_topic"] = service_type  # HIGH-LEVEL type, not variant!
                    session_data["last_discussed_service"] = service_full_name  # Keep full name for booking
                    
                    logger.info(f"💾 SAVED SELECTED SERVICE: id={selected_service.get('id')}, variant={service_full_name}, type={service_type}, type_id={service_type_id}, price={selected_service.get('price')}")
            
            # UPDATE TOPIC: If user mentions a service name, set it as current topic
            if not selected_service and intent in ["selection", "question", "follow_up"]:
                # Check if message contains a service name
                message_lower = message_text.lower()
                service_keywords = ["علاج", "ليزر", "استشارة", "فحص", "therapy", "laser", "consultation"]
                if any(keyword in message_lower for keyword in service_keywords):
                    # Extract service TYPE from user message
                    service_type = self._extract_service_type(message_text.strip())
                    session_data["current_topic"] = service_type  # HIGH-LEVEL type
                    session_data["last_discussed_service"] = message_text.strip()  # Full message
                    logger.info(f"📌 TOPIC SET: {service_type}")
            
            # USER JOURNEY TRACKING: Determine which phase user is in (Issue #20)
            # Phase State Machine:
            # discovery → interest → detail → booking → confirmation
            previous_phase = session_data.get("journey_phase", "discovery")
            
            # CRITICAL: Preserve advanced phases for casual greetings (Issue #54)
            # User saying "هلا" mid-booking shouldn't destroy progress!
            advanced_phases = ["booking", "confirmation", "detail"]
            has_booking_context = session_data.get("booking_state", {}).get("started") or session_data.get("current_topic")
            
            if intent == "chitchat":
                # If user is in advanced phase or has booking context, DON'T reset!
                if previous_phase in advanced_phases or has_booking_context:
                    journey_phase = previous_phase  # PRESERVE phase
                    logger.info(f"💬 Chitchat in {previous_phase} phase - preserving context (topic: {session_data.get('current_topic')})")
                else:
                    journey_phase = "discovery"
            elif intent == "question":
                # Questions can happen at any phase - preserve context
                if previous_phase in advanced_phases or has_booking_context:
                    journey_phase = previous_phase
                    logger.info(f"❓ Question in {previous_phase} phase - preserving context")
                else:
                    journey_phase = "discovery"
            elif intent == "selection":
                # CRITICAL FIX: Don't blindly set to "interest" - check if we're already in booking flow!
                if previous_phase in ["booking", "confirmation"]:
                    journey_phase = previous_phase  # Preserve advanced phase
                    logger.info(f"✅ Selection in {previous_phase} phase - preserving (user refining choice)")
                elif session_data.get("selected_service_id"):
                    # User selected a service - move to detail phase (show service info)
                    journey_phase = "detail"
                    logger.info(f"📋 Service selected - moving to detail phase")
                else:
                    journey_phase = "interest"
                
                # When user selects a service, save it as topic (avoid PII)
                if not session_data.get("current_topic"):
                    import re as _re
                    has_long_digits = _re.search(r"\d{8,12}", message_text) is not None
                    mentioned_service = entities.get("mentioned_service") if 'entities' in locals() else None
                    if mentioned_service and not has_long_digits:
                        # Extract service type from mention
                        service_type = self._extract_service_type(mentioned_service)
                        session_data["current_topic"] = service_type
                        logger.info(f"📌 TOPIC SET (selection): {service_type}")
                    else:
                        logger.debug("⏭️ Skipping topic set for selection: no service mention or message contains potential PII")
            elif intent == "follow_up":
                # Follow-up moves to detail only if we're not already in booking
                if previous_phase in ["booking", "confirmation"]:
                    journey_phase = previous_phase
                else:
                    journey_phase = "detail"
            elif intent == "booking":
                journey_phase = "booking"
            elif intent == "confirmation":
                # Confirmation advances the phase
                if previous_phase == "booking":
                    journey_phase = "confirmation"
                elif previous_phase in ["interest", "detail"]:
                    journey_phase = "booking"  # Advance to booking
                else:
                    journey_phase = "confirmation"
            else:
                journey_phase = previous_phase  # Maintain current phase
            
            # Log phase transitions (Issue #20)
            if journey_phase != previous_phase:
                logger.info(f"🔄 PHASE TRANSITION: {previous_phase} → {journey_phase} (intent={intent})")
            else:
                logger.debug(f"📍 PHASE MAINTAINED: {journey_phase} (intent={intent})")
            
            session_data["journey_phase"] = journey_phase
            session_data["phase_history"] = session_data.get("phase_history", []) + [{
                "phase": journey_phase,
                "intent": intent,
                "timestamp": time.time()
            }]
            # Keep only last 10 phase transitions
            session_data["phase_history"] = session_data["phase_history"][-10:]
            
            # EXPLICIT STATE PERSISTENCE LOG
            updated_topic = session_data.get("current_topic")
            preferred_lang = session_data.get("preferred_language", "arabic")
            logger.info(f"💾 SESSION STATE SAVED: intent={intent}, topic={updated_topic}, phase={journey_phase}, turn={context['conversation_turn']}, language={preferred_lang}")
            await self.session_manager.put_session(session_key, session_data)
            
            # Handle chitchat, questions, and conversational intents with LLM
            # BUT: If booking is active and intent is confirmation, route to booking agent
            if intent == "confirmation" and booking_state.get("started"):
                # Don't log here - will log below when routing
                # Let it fall through to booking agent handling below
                pass
            elif intent == "confirmation" and not booking_state.get("started"):
                # CRITICAL: User confirmed interest in a service but booking not started yet
                # This happens when conversational agent shows services and user says "yes"
                # Solution: Start booking flow with the discussed service
                logger.info(f"✅ [CONFIRMATION] User confirmed interest but no booking started - initiating booking flow")
                
                # Check if there's a service discussed in conversation history
                history = session_data.get("history", [])
                last_bot_message = None
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        last_bot_message = msg.get("content", "")
                        break
                
                # If bot just showed services, extract service name and start booking
                if last_bot_message and any(word in last_bot_message for word in ["بوتوكس", "ليزر", "فيلر", "تنظيف", "ميزو", "خيوط", "تقشير"]):
                    logger.info(f"🎯 Service discussion detected in last message, treating confirmation as booking intent")
                    logger.info(f"🔀 [INTENT_OVERRIDE] confirmation → booking (service context: {last_bot_message[:50]}...)")
                    intent = "booking"  # Change intent to booking to trigger booking flow
                    
                    # CRITICAL: Update session to reflect intent change
                    session_data["last_intent"] = "booking"
                    session_data["router_intent"] = "booking"
                    await self.session_manager.put_session(session_key, session_data)
                    
                    # Fall through to booking handling below - skip chitchat handler
                else:
                    # No clear service context - treat as chitchat
                    logger.warning(f"⚠️ Confirmation without clear service context - treating as chitchat")
                    intent = "chitchat"
            
            # CRITICAL: Smart routing - if in active booking, chitchat/questions stay in booking flow!
            # Human receptionist doesn't abandon booking when customer says "hi" or asks a question
            current_step = booking_state.get("step", "")
            is_error_state = current_step.startswith("error_") or current_step.endswith("_error") if current_step else False
            is_loop_state = current_step in ["loop_detected", "loop_help"] if current_step else False
            
            # CRITICAL: step="start", None, error, or loop means NOT in active booking
            # These states should use conversational agent, not booking agent
            should_use_conversational = current_step in [None, "", "start", "loop_detected", "loop_help"] or is_error_state
            
            # Active booking = in advanced phase AND has actual booking progress (not start/error/loop)
            in_active_booking = (
                journey_phase in ["booking", "confirmation", "detail"] and
                not should_use_conversational
            ) or (
                booking_state.get("started") and 
                current_step and 
                not should_use_conversational
            )
            
            # 🤖 INTELLIGENT AGENT - THE BRAIN (handles EVERYTHING)
            use_intelligent_agent = getattr(settings, 'use_intelligent_agent', False)
            
            if use_intelligent_agent:
                #  🧠 LET INTELLIGENT AGENT HANDLE EVERYTHING (including registration)
                # Intelligent agent is smart enough to handle registration on its own
                patient_data = context.get("patient_data", {})
                needs_registration = not patient_data.get("already_registered", False)
                
                if needs_registration:
                    logger.info(f"ℹ️ [INTELLIGENT AGENT] Patient not registered - will handle registration within intelligent agent")
                    logger.info(f"   Turn: {current_turn}, patient_data: {patient_data}")
                    # NO OVERRIDE - let intelligent agent handle it naturally
                
                # Always use intelligent agent when enabled
                if True:
                    # Patient is registered OR we've tried enough times - let intelligent agent handle
                    logger.info(f"🤖 [INTELLIGENT AGENT] LLM brain handles ALL messages - intent: {intent}")
                    
                    try:
                        from ..agents.intelligent_agent_factory import get_intelligent_agent
                        
                        intelligent_agent = get_intelligent_agent()
                        
                        # Get conversation history
                        conversation_history = context.get("conversation_history", [])
                        patient_data = context.get("patient_data")
                        
                        # CRITICAL: Pass RAW original message, NOT transformed version!
                        raw_message = payload.get("message", "")
                        logger.info(f"🤖 [INTELLIGENT AGENT] Processing RAW message: '{raw_message[:50]}...'")
                        
                        # Call intelligent agent (LLM decides everything!)
                        result = await intelligent_agent.handle(
                            message=raw_message,  # RAW message - let AI be smart!
                            conversation_history=conversation_history,
                            patient_data=patient_data,
                            session_data=session_data
                        )
                        
                        # 🚨 CRITICAL: Save NEW conversation messages (including tool calls!) to Redis
                        if "conversation_history" in result:
                            history_cache = self._get_history_cache()
                            returned_history = result["conversation_history"]
                            
                            # Only save NEW messages added this turn (not old messages from previous turns)
                            # The intelligent agent appends new messages to the history it received
                            original_history_len = len(conversation_history)  # What we sent to agent
                            returned_history_len = len(returned_history)
                            
                            # 🚨 CRITICAL: Agent may trim history (e.g., 30→10 for selection context)
                            # If returned < original, agent trimmed - save last N messages
                            # If returned > original, agent added new - save the difference
                            if returned_history_len < original_history_len:
                                # Agent trimmed history - save the returned messages (they're the most recent)
                                logger.info(f"📊 History trimmed by agent: {original_history_len}→{returned_history_len}")
                                logger.info(f"💾 Saving last {min(5, returned_history_len)} messages from trimmed history")
                                await history_cache.batch_add_messages(session_key, returned_history[-min(5, returned_history_len):])
                            else:
                                # Normal case: Agent added new messages
                                new_messages = returned_history[original_history_len:]
                                logger.info(f"📊 History lengths: original={original_history_len}, returned={returned_history_len}, new={len(new_messages)}")
                                
                                if new_messages:
                                    await history_cache.batch_add_messages(session_key, new_messages)
                                    logger.info(f"💾 Saved {len(new_messages)} NEW messages to Redis")
                                else:
                                    logger.warning(f"⚠️ No new messages - lengths equal. Saving last 5 as fallback")
                                    if returned_history_len > 0:
                                        await history_cache.batch_add_messages(session_key, returned_history[-5:])
                        else:
                            logger.warning("⚠️ No conversation_history returned from intelligent agent!")
                        
                        logger.info(f"✅ [INTELLIGENT AGENT] Response generated: {len(result.get('response', ''))} chars")
                        
                        # CRITICAL: Detect if response contains a numbered list and set session flag
                        response_text = result.get("response", "")
                        import re
                        numbered_patterns = [r'\n\s*\d+[\.\-\)]\s+', r'\n\s*[١-٩]+[\.\-\)]\s+']
                        has_numbered_list = any(re.search(pattern, response_text) for pattern in numbered_patterns)
                        
                        if has_numbered_list:
                            logger.info(f"📋 [INTELLIGENT AGENT] Detected numbered list in response - setting awaiting_selection flag")
                            session_data["awaiting_selection"] = True
                            session_data["last_list_timestamp"] = time.time()
                            # Save immediately so next request can see it
                            await self.session_manager.put_session(session_key, session_data)
                        
                        # 🚨 CRITICAL: Save selection_map to session if agent returned one
                        if result.get("last_selection_map"):
                            session_data["last_selection_map"] = result.get("last_selection_map")
                            logger.warning(f"💾 Saved selection_map to session ({len(result.get('last_selection_map'))} chars)")
                            
                            # Also save selection type
                            if result.get("last_selection_type"):
                                session_data["last_selection_type"] = result.get("last_selection_type")
                                logger.info(f"💾 Saved selection_type: {result.get('last_selection_type')}")
                            
                            await self.session_manager.put_session(session_key, session_data)
                        
                        # 🚨 CRITICAL: Save selected IDs for context preservation
                        needs_session_update = False
                        if result.get("selected_service_id"):
                            session_data["selected_service_id"] = result.get("selected_service_id")
                            logger.info(f"💾 Saved selected_service_id={result.get('selected_service_id')} to session")
                            needs_session_update = True
                        
                        if result.get("selected_doctor_id"):
                            session_data["selected_doctor_id"] = result.get("selected_doctor_id")
                            logger.info(f"💾 Saved selected_doctor_id={result.get('selected_doctor_id')} to session")
                            needs_session_update = True
                        
                        if needs_session_update:
                            await self.session_manager.put_session(session_key, session_data)
                        
                        # 🚨 CRITICAL: Update patient_data in session if registration happened
                        if result.get("patient_data"):
                            session_data["patient_data"] = result.get("patient_data")
                            logger.warning(f"✅ Updated patient_data in session after registration (ID={result.get('patient_data', {}).get('id')})")
                            await self.session_manager.put_session(session_key, session_data)
                        
                        # Send response
                        await self._send_whatsapp_response(phone_number, result.get("response"))
                        
                        return {
                            "response": result.get("response"),
                            "intent": intent,
                            "status": result.get("status", "success"),
                            "agent": "IntelligentBookingAgent"
                        }
                    
                    except Exception as e:
                        logger.error(f"❌ [INTELLIGENT AGENT] Error: {e}", exc_info=True)
                        
                        # 🚨 CRITICAL: Check if error is due to OpenAI API rejection
                        error_str = str(e).lower()
                        is_api_error = any(keyword in error_str for keyword in ["400", "bad request", "invalid", "openai"])
                        
                        if is_api_error:
                            logger.error(f"🚨 OpenAI API error detected - sending error message to user")
                            await self._send_whatsapp_response(
                                phone_number, 
                                "عذراً، حصل خطأ مؤقت في النظام. ممكن تعيد رسالتك؟ أو تواصل معنا على 920033304 📞"
                            )
                            return {
                                "response": "Error sent to user",
                                "intent": intent,
                                "status": "api_error",
                                "agent": "IntelligentBookingAgent"
                            }
                        
                        # For other errors, try to recover
                        logger.warning("⚠️ Attempting to recover from error...")
                        
                        # Check if user was in middle of something
                        has_context = (
                            session_data.get("last_selection_map") or
                            session_data.get("awaiting_selection") or
                            booking_state.get("started")
                        )
                        
                        if has_context:
                            # User was doing something - acknowledge and ask to continue
                            await self._send_whatsapp_response(
                                phone_number,
                                "عذراً، حصلت مشكلة بسيطة. ممكن تعيد طلبك؟ 🙏"
                            )
                        else:
                            # No context - route to resource agent for general help
                            logger.warning("⚠️ No active context - routing to resource agent")
                            from ..agents.resource_agent import ResourceAgent
                            agent = ResourceAgent(session_key)
                            return await agent.handle(payload, context)
                        
                        return {
                            "response": "Error recovery attempted",
                            "intent": intent,
                            "status": "error_recovered",
                            "agent": "ErrorHandler"
                        }
            
            # CRITICAL: If should use conversational OR user sends chitchat/greeting, use conversational agent
            if should_use_conversational and intent in ["chitchat", "question", "greeting"]:
                # CRITICAL FIX: DON'T clear state if there's active context!
                # Check if user has ongoing conversation (topic, service discussion, etc.)
                has_active_context = (
                    current_topic  # Has topic set
                    or session_data.get("last_discussed_service")  # Was discussing service
                    or session_data.get("selected_service_id")  # Selected service
                    or booking_state.get("started")  # Booking started
                )
                
                if has_active_context:
                    logger.info(f"💬 [CONVERSATIONAL] User sent {intent} at step '{current_step}' - PRESERVING context (has active topic/service)")
                    logger.info(f"   Topic: {current_topic}, Service: {session_data.get('last_discussed_service')}")
                    # DON'T clear - user is continuing conversation
                else:
                    logger.info(f"💬 [CONVERSATIONAL] User sent {intent} at step '{current_step}' - using conversational agent")
                    # Clear ALL stale booking state to prevent routing issues
                    # NOTE: Numbered response misclassification is now handled earlier (line 2304-2307)
                    booking_state["step"] = None
                    booking_state["started"] = False
                    session_data["booking_state"] = {}  # Clear entire booking state
                    session_data["error_turn_count"] = 0
                    await self.session_manager.put_session(session_key, session_data)
                    logger.info(f"✅ Cleared all booking state - ready for fresh conversational flow")
                # Don't route to booking agent - fall through to conversational agent
            elif in_active_booking and intent in ["chitchat", "question"]:
                logger.info(f"🎯 [ROUTING] {intent.upper()} during active booking - preserving context")
                logger.info(f"   Phase: {journey_phase}, Step: {current_step}")
                logger.info(f"   Decision: Route to BOOKING AGENT (user clarifying/being polite, don't break flow)")
                # Let it fall through to booking agent routing below
                # User is just being polite or asking clarification - don't exit booking!
            elif intent in ["chitchat", "question", "cancel", "selection", "follow_up"]:
                # CRITICAL ROUTING DECISION: Use conversational agent for pre-booking interactions
                # This allows agent to understand context, store selected service, and prepare for booking
                logger.info(f"💬 [ROUTING] {intent.upper()} → CONVERSATIONAL AGENT")
                logger.info(f"   Reason: Pre-booking interaction (understanding context, storing selection)")
                logger.info(f"   Next: If user confirms booking, will route to BOOKING AGENT with full context")
                
                # Clear booking state if user wants to cancel
                if intent == "cancel" and booking_state.get("started"):
                    session_data["booking_state"] = {}
                    await self.session_manager.put_session(session_key, session_data)
                    logger.info("🗑️ Booking cancelled by user")
                
                # Build rich context for LLM with conversation history
                history = session_data.get("history", [])[-15:]  # Last 10 exchanges for full context
                
                # Check if we should offer booking (after service selection)
                show_booking_offer = session_data.get("show_booking_offer", False)
                
                # Detect greeting type for proper cultural response
                greeting_type = None
                message_lower = message_text.lower().strip()
                if "السلام" in message_lower or "سلام" in message_lower:
                    greeting_type = "salam"  # Reply: وعليكم السلام
                elif "صباح" in message_lower:
                    greeting_type = "morning"  # Reply: صباح النور
                elif "مساء" in message_lower:
                    greeting_type = "evening"  # Reply: مساء النور
                
                # CRITICAL: Fetch real services if user is asking about them
                real_services = None
                is_service_inquiry = any(word in message_text.lower() for word in [
                    "خدمة", "خدمات", "service", "services", "عندكم", "وش عندكم", "ايش عندكم",
                    "متوفر", "موجود", "available", "فيه", "في عندكم"
                ])
                
                # CRITICAL: Also fetch for selection/follow_up - user naming/asking about a service! (Issue #42)
                if is_service_inquiry or intent in ["question", "selection", "follow_up"]:
                    try:
                        from ..api.agent_api import AgentApiClient
                        from ..cache.service_cache import service_cache
                        api_client = AgentApiClient()
                        
                        # 🚀 PERFORMANCE: Use cached services instead of fetching every time (Issue #10)
                        # Cache expires after 5 minutes, reduces API calls from every request to every 5 min
                        message_lower = message_text.lower()
                        services_list = await service_cache.get_services(api_client)
                        
                        # Enhanced logging (Issue #27)
                        if services_list and len(services_list) > 0:
                            logger.debug(f"✅ Using {len(services_list)} services from cache")
                            
                            if settings.app_env != "production":
                                sample_service = services_list[0]
                                logger.debug(f"📋 Sample service structure: {list(sample_service.keys())}")
                                logger.debug(
                                    f"📋 Sample service: name_ar={sample_service.get('name_ar')}, "
                                    f"nameAr={sample_service.get('nameAr')}, "
                                    f"name={sample_service.get('name')}, "
                                    f"nameEn={sample_service.get('nameEn')}, "
                                    f"price={sample_service.get('price')}"
                                )
                                logger.debug(f"📋 First 5 services from API:")
                                for i, s in enumerate(services_list[:5]):
                                    sname = s.get("name_ar") or s.get("nameAr") or s.get("name") or s.get("nameEn") or "?"
                                    logger.debug(f"   {i+1}. {sname} (id={s.get('id')}, price={s.get('price')})")
                        else:
                            # Enhanced diagnostics for empty services (Issue #27)
                            logger.error(f"🚨 SERVICE FETCH FAILED: No services in response")
                            logger.error(f"🚨 DIAGNOSIS:")
                            logger.error(f"   - Response structure: {services_result}")
                            logger.error(f"   - Has 'results' key: {'results' in services_result if isinstance(services_result, dict) else False}")
                            logger.error(f"   - Has 'data' key: {'data' in services_result if isinstance(services_result, dict) else False}")
                            logger.error(f"   - Response length: {len(services_result) if isinstance(services_result, (list, dict)) else 'N/A'}")
                            logger.error(f"🚨 POSSIBLE CAUSES:")
                            logger.error(f"   1. Database is empty (no services configured)")
                            logger.error(f"   2. API query parameters wrong (limit=50 might be invalid)")
                            logger.error(f"   3. API endpoint /services not returning expected format")
                            logger.error(f"   4. Backend database connection issue")
                            logger.error(f"🚨 IMPACT: Users cannot browse or select services")
                        
                        # Filter services that match keywords in the user's message
                        # Enhanced matching: Show all service variants using service_type_id (Issue #47)
                        matched_services = []
                        matched_base_service = None
                        
                        # Extract keywords from user message (clean, significant words)
                        user_keywords = [w for w in message_lower.split() if len(w) >= 3]
                        
                        # CRITICAL: Skip service matching if user already selected a service!
                        # (e.g., user typed "10" after seeing numbered list - don't re-fetch list!)
                        user_already_selected = session_data.get("selected_service_id") is not None
                        
                        if user_already_selected:
                            logger.info(f"✅ User already selected service (ID: {session_data.get('selected_service_id')}) - skipping variant fetch")
                            matched_base_service = None
                            matched_services = []
                        
                        # Step 1: Find the base service (e.g., "فيلر")
                        if not user_already_selected:
                            for service in services_list:
                                # Try multiple possible name fields (Issue #43)
                                service_name_ar = service.get("name_ar") or service.get("nameAr") or service.get("arabic_name") or ""
                                service_name_en = service.get("name") or service.get("name_en") or service.get("nameEn") or service.get("english_name") or ""
                                service_name = (service_name_ar or service_name_en).lower()
                                
                                if not service_name:
                                    logger.debug(f"⚠️ Service has no name: {service}")
                                    continue
                                
                                # Check if service name CONTAINS any keyword from user message (Issue #47)
                                for keyword in user_keywords:
                                    if keyword in service_name:
                                        display_name = service_name_ar or service_name_en or service.get("name") or "خدمة"
                                        service_id = service.get('id')
                                        logger.info(f"🔍 Matched base service: {display_name} (keyword: {keyword}, service_id: {service_id})")
                                        matched_base_service = service
                                        break
                                
                                if matched_base_service:
                                    break
                        
                        # Step 2: If base service found, fetch all its variants (Issue #47)
                        if matched_base_service:
                            service_type_id = matched_base_service.get('id')
                            logger.info(f"🔎 FETCHING VARIANTS: GET /services/?service_type_id={service_type_id}")
                            
                            # CRITICAL FIX (Bug 28): Save service_type_id to session when fetching variants
                            # This prevents error_no_service_type loop on resume!
                            session_data["selected_service_type_id"] = service_type_id
                            await self.session_manager.put_session(session_key, session_data)
                            logger.debug(f"💾 Saved service_type_id={service_type_id} to session (prevents error loop)")
                            
                            try:
                                # Fetch all variants of this service type
                                variants_result = await api_client.get("/services/", params={"service_type_id": service_type_id})
                                variants_list = variants_result.get("results") or variants_result.get("data") or []
                                
                                if variants_list:
                                    logger.info(f"✅ VARIANTS FETCH SUCCESS: {len(variants_list)} variants for {matched_base_service.get('name')}")
                                    matched_services = variants_list
                                    
                                    # Log variants
                                    for i, v in enumerate(variants_list[:5]):
                                        vname = v.get("name_ar") or v.get("nameAr") or v.get("name") or v.get("nameEn") or "?"
                                        logger.info(f"   Variant {i+1}: {vname} (price={v.get('price')})")
                                else:
                                    logger.warning(f"⚠️ VARIANTS FETCH: No variants found for service_type_id={service_type_id}, using base service")
                                    matched_services = [matched_base_service]
                            except Exception as e:
                                logger.error(f"🚨 VARIANTS FETCH FAILED: {e}", exc_info=True)
                                logger.error(f"🚨 FALLBACK: Using base service only")
                                # Fallback to base service
                                matched_services = [matched_base_service]
                        else:
                            logger.debug(f"No base service matched for keywords: {user_keywords}")
                        
                        # Helper: Extract price from multiple possible fields (Issue #18)
                        def extract_price(service):
                            """Try multiple price field variations from API"""
                            price = (
                                service.get("price") or
                                service.get("Price") or
                                service.get("service_price") or
                                service.get("servicePrice") or
                                service.get("amount") or
                                service.get("cost") or
                                service.get("value")
                            )
                            # Log first service price extraction for debugging
                            if service == (matched_services[0] if matched_services else (services_list[0] if services_list else None)):
                                logger.debug(f"💰 PRICE EXTRACTION: Available keys in service: {list(service.keys())}")
                                logger.debug(f"💰 PRICE EXTRACTION: price={service.get('price')}, Price={service.get('Price')}, service_price={service.get('service_price')}")
                                logger.debug(f"💰 PRICE EXTRACTION: Final extracted price={price}")
                            return price
                        
                        # If we found matching services, show them (Issue #47 - preserve raw API price)
                        if matched_services:
                            real_services = [{
                                "name": s.get("name_ar") or s.get("nameAr") or s.get("name") or s.get("nameEn") or "خدمة",
                                "name_en": s.get("name") or s.get("nameEn") or s.get("name_en") or "",
                                "price": extract_price(s),  # Issue #18: Try multiple price fields
                                "id": s.get("id"),
                                "matched": True  # Flag to indicate these are matched services
                            } for s in matched_services]  # Show ALL matched services
                            logger.info(f"✅ Found {len(real_services)} matching services for user query")
                        elif is_service_inquiry or services_list:
                            # Generic service inquiry OR any question/selection - show ALL services (no limit)
                            # CRITICAL: Filter out test services (test, zzzz, etc.)
                            filtered_services = [s for s in services_list 
                                                if s.get("name", "").lower() not in ["test", "zzzz"] 
                                                and s.get("name_ar", "").lower() not in ["test", "zzzz"]]
                            
                            real_services = [{
                                "name": s.get("name_ar") or s.get("nameAr") or s.get("name") or s.get("nameEn") or "خدمة",
                                "name_en": s.get("name") or s.get("nameEn") or s.get("name_en") or "",
                                "price": extract_price(s),  # Issue #18: Try multiple price fields
                                "id": s.get("id"),
                                "matched": False  # Generic list
                            } for s in filtered_services]  # Show ALL services (excluding test)
                            logger.info(f"✅ Showing {len(real_services)} services in context (matched: {is_service_inquiry}, filtered out test services)")
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch services for context: {e}")
                
                # Save services to session for selection matching
                if real_services:
                    session_data["last_shown_services"] = real_services
                    logger.debug(f"💾 Saved {len(real_services)} services to session for selection")
                
                # CRITICAL FIX (Bug 27): Show variants as numbered list, don't let LLM talk about them!
                # BUT: If user already selected a service (by number), DON'T show list again!
                user_already_selected = session_data.get("selected_service_id") is not None
                
                if matched_services and len(matched_services) > 1 and not user_already_selected:
                    logger.info(f"📋 SHOWING VARIANTS: Displaying {len(matched_services)} options as numbered list")
                    
                    # Build numbered list of variants
                    service_list = f"عندنا {len(matched_services)} خيار من {matched_base_service.get('name_ar') or matched_base_service.get('name')}:\n\n"
                    
                    for i, svc in enumerate(matched_services[:15], 1):  # Limit to 15
                        name = svc.get("name_ar") or svc.get("name") or svc.get("nameAr") or "خدمة"
                        price = extract_price(svc)
                        price_text = f" - {price} ريال" if price else ""
                        service_list += f"{i}. {name}{price_text}\n"
                    
                    service_list += f"\n💡 اختر الرقم أو اكتب اسم الخدمة"
                    
                    # Send directly without LLM
                    response_text = service_list
                    
                    # Save and update session
                    session_data["last_intent"] = intent
                    session_data["last_message"] = message_text
                    await self.session_manager.put_session(session_key, session_data)
                    
                    # Send response
                    await self._send_whatsapp_response(phone_number, response_text)
                    
                    logger.info(f"✅ Sent {len(matched_services)} service variants as numbered list")
                    
                    return {
                        "intent": intent,
                        "status": "success",
                        "response": response_text,
                        "source": "service_variants_list",
                        "confidence": confidence
                    }
                
                llm_context = {
                    "sender_name": sender_name,
                    "phone_number": phone_number,
                    "intent": intent,
                    "entities": entities,
                    "conversation_history": history,
                    "booking_active": booking_state.get("started", False),
                    "show_booking_offer": show_booking_offer,  # Flag to offer booking
                    "selected_service": session_data.get("last_discussed_service"),
                    "selected_service_details": {
                        "id": session_data.get("selected_service_id"),
                        "name": session_data.get("selected_service_name"),
                        "price": session_data.get("selected_service_price"),
                        "changed": session_data.get("selection_changed", False),  # Issue #11
                        "previous": session_data.get("previous_selection"),  # Issue #11
                        "topic_changed": session_data.get("topic_changed", False),  # Issue #22
                        "previous_topic": session_data.get("previous_topic_before_change")  # Issue #22
                    } if session_data.get("selected_service_id") else None,
                    "available_services": real_services,  # Real services from API
                    "greeting_type": greeting_type  # For proper cultural greeting response
                }
                
                # DIAGNOSTIC: Context window analysis (Issue #28, enhanced #45)
                history_count = len(history)
                history_chars = sum(len(str(msg.get("content", ""))) for msg in history)
                context_total_chars = len(str(llm_context))
                
                # Estimate tokens (rough: 1 token ≈ 4 characters for English, ≈ 2 for Arabic)
                # Arabic is more token-efficient than English
                # Arabic token estimation: ~1.7 chars per token (more accurate than 2.5)
                # Arabic is more token-dense than English due to diacritics and morphology
                estimated_tokens = int(context_total_chars / 1.7)  # Accurate for Arabic text
                
                # Check if history was truncated
                full_history_count = len(session_data.get("history", []))
                history_truncated = full_history_count > history_count
                
                # Current conversation turn for context
                current_conv_turn = context.get('conversation_turn', 0)
                
                logger.info(f"📊 CONTEXT WINDOW:")
                logger.info(f"  ├─ Turn: {current_conv_turn} (history expected: {current_conv_turn > 1})")
                logger.info(f"  ├─ History: {history_count} messages ({history_chars:,} chars)" + 
                       (f" [TRUNCATED from {full_history_count}]" if history_truncated else 
                        " [Turn 1 - no prior history]" if current_conv_turn == 1 else ""))
                logger.info(f"  ├─ Services: {len(real_services) if real_services else 0} items")
                logger.info(f"  ├─ Total context: {context_total_chars:,} chars")
                logger.info(f"  └─ Estimated tokens: ~{estimated_tokens:,} (max: ~8k, usage: {(estimated_tokens/8000)*100:.1f}%)")
                
                # Clear the flag after using it
                if show_booking_offer:
                    session_data["show_booking_offer"] = False
                    await self.session_manager.put_session(session_key, session_data)
                
                # LLM-ONLY: Always use LLM for service presentation
                logger.info(f"🤖 Using LLM for service inquiry - natural and context-aware")
                    
                # Build greeting - avoid double "يا" (Issue #43)
                if sender_name:
                    greeting = f"يا {sender_name}"
                else:
                    greeting = ""
                
                # Generate natural, context-aware response (LLM-ONLY)
                if is_service_inquiry and not real_services:
                    # Service inquiry but fetch failed - don't use LLM, give direct error
                    logger.error(f"🚨 CRITICAL: Service inquiry but no services fetched - preventing hanging response")
                    # Avoid double يا (Issue #45)
                    response_text = f"يا عيني {greeting} 😅\n\nما قدرت أوصل لقائمة الخدمات الحين\n\nجرب:\n• انتظر شوي وجرب مرة ثانية\n• أو اتصل على: 920033304 📞"
                else:
                    # 🧠 USE CONVERSATIONAL AGENT - Truly intelligent responses!
                    llm_start_time = time.time()
                    
                    # Pass session history for better context (Issue #9)
                    session_hist = session_data.get("history", [])
                    
                    # Try conversational agent first (intelligent)
                    try:
                        from ..core.conversational_agent import get_conversational_agent
                        from ..api.agent_api import AgentApiClient
                        
                        api_client = AgentApiClient()
                        agent = get_conversational_agent(self.llm_reasoner, api_client)
                        
                        # Convert session history to conversational agent format
                        conv_history = []
                        for msg in session_hist[-10:]:  # Last 10 messages
                            role = msg.get("role")
                            content = msg.get("content")
                            if role and content:
                                conv_history.append({"role": role, "content": content})
                        
                        logger.info("🧠 Using CONVERSATIONAL AGENT - Intelligent responses enabled")
                        
                        # Get patient data from session (already loaded early in flow)
                        user_name = session_data.get("sender_name") or sender_name
                        patient_data = session_data.get("patient_data")
                        
                        # Get selected service if user just selected one this turn
                        selected_svc = None
                        if session_data.get("selected_service_id") and session_data.get("selected_service_name"):
                            selected_svc = {
                                "id": session_data.get("selected_service_id"),
                                "name": session_data.get("selected_service_name"),
                                "price": session_data.get("selected_service_price")
                            }
                            logger.info(f"📌 Passing selected service to agent: {selected_svc['name']}")
                        
                        result = await agent.chat(
                            user_message=message_text,
                            conversation_history=conv_history,
                            user_name=user_name,
                            user_phone=phone_number,
                            patient_data=patient_data,  # Pass patient data to agent
                            selected_service=selected_svc  # Pass selected service to prevent wrong recovery
                        )
                        response_text = result["response"]
                        logger.info(f"✨ Conversational agent responded (function: {result.get('function_called')})")
                        
                        # If agent detected a name from conversation, save it
                        if result.get("detected_name"):
                            session_data["sender_name"] = result["detected_name"]
                            await self.session_manager.put_session(session_key, session_data)
                            logger.info(f"💾 Saved detected name: {result['detected_name']}")
                        
                    except Exception as conv_error:
                        # Fallback to old LLM reasoner if conversational agent fails
                        logger.warning(f"⚠️ Conversational agent failed: {conv_error}, falling back to LLM reasoner")
                        response_text = self.llm_reasoner.generate_reply(
                            user_id=phone_number,
                            user_message=message_text,
                            context=llm_context,
                            session_history=session_hist
                        )
                    
                    # Log context window improvement
                    if session_hist:
                        hist_chars = sum(len(str(m.get("content", ""))) for m in session_hist[-30:])
                        logger.info(f"📚 CONTEXT ENRICHED: {len(session_hist[-30:])} messages, ~{hist_chars} chars from session history")
                    
                    llm_duration = time.time() - llm_start_time
                    
                    # DIAGNOSTIC: Response metrics (Issue #28)
                    response_chars = len(response_text)
                    # Arabic token estimation: ~1.7 chars per token
                    response_tokens = int(response_chars / 1.7)  # Accurate for Arabic
                    
                    logger.info(f"📊 LLM RESPONSE METRICS (Issue #28):")
                    logger.info(f"  ├─ Duration: {llm_duration:.2f}s")
                    logger.info(f"  ├─ Response: {response_chars:,} chars")
                    logger.info(f"  └─ Estimated tokens: ~{response_tokens:,}")
                    
                    # CRITICAL: Log LLM response content (full response for debugging)
                    # Don't truncate - need full response for debugging issues
                    if len(response_text) <= 500:
                        logger.info(f"🤖 LLM RESPONSE: {response_text}")
                    else:
                        # For very long responses, log first 300 and last 200 chars
                        logger.info(f"🤖 LLM RESPONSE (first 300 chars): {response_text[:300]}")
                        logger.info(f"🤖 LLM RESPONSE (last 200 chars): ...{response_text[-200:]}")
                        logger.info(f"   💬 Full length: {len(response_text)} chars")
                    
                    # CRITICAL: Validate LLM response - check for hallucinated services
                    if real_services:
                        # Get list of valid service names
                        valid_service_names = set()
                        for svc in real_services:
                            name_ar = svc.get('name_ar') or svc.get('name') or svc.get('nameAr') or ""
                            if name_ar:
                                valid_service_names.add(name_ar.lower())
                        
                        # Common hallucinated services to watch for
                        hallucination_keywords = [
                            'علاج طبيعي', 'physical therapy',
                            'عملية', 'surgery', 'جراحة',
                            'تدليك', 'massage',
                            'حقن', 'injection' # Unless it's in our list
                        ]
                        
                        # Check if LLM mentioned non-existent services
                        response_lower = response_text.lower()
                        for keyword in hallucination_keywords:
                            if keyword in response_lower:
                                # Check if this is actually a valid service
                                is_valid = any(keyword in name for name in valid_service_names)
                                if not is_valid:
                                    logger.warning(f"⚠️ LLM may have hallucinated service: '{keyword}' not in our service list")
                                    logger.warning(f"   Valid services: {list(valid_service_names)[:5]}")
                    
                    # CRITICAL: Post-process to catch placeholder failures (Issue #14)
                    if '[السعر]' in response_text or '[اسم الخدمة]' in response_text:
                        logger.error(f"🚨 PLACEHOLDER FAILURE: LLM returned template with placeholders instead of actual data!")
                        logger.error(f"🚨 Response contains: {response_text}")
                        
                        # Replace placeholders with generic text
                        response_text = response_text.replace('[السعر]', 'حسب الاستشارة')
                        response_text = response_text.replace('[اسم الخدمة]', 'الخدمة')
                        
                        logger.warning(f"⚠️ Replaced placeholders with generic text")
                    
                    # Send response
                    try:
                        await self.wasender_client.send_message(
                            phone_number=phone_number,
                            message=response_text
                        )
                        logger.info("✅ LLM conversational response sent")
                    except Exception as send_exc:
                        logger.error(f"Failed to send response: {send_exc}")
                
                # Add assistant response to history (keep last 30 messages for better context)
                await self.session_manager.add_to_history(session_key, "assistant", response_text, max_history=30)
                
                # CRITICAL STATE SYNC FIX: DO NOT clear selected_service if booking might happen next!
                # The user selected a service and might say "احجز" immediately after
                # Clearing it causes amnesia bug where booking agent forgets the selection
                # Only clear if:
                # 1. User explicitly changed topic to something else, OR
                # 2. Booking completed/cancelled, OR  
                # 3. User sent non-booking intent (chitchat about different topic)
                # 
                # For now: PRESERVE selected_service for booking flow continuity
                # It will be cleared when booking completes or user changes topic
                if session_data.get("selected_service_id"):
                    # Check if next intent will likely be booking
                    journey_phase = session_data.get("journey_phase", "discovery")
                    if journey_phase == "booking" or session_data.get("show_booking_offer"):
                        logger.debug(f"💾 PRESERVING selected_service_id for booking continuity (phase={journey_phase})")
                        # Don't clear - user might book next turn!
                    else:
                        # User moved to different topic/phase - safe to clear
                        logger.debug(f"🗑️ Clearing selected_service_id - user moved to different phase")
                        session_data.pop("selected_service_id", None)
                        session_data.pop("selected_service_name", None)
                        session_data.pop("selected_service_price", None)
                        session_data.pop("selection_changed", None)  # Issue #11: Clear change flag
                        session_data.pop("previous_selection", None)  # Issue #11: Clear previous
                        session_data.pop("topic_changed", None)  # Issue #22: Clear topic change flag
                        session_data.pop("previous_topic_before_change", None)  # Issue #22: Clear previous topic
                
                # Update session
                session_data["last_intent"] = intent
                session_data["last_message"] = message_text
                await self.session_manager.put_session(session_key, session_data)
                
                return {
                    "intent": intent,
                    "status": "success",
                    "response": response_text,
                    "source": "llm_conversational",
                    "confidence": confidence
                }
            
            # Determine action intent from classified intent
            # Route "confirmation" to booking agent if booking is active OR user just selected a service (Issue #43)
            
            # 🛡️ CRITICAL PROTECTION: Detect and correct numbered response misclassification
            # If user responds with number after numbered list, force correct intent BEFORE routing
            import re
            last_bot_msg = last_bot_message or ""
            numbered_patterns = [r'\n\s*\d+[\.\-\)]\s+', r'\n\s*[١-٩]+[\.\-\)]\s+']
            has_numbered_list = any(re.search(pattern, last_bot_msg) for pattern in numbered_patterns)
            user_sent_number = message_text.strip().isdigit()
            
            if has_numbered_list and user_sent_number and intent in ["chitchat", "question", "greeting"]:
                logger.warning(f"🛡️ MISCLASSIFICATION DETECTED: User sent '{message_text}' after numbered list (intent was: {intent})")
                logger.warning(f"🔀 Auto-correcting: {intent} → selection (PROTECTION: prevents context loss)")
                intent = "selection"  # Force correct classification
            
            # CRITICAL: Preserve original intent for agent to understand user's actual intent!
            original_intent = intent  # Save before modification
            if intent == "confirmation" and (booking_state.get("started") or session_data.get("show_booking_offer") or current_topic):
                logger.info(f"🎯 Confirmation detected (booking_active={booking_state.get('started')}, has_topic={bool(current_topic)}) - routing to booking agent")
                # Change routing intent but preserve original for agent
                intent = "booking"  # For routing decision
                # original_intent stays "confirmation" for agent to understand
            elif intent == "selection" and booking_state.get("started"):
                logger.info(f"🎯 Selection in active booking (user answering question) - routing to booking agent")
                intent = "booking"
            elif intent == "selection" and not booking_state.get("started"):
                logger.info(f"🎯 Service SELECTED (interest phase) - offering to book: {message_text}")
                
                # 🧠 INTELLIGENT NUMBERED SELECTION: If user said "5" and we have displayed items
                displayed_items = session_data.get("displayed_items", [])
                logger.info(f"🔍 SMART SELECTION CHECK: user_message='{message_text}', displayed_items_count={len(displayed_items)}")
                
                if message_text.strip().isdigit() and displayed_items:
                    selection_index = int(message_text.strip()) - 1  # Convert to 0-indexed
                    logger.info(f"🔍 Attempting selection: index={selection_index}, available={len(displayed_items)}")
                    
                    if 0 <= selection_index < len(displayed_items):
                        selected_item = displayed_items[selection_index]
                        service_name = selected_item.get("name", selected_item.get("name_ar", message_text))
                        logger.info(f"✅ SMART SELECTION: User chose #{message_text} → '{service_name}'")
                        # Override message_text with actual service name for natural processing
                        message_text = service_name
                        payload["message"] = service_name
                    else:
                        logger.warning(f"⚠️ Selection index {selection_index} out of range (0-{len(displayed_items)-1})")
                else:
                    if not displayed_items:
                        logger.warning(f"⚠️ SMART SELECTION FAILED: No displayed_items in session!")
                    elif not message_text.strip().isdigit():
                        logger.debug(f"Not a numeric selection: '{message_text}'")
                
                # Save selected service as topic
                session_data["current_topic"] = message_text
                session_data["last_discussed_service"] = message_text
                session_data["show_booking_offer"] = True  # Flag to offer booking
                await self.session_manager.put_session(session_key, session_data)
                logger.info(f"💾 Saved service interest: topic={message_text}, show_booking_offer=True")
                # Keep as selection, will offer booking in response
            elif intent == "follow_up":
                logger.info(f"🔄 Follow-up request about: {current_topic} - continuing conversation")
                # Keep as follow_up, will be handled by LLM with context
            
            logger.info(f"🎯 Action needed - routing to {intent} agent")
            
            # CRITICAL: Store intent in payload for agent to access
            # Note: original_intent = what user meant, intent = where we're routing
            payload["intent"] = original_intent  # Primary field - what user meant
            payload["original_intent"] = original_intent  # Deprecated - kept for compatibility
            payload["routing_intent"] = intent  # Where message is actually routed
            
            # Add extracted entities to payload
            payload["extracted_entities"] = entities
            payload["intent_confidence"] = confidence
            
            # Build RICH conversation context with FULL state (human-like memory)
            context = {
                # Identity
                "sender_name": sender_name,
                "phone_number": phone_number,
                "session_key": session_key,
                
                # Conversation flow
                "conversation_history": conversation_history,  # Full history, not truncated
                "last_bot_message": last_bot_message,  # COMPLETE message (already in context from line 1098)
                "last_intent": intent,
                "conversation_turn": context.get("conversation_turn"),
                "journey_phase": session_data.get("journey_phase"),
                
                # Current context
                "current_topic": current_topic,
                "last_discussed_service": last_discussed_service,
                "entities": entities,
                
                # CRITICAL: Selected service details for booking agent (State Sync Fix)
                "selected_service_id": session_data.get("selected_service_id"),
                "selected_service_name": session_data.get("selected_service_name"),
                "selected_service_price": session_data.get("selected_service_price"),
                "selected_service_type_id": session_data.get("selected_service_type_id"),  # ✅ NEW: For LangGraph
                
                # State & data
                "patient_data": session_data.get("patient_data"),
                "booking_state": booking_state,  # CRITICAL: Full booking state
                
                # UI State (what was just shown to user)
                "displayed_items": session_data.get("displayed_items", []),
                "last_shown_list": session_data.get("last_shown_list", []),
                
                # Preferences
                "preferred_language": session_data.get("preferred_language", "arabic")
            }
            
            # Route to appropriate agent WITH context
            timing_checkpoints["before_agent"] = time.time()
            agent_result = await self._route_to_agent(
                intent=intent,
                session_key=session_key,
                payload=payload,
                context=context,
                confidence=confidence
            )
            timing_checkpoints["after_agent"] = time.time()
            
            # 🧠 INTELLIGENT STATE TRACKING: Save what agent displayed to user
            # This enables human-like memory: "I just showed you 8 services, you said '5'"
            if agent_result.get("displayed_services"):
                services = agent_result["displayed_services"]
                logger.info(f"💾 STORING {len(services)} displayed_items in session")
                session_data["displayed_items"] = services
                session_data["last_shown_list"] = services
                await self.session_manager.put_session(session_key, session_data)
                logger.info(f"✅ STORED {len(services)} displayed items successfully (enables smart selection on next turn)")
            else:
                logger.warning(f"⚠️ No displayed_services in agent_result - smart selection will not work")
            
            # Extract response text
            response_text = agent_result.get("response", "Thank you for your message.")
            
            # Personalize response with sender name
            if sender_name and sender_name != "Unknown":
                # Detect language from response
                from ..utils.language_detector import detect_language
                language = detect_language(response_text)
                
                # Check if greeting already includes name
                has_name = sender_name.lower() in response_text.lower()
                
                # Add greeting with name if not already present
                if not has_name:
                    # DON'T add greetings - let the agent's response stand as-is
                    # Agent responses already follow the conversational style
                    pass
            
            # Log response with configurable truncation (Issue #41-10)
            LOG_TRUNCATE_LENGTH = 500  # Increase from 100 to 500 for better debugging
            response_preview = response_text[:LOG_TRUNCATE_LENGTH] if len(response_text) > LOG_TRUNCATE_LENGTH else response_text
            logger.info(f"💬 Agent response ({len(response_text)} chars): {response_preview}{'...' if len(response_text) > LOG_TRUNCATE_LENGTH else ''}")
            
            # For very long responses, log full text at DEBUG level
            if len(response_text) > LOG_TRUNCATE_LENGTH:
                logger.debug(f"📄 Full response:\n{response_text}")
            
            # Reset failure counter on successful routing
            session_key = f"whatsapp:{phone_number}"
            if session_key in IntentRouter._routing_failures:
                del IntentRouter._routing_failures[session_key]
                logger.debug("✅ Reset router failure counter after success")
            
            # Send final response to WhatsApp
            timing_checkpoints["before_whatsapp"] = time.time()
            send_result = await self._send_whatsapp_response(
                phone_number=phone_number,
                message=agent_result.get("response", "حصل خطأ")
            )
            timing_checkpoints["after_whatsapp"] = time.time()
            
            # ⏱️ Performance Summary
            total_time = timing_checkpoints["after_whatsapp"] - timing_checkpoints["start"]
            patient_lookup_time = timing_checkpoints.get("patient_lookup", timing_checkpoints["start"]) - timing_checkpoints["start"]
            agent_time = timing_checkpoints["after_agent"] - timing_checkpoints["before_agent"]
            whatsapp_time = timing_checkpoints["after_whatsapp"] - timing_checkpoints["before_whatsapp"]
            
            logger.info(f"⏱️ PERFORMANCE: total={total_time:.2f}s | patient_lookup={patient_lookup_time:.2f}s | agent={agent_time:.2f}s | whatsapp={whatsapp_time:.2f}s")
            
            if send_result.get("success"):
                logger.info(f"✅ Response sent successfully to {phone_number}")
                
                # CRITICAL: Add bot response to conversation history (Issue #14)
                # This enables context-aware responses in future turns
                await self.session_manager.add_to_history(session_key, "assistant", response_text)
                logger.debug(f"📝 Added bot response to conversation history")
                
                # Update session with successful interaction
                await self._update_session(
                    session_key=session_key,
                    payload=payload,
                    intent=intent,
                    response=response_text,
                    status="success"
                )
                
                return {
                    "intent": intent,
                    "status": "success",
                    "response_sent": True,
                    "response_text": response_text,
                    "agent_result": agent_result
                }
            else:
                logger.error(f"❌ Failed to send response to {phone_number}")
                return {
                    "intent": intent,
                    "status": "send_failed",
                    "response_sent": False,
                    "error": send_result.get("error"),
                    "agent_result": agent_result
                }
                
        except Exception as exc:
            request_id = payload.get("request_id", "unknown")
            logger.error(f"🔴 [REQ:{request_id}] Intent routing error: {exc}", exc_info=True)
            
            # CRITICAL: Track error turn separately without incrementing main counter (Issue #20)
            session_key = f"whatsapp:{phone_number}" if phone_number else "unknown"
            try:
                session_data = await self.session_manager.get_session(session_key) or {}
                error_turn_count = session_data.get("error_turn_count", 0) + 1
                session_data["error_turn_count"] = error_turn_count
                await self.session_manager.put_session(session_key, session_data)
                logger.warning(f"📊 [REQ:{request_id}] Error turn tracked separately: {error_turn_count} errors (conversation_turn not incremented)")
            except:
                pass  # Don't fail if tracking fails
            
            # Build progressive error response
            sender_name = payload.get("sender_name", "")
            
            error_response = self._build_router_error_response(
                session_key=session_key,
                sender_name=sender_name,
                error_type="routing"
            )
            
            if phone_number:
                try:
                    await self._send_whatsapp_response(
                        phone_number=phone_number,
                        message=error_response
                    )
                    logger.info("✅ Sent progressive error message to user")
                except Exception as send_exc:
                    logger.error(f"Failed to send error message: {send_exc}")
            
            return {
                "intent": "error",
                "status": "error",
                "error": str(exc),
                "response_sent": True
            }
    
    def _validate_agent_ready(self, agent_class, agent_name: str) -> tuple[bool, str]:
        """
        Validate that agent class is properly defined and ready to use (Issue #15).
        
        Args:
            agent_class: The agent class to validate
            agent_name: Name of the agent for logging
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check 1: Class exists
            if agent_class is None:
                return False, f"{agent_name} class is None"
            
            # Check 2: Has handle method
            if not hasattr(agent_class, 'handle'):
                return False, f"{agent_name} missing 'handle' method"
            
            # Check 3: Has __init__ method
            if not hasattr(agent_class, '__init__'):
                return False, f"{agent_name} missing '__init__' method"
            
            # Check 4: For BookingAgent, verify critical class attributes
            if agent_name == "BookingAgent":
                required_attrs = ['_instances', '_initialized', '_request_cache', '_failure_counters']
                missing_attrs = [attr for attr in required_attrs if not hasattr(agent_class, attr)]
                if missing_attrs:
                    return False, f"{agent_name} missing class attributes: {missing_attrs}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    async def _route_to_agent(self, intent: str, session_key: str, payload: dict, context: dict = None, confidence: float = 0.0) -> dict:
        """
        Route to appropriate specialized agent based on intent with conversation context.
        Includes validation to prevent runtime errors (Issue #15).
        
        Args:
            intent: Classified intent
            session_key: Session identifier
            payload: Message payload
            context: Conversation context (history, sender info, etc.)
            confidence: Intent classification confidence score
            
        Returns:
            Agent processing result
        """
        try:
            logger.info(f"🤖 Routing to {intent} agent...")
            
            # CRITICAL: Add router intent to context for agent use
            if context is None:
                context = {}
            # Pass ORIGINAL intent (before routing modifications) so agent understands user's true intent
            # Get from payload where it was stored earlier
            original_user_intent = payload.get("original_intent", intent)
            context["classified_intent"] = original_user_intent
            context["intent_confidence"] = confidence
            context["routing_intent"] = intent  # The modified intent used for routing
            
            # DEBUG: Log intent passing
            logger.info(f"🎯 Intent passing: original={original_user_intent}, routing={intent}, context_intent={context['classified_intent']}")
            
            # CRITICAL: Route to booking agent if intent is booking OR if in active booking (chitchat/question during booking)
            # Retrieve session state to check if in active booking
            session_data = await self.session_manager.get_session(session_key) or {}
            journey_phase = session_data.get("journey_phase", "discovery")
            booking_state = session_data.get("booking_state", {})
            
            # Check step status
            current_step = booking_state.get("step", "")
            is_error_state = current_step.startswith("error_") or current_step.endswith("_error") if current_step else False
            
            # CRITICAL: These states should use conversational agent
            should_use_conversational = current_step in [None, "", "start", "loop_detected", "loop_help"] or is_error_state
            
            # Active booking = in advanced phase AND has actual progress (not start/error/loop)
            in_active_booking = (
                journey_phase in ["booking", "confirmation", "detail"] and
                not should_use_conversational
            ) or (
                booking_state.get("started") and 
                current_step and 
                not should_use_conversational
            )
            
            should_use_booking_agent = (
                intent == "booking" or
                (in_active_booking and intent in ["chitchat", "question"])
            )
            
            if should_use_booking_agent:
                if intent in ["chitchat", "question"]:
                    logger.info(f"🎯 [SMART ROUTING] Using BOOKING AGENT for {intent.upper()} during active booking")
                else:
                    logger.info(f"🎯 [ROUTING] Using BOOKING AGENT (LangGraph) - conversational agent already handled or skipped")
                
                # 🤖 NEW: INTELLIGENT AGENT - LLM with Function Calling (TRUE AI)
                # Enable this to use the new intelligent agent instead of hardcoded routing
                use_intelligent_agent = getattr(settings, 'use_intelligent_agent', False)
                
                # 🚨 CRITICAL: Check if we're forcing LangGraph for registration
                force_langgraph = session_data.get("force_langgraph_registration", False)
                
                if force_langgraph:
                    logger.warning(f"🚨 [FORCE LANGGRAPH] Bypassing intelligent agent - routing to LangGraph for patient registration")
                
                if use_intelligent_agent and not force_langgraph:
                    logger.info(f"🤖 [INTELLIGENT AGENT] Routing to AI-powered agent with function calling")
                    
                    try:
                        from ..agents.intelligent_agent_factory import get_intelligent_agent
                        
                        intelligent_agent = get_intelligent_agent()
                        
                        # Get conversation history
                        conversation_history = context.get("conversation_history", [])
                        patient_data = context.get("patient_data")
                        
                        # Call intelligent agent (LLM decides everything!)
                        result = await intelligent_agent.handle(
                            message=payload.get("message", ""),
                            conversation_history=conversation_history,
                            patient_data=patient_data,
                            session_data=session_data
                        )
                        
                        # History is now automatically saved to Redis at the end (no need to store in session)
                        logger.info(f"✅ [INTELLIGENT AGENT] Response generated: {len(result.get('response', ''))} chars")
                        
                        return {
                            "response": result.get("response"),
                            "intent": intent,
                            "status": result.get("status", "success"),
                            "agent": "IntelligentBookingAgent"
                        }
                    
                    except Exception as e:
                        logger.error(f"❌ [INTELLIGENT AGENT] Error: {e}", exc_info=True)
                        logger.warning("⚠️ Falling back to BookingAgentLangGraph")
                        # Continue to old booking agent below
                
                # CIRCUIT BREAKER: Check if BookingAgentLangGraph is disabled (Issue #24)
                is_open, breaker_reason = self._is_agent_circuit_open("BookingAgentLangGraph")
                if is_open:
                    logger.error(f"🚨 [REQ:{payload.get('request_id', 'unknown')}] CIRCUIT BREAKER OPEN: {breaker_reason}")
                    sender_name = payload.get("sender_name", "")
                    
                    # Language-aware circuit breaker message (Issue #25)
                    user_language = self._get_user_language(session_key)
                    
                    if user_language == "english":
                        circuit_msg = f"""Dear {sender_name} 🙏

The booking system is experiencing a technical issue right now

**To book now:**
📞 **Call: 920033304**

Our team is ready to help you directly
From 9 AM to 9 PM

We apologize for the inconvenience 💙"""
                    else:
                        circuit_msg = f"""عزيزي {sender_name} 🙏

نظام الحجز يواجه مشكلة تقنية حالياً

**للحجز الآن:**
📞 **اتصل: 920033304**

فريقنا جاهز يساعدك مباشرة
من ٩ صباحاً إلى ٩ مساءً

نعتذر عن الإزعاج 💙"""
                    
                    return {
                        "response": circuit_msg,
                        "intent": intent,
                        "status": "circuit_breaker_open",
                        "error": breaker_reason,
                        "language": user_language
                    }
                
                # VALIDATION: Check BookingAgentLangGraph is ready before instantiation (Issue #15)
                is_valid, error_msg = self._validate_agent_ready(BookingAgentLangGraph, "BookingAgentLangGraph")
                if not is_valid:
                    logger.error(f"🚨 [REQ:{payload.get('request_id', 'unknown')}] AGENT VALIDATION FAILED: {error_msg}")
                    self._track_agent_failure("BookingAgentLangGraph")  # Track for circuit breaker
                    sender_name = payload.get("sender_name", "")
                    return {
                        "response": self._build_router_error_response(
                            session_key=session_key,
                            sender_name=sender_name,
                            error_type="agent_validation"
                        ),
                        "intent": intent,
                        "status": "validation_failed",
                        "error": error_msg
                    }
                
                # Safe to instantiate now - USE FACTORY PATTERN (Issue #14)
                try:
                    # Factory creates agent with its own api_client if needed
                    # Note: Agent uses instance pooling, so this reuses existing instance if available
                    agent = BookingAgentFactory.create(session_key)
                    agent_type = type(agent).__name__
                    logger.info(f"✅ [REQ:{payload.get('request_id', 'unknown')}] {agent_type} obtained from factory (pooled)")
                except Exception as init_exc:
                    logger.error(f"🚨 [REQ:{payload.get('request_id', 'unknown')}] AGENT INSTANTIATION FAILED: {init_exc}", exc_info=True)
                    self._track_agent_failure("BookingAgentLangGraph")  # Track for circuit breaker
                    sender_name = payload.get("sender_name", "")
                    return {
                        "response": self._build_router_error_response(
                            session_key=session_key,
                            sender_name=sender_name,
                            error_type="agent_instantiation"
                        ),
                        "intent": intent,
                        "status": "instantiation_failed",
                        "error": str(init_exc)
                    }
                
                # Validate handle method exists before calling
                if not hasattr(agent, 'handle') or not callable(getattr(agent, 'handle')):
                    logger.error(f"🚨 AGENT METHOD VALIDATION FAILED: handle method not callable")
                    sender_name = payload.get("sender_name", "")
                    return {
                        "response": self._build_router_error_response(
                            session_key=session_key,
                            sender_name=sender_name,
                            error_type="agent_method"
                        ),
                        "intent": intent,
                        "status": "method_validation_failed",
                        "error": "handle method not callable"
                    }
                
                result = await agent.handle(payload, context)
                
                # CIRCUIT BREAKER: Reset on success (Issue #24)
                self._reset_agent_circuit_breaker("BookingAgentLangGraph")
                
                # 🚨 Clear force_langgraph flag after LangGraph completes
                if session_data.get("force_langgraph_registration"):
                    logger.info(f"✅ Clearing force_langgraph_registration flag after LangGraph completion")
                    session_data.pop("force_langgraph_registration", None)
                
                # ═══════════════════════════════════════════════════════════════
                # BIDIRECTIONAL STATE SYNCHRONIZATION (Bug 21, 23 Fix)
                # ═══════════════════════════════════════════════════════════════
                #
                # Bug 21: Two agents with no coordination → AMNESIA
                # Bug 23: Phase sync attempted but incomplete
                #
                # DIRECTION 1: Router → LangGraph (BEFORE invocation - lines 2478-2492)
                #   ✅ selected_service_id, selected_service_name (from conversational agent)
                #   ✅ patient_data with already_registered flag
                #   ✅ conversation_history, last_bot_message
                #   ✅ All Router state passed in context parameter
                #
                # DIRECTION 2: LangGraph → Router (AFTER invocation - HERE)
                #   ✅ booking_state (step, patient_verified, selected_service, etc.)
                #   ✅ journey_phase sync (map LangGraph step → Router phase)
                #   ✅ Atomic save to prevent race conditions
                #
                # Result: Full bidirectional sync - agents coordinate perfectly!
                # ═══════════════════════════════════════════════════════════════
                
                updated_session = await self.session_manager.get_session(session_key) or {}
                updated_booking_state = updated_session.get("booking_state", {})
                updated_step = updated_booking_state.get("step")
                
                # SYNC STEP 1: Update Router's booking_state from LangGraph
                session_data = updated_session
                session_data["booking_state"] = updated_booking_state
                booking_state = updated_booking_state
                
                # SYNC STEP 2: Map LangGraph step to Router phase
                step_to_phase_mapping = {
                    "start": "discovery",
                    "awaiting_service_selection": "discovery",
                    "awaiting_service_type": "detail",
                    "awaiting_service": "detail",
                    "needs_registration": "registration",
                    "awaiting_name": "registration",
                    "registration_id": "registration",
                    "registration_complete": "booking",
                    "patient_verified": "booking",
                    "awaiting_confirmation": "confirmation",
                    "completed": "completed"
                }
                
                if updated_step in step_to_phase_mapping:
                    synced_phase = step_to_phase_mapping[updated_step]
                    if session_data.get("journey_phase") != synced_phase:
                        old_phase = session_data.get("journey_phase")
                        session_data["journey_phase"] = synced_phase
                        logger.info(f"🔄 [SYNC] LangGraph → Router: phase {old_phase} → {synced_phase} (step={updated_step})")
                
                # SYNC STEP 3: Atomic save (prevent race conditions)
                await self.session_manager.put_session(session_key, session_data)
                
                logger.info(f"🔄 [SYNC] Bidirectional sync complete: step={updated_step}, phase={session_data.get('journey_phase')}")
                logger.debug(f"✅ Both agents now synchronized and coordinating")
                
                # CRITICAL: Check if conversational agent should generate response
                if result.get("use_conversational_agent"):
                    logger.info(f"🤖 LangGraph requests conversational agent for response generation")
                    
                    # Get conversational agent
                    from app.core.conversational_agent import get_conversational_agent
                    from app.core.llm_reasoner import LLMReasoner
                    from app.api.agent_api import get_api_client
                    
                    llm = LLMReasoner()
                    api_client = get_api_client()
                    conv_agent = get_conversational_agent(llm, api_client)
                    
                    # Build context for conversational agent
                    history = session_data.get("history", [])[-10:]
                    patient_data = session_data.get("patient_data")
                    user_phone = payload.get("sender_phone")
                    user_name = session_data.get("sender_name", session_data.get("confirmed_name"))
                    
                    # Add service types to context if needed
                    context_for_agent = updated_booking_state.get("context_for_agent", {})
                    service_types = updated_booking_state.get("service_types", [])
                    
                    # Generate response with conversational agent
                    agent_response = await conv_agent.chat(
                        user_message=message_text,
                        conversation_history=history,
                        patient_data=patient_data,
                        user_phone=user_phone,
                        user_name=user_name,
                        services_context=service_types if service_types else None
                    )
                    
                    # Update result with conversational agent response
                    result["response"] = agent_response
                    logger.info(f"✅ Conversational agent generated response: {agent_response[:100]}...")
                
                return result
            
            elif intent == "registration":
                # Registration data should be handled by the booking agent flow (via factory - Issue #14)
                logger.info(f"🎯 Registration intent → Routing to booking agent via factory")
                agent = BookingAgentFactory.create(session_key)
                return await agent.handle(payload, context)
            elif intent == "patient":
                agent = PatientAgent(session_key)
                return await agent.handle(payload, context)
            
            elif intent == "feedback":
                agent = FeedbackAgent(session_key)
                return await agent.handle(payload, context)
            
            elif intent == "resource":
                agent = ResourceAgent(session_key)
                return await agent.handle(payload, context)
            
            elif intent == "confirmation":
                # CRITICAL FIX: Confirmation in booking context routes to booking agent (Issue #6, #44, #14)
                # This is INTENTIONAL - confirmations are part of booking flow
                logger.info(f"🎯 Confirmation intent → Routing to booking agent via factory")
                agent = BookingAgentFactory.create(session_key)
                return await agent.handle(payload, context)
            
            elif intent in ["chitchat", "question", "cancel", "selection", "follow_up"]:
                # Immediate explicit cancellation handling (Issue #23)
                if intent == "cancel":
                    try:
                        # Clear booking state and reset registration expectations
                        session_data = await self.session_manager.get_session(session_key) or {}
                        session_data["booking_state"] = {}
                        session_data["registration_expected"] = None
                        session_data["current_topic"] = None
                        session_data["journey_phase"] = "discovery"
                        await self.session_manager.put_session(session_key, session_data)
                        logger.info("🗑️ Booking/registration cancelled by user - state cleared")

                        # Localized acknowledgement (returned to main sender)
                        cancel_msg = (
                            "تم إلغاء العملية ✅\n"
                            "إذا حاب نبدأ من جديد، اكتب 'اخجز' أو 'ابدأ' ✨"
                        )

                        return {
                            "intent": intent,
                            "status": "success",
                            "response": cancel_msg,
                            "source": "fast_path_cancel"
                        }
                    except Exception as cancel_exc:
                        logger.error(f"Cancel handling error: {cancel_exc}")
                
                # These conversational intents should be handled by LLM before routing
                # If they reach here, something went wrong - route to resource agent
                logger.warning(f"⚠️ Conversational intent '{intent}' reached agent routing - using resource agent")
                agent = ResourceAgent(session_key)
                return await agent.handle(payload, context)
            
            elif intent == "restart":
                # Clear booking state and reset registration expectations
                session_data = await self.session_manager.get_session(session_key) or {}
                session_data["booking_state"] = {}
                session_data["registration_expected"] = None
                session_data["current_topic"] = None
                session_data["journey_phase"] = "discovery"
                await self.session_manager.put_session(session_key, session_data)
                logger.info("🗑️ Booking/registration restarted by user - state cleared")

                # Localized acknowledgement (returned to main sender)
                restart_msg = (
                    "تم إعادة تشغيل العملية ✅\n"
                    "إذا حاب نبدأ من جديد، اكتب 'اخجز' أو 'ابدأ' ✨"
                )

                return {
                    "intent": intent,
                    "status": "success",
                    "response": restart_msg,
                    "source": "fast_path_restart"
                }
            
            elif intent == "back":
                # Clear booking state and reset registration expectations
                session_data = await self.session_manager.get_session(session_key) or {}
                session_data["booking_state"] = {}
                session_data["registration_expected"] = None
                session_data["current_topic"] = None
                session_data["journey_phase"] = "discovery"
                await self.session_manager.put_session(session_key, session_data)
                logger.info("🗑️ Booking/registration went back by user - state cleared")

                # Localized acknowledgement (returned to main sender)
                back_msg = (
                    "تم العودة إلى بداية العملية ✅\n"
                    "إذا حاب نبدأ من جديد، اكتب 'اخجز' أو 'ابدأ' ✨"
                )

                return {
                    "intent": intent,
                    "status": "success",
                    "response": back_msg,
                    "source": "fast_path_back"
                }
            
            else:
                # Fallback to resource agent for unknown/None intents
                logger.warning(f"⚠️ Unknown intent: {intent}, routing to resource agent as fallback")
                agent = ResourceAgent(session_key)
                return await agent.handle(payload, context)
                
        except Exception as exc:
            request_id = payload.get("request_id", "unknown")
            logger.error(f"🔴 [REQ:{request_id}] Agent routing error: {exc}", exc_info=True)
            
            # CIRCUIT BREAKER: Track agent failure (Issue #24)
            if intent in ["booking", "confirmation"]:
                failure_count = self._track_agent_failure("BookingAgentLangGraph")
                logger.warning(f"🔴 [REQ:{request_id}] BookingAgentLangGraph failure {failure_count}/3")
            
            # CRITICAL: Track error turn separately (Issue #20)
            try:
                session_data = await self.session_manager.get_session(session_key) or {}
                error_turn_count = session_data.get("error_turn_count", 0) + 1
                session_data["error_turn_count"] = error_turn_count
                await self.session_manager.put_session(session_key, session_data)
                logger.warning(f"📊 [REQ:{request_id}] Error turn tracked separately: {error_turn_count} errors (conversation_turn not incremented)")
            except:
                pass  # Don't fail if tracking fails
            
            # Track error pattern (Issue #17)
            import hashlib
            error_signature = hashlib.md5(f"{type(exc).__name__}:{str(exc)[:50]}".encode()).hexdigest()
            error_count = self._track_error_pattern(session_key, error_signature)
            
            # Check if repeated error should escalate (Issue #17)
            if self._should_escalate_repeated_error(session_key, error_signature):
                sender_name = payload.get("sender_name", "عزيزي")
                logger.error(f"🚨 ESCALATING: User hit same error {error_count} times - terminating with support contact")
                
                # Language-aware escalation message (Issue #25)
                user_language = self._get_user_language(session_key)
                
                if user_language == "english":
                    escalation_msg = f"""Dear {sender_name} 🙏

Unfortunately, the same issue is recurring

**Please contact us directly:**
📞 **Call: 920033304**

Our support team is ready to help
From 9 AM to 9 PM

We'll resolve this as quickly as possible 💙"""
                else:
                    escalation_msg = f"""عزيزي {sender_name} 🙏

للأسف، نفس المشكلة تتكرر

**يرجى التواصل المباشر:**
📞 **اتصل: 920033304**

فريق الدعم جاهز يساعدك
من ٩ صباحاً إلى ٩ مساءً

سنحل المشكلة بأسرع وقت 💙"""
                
                return {
                    "response": escalation_msg,
                    "intent": intent,
                    "status": "repeated_error_escalated",
                    "error": str(exc),
                    "error_count": error_count,
                    "language": user_language
                }
            
            # Build progressive error response
            sender_name = payload.get("sender_name", "")
            error_response = self._build_router_error_response(
                session_key=session_key,
                sender_name=sender_name,
                error_type="agent"
            )
            
            return {
                "response": error_response,
                "intent": intent,
                "status": "error",
                "error": str(exc),
                "error_count": error_count
            }
    
    async def _send_whatsapp_response(self, phone_number: str, message: str) -> dict:
        """
        Send response message back to WhatsApp via WaSender.
        
        Args:
            phone_number: Recipient phone number
            message: Message text to send
            
        Returns:
            Send result with success status
        """
        try:
            logger.info(f"📤 Sending WhatsApp message to {phone_number}")
            
            # CRITICAL: Strip any HTML comments before sending (they're visible in WhatsApp!)
            import re
            original_message = message
            message = re.sub(r'<!--.*?-->', '', message, flags=re.DOTALL).strip()
            if message != original_message:
                logger.warning(f"⚠️ Stripped HTML comments from message (would be visible to user!)")
            
            # IMPROVEMENT: Send typing indicator first (Issue #41-D)
            # Shows user we're processing their request
            try:
                await self.wasender_client.send_typing_indicator(phone_number)
                logger.debug("⌨️ Typing indicator sent")
            except Exception as typing_exc:
                # Non-critical - continue even if fails
                logger.debug(f"Typing indicator failed (non-critical): {typing_exc}")
            
            # Send via WaSender client
            result = await self.wasender_client.send_message(
                phone_number=phone_number,
                message=message
            )
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as exc:
            # Check if circuit breaker or rate limit
            error_msg = str(exc)
            
            if "Circuit breaker open" in error_msg:
                logger.error(f"🚫 Circuit breaker open - cannot send message: {exc}")
                # Return user-friendly error
                return {
                    "success": False,
                    "error": "service_overloaded",
                    "user_message": "عذراً، النظام مشغول حالياً. يرجى المحاولة بعد قليل أو الاتصال على 920033304"
                }
            elif "Rate limit" in error_msg:
                logger.error(f"⚠️ Rate limit exceeded - message failed: {exc}")
                return {
                    "success": False,
                    "error": "rate_limited",
                    "user_message": "عذراً، النظام مشغول. يرجى الانتظار قليلاً"
                }
            else:
                logger.error(f"WhatsApp send error: {exc}", exc_info=True)
                return {
                    "success": False,
                    "error": str(exc)
                }
    
    async def _update_session(
        self,
        session_key: str,
        payload: dict,
        intent: str,
        response: str,
        status: str
    ) -> None:
        """
        Update session with interaction history.
        Only called on successful interactions (Issue #20).
        
        Args:
            session_key: Session identifier
            payload: Original message payload
            intent: Classified intent
            response: Response sent to user
            status: Processing status
        """
        try:
            # Get existing session data
            # CRITICAL: Use async session methods for proper persistence (Issue #45)
            session_data = await self.session_manager.get_session(session_key) or {}
            
            # NOTE: Turn counter already incremented at beginning (Issue #13)
            # This ensures consistent turn tracking across all code paths
            
            # CRITICAL FIX: Save to BOTH "history" (for conversation) AND "interaction_history" (for analytics)
            # Initialize histories if not exist
            if "interaction_history" not in session_data:
                session_data["interaction_history"] = []
            if "history" not in session_data:
                session_data["history"] = []
            
            # Add interaction to analytics history
            interaction = {
                "timestamp": payload.get("timestamp"),
                "user_message": payload.get("message"),
                "intent": intent,
                "response": response,
                "status": status,
                "sender_name": payload.get("sender_name"),
                "turn": session_data.get("conversation_turn")  # Already incremented at beginning
            }
            
            session_data["interaction_history"].append(interaction)
            
            # Keep only last 20 interactions for audit
            session_data["interaction_history"] = session_data["interaction_history"][-20:]
            
            # CRITICAL: Save conversation history to REDIS (not in session!)
            # NOTE: If intelligent agent was used, history already saved (including tool calls)
            use_intelligent_agent = getattr(settings, 'use_intelligent_agent', False)
            
            # Initialize history_cache regardless of agent type (needed for migration later)
            history_cache = self._get_history_cache()
            
            if not use_intelligent_agent:
                
                # Add user message
                await history_cache.add_message(
                    session_key,
                    role="user",
                    content=payload.get("message", ""),
                    metadata={
                        "intent": intent,
                        "turn": session_data.get("conversation_turn"),
                        "language": session_data.get("preferred_language"),
                        "timestamp": payload.get("timestamp")
                    }
                )
                
                # Add assistant message
                await history_cache.add_message(
                    session_key,
                    role="assistant",
                    content=response,
                    metadata={
                        "intent": intent,
                        "agent": "unknown",  # route_result not available in this scope
                        "status": status
                    }
                )
            else:
                logger.info(f"⏭️ Skipping duplicate history save - intelligent agent already saved full history")
            
            # Migrate old history to Redis if exists (one-time migration)
            if "history" in session_data and session_data["history"]:
                old_history = session_data["history"]
                # Only migrate if Redis history is empty or smaller
                current_count = await history_cache.get_message_count(session_key)
                if current_count < len(old_history):
                    logger.info(f"📦 Migrating {len(old_history)} old history messages to Redis")
                    await history_cache.batch_add_messages(session_key, old_history)
                # Remove from session after migration
                session_data.pop("history", None)
            
            logger.debug(f"✅ Conversation history saved to Redis (session: {session_key[:30]}...)")
            
            # Update last interaction info
            session_data["last_intent"] = intent
            session_data["last_message"] = payload.get("message")
            session_data["last_response"] = response
            
            # Save session (TTL: 2 hours) - Use async method (Issue #45)
            await self.session_manager.put_session(session_key, session_data)
            
            logger.debug(f"✅ Session updated and persisted: {session_key} (turn={session_data['conversation_turn']})")
            
        except Exception as exc:
            logger.error(f"Session update error: {exc}", exc_info=True)



