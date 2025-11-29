"""
LLM Reasoning Layer
===================
Handles all LLM-based reasoning and response generation.
This is the core intelligence layer that replaces static rule-based responses.
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional
from openai import OpenAI
from loguru import logger

from ..config import settings


class LLMReasoner:
    """
    LLM-based reasoning engine for natural conversations.
    Uses GPT to generate contextual, human-like responses.
    Singleton pattern to avoid re-initialization.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMReasoner, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize LLM reasoner with OpenAI client (once)"""
        # Only initialize once - prevent duplicate initialization
        if LLMReasoner._initialized:
            logger.debug("♻️ Reusing existing LLM Reasoner instance (singleton)")
            return  # Already initialized - silent reuse
        
        logger.info("🚀 Initializing LLM Reasoner for the first time...")
        
        # Always initialize chat memory first
        self.chat_memory: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = 30  # Increased from 10 to 30 to use more context (Issue #9)
        
        # Track recent phrases to avoid repetition
        self.recent_phrases: Dict[str, List[str]] = defaultdict(list)
        self.max_phrase_memory = 5
        
        # Initialize to None first
        self.client = None
        self.system_prompt = None
        
        try:
            # Check if API key exists
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            # Configure client with timeout and retries (Issue #34)
            self.model = settings.openai_model
            # CRITICAL: Reduce timeout to prevent long waits (10s max instead of 15s)
            self.request_timeout = getattr(settings, "openai_timeout_seconds", 10)
            # CRITICAL: No retries for timeout errors (fail fast instead of 30s wait)
            self.max_retries = 0  # Fail fast - don't retry timeouts
            self.client = OpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                max_retries=self.max_retries,
                timeout=self.request_timeout,
            )
            logger.info(f"⚙️ OpenAI client configured: timeout={self.request_timeout}s, retries={self.max_retries}")
            
            # Load system prompt
            prompt_path = os.path.join(
                os.path.dirname(__file__), 
                'prompts', 
                'wajen_assistant.txt'
            )
            
            if not os.path.exists(prompt_path):
                logger.error(f"❌ System prompt file not found: {prompt_path}")
                raise FileNotFoundError(f"System prompt file missing: {prompt_path}")
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            
            logger.info(f"✅ System prompt loaded ({len(self.system_prompt)} chars)")
            
            # ONLY set initialized=True if everything succeeded
            LLMReasoner._initialized = True
            logger.info(f"✅ LLM Reasoner initialized with {self.model} (singleton) - Ready!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM Reasoner: {e}")
            logger.error(f"❌ OpenAI client will be unavailable")
            logger.error(f"❌ System will return fallback responses")
            self.client = None
            self.system_prompt = "أنت ريم - وكيلة خدمة عملاء في مركز وجن الطبي. تحدث باللهجة السعودية."
            # Don't set _initialized = True on failure!
    
    def detect_language(self, text: str) -> str:
        """
        Detect if message is Arabic or English.
        
        Args:
            text: Input message
            
        Returns:
            'ar' for Arabic, 'en' for English
        """
        arabic_chars = re.compile(r"[\u0600-\u06FF]")
        return "ar" if arabic_chars.search(text) else "en"
    
    def add_to_memory(self, user_id: str, role: str, content: str):
        """
        Add message to conversation memory.
        
        Args:
            user_id: User identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        self.chat_memory[user_id].append({
            "role": role,
            "content": content
        })
        
        # Keep only recent messages
        if len(self.chat_memory[user_id]) > self.max_history:
            # Keep first (context) and recent messages
            self.chat_memory[user_id] = self.chat_memory[user_id][-self.max_history:]
    
    def get_conversation_context(self, user_id: str) -> List[Dict]:
        """
        Get conversation history for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of message dictionaries
        """
        return self.chat_memory.get(user_id, [])
    
    def clear_memory(self, user_id: str):
        """Clear conversation memory for user"""
        if user_id in self.chat_memory:
            del self.chat_memory[user_id]
            logger.debug(f"Cleared memory for user {user_id}")
    
    def classify_intent(
        self,
        user_id: str,
        user_message: str,
        context: Optional[Dict] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Classify intent WITHOUT polluting conversational memory.
        
        This is a stateless call - doesn't add to chat memory.
        Used for intent classification only.
        """
        try:
            if not self.client:
                return '{"intent": "chitchat", "confidence": 0.5}'
            
            # Build messages WITHOUT adding to memory
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            if context:
                context_str = "\n\n**Classification Context:**\n"
                for key, value in context.items():
                    if key not in ["sender_name", "phone_number"]:
                        context_str += f"- {key}: {value}\n"
                messages.append({"role": "system", "content": context_str})
            
            # Add the classification prompt (user_message is the prompt here)
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=150,
                timeout=getattr(self, "request_timeout", None)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return '{"intent": "chitchat", "confidence": 0.5}'
    
    def generate_reply(
        self, 
        user_id: str, 
        user_message: str,
        context: Optional[Dict] = None,
        temperature: float = 0.9,
        session_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate LLM-based reply to user message.
        
        Args:
            user_id: User identifier
            user_message: User's message
            context: Additional context (sender_name, etc.)
            temperature: Response creativity (0.0-1.0)
            session_history: Full session history from SessionManager (Issue #9)
            
        Returns:
            Generated response text
        """
        try:
            if not self.client:
                return self._fallback_response(user_message)
            
            # Detect language
            language = self.detect_language(user_message)
            
            # Seed memory from session history if provided (Issue #9)
            if session_history and not self.chat_memory[user_id]:
                logger.info(f"📚 Seeding LLM memory from session: {len(session_history)} messages")
                for msg in session_history[-self.max_history:]:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role and content:
                        self.chat_memory[user_id].append({"role": role, "content": content})
            
            # Add user message to memory
            self.add_to_memory(user_id, "user", user_message)
            # Build messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Build context from additional info
            if context:
                context_str = "\n\n**Current Context:**\n"
                for key, value in context.items():
                    if key not in ["sender_name", "phone_number"]:
                        context_str += f"- {key}: {value}\n"
                messages.append({"role": "system", "content": context_str})
            
            # Add recent phrases to avoid repetition
            recent = self.recent_phrases.get(user_id, [])
            if recent:
                avoid_str = f"\n\n**CRITICAL - Avoid Repetition:**\nDo NOT start with these phrases (vary your language):\n"
                for phrase in recent[-3:]:  # Last 3 phrases
                    avoid_str += f"- {phrase}\n"
                messages.append({"role": "system", "content": avoid_str})
                
                # Add intent
                if context.get('intent'):
                    context_str += f"Intent: {context.get('intent')}\n"
                
                # Add booking step
                if context.get('booking_step'):
                    context_str += f"Booking Step: {context.get('booking_step')}\n"
                
                # Add collected info
                if context.get('collected_info'):
                    context_str += f"Collected Info: {context.get('collected_info')}\n"
                
                # Add selected service details (Issue #10, #11, #22: Don't show list, detect changes)
                has_selected_service = context.get('selected_service_details') is not None
                
                if has_selected_service:
                    svc_details = context.get('selected_service_details')
                    is_changed = svc_details.get('changed', False)
                    previous = svc_details.get('previous')
                    topic_changed = svc_details.get('topic_changed', False)
                    previous_topic = svc_details.get('previous_topic')
                    
                    if is_changed and previous:
                        # User changed their selection (Issue #11)
                        context_str += f"\n🔄 USER CHANGED THEIR SELECTION:\n"
                        context_str += f"   - Previous: {previous}\n"
                        context_str += f"   - New choice: {svc_details.get('name')}\n"
                        if svc_details.get('price'):
                            context_str += f"   - Price: {svc_details.get('price')} ريال\n"
                        context_str += f"\n🚨 ACKNOWLEDGE THE CHANGE!\n"
                        context_str += f"Your response MUST:\n"
                        context_str += f"1. Acknowledge they changed: 'تمام! غيّرت اختيارك من {previous} إلى {svc_details.get('name')}'\n"
                        context_str += f"2. Confirm the new selection and price\n"
                        context_str += f"3. Ask if they're sure or want to proceed\n"
                        context_str += f"4. DO NOT show the service list!\n\n"
                    elif topic_changed and previous_topic:
                        # User changed topic/service (Issue #22)
                        context_str += f"\n🔄 USER SWITCHED TO DIFFERENT SERVICE:\n"
                        context_str += f"   - Previous discussion: {previous_topic}\n"
                        context_str += f"   - Now selected: {svc_details.get('name')}\n"
                        if svc_details.get('price'):
                            context_str += f"   - Price: {svc_details.get('price')} ريال\n"
                        context_str += f"\n🚨 ACKNOWLEDGE THE TOPIC CHANGE!\n"
                        context_str += f"Your response MUST:\n"
                        context_str += f"1. Acknowledge switch: 'تمام! كنا نتكلم عن {previous_topic}، الحين اخترت {svc_details.get('name')}'\n"
                        context_str += f"2. Confirm the new service and price\n"
                        context_str += f"3. Ask if they want to proceed\n"
                        context_str += f"4. DO NOT show the service list!\n\n"
                    else:
                        # First selection (Issue #10)
                        context_str += f"\n✅ USER JUST SELECTED SERVICE:\n"
                        context_str += f"   - Name: {svc_details.get('name')}\n"
                        if svc_details.get('price'):
                            context_str += f"   - Price: {svc_details.get('price')} ريال\n"
                        context_str += f"   - ID: {svc_details.get('id')}\n"
                        context_str += f"\n🚨 CRITICAL - DO NOT LIST SERVICES AGAIN!\n"
                        context_str += f"The user ALREADY selected this service.\n"
                        context_str += f"Your response MUST:\n"
                        context_str += f"1. Acknowledge their selection: 'تمام! اخترت {svc_details.get('name')}'\n"
                        context_str += f"2. Confirm price if applicable\n"
                        context_str += f"3. Ask if they want to proceed with booking\n"
                        context_str += f"4. DO NOT show the service list again!\n\n"
                
                # Add available options (Issue #27 - Smart service context)
                # Only show if user hasn't selected a service yet (Issue #10)
                if context.get('available_services') and not has_selected_service:
                    services = context.get('available_services')
                    if services and len(services) > 0:
                        context_str += f"\n📋 Available Services Database ({len(services)} total):\n"
                        for i, svc in enumerate(services, 1):  # Show ALL services, no limit
                            svc_name = svc.get('name', 'خدمة')
                            svc_price = svc.get('price')
                            if svc_price and svc_price > 0:
                                context_str += f"{i}. {svc_name} - {svc_price} ريال\n"
                            else:
                                context_str += f"{i}. {svc_name}\n"
                        
                        context_str += f"\n🎯 SMART INSTRUCTIONS:\n"
                        context_str += f"- If user asks 'what services?' or 'show me services': LIST them with numbers\n"
                        context_str += f"- If user asks about SPECIFIC service(s): Answer their question using the data above\n"
                        context_str += f"- If user asks for comparison/difference: COMPARE the specific services mentioned\n"
                        context_str += f"- If user asks 'which is better?': Give personalized recommendation\n"
                        context_str += f"- DON'T list all services unless they specifically asked to see them\n"
                        context_str += f"- User can select by NUMBER (1-{len(services)}) or by NAME (Arabic/English)\n\n"
                    else:
                        context_str += f"Available Services: EMPTY (tell user you can't access services right now)\n"
                
                if context.get('available_doctors'):
                    context_str += f"Available Doctors: {context.get('available_doctors')}\n"
                
                # Add greeting type for proper cultural response
                if context.get('greeting_type'):
                    greeting_type = context.get('greeting_type')
                    if greeting_type == "salam":
                        context_str += f"\n🚨 CRITICAL: User said 'السلام عليكم' - You MUST reply with 'وعليكم السلام' first!\n"
                    elif greeting_type == "morning":
                        context_str += f"\n🚨 CRITICAL: User said 'صباح الخير' - You MUST reply with 'صباح النور' first!\n"
                    elif greeting_type == "evening":
                        context_str += f"\n🚨 CRITICAL: User said 'مساء الخير' - You MUST reply with 'مساء النور' first!\n"
                
                # Add special instructions
                if context.get('message'):
                    context_str += f"\nINSTRUCTION: {context.get('message')}\n"
                
                context_str += "===================\n"
                
                messages[0]["content"] += context_str
                        
            # Add conversation history
            messages.extend(self.get_conversation_context(user_id))
            
            # Generate response
            logger.debug(f"Generating LLM response for: '{user_message[:50]}...'")
            
            logger.debug(
                f"LLM request: model={self.model}, timeout={getattr(self, 'request_timeout', 'n/a')}s, max_retries={getattr(self, 'max_retries', 'n/a')}"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=300,
                timeout=getattr(self, "request_timeout", None)
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Clean up response (remove typos, duplicates)
            reply = self._cleanup_response(reply)
            
            # Track opening phrase to avoid repetition
            opening = reply.split('.')[0][:50] if '.' in reply else reply[:50]
            self.recent_phrases[user_id].append(opening)
            if len(self.recent_phrases[user_id]) > self.max_phrase_memory:
                self.recent_phrases[user_id] = self.recent_phrases[user_id][-self.max_phrase_memory:]
            
            # Add assistant reply to memory
            self.add_to_memory(user_id, "assistant", reply)
            
            logger.debug(f"Generated reply: '{reply[:50]}...'")
            
            return reply
            
        except Exception as e:
            # Highlight timeouts explicitly
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.error(
                    f"LLM generation timeout after {getattr(self, 'request_timeout', 'n/a')}s: {e}",
                    exc_info=True,
                )
            else:
                logger.error(f"LLM generation error: {e}", exc_info=True)
            return self._fallback_response(user_message)
    
    def _cleanup_response(self, text: str) -> str:
        """
        Clean up LLM response (remove duplicates, typos).
        
        Args:
            text: Raw LLM response
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove duplicate characters (e.g., "ي ي" -> "ي")
        text = re.sub(r'(\S)\s+\1(?=\s|$)', r'\1', text)
        
        # Remove duplicate words
        words = text.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                cleaned_words.append(word)
        
        text = ' '.join(cleaned_words)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove duplicate punctuation
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        return text
    
    def _fallback_response(self, message: str) -> str:
        """
        Fallback response when LLM fails (Saudi dialect).
        
        Args:
            message: User message
            
        Returns:
            Simple fallback response
        """
        # Always respond in Saudi dialect for consistency
        return "عذرًا يا الغالي 🙏 صار عندي خلل بسيط الحين. ممكن تعيد رسالتك بعدين؟"
    
    def generate_with_intent(
        self,
        user_id: str,
        user_message: str,
        intent: str,
        entities: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate reply with known intent and entities.
        
        Args:
            user_id: User identifier
            user_message: User's message
            intent: Detected intent (booking, info, etc.)
            entities: Extracted entities
            context: Additional context
            
        Returns:
            Generated response
        """
        # Add intent context to system message
        intent_context = f"\n\nDetected Intent: {intent}"
        if entities:
            intent_context += f"\nEntities: {entities}"
        
        # Temporarily modify system prompt
        original_prompt = self.system_prompt
        self.system_prompt += intent_context
        
        try:
            reply = self.generate_reply(user_id, user_message, context)
        finally:
            self.system_prompt = original_prompt
        
        return reply
    
    async def chat_completion(
        self,
        messages: List[Dict],
        functions: Optional[List[Dict]] = None,
        function_call: str = "auto",
        temperature: float = 0.7
    ) -> Dict:
        """
        Chat completion with function calling support (for conversational agent)
        
        Args:
            messages: Conversation messages
            functions: Available functions for GPT to call
            function_call: "auto" or "none"
            temperature: Response randomness (0-1)
        
        Returns:
            {
                "content": "response text",
                "function_call": {"name": "...", "arguments": "{...}"} or None
            }
        """
        
        if not self.client:
            logger.error("❌ OpenAI client not initialized")
            logger.error(f"⚠️ Initialization status: {LLMReasoner._initialized}")
            logger.error(f"⚠️ API key configured: {bool(getattr(settings, 'openai_api_key', None))}")
            logger.error("⚠️ Check OPENAI_API_KEY environment variable in .env file")
            return {
                "content": "عذراً، النظام مشغول حالياً. يرجى الاتصال بنا على 966508344653 🏥",
                "function_call": None
            }
        
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            
            # Add function calling if provided
            if functions:
                params["functions"] = functions
                params["function_call"] = function_call
            
            logger.info(f"🤖 Calling GPT with {len(messages)} messages, functions={len(functions) if functions else 0}")
            
            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message
            
            # Check if function call
            if hasattr(message, 'function_call') and message.function_call:
                logger.info(f"🔧 GPT wants to call function: {message.function_call.name}")
                return {
                    "content": message.content,
                    "function_call": {
                        "name": message.function_call.name,
                        "arguments": message.function_call.arguments
                    }
                }
            else:
                return {
                    "content": message.content,
                    "function_call": None
                }
        
        except Exception as e:
            error_msg = str(e)
            
            # CRITICAL: Better error messages based on error type
            if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                logger.error(f"❌ OpenAI TIMEOUT after {self.request_timeout}s: {error_msg}")
                return {
                    "content": "عذراً، الاستجابة تأخرت شوي. ممكن تعيد رسالتك؟ 🙏",
                    "function_call": None
                }
            elif "rate limit" in error_msg.lower():
                logger.error(f"❌ OpenAI RATE LIMIT: {error_msg}")
                return {
                    "content": "عذراً، عندنا ضغط كبير حالياً. جرب بعد ثواني؟ 🙏",
                    "function_call": None
                }
            else:
                logger.error(f"❌ OpenAI error: {error_msg}", exc_info=True)
                return {
                    "content": "عذراً، حدث خطأ مؤقت. ممكن تعيد الرسالة؟",
                    "function_call": None
                }


# Global instance
def get_llm_reasoner() -> LLMReasoner:
    """Get LLM reasoner singleton instance"""
    return LLMReasoner()


def generate_llm_reply(
    user_id: str,
    user_message: str,
    context: Optional[Dict] = None
) -> str:
    """
    Convenient function to generate LLM reply.
    
    Args:
        user_id: User identifier
        user_message: User's message
        context: Additional context
        
    Returns:
        Generated response
    """
    reasoner = get_llm_reasoner()
    return reasoner.generate_reply(user_id, user_message, context)
