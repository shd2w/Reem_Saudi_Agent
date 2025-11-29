"""
LLM-Based Intent Classifier
============================
Uses GPT to classify user intent dynamically instead of rules.
"""

from typing import Dict, Optional
from openai import OpenAI
from loguru import logger

from ..config import settings


class LLMIntentClassifier:
    """
    LLM-powered intent classification.
    Uses GPT to understand user intent contextually.
    """
    
    INTENT_PROMPT = """You are an intent classification system for a medical center WhatsApp assistant.

Analyze the user's message and classify it into ONE of these intents:

1. **booking** - User wants to book/schedule an appointment
   Examples: "ابي احجز", "I want to book", "schedule appointment", "موعد"

2. **patient** - User wants to register or update patient information
   Examples: "سجل اسمي", "register", "update my info", "تسجيل"

3. **resource** - User asking about services, doctors, offers, promotions, prices, or general info
   Examples: "وش عندكم خدمات", "show me doctors", "الدكاترة", "ليزر", "عندكم عروض", "what offers do you have", "prices", "أسعار"

4. **feedback** - User giving feedback, thanks, complaints
   Examples: "شكراً", "thank you", "ممتاز", "complaint"

5. **greeting** - Simple greetings without specific request
   Examples: "هلا", "hello", "مرحبا", "hi"

Return ONLY a JSON object with:
{
  "intent": "intent_name",
  "confidence": 0.95,
  "reasoning": "brief explanation",
  "entities": {"key": "value"}
}

User's message: """
    
    def __init__(self):
        """Initialize intent classifier"""
        try:
            self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
            self.model = settings.openai_model
            logger.info("✅ LLM Intent Classifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Intent Classifier: {e}")
            self.client = None
    
    def classify(
        self, 
        message: str, 
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Classify user intent using LLM.
        
        Args:
            message: User's message
            context: Conversation context
            
        Returns:
            Dict with intent, confidence, entities
        """
        try:
            if not self.client:
                return self._fallback_classify(message)
            
            # Build prompt with context
            prompt = self.INTENT_PROMPT + f'"{message}"'
            
            if context:
                last_intent = context.get("last_intent")
                if last_intent:
                    prompt += f"\n\nPrevious intent: {last_intent}"
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intent classification expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistent classification
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            logger.info(f"Intent classified: {result.get('intent')} (confidence: {result.get('confidence')})")
            
            return result
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return self._fallback_classify(message)
    
    def _fallback_classify(self, message: str) -> Dict:
        """
        Simple rule-based fallback classification.
        
        Args:
            message: User message
            
        Returns:
            Basic intent classification
        """
        message_lower = message.lower()
        
        # Booking keywords
        if any(word in message_lower for word in ["احجز", "موعد", "book", "appointment", "schedule"]):
            return {
                "intent": "booking",
                "confidence": 0.8,
                "reasoning": "Booking keyword detected",
                "entities": {}
            }
        
        # Patient keywords
        if any(word in message_lower for word in ["سجل", "تسجيل", "register", "sign up"]):
            return {
                "intent": "patient",
                "confidence": 0.8,
                "reasoning": "Registration keyword detected",
                "entities": {}
            }
        
        # Greeting keywords
        if any(word in message_lower for word in ["هلا", "مرحبا", "hello", "hi", "hey"]):
            return {
                "intent": "greeting",
                "confidence": 0.9,
                "reasoning": "Greeting detected",
                "entities": {}
            }
        
        # Default to resource/info
        return {
            "intent": "resource",
            "confidence": 0.6,
            "reasoning": "Default classification",
            "entities": {}
        }


# Global instance
_classifier = None

def get_intent_classifier() -> LLMIntentClassifier:
    """Get or create global intent classifier"""
    global _classifier
    if _classifier is None:
        _classifier = LLMIntentClassifier()
    return _classifier


def classify_intent(message: str, context: Optional[Dict] = None) -> Dict:
    """
    Convenient function to classify intent.
    
    Args:
        message: User message
        context: Optional context
        
    Returns:
        Intent classification result
    """
    classifier = get_intent_classifier()
    return classifier.classify(message, context)
