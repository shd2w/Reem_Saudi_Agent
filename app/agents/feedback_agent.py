"""
Feedback Agent - User feedback and satisfaction collection
===========================================================
Handles user feedback, ratings, complaints, and suggestions.

Features:
- Feedback collection
- Rating submissions
- Complaint handling
- Suggestion processing
- Sentiment analysis

Author: Agent Orchestrator Team
Version: 1.0.0
"""

from typing import Dict, Any
from loguru import logger

from ..memory.session_manager import SessionManager
from ..utils.language_detector import detect_language
from ..services.llm_response_generator import get_llm_response_generator


class FeedbackAgent:
    """
    Professional feedback agent for collecting user feedback.
    
    Handles:
    - General feedback
    - Service ratings
    - Complaints
    - Suggestions
    """
    
    def __init__(self, session_key: str):
        self.session_key = session_key
        self.session_manager = SessionManager()
        self.llm_generator = get_llm_response_generator()
    
    async def handle(self, payload: dict, context: dict = None) -> dict:
        """
        Handle feedback-related requests with conversation context.
        
        Args:
            payload: Message payload with feedback
            context: Conversation context (history, sender info, etc.)
            
        Returns:
            Response dictionary with acknowledgment
        """
        try:
            message = payload.get("message", "")
            sender_name = payload.get("sender_name", "")
            phone_number = payload.get("phone_number")
            
            logger.info(f"💬 Feedback agent processing from {sender_name}")
            
            # Determine feedback type
            message_lower = message.lower()
            
            if any(word in message_lower for word in ["complaint", "complain", "issue", "problem", "bad", "terrible", "worst"]):
                feedback_type = "complaint"
                sentiment = "negative"
            elif any(word in message_lower for word in ["suggest", "suggestion", "recommend", "should", "could"]):
                feedback_type = "suggestion"
                sentiment = "neutral"
            elif any(word in message_lower for word in ["thank", "great", "excellent", "good", "amazing", "wonderful"]):
                feedback_type = "praise"
                sentiment = "positive"
            else:
                feedback_type = "general"
                sentiment = "neutral"
            
            # Store feedback
            feedback_data = {
                "type": feedback_type,
                "sentiment": sentiment,
                "message": message,
                "phone_number": phone_number,
                "sender_name": sender_name
            }
            
            await self._store_feedback(feedback_data)
            
            # Generate natural, empathetic response using LLM
            response_text = await self.llm_generator.generate_response(
                intent="feedback",
                user_message=message,
                context=context,
                data={"feedback_type": feedback_type, "sentiment": sentiment},
                sender_name=sender_name
            )
            
            logger.info(f"✅ Feedback recorded: {feedback_type} ({sentiment})")
            
            return {
                "response": response_text,
                "intent": "feedback",
                "status": "received",
                "feedback_type": feedback_type,
                "sentiment": sentiment
            }
            
        except Exception as exc:
            logger.error(f"Feedback agent error: {exc}", exc_info=True)
            return {
                "response": "Thank you for your feedback. We've noted your message and appreciate you reaching out to us.",
                "intent": "feedback",
                "status": "error"
            }
    
    async def _store_feedback(self, feedback_data: dict) -> None:
        """Store feedback in session for later processing"""
        try:
            session_data = self.session_manager.get(self.session_key) or {}
            
            if "feedback_history" not in session_data:
                session_data["feedback_history"] = []
            
            session_data["feedback_history"].append(feedback_data)
            
            # Keep only last 10 feedback entries
            session_data["feedback_history"] = session_data["feedback_history"][-10:]
            
            self.session_manager.put(self.session_key, session_data, ttl_minutes=120)
            
            logger.debug(f"Feedback stored for session: {self.session_key}")
            
        except Exception as exc:
            logger.error(f"Store feedback error: {exc}")
