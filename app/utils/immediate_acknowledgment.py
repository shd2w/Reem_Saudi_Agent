"""
Immediate User Acknowledgment System

Sends instant "Got it!" message while processing happens in background.
Prevents user confusion during long processing times.
"""
from typing import Dict, Optional
from loguru import logger


class ImmediateAck:
    """
    Provides immediate acknowledgment to users.
    
    Problem: 6+ second delays make users think bot crashed
    Solution: Instant "Got it!" message + actual response later
    """
    
    # Arabic acknowledgment messages
    ACK_MESSAGES = {
        "booking": "تمام، فهمتك ✅ جاري التحضير...",  # Got it, preparing...
        "question": "فهمت سؤالك 👍 دعني أتحقق...",  # Got your question, let me check...
        "confirmation": "ممتاز! ✅ جاري المتابعة...",  # Excellent! Following up...
        "default": "تمام ✅ لحظة واحدة...",  # Got it, one moment...
    }
    
    @classmethod
    def should_send_ack(
        cls,
        intent: str,
        expected_processing_time: float = 2.0
    ) -> bool:
        """
        Determine if immediate acknowledgment needed.
        
        Args:
            intent: User intent (booking, question, etc.)
            expected_processing_time: Estimated processing time (seconds)
        
        Returns:
            True if should send immediate ack
        """
        # Send ack if processing might take >2 seconds
        if expected_processing_time > 2.0:
            return True
        
        # Always send for complex operations
        complex_intents = ["booking", "registration", "appointment"]
        if intent in complex_intents:
            return True
        
        return False
    
    @classmethod
    def get_ack_message(
        cls,
        intent: str,
        custom_message: Optional[str] = None
    ) -> str:
        """
        Get appropriate acknowledgment message.
        
        Args:
            intent: User intent
            custom_message: Optional custom message
        
        Returns:
            Acknowledgment message
        """
        if custom_message:
            return custom_message
        
        return cls.ACK_MESSAGES.get(intent, cls.ACK_MESSAGES["default"])
    
    @classmethod
    def format_delayed_response(
        cls,
        original_response: str,
        processing_time: float
    ) -> str:
        """
        Format the delayed response (after ack).
        
        Args:
            original_response: The actual response
            processing_time: How long it took
        
        Returns:
            Formatted response
        """
        # If processing was very slow (>5s), apologize
        if processing_time > 5.0:
            apology = "عذراً على التأخير 🙏\n\n"
            return apology + original_response
        
        return original_response


# Integration example for router.py
async def route_with_acknowledgment(router, payload, wasender_client):
    """
    Example of how to integrate immediate acknowledgment.
    
    Usage in router.py:
        if ImmediateAck.should_send_ack(intent, expected_time=5.0):
            # Send instant ack
            ack_msg = ImmediateAck.get_ack_message(intent)
            await wasender_client.send_message(phone, ack_msg)
            
            # Process (takes time, but user already acknowledged)
            response = await agent.handle(...)
            
            # Send actual response
            await wasender_client.send_message(phone, response)
    """
    phone_number = payload.get("phone_number")
    intent = payload.get("intent", "default")
    
    # Estimate processing time
    expected_time = 5.0 if intent == "booking" else 1.0
    
    # Check if ack needed
    if ImmediateAck.should_send_ack(intent, expected_time):
        # Send immediate ack (fast!)
        ack_message = ImmediateAck.get_ack_message(intent)
        await wasender_client.send_message(phone_number, ack_message)
        logger.info(f"✅ Immediate ack sent to {phone_number}")
    
    # Now process (can take time, user already knows we're working)
    import time
    start_time = time.time()
    
    # ... actual processing ...
    response = "Your actual response here"
    
    processing_time = time.time() - start_time
    
    # Format response based on processing time
    final_response = ImmediateAck.format_delayed_response(
        response,
        processing_time
    )
    
    # Send actual response
    await wasender_client.send_message(phone_number, final_response)
    logger.info(f"📤 Final response sent to {phone_number} (took {processing_time:.1f}s)")
