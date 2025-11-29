"""
Fallback Communication Channels

Provides alternative delivery methods when primary channel (WhatsApp) fails.

Priority order:
1. WhatsApp (primary)
2. SMS (fallback 1)
3. Email (fallback 2)
4. Manual queue (last resort)
"""
from typing import Optional, Dict, Any
from loguru import logger
from enum import Enum


class Channel(Enum):
    """Communication channels."""
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"
    MANUAL = "manual"


class FallbackChannelManager:
    """
    Manages fallback communication channels.
    
    Usage:
        manager = FallbackChannelManager(
            whatsapp_client=wasender,
            sms_client=twilio,
            email_client=sendgrid
        )
        
        result = await manager.send_with_fallback(
            user_id="123",
            phone="1234567890",
            email="user@example.com",
            message="Your appointment is tomorrow"
        )
    """
    
    def __init__(
        self,
        whatsapp_client,
        sms_client=None,
        email_client=None,
        manual_queue=None
    ):
        self.whatsapp = whatsapp_client
        self.sms = sms_client
        self.email = email_client
        self.manual_queue = manual_queue
    
    async def send_with_fallback(
        self,
        user_id: str,
        phone: str,
        message: str,
        email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send message with automatic fallback to alternative channels.
        
        Args:
            user_id: User identifier
            phone: Phone number (for WhatsApp/SMS)
            message: Message text
            email: Email address (optional, for email fallback)
            metadata: Additional metadata
        
        Returns:
            Result dict with channel used and status
        """
        channels_tried = []
        last_error = None
        
        # Try WhatsApp first (primary channel)
        try:
            logger.info(f"📱 Attempting WhatsApp to {phone}")
            result = await self.whatsapp.send_message(phone, message)
            
            logger.info(f"✅ Message sent via WhatsApp to {phone}")
            return {
                "success": True,
                "channel": Channel.WHATSAPP.value,
                "channels_tried": [Channel.WHATSAPP.value],
                "result": result
            }
            
        except Exception as exc:
            last_error = str(exc)
            channels_tried.append(Channel.WHATSAPP.value)
            
            # Check if it's a rate limit (might resolve soon)
            if "rate limit" in last_error.lower() or "429" in last_error:
                logger.warning(f"⚠️ WhatsApp rate limited for {phone} - trying SMS fallback")
                
                # Try SMS fallback
                if self.sms:
                    try:
                        logger.info(f"📲 Attempting SMS to {phone}")
                        sms_result = await self.sms.send_sms(phone, message)
                        
                        logger.info(f"✅ Message sent via SMS to {phone}")
                        return {
                            "success": True,
                            "channel": Channel.SMS.value,
                            "channels_tried": channels_tried + [Channel.SMS.value],
                            "fallback_reason": "whatsapp_rate_limited",
                            "result": sms_result
                        }
                    except Exception as sms_exc:
                        last_error = str(sms_exc)
                        channels_tried.append(Channel.SMS.value)
                        logger.error(f"❌ SMS fallback failed: {sms_exc}")
            
            # If SMS failed or not available, try email
            if self.email and email:
                try:
                    logger.info(f"📧 Attempting email to {email}")
                    email_result = await self.email.send_email(
                        to=email,
                        subject="رسالة من عيادة وجن",
                        body=message
                    )
                    
                    logger.info(f"✅ Message sent via email to {email}")
                    return {
                        "success": True,
                        "channel": Channel.EMAIL.value,
                        "channels_tried": channels_tried + [Channel.EMAIL.value],
                        "fallback_reason": "whatsapp_and_sms_failed",
                        "result": email_result
                    }
                except Exception as email_exc:
                    last_error = str(email_exc)
                    channels_tried.append(Channel.EMAIL.value)
                    logger.error(f"❌ Email fallback failed: {email_exc}")
            
            # All channels failed - add to manual queue
            if self.manual_queue:
                try:
                    logger.warning(f"⚠️ All channels failed - adding to manual queue")
                    await self.manual_queue.add({
                        "user_id": user_id,
                        "phone": phone,
                        "email": email,
                        "message": message,
                        "metadata": metadata,
                        "channels_tried": channels_tried,
                        "last_error": last_error,
                        "timestamp": __import__("time").time()
                    })
                    
                    logger.info(f"📋 Message queued for manual follow-up: {user_id}")
                    return {
                        "success": False,
                        "channel": Channel.MANUAL.value,
                        "channels_tried": channels_tried + [Channel.MANUAL.value],
                        "fallback_reason": "all_automated_channels_failed",
                        "queued_for_manual": True
                    }
                except Exception as queue_exc:
                    logger.error(f"❌ Manual queue failed: {queue_exc}")
            
            # Complete failure
            logger.error(
                f"❌ All communication channels failed for {user_id}: {last_error}"
            )
            return {
                "success": False,
                "channel": None,
                "channels_tried": channels_tried,
                "error": last_error
            }


# SMS Provider Interface (Twilio example)
class SMSProvider:
    """SMS provider abstraction."""
    
    async def send_sms(self, phone: str, message: str) -> Dict:
        """Send SMS via provider (e.g., Twilio)."""
        # TODO: Integrate with actual SMS provider
        raise NotImplementedError("SMS provider not configured")


# Email Provider Interface (SendGrid example)
class EmailProvider:
    """Email provider abstraction."""
    
    async def send_email(self, to: str, subject: str, body: str) -> Dict:
        """Send email via provider (e.g., SendGrid)."""
        # TODO: Integrate with actual email provider
        raise NotImplementedError("Email provider not configured")


# Manual Queue (Redis-based)
class ManualFollowUpQueue:
    """Queue for messages that need manual follow-up."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.queue_key = "manual_followup:pending"
    
    async def add(self, item: Dict) -> None:
        """Add item to manual follow-up queue."""
        import json
        await self.redis.rpush(
            self.queue_key,
            json.dumps(item)
        )
    
    async def get_pending(self) -> list:
        """Get all pending manual follow-ups."""
        import json
        items = await self.redis.lrange(self.queue_key, 0, -1)
        return [json.loads(item) for item in items]
