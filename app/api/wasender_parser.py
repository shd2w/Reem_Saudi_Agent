"""
WaSender Payload Parser
=======================
Professional parser for WaSender webhook payloads with comprehensive validation,
error handling, and support for multiple message types.

Handles:
- Text messages
- Media messages (images, audio, video, documents)
- Location messages
- Contact messages
- Message validation and sanitization
- Error recovery

Author: Agent Orchestrator Team
Version: 1.0.0
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field, validator


class WaSenderMessageKey(BaseModel):
    """WaSender message key structure"""
    remoteJid: str = Field(..., description="Sender JID (e.g., 1234567890@s.whatsapp.net)")
    fromMe: bool = Field(default=False, description="Whether message is from bot")
    id: Optional[str] = Field(default=None, description="Unique message ID (optional in new format)")
    
    # New WASender format fields
    senderLid: Optional[str] = Field(default=None, description="Sender LID")
    senderPn: Optional[str] = Field(default=None, description="Sender phone number")
    cleanedSenderPn: Optional[str] = Field(default=None, description="Cleaned sender phone number")
    addressingMode: Optional[str] = Field(default=None, description="Addressing mode (pn/lid)")


class WaSenderTextMessage(BaseModel):
    """Text message content"""
    conversation: Optional[str] = None
    extendedTextMessage: Optional[Dict[str, Any]] = None


class WaSenderMessage(BaseModel):
    """Complete WaSender message structure"""
    key: WaSenderMessageKey
    message: Dict[str, Any]
    pushName: Optional[str] = Field(default="Unknown", description="Sender display name")
    messageTimestamp: Optional[int] = None


class ParsedMessage(BaseModel):
    """Parsed and validated message ready for agent processing"""
    phone_number: str = Field(..., description="Sender phone number (without @s.whatsapp.net)")
    message_text: str = Field(..., description="Message text content")
    sender_name: str = Field(default="Unknown", description="Sender display name")
    message_id: str = Field(..., description="Unique message identifier")
    message_type: str = Field(default="text", description="Message type (text, image, audio, etc.)")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    session_key: str = Field(..., description="Session identifier for conversation tracking")
    raw_payload: Optional[Dict[str, Any]] = Field(default=None, description="Original payload for debugging")
    
    @validator('phone_number')
    def validate_phone(cls, v):
        """Validate phone number format"""
        if not v or len(v) < 10:
            raise ValueError("Invalid phone number")
        # Remove any non-digit characters except +
        cleaned = ''.join(c for c in v if c.isdigit() or c == '+')
        return cleaned
    
    @validator('message_text')
    def validate_message_text(cls, v):
        """Validate and sanitize message text (Issue #45 - enhanced logging)"""
        if not v or not v.strip():
            raise ValueError("Message text cannot be empty")
        
        # Trim whitespace and limit length
        sanitized = v.strip()[:4000]
        
        # Log if sanitization changed the message (Issue #45)
        if sanitized != v:
            from loguru import logger
            if len(v) != len(sanitized):
                logger.debug(f"🧹 Message sanitized: '{v[:50]}' → '{sanitized[:50]}' (len: {len(v)}→{len(sanitized)})")
        
        return sanitized


class WaSenderPayloadParser:
    """
    Professional WaSender payload parser with validation and error handling.
    
    Features:
    - Validates payload structure
    - Extracts phone number, message text, sender info
    - Handles multiple message types
    - Provides detailed error messages
    - Skips bot's own messages
    - Creates session keys for conversation tracking
    """
    
    def __init__(self):
        self.supported_message_types = [
            "conversation",
            "extendedTextMessage",
            "imageMessage",
            "audioMessage",
            "videoMessage",
            "documentMessage",
            "locationMessage",
            "contactMessage"
        ]
    
    def parse(self, raw_payload: Dict[str, Any]) -> Optional[ParsedMessage]:
        """
        Parse WaSender webhook payload into structured message.
        
        Args:
            raw_payload: Raw webhook payload from WaSender
            
        Returns:
            ParsedMessage object or None if parsing fails
            
        Raises:
            ValueError: If payload structure is invalid
        """
        try:
            logger.debug(f"Parsing WaSender payload: {raw_payload.keys()}")
            
            # Handle new WaSender format
            if "event" in raw_payload and raw_payload.get("event") == "messages.received":
                # New format: data.messages (single object, not array)
                data = raw_payload.get("data", {})
                message_data = data.get("messages", {})
                
                if not message_data:
                    logger.warning("No messages found in new format payload")
                    return None
            else:
                # Old format: body.data.messages (array)
                body = raw_payload.get("body", {})
                data = body.get("data", {})
                messages = data.get("messages", [])
                
                if not messages:
                    logger.warning("No messages found in payload")
                    return None
                
                # Get first message (WaSender typically sends one message per webhook)
                message_data = messages[0]
            
            # Validate and parse message structure
            try:
                message = WaSenderMessage(**message_data)
            except Exception as validation_error:
                logger.error(f"Message validation failed: {validation_error}")
                raise ValueError(f"Invalid message structure: {validation_error}")
            
            # Skip messages sent by the bot itself
            if message.key.fromMe:
                logger.debug("Skipping message sent by bot")
                return None
            
            # Extract phone number (NEW: prioritize cleanedSenderPn from new format)
            logger.info(f"📞 RAW remoteJid from WhatsApp: {message.key.remoteJid}")
            
            # NEW FORMAT: Check for cleanedSenderPn first (most reliable)
            if message.key.cleanedSenderPn:
                phone_number = message.key.cleanedSenderPn
                logger.info(f"✅ Using cleanedSenderPn from new format: {phone_number}")
            elif message.key.senderPn:
                phone_number = self._extract_phone_number(message.key.senderPn)
                logger.info(f"✅ Using senderPn from new format: {phone_number}")
            elif message.key.addressingMode == "lid":
                # OLD LID mode fallback
                logger.info(f"🔗 LID addressing mode detected - using senderPn instead of remoteJid")
                raw_key_dict = message_data.get("key", {})
                cleaned_sender = raw_key_dict.get("cleanedSenderPn")
                sender_pn = raw_key_dict.get("senderPn")
                
                if cleaned_sender:
                    phone_number = cleaned_sender
                    logger.info(f"✅ Using cleanedSenderPn: {phone_number}")
                elif sender_pn:
                    phone_number = self._extract_phone_number(sender_pn)
                    logger.info(f"✅ Using senderPn: {phone_number}")
                else:
                    logger.error(f"❌ LID mode but no senderPn/cleanedSenderPn found!")
                    return None
            else:
                # Regular mode: Use remoteJid as before
                phone_number = self._extract_phone_number(message.key.remoteJid)
            
            if not phone_number:
                logger.error(f"Failed to extract phone number from: {message.key.remoteJid}")
                return None
            logger.info(f"📞 EXTRACTED phone_number: {phone_number} ({len(phone_number)} digits)")
            
            # Extract message text based on message type
            message_text_raw, message_type = self._extract_message_content(message.message)
            if not message_text_raw:
                logger.warning(f"No text content found in message type: {message_type}")
                # For non-text messages, use placeholder
                message_text_raw = f"[{message_type.upper()}]"
            
            # Log raw message for debugging (Issue #45)
            if len(message_text_raw) != len(message_text_raw.strip()):
                logger.debug(f"📥 Raw message (before sanitization): '{message_text_raw}' (len={len(message_text_raw)})")
            
            message_text = message_text_raw  # Will be sanitized by Pydantic validator
            
            # Extract sender name with validation (Issue: Bot calls users ".")
            raw_name = message.pushName or ""
            
            # Validate sender name - reject invalid values
            invalid_names = [".", "..", "...", "null", "undefined", "None", ""]
            if (not raw_name or 
                raw_name in invalid_names or 
                len(raw_name.strip()) < 2 or 
                raw_name.strip() in ['?', '!', '-', '_', '/', '\\']):
                sender_name = ""  # Let LLM handle missing name naturally
                if raw_name and raw_name != "Unknown":
                    logger.debug(f"Invalid pushName received: '{raw_name}' - passing empty to LLM")
            else:
                sender_name = raw_name.strip()
            
            # Create session key
            session_key = f"whatsapp:{phone_number}"
            
            # Parse timestamp
            timestamp = None
            if message.messageTimestamp:
                try:
                    timestamp = datetime.fromtimestamp(message.messageTimestamp)
                except Exception as ts_error:
                    logger.warning(f"Failed to parse timestamp: {ts_error}")
            
            # Generate message ID if not provided (new format doesn't include it)
            import hashlib
            if message.key.id:
                message_id = message.key.id
            else:
                # Generate deterministic ID from phone + timestamp + message
                id_source = f"{phone_number}_{message.messageTimestamp}_{message_text[:50]}"
                message_id = hashlib.md5(id_source.encode()).hexdigest()[:16]
                logger.debug(f"Generated message ID: {message_id}")
            
            # Create parsed message
            parsed = ParsedMessage(
                phone_number=phone_number,
                message_text=message_text,
                sender_name=sender_name,
                message_id=message_id,
                message_type=message_type,
                timestamp=timestamp,
                session_key=session_key,
                raw_payload=raw_payload
            )
            
            # Log parsed message (Issue #45 - enhanced to show sanitization)
            message_preview = parsed.message_text[:50] + ('...' if len(parsed.message_text) > 50 else '')
            logger.info(f"✓ Parsed message from {sender_name} ({phone_number}): {message_preview}")
            
            # Show if message was changed by sanitization (Issue #45)
            if parsed.message_text != message_text_raw:
                logger.debug(f"  ℹ️ Note: Message was sanitized (whitespace/length adjusted)")
            
            return parsed
            
        except ValueError as ve:
            logger.error(f"Validation error parsing WaSender payload: {ve}")
            raise
        except Exception as exc:
            logger.error(f"Unexpected error parsing WaSender payload: {exc}", exc_info=True)
            return None
    
    def _extract_phone_number(self, remote_jid: str) -> Optional[str]:
        """
        Extract phone number from WaSender JID format with normalization.
        
        Args:
            remote_jid: JID in format "1234567890@s.whatsapp.net"
            
        Returns:
            Normalized phone number without suffix or None if invalid
        """
        try:
            if not remote_jid:
                return None
            
            # Split by @ and take first part
            parts = remote_jid.split("@")
            if not parts:
                return None
            
            phone = parts[0].strip()
            
            # Remove all non-digit characters
            import re
            phone = re.sub(r'\D', '', phone)
            
            # Validate it's not empty
            if not phone:
                logger.warning(f"No digits found in JID: {remote_jid}")
                return None
            
            # CRITICAL: DO NOT TRUNCATE - preserve full phone number for data integrity
            # Different countries have different phone number lengths
            # Truncation causes session tracking errors and message delivery failures
            original_phone = phone
            
            # Validate minimum length
            if len(phone) < 10:
                logger.warning(f"Phone number too short: {phone} from JID: {remote_jid}")
                return None
            
            # Log if unusually long (for monitoring, but don't truncate)
            if len(phone) > 15:
                logger.warning(f"⚠️ Unusually long phone number: {phone} ({len(phone)} digits) from JID: {remote_jid}")
            
            logger.debug(f"✅ Extracted phone: {phone} ({len(phone)} digits) from JID: {remote_jid}")
            
            return phone
            
        except Exception as exc:
            logger.error(f"Error extracting phone number: {exc}")
            return None
    
    def _extract_message_content(self, message_obj: Dict[str, Any]) -> tuple[str, str]:
        """
        Extract message text content based on message type.
        
        Args:
            message_obj: Message object from WaSender
            
        Returns:
            Tuple of (message_text, message_type)
        """
        try:
            # Check for simple text conversation
            if "conversation" in message_obj:
                return message_obj["conversation"], "text"
            
            # Check for extended text message (with formatting, links, etc.)
            if "extendedTextMessage" in message_obj:
                extended = message_obj["extendedTextMessage"]
                text = extended.get("text", "")
                return text, "text"
            
            # Check for image message with caption
            if "imageMessage" in message_obj:
                image = message_obj["imageMessage"]
                caption = image.get("caption", "[Image]")
                return caption, "image"
            
            # Check for audio message
            if "audioMessage" in message_obj:
                return "[Audio Message]", "audio"
            
            # Check for video message with caption
            if "videoMessage" in message_obj:
                video = message_obj["videoMessage"]
                caption = video.get("caption", "[Video]")
                return caption, "video"
            
            # Check for document message
            if "documentMessage" in message_obj:
                doc = message_obj["documentMessage"]
                filename = doc.get("fileName", "[Document]")
                return f"[Document: {filename}]", "document"
            
            # Check for location message
            if "locationMessage" in message_obj:
                location = message_obj["locationMessage"]
                name = location.get("name", "Location")
                return f"[Location: {name}]", "location"
            
            # Check for contact message
            if "contactMessage" in message_obj:
                contact = message_obj["contactMessage"]
                display_name = contact.get("displayName", "Contact")
                return f"[Contact: {display_name}]", "contact"
            
            # Unknown message type
            logger.warning(f"Unknown message type: {message_obj.keys()}")
            return "[Unsupported Message Type]", "unknown"
            
        except Exception as exc:
            logger.error(f"Error extracting message content: {exc}")
            return "", "error"
    
    def validate_payload_structure(self, raw_payload: dict) -> bool:
        """
        Validate that the payload has the expected WaSender structure.
        
        Args:
            raw_payload: Raw webhook payload
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # New WaSender format: event, sessionId, data, timestamp
            if "event" in raw_payload and "data" in raw_payload:
                # New format validation
                if raw_payload.get("event") != "messages.received":
                    logger.error(f"Unexpected event type: {raw_payload.get('event')}")
                    return False
                
                data = raw_payload.get("data", {})
                if "messages" not in data:
                    logger.error("Missing 'messages' in data")
                    return False
                
                return True
            
            # Old format: Check for required top-level keys
            if "body" not in raw_payload:
                logger.error("Missing 'body' key in payload")
                return False
            
            body = raw_payload["body"]
            if not isinstance(body, dict):
                logger.error("'body' is not a dictionary")
                return False
            
            if "data" not in body:
                logger.error("Missing 'data' key in body")
                return False
            
            data = body["data"]
            if not isinstance(data, dict):
                logger.error("'data' is not a dictionary")
                return False
            
            if "messages" not in data:
                logger.error("Missing 'messages' key in data")
                return False
            
            messages = data["messages"]
            if not isinstance(messages, list):
                logger.error("'messages' is not a list")
                return False
            
            if not messages:
                logger.warning("Empty messages list")
                return False
            
            return True
            
        except Exception as exc:
            logger.error(f"Payload validation error: {exc}")
            return False


# Singleton instance for easy import
_parser_instance = None

def get_wasender_parser() -> WaSenderPayloadParser:
    """Get singleton parser instance"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = WaSenderPayloadParser()
    return _parser_instance
