"""
WaSender Webhook Handler
=========================
Professional webhook endpoint for receiving WhatsApp messages from WaSender.
Handles payload parsing, validation, and routing to agent orchestration.

Features:
- Payload validation and parsing
- Error handling and recovery
- Logging and monitoring
- Session management
- Response formatting

Author: Agent Orchestrator Team
Version: 1.0.0
"""
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
import time
import hashlib
import json
from typing import Dict, Tuple, Optional

from ..orchestration.router import IntentRouter
from ..api.wasender_parser import get_wasender_parser, ParsedMessage
from ..memory.session_manager import SessionManager


webhook = APIRouter()


# ============================================================================
# MESSAGE DEDUPLICATION - REDIS-BACKED (Issue #12)
# ============================================================================
# Prevents duplicate processing of same message (WhatsApp retries, webhook duplicates)
# Uses Redis for distributed deduplication across multiple processes
# Key: idempotency:{message_id} or dedup:{hash(phone+message+timestamp)}
# TTL: 5 minutes (covers all realistic duplicate scenarios)
_DEDUP_WINDOW_SECONDS = 300  # 5 minutes


class MessageDeduplicator:
    """Redis-backed message deduplication for idempotent webhook processing"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.window_seconds = _DEDUP_WINDOW_SECONDS
    
    def _get_idempotency_key(self, message_id: Optional[str], phone_number: str, message_text: str, timestamp: Optional[int] = None) -> str:
        """
        Generate idempotency key for message.
        
        Priority:
        1. Use message_id if available (WhatsApp unique ID)
        2. Fallback to hash(phone + message + timestamp)
        """
        if message_id:
            # Use WhatsApp message ID as idempotency key
            return f"idempotency:msg:{message_id}"
        else:
            # Fallback: hash of phone + message + rounded timestamp
            # Round timestamp to 5-second buckets to catch rapid duplicates
            ts_bucket = (timestamp or int(time.time())) // 5 * 5
            combined = f"{phone_number}:{message_text}:{ts_bucket}"
            msg_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
            return f"idempotency:hash:{msg_hash}"
    
    async def is_duplicate(self, message_id: Optional[str], phone_number: str, message_text: str, timestamp: Optional[int] = None) -> Tuple[bool, Optional[dict]]:
        """
        Check if message is duplicate using Redis.
        
        Returns:
            Tuple of (is_duplicate, cached_response)
        """
        idempotency_key = self._get_idempotency_key(message_id, phone_number, message_text, timestamp)
        
        try:
            # Check Redis for cached processing result
            redis_key = f"dedup:{idempotency_key}"
            cached_data = await self.session_manager.redis.get(redis_key)
            
            if cached_data:
                try:
                    cached_response = json.loads(cached_data)
                    cached_time = cached_response.get("processed_at", 0)
                    current_time = time.time()
                    time_diff = current_time - cached_time
                    
                    logger.warning(
                        f"🚫 DUPLICATE WEBHOOK DETECTED: message_id={message_id or 'hash'} from {phone_number[:15]}... "
                        f"within {time_diff:.1f}s - SKIPPING reprocessing (Issue #12)"
                    )
                    logger.warning(f"🚫 IDEMPOTENCY: Returning cached response to prevent duplicate actions")
                    
                    return True, cached_response.get("response")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse cached response for {idempotency_key}")
            
            return False, None
            
        except Exception as exc:
            logger.error(f"Deduplication check failed: {exc}")
            # On error, allow processing (fail open) but log the issue
            return False, None
    
    async def mark_processing(self, message_id: Optional[str], phone_number: str, message_text: str, timestamp: Optional[int] = None):
        """
        Mark message as being processed to prevent race conditions.
        Sets a placeholder in Redis immediately after duplicate check.
        """
        idempotency_key = self._get_idempotency_key(message_id, phone_number, message_text, timestamp)
        
        try:
            redis_key = f"dedup:{idempotency_key}"
            cache_data = {
                "processing": True,
                "started_at": time.time(),
                "message_id": message_id,
                "phone_number": phone_number[:15] + "...",
                "response": {
                    "status": "processing",
                    "message": "Message is currently being processed"
                }
            }
            
            # Store with short TTL (will be overwritten by cache_response)
            await self.session_manager.redis.setex(
                redis_key,
                30,  # 30 seconds - just to prevent race condition window
                json.dumps(cache_data)
            )
            
            logger.debug(f"🔄 IDEMPOTENCY: Marked {idempotency_key} as processing")
            
        except Exception as exc:
            logger.error(f"Failed to mark message as processing: {exc}")
            # Non-critical - don't fail the request
    
    async def cache_response(self, message_id: Optional[str], phone_number: str, message_text: str, response: dict, timestamp: Optional[int] = None):
        """
        Cache message processing result in Redis with TTL.
        """
        idempotency_key = self._get_idempotency_key(message_id, phone_number, message_text, timestamp)
        
        try:
            redis_key = f"dedup:{idempotency_key}"
            cache_data = {
                "processed_at": time.time(),
                "message_id": message_id,
                "phone_number": phone_number[:15] + "...",  # Truncate for privacy
                "response": response
            }
            
            # Store with TTL
            await self.session_manager.redis.setex(
                redis_key,
                self.window_seconds,
                json.dumps(cache_data)
            )
            
            logger.debug(f"✅ IDEMPOTENCY: Cached response for {idempotency_key} (TTL: {self.window_seconds}s)")
            
        except Exception as exc:
            logger.error(f"Failed to cache response for deduplication: {exc}")
            # Non-critical - don't fail the request


# Global deduplicator instance
_deduplicator = MessageDeduplicator()


@webhook.post("/webhook")
async def receive_wasender(request: Request):
    """
    WaSender webhook endpoint for receiving WhatsApp messages.
    
    Expected payload structure from WaSender:
    {
        "body": {
            "data": {
                "messages": [{
                    "key": {
                        "remoteJid": "1234567890@s.whatsapp.net",
                        "fromMe": false,
                        "id": "message_id"
                    },
                    "message": {
                        "conversation": "Hello, I want to book an appointment"
                    },
                    "pushName": "John Doe",
                    "messageTimestamp": 1234567890
                }]
            }
        }
    }
    
    Returns:
        JSON response with processing status
    """
    try:
        # Get raw payload as text first for sanitization
        raw_text = await request.body()
        raw_text = raw_text.decode('utf-8')
        
        # SANITIZE: Fix common JSON errors from WaSender
        raw_text = raw_text.replace('Falsee', 'false')  # Fix typo
        raw_text = raw_text.replace('Truee', 'true')    # Just in case
        
        # Parse sanitized JSON
        logger.info("=" * 60)
        logger.info(f"📥 WEBHOOK RECEIVED FROM WASender")
        logger.info("=" * 60)
        
        # Parse raw body
        raw_payload = await request.json()
        logger.info(f"📦 RAW PAYLOAD: {raw_payload}")  # Show full payload for debugging
        
        # Parse and validate payload
        parser = get_wasender_parser()
        
        # Validate structure first
        if not parser.validate_payload_structure(raw_payload):
            logger.error("Invalid WaSender payload structure")
            logger.error(f"Payload keys: {list(raw_payload.keys())}")  # DEBUG: See what keys exist
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "error": "invalid_payload_structure",
                    "message": "Payload does not match WaSender format"
                }
            )
        
        # Parse payload
        parsed_message = parser.parse(raw_payload)
        
        if not parsed_message:
            logger.warning("Failed to parse message or message should be skipped")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "skipped",
                    "message": "Message skipped (bot message or parsing failed)"
                }
            )
        
        logger.info(f"📞 From: {parsed_message.sender_name} ({parsed_message.phone_number})")
        logger.info(f"💬 Message: {parsed_message.message_text[:100]}")
        logger.info(f"🔑 Session: {parsed_message.session_key}")
        logger.info(f"🎫 Message ID: {parsed_message.message_id or 'N/A'}")
        
        # CRITICAL: Check for duplicate message using Redis-backed idempotency (Issue #12)
        # Handles: WhatsApp retries, webhook redeliveries, network duplicates
        is_duplicate, cached_response = await _deduplicator.is_duplicate(
            parsed_message.message_id,
            parsed_message.phone_number,
            parsed_message.message_text,
            int(parsed_message.timestamp.timestamp()) if parsed_message.timestamp else None
        )
        
        if is_duplicate:
            # Return cached response immediately without reprocessing
            logger.info("✅ IDEMPOTENCY: Returning cached response (no reprocessing)")
            logger.info("✅ IMPACT: User won't receive duplicate response, state unchanged")
            logger.info("=" * 60)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=cached_response or {
                    "status": "success",
                    "message": "Duplicate webhook - already processed",
                    "session_key": parsed_message.session_key,
                    "idempotent": True
                }
            )
        
        # CRITICAL: Mark as processing IMMEDIATELY to prevent race condition
        # If another duplicate arrives in the next few milliseconds, it will see "processing"
        await _deduplicator.mark_processing(
            parsed_message.message_id,
            parsed_message.phone_number,
            parsed_message.message_text,
            int(parsed_message.timestamp.timestamp()) if parsed_message.timestamp else None
        )
        logger.info("🔄 Message marked as processing - duplicates will now be blocked")
        
        # Convert parsed message to dict for router
        payload_for_router = {
            "phone_number": parsed_message.phone_number,
            "message": parsed_message.message_text,
            "sender_name": parsed_message.sender_name,
            "message_id": parsed_message.message_id,
            "message_type": parsed_message.message_type,
            "session_key": parsed_message.session_key,
            "timestamp": parsed_message.timestamp.isoformat() if parsed_message.timestamp else None,
            # Extract IP headers for rate limiting/analytics (Issue #36, #46)
            "x-forwarded-for": request.headers.get("x-forwarded-for"),
            "x-real-ip": request.headers.get("x-real-ip"),
            "remote_addr": request.client.host if request.client else None
        }
        
        # CRITICAL: Acquire per-user lock to prevent race conditions (Issue: Quick messages lost)
        # When messages arrive too quickly, they can overwrite each other's state
        # Solution: Use Redis lock to ensure sequential processing per user
        from redis.asyncio.lock import Lock as AsyncRedisLock
        import asyncio
        
        session_manager = SessionManager()
        lock_key = f"lock:{parsed_message.session_key}"
        
        # Create distributed lock (120 second timeout, 0.1s retry)
        # NOTE: Lock held during API calls - if API hangs, lock auto-releases after 120s
        # TODO: Refactor to lock only state read/write, not external API calls
        # Increased from 45s to 120s due to slow API/LLM calls (observed: 105s processing time)
        import time
        lock_start = time.time()
        
        async with AsyncRedisLock(
            session_manager.redis,
            lock_key,
            timeout=120,  # Auto-release after 120s if process crashes (handles slow API/LLM calls)
            blocking_timeout=5  # Wait max 5s to acquire lock
        ):
            logger.info(f"🔒 Lock acquired for {parsed_message.session_key}")
            
            # Route to intent router for agent processing
            # CRITICAL: This includes external API calls (customer creation, etc.)
            # If API hangs, lock will auto-release after 120s timeout
            logger.info("🤖 Routing to agent orchestration...")
            router = IntentRouter(session_key=parsed_message.session_key)
            result = await router.route(payload_for_router)
            
            lock_duration = time.time() - lock_start
            logger.info(f"🔓 Lock released for {parsed_message.session_key} (held for {lock_duration:.2f}s)")
            
            # Warn if lock held too long (indicates slow API or processing)
            if lock_duration > 3.0:
                logger.warning(f"⚠️ Lock held for {lock_duration:.2f}s - consider optimizing or refactoring")
        
        logger.info(f"✅ Processing complete - Intent: {result.get('intent', 'unknown')}")
        logger.info("=" * 60)
        
        # Cache response for deduplication (Issue #12)
        response_content = {
            "status": "success",
            "message": "Message processed successfully",
            "intent": result.get("intent"),
            "session_key": parsed_message.session_key,
            "idempotent": False  # First processing
        }
        
        # Store in Redis with 5-minute TTL for idempotency
        await _deduplicator.cache_response(
            parsed_message.message_id,
            parsed_message.phone_number,
            parsed_message.message_text,
            response_content,
            int(parsed_message.timestamp.timestamp()) if parsed_message.timestamp else None
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_content
        )
        
    except ValueError as ve:
        logger.error(f"❌ Validation error: {ve}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error": "validation_error",
                "message": str(ve)
            }
        )
    
    except Exception as exc:
        logger.error(f"❌ Webhook processing error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error": "internal_error",
                "message": "Failed to process webhook"
            }
        )


