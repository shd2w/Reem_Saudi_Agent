"""
Hybrid Architecture Webhook Handler
====================================
Simplified webhook handler that delegates to ConversationOrchestrator.

This is MUCH simpler than the old Router-based handler because
all logic is now in the orchestrator.
"""
from fastapi import APIRouter, Request
from typing import Dict, Any, Optional, Tuple
from loguru import logger
import time
import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

from ..orchestration.conversation_orchestrator import ConversationOrchestrator
from ..api.wasender_client import WaSenderClient
from ..api.wasender_parser import get_wasender_parser
from ..memory.session_manager import SessionManager


router = APIRouter()


# ============================================================================
# MESSAGE DEDUPLICATION (Inline - from webhook_handler.py)
# ============================================================================

class MessageDeduplicator:
    """Redis-backed message deduplication"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.window_seconds = 300  # 5 minutes
    
    def _get_key(self, message_id: str, session_key: str) -> str:
        """Generate Redis key for deduplication"""
        if message_id:
            return f"dedup:msg:{message_id}"
        else:
            msg_hash = hashlib.sha256(f"{session_key}:{time.time()}".encode()).hexdigest()[:16]
            return f"dedup:hash:{msg_hash}"
    
    async def check_duplicate(self, message_id: str, session_key: str) -> Optional[str]:
        """Check if message is duplicate, return cached response if found"""
        try:
            redis_key = self._get_key(message_id, session_key)
            cached_data = await self.session_manager.redis.get(redis_key)
            
            if cached_data:
                try:
                    cached_response = json.loads(cached_data)
                    logger.warning(f"♻️ Duplicate message detected - returning cached response")
                    return cached_response.get("response")
                except json.JSONDecodeError:
                    pass
            
            return None
        except Exception as e:
            logger.error(f"Deduplication check failed: {e}")
            return None
    
    async def mark_processing(self, message_id: str, session_key: str):
        """Mark message as being processed"""
        try:
            redis_key = self._get_key(message_id, session_key)
            cache_data = {
                "processing": True,
                "started_at": time.time(),
                "response": {"status": "processing"}
            }
            await self.session_manager.redis.setex(redis_key, 30, json.dumps(cache_data))
        except Exception as e:
            logger.error(f"Failed to mark processing: {e}")
    
    async def cache_response(self, message_id: str, session_key: str, response: str):
        """Cache response for deduplication"""
        try:
            redis_key = self._get_key(message_id, session_key)
            cache_data = {
                "processed_at": time.time(),
                "response": response
            }
            await self.session_manager.redis.setex(redis_key, self.window_seconds, json.dumps(cache_data))
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")


# ============================================================================
# SESSION LOCK MANAGER (Inline - simplified)
# ============================================================================

class SessionLockManager:
    """Simple session lock using Redis"""
    
    def __init__(self):
        self.session_manager = SessionManager()
    
    @asynccontextmanager
    async def acquire_lock(self, session_key: str, timeout: float = 30.0):
        """Acquire lock for session"""
        lock_key = f"lock:{session_key}"
        acquired = False
        
        try:
            # Try to acquire lock
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Try to set lock with NX (only if not exists)
                result = await self.session_manager.redis.set(
                    lock_key, 
                    "locked", 
                    ex=int(timeout),
                    nx=True
                )
                
                if result:
                    acquired = True
                    break
                
                # Wait a bit before retry
                await asyncio.sleep(0.1)
            
            yield acquired
            
        finally:
            # Release lock
            if acquired:
                try:
                    await self.session_manager.redis.delete(lock_key)
                except:
                    pass


# Singletons
orchestrator = ConversationOrchestrator()
deduplicator = MessageDeduplicator()
lock_manager = SessionLockManager()
whatsapp_client = WaSenderClient()
wasender_parser = get_wasender_parser()


@router.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """
    Webhook endpoint for WhatsApp messages.
    
    This is the ONLY entry point for user messages.
    
    Architecture:
        Webhook → Validate → Deduplicate → Lock → Orchestrator → WhatsApp Send
    
    Much simpler than before!
    """
    
    start_time = time.time()
    
    try:
        # 1. Parse payload using WaSender parser
        payload = await request.json()
        
        # Try to parse with WaSender parser first
        parsed_message = wasender_parser.parse(payload)
        
        if parsed_message:
            # WaSender format
            phone_number = parsed_message.phone_number
            message_text = parsed_message.message_text
            message_id = parsed_message.message_id
            sender_name = parsed_message.sender_name
        else:
            # Fallback: Simple format (for testing)
            phone_number = payload.get("phone") or payload.get("from")
            message_text = payload.get("message") or payload.get("text") or payload.get("body", "")
            message_id = payload.get("id", f"{phone_number}_{int(time.time())}")
            sender_name = payload.get("name", "")
        
        if not phone_number or not message_text:
            logger.warning(f"⚠️ Invalid webhook payload - missing phone or message. Keys: {list(payload.keys())}")
            return {"status": "error", "message": "Invalid payload"}
        
        # Normalize phone (remove + and spaces)
        phone_number = phone_number.replace("+", "").replace(" ", "").strip()
        session_key = f"whatsapp:{phone_number}"
        
        logger.info("=" * 60)
        logger.info(f"📩 [WEBHOOK] New message from {phone_number}")
        if sender_name:
            logger.info(f"👤 Sender: {sender_name}")
        logger.info(f"📝 Message: {message_text[:100]}...")
        logger.info(f"🔑 Session: {session_key}")
        logger.info("=" * 60)
        
        # 2. Deduplication check
        cached_response = await deduplicator.check_duplicate(message_id, session_key)
        if cached_response:
            logger.info(f"♻️ Duplicate message detected - returning cached response")
            return {"status": "cached", "response": cached_response}
        
        # 3. Mark as processing (prevents race conditions)
        await deduplicator.mark_processing(message_id, session_key)
        
        # 4. Acquire session lock (prevent concurrent processing)
        lock_start = time.time()
        async with lock_manager.acquire_lock(session_key, timeout=30.0) as acquired:
            if not acquired:
                logger.warning(f"⏱️ Lock timeout for {session_key}")
                return {
                    "status": "timeout",
                    "message": "Session busy, please try again"
                }
            
            lock_duration = time.time() - lock_start
            logger.info(f"🔒 Lock acquired for {session_key} (waited {lock_duration:.2f}s)")
            
            try:
                # 5. Process message through orchestrator
                response = await orchestrator.handle_message(
                    message=message_text,
                    session_id=session_key,
                    user_phone=phone_number,
                    metadata={
                        "message_id": message_id,
                        "request_time": start_time,
                        "sender_name": sender_name if sender_name else None
                    }
                )
                
                # 6. Send response to WhatsApp
                send_start = time.time()
                await whatsapp_client.send_message(
                    phone_number=phone_number,
                    message=response
                )
                send_duration = time.time() - send_start
                
                # 7. Cache response for deduplication
                await deduplicator.cache_response(message_id, session_key, response)
                
                # 8. Log performance
                total_duration = time.time() - start_time
                logger.info(f"⏱️ PERFORMANCE: total={total_duration:.2f}s | send={send_duration:.2f}s")
                logger.info(f"✅ Processing complete for {phone_number}")
                logger.info("=" * 60)
                
                return {
                    "status": "success",
                    "response": response,
                    "duration": total_duration
                }
            
            finally:
                # Lock is released automatically by context manager
                lock_held = time.time() - (lock_start + lock_duration)
                logger.info(f"🔓 Lock released for {session_key} (held {lock_held:.2f}s)")
    
    except Exception as e:
        logger.error(f"❌ [WEBHOOK] Error processing message: {e}", exc_info=True)
        
        # Send error message to user
        try:
            await whatsapp_client.send_message(
                phone_number=phone_number,
                message="آسف! حصل خطأ تقني. الرجاء المحاولة مرة أخرى."
            )
        except:
            pass
        
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "architecture": "hybrid",
        "components": {
            "orchestrator": "active",
            "reem_agent": "active",
            "workflow_executor": "active"
        }
    }
