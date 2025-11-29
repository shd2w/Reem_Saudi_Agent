import json
import time
from datetime import timedelta
import redis.asyncio as redis
import redis as sync_redis
from loguru import logger

from ..config import settings
from ..utils.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError

# CRITICAL FIX: Import exceptions correctly for redis-py compatibility
try:
    # Try new redis-py >= 4.2.0 structure
    from redis.asyncio import exceptions as redis_exceptions
except (ImportError, AttributeError):
    # Fallback to older structure
    from redis import exceptions as redis_exceptions


class SessionManager:
    """
    Manages user sessions using Redis for state persistence.
    Supports both synchronous and asynchronous operations.
    Singleton pattern to avoid re-initialization.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once - prevent duplicate initialization
        if SessionManager._initialized:
            return  # Already initialized - silent reuse
        
        # Async Redis client with timeouts to prevent blocking (Issue: 45s response times)
        self.redis = redis.from_url(
            settings.redis_url, 
            decode_responses=True,
            socket_connect_timeout=3,  # Connection timeout: 3 seconds
            socket_timeout=5,  # Operation timeout: 5 seconds
            socket_keepalive=True,  # Keep connections alive
            retry_on_timeout=False  # Don't retry on timeout - fail fast
        )
        # Sync Redis client for backward compatibility
        self.sync_redis = sync_redis.from_url(
            settings.redis_url, 
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5
        )
        
        SessionManager._initialized = True
        logger.info("✅ Session Manager initialized with Redis (singleton) - First init")

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"

    async def get_session(self, session_id: str) -> dict:
        """
        Get session data for a given session ID.
        Returns empty dict if session doesn't exist.
        """
        circuit_breaker = get_circuit_breaker("redis_read", failure_threshold=3, recovery_timeout=30)
        
        try:
            logger.debug(f"📖 REDIS READ: {self._key(session_id)[:50]}...")
            
            def _read():
                return self.redis.get(self._key(session_id))
            
            raw = circuit_breaker.call(_read)
            raw = await raw  # Await the coroutine
            
            if not raw:
                logger.debug(f"📖 REDIS READ RESULT: No session found for {session_id[:30]}...")
                return {}
            
            data = json.loads(raw)
            logger.debug(f"✅ REDIS READ SUCCESS: {len(raw)} bytes, {len(data)} keys")
            return data
            
        except CircuitBreakerOpenError as e:
            logger.error(f"🔴 REDIS CIRCUIT OPEN: Cannot read session {session_id} - {e}")
            logger.error(f"🔴 FALLBACK: Returning empty session (user treated as new)")
            return {}
        except redis_exceptions.ConnectionError as exc:
            logger.error(f"🔴 REDIS CONNECTION FAILED: {exc}")
            logger.error(f"🔴 FALLBACK: Returning empty session for {session_id}")
            return {}
        except redis_exceptions.TimeoutError as exc:
            logger.error(f"🔴 REDIS TIMEOUT: {exc}")
            logger.error(f"🔴 FALLBACK: Returning empty session for {session_id}")
            return {}
        except Exception as exc:
            logger.error(f"🔴 REDIS ERROR (unexpected): {exc}", exc_info=True)
            logger.error(f"🔴 FALLBACK: Returning empty session for {session_id}")
            return {}

    async def update_session(self, session_id: str, data: dict, ttl_minutes: int = 120) -> None:
        """
        Update session data. Merges with existing data if present.
        
        CRITICAL: Handles Redis memory exhaustion gracefully (Issue: Redis maxmemory)
        PERFORMANCE: 5-second timeout to prevent blocking (Issue: 45s response times)
        """
        try:
            import asyncio
            
            # Get existing session
            existing = await self.get_session(session_id)
            
            # Merge with new data
            existing.update(data)
            
            # Save back to Redis with TTL and TIMEOUT (5 seconds max)
            await asyncio.wait_for(
                self.redis.setex(
                    self._key(session_id),
                    timedelta(minutes=ttl_minutes),
                    json.dumps(existing)
                ),
                timeout=5.0  # CRITICAL: Prevent blocking for 45+ seconds
            )
            logger.debug(f"✅ Session updated for {session_id}")
        
        except asyncio.TimeoutError:
            # CRITICAL: Redis write took too long - don't block user response
            logger.error(f"⚠️ REDIS WRITE TIMEOUT (>5s) for {session_id[:30]}... - continuing without persistence")
            logger.error(f"⚠️ IMPACT: Session state may be lost, but user gets fast response")
            
        except redis_exceptions.ConnectionError as exc:
            # CRITICAL: Redis connection lost (Issue: Error 10054)
            logger.error(f"🔴 REDIS CONNECTION FAILED during write!")
            logger.error(f"🔴 Error: {exc}")
            
            # Retry with minimal backoff (fast fail for better UX)
            for attempt in range(2):  # Reduced to 2 retries
                try:
                    import asyncio
                    wait_time = 0.1 * (attempt + 1)  # 0.1s, 0.2s (was 0.5s, 1s, 2s)
                    logger.warning(f"⚠️ Retry attempt {attempt + 1}/2 after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    
                    # CRITICAL: Recreate Redis connection on retry (Error 10054: connection closed)
                    logger.warning(f"🔄 Recreating Redis connection for retry {attempt + 1}...")
                    try:
                        await self.redis.close()
                    except:
                        pass
                    self.redis = redis.from_url(
                        settings.redis_url,
                        decode_responses=True,
                        socket_connect_timeout=1,  # Reduced from 3s
                        socket_timeout=2,  # Reduced from 5s
                        socket_keepalive=True,
                        retry_on_timeout=False
                    )
                    
                    # Retry save
                    await self.redis.setex(
                        self._key(session_id),
                        timedelta(minutes=ttl_minutes),
                        json.dumps(existing)
                    )
                    logger.warning(f"✅ Session saved after retry {attempt + 1}")
                    return
                    
                except Exception as retry_exc:
                    logger.error(f"❌ Retry {attempt + 1} failed: {retry_exc}")
                    if attempt == 1:  # Last attempt (0, 1)
                        logger.error("❌ All retries exhausted - session not saved!")
                        
        except redis_exceptions.ResponseError as exc:
            # CRITICAL: Handle Redis memory exhaustion
            if "maxmemory" in str(exc).lower():
                logger.error(f"🚨 REDIS MEMORY FULL! Cannot save session {session_id}")
                logger.error(f"🚨 Error: {exc}")
                
                # Try emergency cleanup
                try:
                    logger.warning("⚠️ Attempting emergency cleanup of old sessions...")
                    await self._emergency_cleanup()
                    
                    # Retry save after cleanup
                    await self.redis.setex(
                        self._key(session_id),
                        timedelta(minutes=ttl_minutes),
                        json.dumps(existing)
                    )
                    logger.warning("✅ Session saved after emergency cleanup")
                    return
                    
                except Exception as cleanup_exc:
                    logger.error(f"❌ Emergency cleanup failed: {cleanup_exc}")
                
                # If still failing, alert and continue (graceful degradation)
                logger.error("❌ SESSION STATE NOT SAVED - User may experience loops!")
                logger.error("❌ ACTION REQUIRED: Increase Redis maxmemory or enable eviction policy")
                
                # TODO: Implement in-memory fallback cache if needed
                # self._memory_cache[session_id] = existing
                
            else:
                # Other Redis error (including protocol errors)
                error_msg = str(exc).lower()
                
                if "protocol error" in error_msg or "invalid bulk length" in error_msg:
                    # CRITICAL: Protocol error - likely data corruption or encoding issue
                    logger.error(f"🚨 REDIS PROTOCOL ERROR for session {session_id}: {exc}")
                    logger.error(f"🚨 This indicates corrupted data or encoding issues!")
                    
                    # Try to identify the problematic data
                    try:
                        serialized = json.dumps(existing)
                        logger.error(f"📊 Data size: {len(serialized)} bytes")
                        logger.error(f"📊 Data keys: {list(existing.keys())}")
                        
                        # Check for non-serializable objects
                        for key, value in existing.items():
                            try:
                                json.dumps({key: value})
                            except (TypeError, ValueError) as e:
                                logger.error(f"❌ Problematic field: {key} = {type(value)} - {e}")
                                
                    except Exception as debug_exc:
                        logger.error(f"❌ Could not debug data: {debug_exc}")
                    
                    # Try to save without problematic fields
                    try:
                        # Create clean copy with only JSON-serializable data
                        clean_data = {}
                        for key, value in existing.items():
                            try:
                                json.dumps({key: value})
                                clean_data[key] = value
                            except:
                                logger.warning(f"⚠️ Skipping non-serializable field: {key}")
                        
                        # Retry with clean data
                        await self.redis.setex(
                            self._key(session_id),
                            timedelta(minutes=ttl_minutes),
                            json.dumps(clean_data)
                        )
                        logger.warning(f"✅ Session saved with cleaned data (removed {len(existing) - len(clean_data)} fields)")
                        return
                        
                    except Exception as clean_exc:
                        logger.error(f"❌ Even cleaned data failed: {clean_exc}")
                else:
                    # Other Redis error
                    logger.error(f"❌ Redis error updating session {session_id}: {exc}")
                
        except Exception as exc:
            logger.error(f"❌ Error updating session {session_id}: {exc}", exc_info=True)

    async def _emergency_cleanup(self) -> None:
        """
        Emergency cleanup when Redis is full.
        Deletes oldest sessions to free up memory.
        """
        try:
            # Find all session keys
            pattern = "session:*"
            keys_deleted = 0
            
            # Use SCAN to avoid blocking Redis
            cursor = 0
            all_keys = []
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                all_keys.extend(keys)
                
                if cursor == 0:
                    break
            
            if not all_keys:
                logger.warning("⚠️ No session keys found to clean")
                return
            
            # Get TTL for each key to find oldest
            key_ttls = []
            for key in all_keys:
                ttl = await self.redis.ttl(key)
                key_ttls.append((key, ttl))
            
            # Sort by TTL (lowest = oldest)
            key_ttls.sort(key=lambda x: x[1])
            
            # Delete oldest 20% of sessions
            delete_count = max(1, len(key_ttls) // 5)
            
            for key, ttl in key_ttls[:delete_count]:
                await self.redis.delete(key)
                keys_deleted += 1
            
            logger.warning(f"🧹 Emergency cleanup: deleted {keys_deleted}/{len(all_keys)} old sessions")
            
        except Exception as exc:
            logger.error(f"❌ Emergency cleanup error: {exc}")
            raise
    
    async def cleanup_stale_session(self, session_id: str, max_idle_seconds: int = 7200) -> None:
        """
        Clean up stale session data after inactivity.
        
        Args:
            session_id: Session identifier
            max_idle_seconds: Maximum idle time before cleanup (default 2 hours for async chat)
        
        Note: Increased from 30 min to 2 hours to accommodate async messaging patterns.
              Chat conversations can span hours between messages.
        """
        import time
        
        session_data = await self.get_session(session_id)
        last_activity = session_data.get('last_activity_timestamp', 0)
        
        current_time = time.time()
        
        # FIX: Handle uninitialized timestamps (0 = never set)
        if last_activity == 0:
            # Session never had activity timestamp - initialize it now
            logger.debug(f"⚠️ Session {session_id} has no activity timestamp - initializing")
            session_data['last_activity_timestamp'] = current_time
            await self.put_session(session_id, session_data)
            return  # Don't clean a session we just initialized
        
        idle_time = current_time - last_activity
        
        # FIX: Sanity check - if idle time is absurd (> 1 year), reset timestamp
        if idle_time > 31536000:  # 1 year in seconds
            logger.warning(f"⚠️ Detected absurd idle time ({idle_time:.0f}s = {idle_time/31536000:.1f} years) for {session_id} - resetting timestamp")
            session_data['last_activity_timestamp'] = current_time
            await self.put_session(session_id, session_data)
            return
        
        if idle_time > max_idle_seconds:
            logger.info(f"🧹 Cleaning stale session (idle: {idle_time:.0f}s): {session_id}")
            
            # Clear ephemeral state
            session_data.pop('last_intent', None)
            session_data.pop('booking_state', None)
            session_data.pop('name_confirmation_pending', None)
            session_data.pop('pending_entities', None)
            
            # Keep only confirmed data + essential session tracking
            cleaned_data = {
                'confirmed_name': session_data.get('confirmed_name'),
                'confirmed_phone': session_data.get('confirmed_phone'),
                'conversation_turn': session_data.get('conversation_turn', 0),  # CRITICAL: Preserve turn counter
                'patient_data': session_data.get('patient_data'),  # CRITICAL: Preserve patient data
                'sender_name': session_data.get('sender_name'),  # Preserve name
                'last_activity_timestamp': current_time
            }
            
            await self.put_session(session_id, cleaned_data)
    
    def put_session_async(self, session_id: str, data: dict, ttl_minutes: int = 120) -> None:
        """
        Fire-and-forget async write to Redis. Does NOT block the caller.
        
        PERFORMANCE: Use this for non-critical writes to prevent blocking user response.
        Example: After sending response to user, persist state in background.
        """
        import asyncio
        
        async def _async_write():
            try:
                await self.put_session(session_id, data, ttl_minutes)
            except Exception as e:
                logger.error(f"⚠️ Background Redis write failed for {session_id[:30]}: {e}")
        
        # Fire and forget - don't wait for result
        asyncio.create_task(_async_write())
        logger.debug(f"🔥 Queued async Redis write for {session_id[:30]}...")
    
    async def put_session(self, session_id: str, data: dict, ttl_minutes: int = 120) -> None:
        """
        Put/create session data. Overwrites existing data.
        
        PERFORMANCE: 5-second timeout to prevent blocking (Issue: 45s response times)
        """
        import asyncio
        circuit_breaker = get_circuit_breaker("redis_write", failure_threshold=3, recovery_timeout=30)
        
        try:
            serialized = json.dumps(data)
            logger.debug(f"💾 REDIS WRITE: {self._key(session_id)[:50]}... ({len(serialized)} bytes, TTL={ttl_minutes}m)")
            
            def _write():
                return self.redis.setex(
                    self._key(session_id),
                    timedelta(minutes=ttl_minutes),
                    serialized
                )
            
            result = circuit_breaker.call(_write)
            
            # CRITICAL: Add timeout to prevent 45+ second blocks
            await asyncio.wait_for(result, timeout=5.0)
            logger.debug(f"✅ REDIS WRITE SUCCESS: {session_id[:30]}... ({len(data)} keys persisted)")
        
        except asyncio.TimeoutError:
            # CRITICAL: Redis write took too long - don't block user response
            logger.error(f"⚠️ REDIS WRITE TIMEOUT (>5s) for {session_id[:30]}... - continuing without persistence")
            logger.error(f"⚠️ IMPACT: Session state may be lost, but user gets fast response")
            
        except CircuitBreakerOpenError as e:
            logger.error(f"🔴 REDIS CIRCUIT OPEN: Cannot save session {session_id} - {e}")
            logger.error(f"🔴 IMPACT: Session state will be lost on next request")
        except redis_exceptions.ConnectionError as exc:
            logger.error(f"🔴 REDIS CONNECTION FAILED: Cannot save session - {exc}")
            logger.error(f"🔴 IMPACT: User will lose conversation context")
        except redis_exceptions.TimeoutError as exc:
            logger.error(f"🔴 REDIS TIMEOUT: Write operation timed out - {exc}")
            logger.error(f"🔴 IMPACT: Session may not be persisted")
        except redis_exceptions.ResponseError as exc:
            # Handle protocol errors and other Redis response errors
            error_msg = str(exc).lower()
            
            if "protocol error" in error_msg or "invalid bulk length" in error_msg:
                logger.error(f"🚨 REDIS PROTOCOL ERROR for session {session_id}: {exc}")
                logger.error(f"🚨 Data size: {len(serialized)} bytes")
                logger.error(f"🚨 Data keys: {list(data.keys())}")
                
                # Try to identify problematic fields
                for key, value in data.items():
                    try:
                        json.dumps({key: value})
                    except (TypeError, ValueError) as e:
                        logger.error(f"❌ Problematic field: {key} = {type(value)} - {e}")
                
                # Try with cleaned data
                try:
                    clean_data = {}
                    for key, value in data.items():
                        try:
                            json.dumps({key: value})
                            clean_data[key] = value
                        except:
                            logger.warning(f"⚠️ Skipping non-serializable field: {key}")
                    
                    clean_serialized = json.dumps(clean_data)
                    await self.redis.setex(
                        self._key(session_id),
                        timedelta(minutes=ttl_minutes),
                        clean_serialized
                    )
                    logger.warning(f"✅ Session saved with cleaned data (removed {len(data) - len(clean_data)} fields)")
                    
                except Exception as clean_exc:
                    logger.error(f"❌ Even cleaned data failed: {clean_exc}")
                    logger.error(f"🔴 IMPACT: Session state lost for {session_id}")
            else:
                logger.error(f"🔴 REDIS RESPONSE ERROR: {exc}", exc_info=True)
                logger.error(f"🔴 IMPACT: Session state lost for {session_id}")
                
        except Exception as exc:
            logger.error(f"🔴 REDIS ERROR (unexpected): {exc}", exc_info=True)
            logger.error(f"🔴 IMPACT: Session state lost for {session_id}")

    async def audit_slot_change(
        self,
        session_id: str,
        slot_name: str,
        old_value: any,
        new_value: any,
        action: str,
        confidence: str = None
    ) -> None:
        """
        Audit trail for slot changes.
        
        Args:
            session_id: Session ID
            slot_name: Name of slot
            old_value: Previous value
            new_value: New value
            action: Action type (set, confirm, reject, clear)
            confidence: Confidence level
        """
        import time
        
        audit_key = f"audit:{session_id}:{slot_name}"
        
        audit_entry = {
            "timestamp": time.time(),
            "slot": slot_name,
            "old_value": old_value,
            "new_value": new_value,
            "action": action,
            "confidence": confidence
        }
        
        # Store in Redis list
        await self.redis.lpush(audit_key, json.dumps(audit_entry, ensure_ascii=False))
        await self.redis.ltrim(audit_key, 0, 99)  # Keep last 100 entries
        await self.redis.expire(audit_key, 86400 * 7)  # 7 days TTL
        
        logger.info(f"📝 audit_trail: session={session_id}, slot={slot_name}, action={action}, old={old_value}, new={new_value}")
    
    async def delete_session(self, session_id: str) -> None:
        """Delete session data"""
        try:
            logger.debug(f"🗑️ REDIS DELETE: {self._key(session_id)[:50]}...")
            result = await self.redis.delete(self._key(session_id))
            logger.debug(f"✅ REDIS DELETE SUCCESS: {session_id[:30]}... (keys_deleted={result})")
        except CircuitBreakerOpenError as e:
            logger.error(f"🔴 REDIS CIRCUIT OPEN: Cannot delete session {session_id} - {e}")
            logger.error(f"🔴 IMPACT: Session state will be lost on next request")
        except redis_exceptions.ConnectionError as exc:
            logger.error(f"🔴 REDIS CONNECTION FAILED: Cannot delete session - {exc}")
            logger.error(f"🔴 IMPACT: Session state will be lost on next request")
        except redis_exceptions.TimeoutError as exc:
            logger.error(f"🔴 REDIS TIMEOUT: Delete operation timed out - {exc}")
            logger.error(f"🔴 IMPACT: Session state may not be deleted")
        except Exception as exc:
            logger.error(f"🔴 REDIS ERROR (unexpected): {exc}", exc_info=True)
            logger.error(f"🔴 IMPACT: Session state lost for {session_id}")
    
    async def add_to_history(self, session_id: str, role: str, message: str, max_history: int = 10) -> None:
        """
        Add message to conversation history.
        Maintains rolling window of last N messages.
{{ ... }}
        Args:
            session_id: Session ID
            role: 'user' or 'assistant'
            message: Message content
            max_history: Maximum messages to keep (default 10)
        """
        try:
            session = await self.get_session(session_id)
            history = session.get("history", [])
            
            # Add new message with timestamp (Issue #8: Use "content" key for consistency)
            from datetime import datetime
            history.append({
                "role": role,
                "content": message,  # Changed from "message" to "content" for LLM compatibility
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last max_history messages
            if len(history) > max_history:
                history = history[-max_history:]
            
            session["history"] = history
            session["last_message"] = message
            session["last_role"] = role
            
            await self.put_session(session_id, session)
            logger.debug(f"Added to history for {session_id} (total: {len(history)})")
            
        except Exception as exc:
            logger.error(f"Error adding to history for {session_id}: {exc}")
    
    # Synchronous methods using sync Redis client
    def get(self, chat_id: str) -> dict | None:
        """
        Synchronous get - uses sync Redis client.
        Returns empty dict if session doesn't exist.
        """
        try:
            raw = self.sync_redis.get(self._key(chat_id))
            if not raw:
                logger.debug(f"No session found for {chat_id}")
                return {}
            data = json.loads(raw)
            logger.debug(f"Retrieved session for {chat_id}: {data}")
            return data
        except Exception as exc:
            logger.error(f"Error getting session {chat_id}: {exc}", exc_info=True)
            return {}

    def put(self, chat_id: str, data: dict, ttl_minutes: int = 120) -> None:
        """
        Synchronous put - uses sync Redis client.
        Overwrites existing session data.
        """
        try:
            self.sync_redis.setex(
                self._key(chat_id),
                timedelta(minutes=ttl_minutes),
                json.dumps(data)
            )
            logger.debug(f"Saved session for {chat_id}: {data}")
        except Exception as exc:
            logger.error(f"Error putting session {chat_id}: {exc}", exc_info=True)


