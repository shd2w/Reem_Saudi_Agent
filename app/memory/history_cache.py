"""
Redis-based Conversation History Cache
Efficient storage and retrieval of conversation history with automatic truncation
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger


class HistoryCache:
    """
    Manages conversation history in Redis with efficient caching.
    Stores history in a Redis list for O(1) append and automatic trimming.
    """
    
    def __init__(self, redis_client):
        """
        Initialize history cache with Redis client.
        
        Args:
            redis_client: Async Redis client from session manager
        """
        self.redis = redis_client
        self.max_messages = 60  # Keep last 60 messages (30 exchanges)
        self.ttl_seconds = 7200  # 2 hours TTL
    
    def _history_key(self, session_id: str) -> str:
        """Generate Redis key for conversation history"""
        return f"history:{session_id}"
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (intent, confidence, etc.)
        """
        try:
            key = self._history_key(session_id)
            
            # Build message object
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                message["metadata"] = metadata
            
            # Add to Redis list (RPUSH = append to end)
            await self.redis.rpush(key, json.dumps(message, ensure_ascii=False))
            
            # Trim to keep only last N messages
            await self.redis.ltrim(key, -self.max_messages, -1)
            
            # Set TTL to auto-expire old conversations
            await self.redis.expire(key, self.ttl_seconds)
            
            logger.debug(f"📝 Added {role} message to history: {session_id[:30]}...")
            
        except Exception as e:
            logger.error(f"❌ Failed to add message to history cache: {e}")
    
    async def get_history(
        self, 
        session_id: str, 
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            last_n: Get only last N messages (default: all)
        
        Returns:
            List of message dicts with role, content, timestamp
        """
        try:
            key = self._history_key(session_id)
            
            # Get from Redis list
            if last_n:
                # Get last N messages
                raw_messages = await self.redis.lrange(key, -last_n, -1)
            else:
                # Get all messages
                raw_messages = await self.redis.lrange(key, 0, -1)
            
            # Parse JSON messages
            history = []
            for raw in raw_messages:
                try:
                    message = json.loads(raw)
                    history.append(message)
                except json.JSONDecodeError as e:
                    logger.error(f"⚠️ Failed to parse history message: {e}")
                    continue
            
            logger.debug(f"📖 Retrieved {len(history)} messages from history: {session_id[:30]}...")
            return history
            
        except Exception as e:
            logger.error(f"❌ Failed to get history from cache: {e}")
            return []
    
    async def get_recent_context(
        self, 
        session_id: str, 
        last_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history in full format (includes tool_calls, metadata, etc).
        
        Args:
            session_id: Session identifier
            last_n: Number of recent messages to retrieve
        
        Returns:
            List of message dicts with ALL fields preserved (role, content, tool_calls, etc)
        """
        history = await self.get_history(session_id, last_n)
        
        # Return full messages with ALL fields (tool_calls, tool_call_id, metadata)
        # This is critical for intelligent agent to detect previous tool calls!
        return history
    
    async def clear_history(self, session_id: str) -> None:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        try:
            key = self._history_key(session_id)
            await self.redis.delete(key)
            logger.info(f"🗑️ Cleared history for session: {session_id[:30]}...")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear history: {e}")
    
    async def get_last_message(
        self, 
        session_id: str, 
        role: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the last message from history.
        
        Args:
            session_id: Session identifier
            role: Filter by role ('user' or 'assistant')
        
        Returns:
            Last message dict or None
        """
        history = await self.get_history(session_id)
        
        if not history:
            return None
        
        if role:
            # Find last message with specified role
            for msg in reversed(history):
                if msg.get("role") == role:
                    return msg
            return None
        else:
            # Return last message
            return history[-1]
    
    async def get_message_count(self, session_id: str) -> int:
        """
        Get total number of messages in history.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Number of messages
        """
        try:
            key = self._history_key(session_id)
            count = await self.redis.llen(key)
            return count
        except Exception as e:
            logger.error(f"❌ Failed to get message count: {e}")
            return 0
    
    async def batch_add_messages(
        self, 
        session_id: str, 
        messages: List[Dict[str, Any]]
    ) -> None:
        """
        Add multiple messages at once (for migration/bulk operations).
        
        Args:
            session_id: Session identifier
            messages: List of message dicts with 'role' and 'content'
        """
        try:
            key = self._history_key(session_id)
            
            # Convert to JSON strings
            json_messages = [
                json.dumps(msg, ensure_ascii=False) 
                for msg in messages
            ]
            
            if json_messages:
                # Add all at once
                await self.redis.rpush(key, *json_messages)
                
                # Trim to keep only last N messages
                await self.redis.ltrim(key, -self.max_messages, -1)
                
                # Set TTL
                await self.redis.expire(key, self.ttl_seconds)
                
                logger.info(f"📝 Batch added {len(messages)} messages to history: {session_id[:30]}...")
            
        except Exception as e:
            logger.error(f"❌ Failed to batch add messages: {e}")


# Singleton instance
_history_cache = None


def get_history_cache(redis_client) -> HistoryCache:
    """Get or create singleton history cache instance"""
    global _history_cache
    if _history_cache is None:
        _history_cache = HistoryCache(redis_client)
        logger.info("✅ HistoryCache initialized")
    return _history_cache
