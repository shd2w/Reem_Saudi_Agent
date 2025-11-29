"""
Message Queue for Asynchronous WhatsApp Message Sending

Implements queue-based architecture to decouple webhook processing
from message sending, preventing webhook timeouts.

Architecture:
  Webhook → Add to queue → Return 200 OK (fast!)
  Background worker → Process queue → Send messages (slow, but doesn't block)
"""
import asyncio
import json
import time
from typing import Dict, Any, Optional
from loguru import logger
import redis.asyncio as redis


class MessageQueue:
    """
    Redis-based message queue for async WhatsApp sending.
    
    Features:
    - Instant webhook response (add to queue = fast)
    - Background processing (send = slow, but doesn't block)
    - Rate limiting at queue level
    - Retry logic for failed messages
    - Priority queue support
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.queue_key = "message_queue:pending"
        self.processing_key = "message_queue:processing"
        self.failed_key = "message_queue:failed"
        self.dlq_key = "message_queue:dlq"  # Dead letter queue
    
    async def enqueue(
        self, 
        phone_number: str, 
        message: str, 
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add message to queue (FAST - returns immediately).
        
        Args:
            phone_number: Recipient phone
            message: Message text
            priority: "high", "normal", "low"
            metadata: Optional metadata (session_id, etc.)
        
        Returns:
            message_id: Unique ID for tracking
        """
        message_id = f"msg_{int(time.time() * 1000)}_{phone_number}"
        
        queue_item = {
            "message_id": message_id,
            "phone_number": phone_number,
            "message": message,
            "priority": priority,
            "metadata": metadata or {},
            "enqueued_at": time.time(),
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Add to appropriate queue based on priority
        if priority == "high":
            queue_key = f"{self.queue_key}:high"
            score = time.time() - 1000  # Process ASAP
        elif priority == "low":
            queue_key = f"{self.queue_key}:low"
            score = time.time() + 300  # Delay 5 minutes
        else:  # normal
            queue_key = self.queue_key
            score = time.time()
        
        # Add to Redis sorted set (score = timestamp)
        await self.redis.zadd(
            queue_key,
            {json.dumps(queue_item): score}
        )
        
        logger.info(
            f"📥 Message queued: {message_id} "
            f"(priority: {priority}, phone: {phone_number})"
        )
        
        return message_id
    
    async def dequeue(self, batch_size: int = 1) -> list[Dict[str, Any]]:
        """
        Get next message(s) from queue.
        
        Args:
            batch_size: Number of messages to fetch
        
        Returns:
            List of message items
        """
        messages = []
        
        # Check priority queues first
        for queue_suffix in [":high", "", ":low"]:
            queue_key = f"{self.queue_key}{queue_suffix}"
            
            # Get messages whose score <= now (due for processing)
            now = time.time()
            results = await self.redis.zrangebyscore(
                queue_key,
                min=0,
                max=now,
                start=0,
                num=batch_size - len(messages)
            )
            
            for item_json in results:
                item = json.loads(item_json)
                messages.append(item)
                
                # Move to processing queue
                await self.redis.zadd(
                    self.processing_key,
                    {item_json: time.time()}
                )
                
                # Remove from pending queue
                await self.redis.zrem(queue_key, item_json)
            
            if len(messages) >= batch_size:
                break
        
        return messages
    
    async def mark_success(self, message_id: str):
        """Mark message as successfully sent."""
        # Remove from processing queue
        items = await self.redis.zrange(self.processing_key, 0, -1)
        for item_json in items:
            item = json.loads(item_json)
            if item["message_id"] == message_id:
                await self.redis.zrem(self.processing_key, item_json)
                logger.info(f"✅ Message sent successfully: {message_id}")
                break
    
    async def mark_failed(self, message_id: str, error: str):
        """
        Mark message as failed and retry or move to DLQ.
        
        Args:
            message_id: Message ID
            error: Error description
        """
        items = await self.redis.zrange(self.processing_key, 0, -1)
        
        for item_json in items:
            item = json.loads(item_json)
            if item["message_id"] == message_id:
                item["retry_count"] += 1
                item["last_error"] = error
                item["last_attempt"] = time.time()
                
                # Remove from processing
                await self.redis.zrem(self.processing_key, item_json)
                
                if item["retry_count"] >= item["max_retries"]:
                    # Max retries exceeded - move to DLQ
                    await self.redis.zadd(
                        self.dlq_key,
                        {json.dumps(item): time.time()}
                    )
                    logger.error(
                        f"❌ Message failed permanently: {message_id} "
                        f"(retries: {item['retry_count']}) - moved to DLQ"
                    )
                else:
                    # Retry with exponential backoff
                    retry_delay = 2 ** item["retry_count"] * 60  # 2m, 4m, 8m
                    retry_at = time.time() + retry_delay
                    
                    await self.redis.zadd(
                        self.failed_key,
                        {json.dumps(item): retry_at}
                    )
                    logger.warning(
                        f"⚠️ Message failed: {message_id} "
                        f"(attempt {item['retry_count']}/{item['max_retries']}) "
                        f"- retry in {retry_delay}s"
                    )
                break
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {
            "pending_high": await self.redis.zcard(f"{self.queue_key}:high"),
            "pending_normal": await self.redis.zcard(self.queue_key),
            "pending_low": await self.redis.zcard(f"{self.queue_key}:low"),
            "processing": await self.redis.zcard(self.processing_key),
            "failed": await self.redis.zcard(self.failed_key),
            "dlq": await self.redis.zcard(self.dlq_key)
        }
        stats["pending_total"] = (
            stats["pending_high"] + 
            stats["pending_normal"] + 
            stats["pending_low"]
        )
        return stats


class QueueWorker:
    """
    Background worker that processes message queue.
    
    Run this as a separate process/service:
        python -m app.queue.worker
    """
    
    def __init__(
        self, 
        message_queue: MessageQueue,
        wasender_client,
        max_concurrent: int = 5
    ):
        self.queue = message_queue
        self.wasender_client = wasender_client
        self.max_concurrent = max_concurrent
        self.running = False
    
    async def start(self):
        """Start processing queue."""
        self.running = True
        logger.info(f"🚀 Queue worker started (max_concurrent: {self.max_concurrent})")
        
        tasks = []
        for i in range(self.max_concurrent):
            task = asyncio.create_task(self._worker_loop(i))
            tasks.append(task)
        
        # Wait for all workers
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop processing queue."""
        self.running = False
        logger.info("🛑 Queue worker stopping...")
    
    async def _worker_loop(self, worker_id: int):
        """Worker loop - processes messages continuously."""
        logger.info(f"👷 Worker {worker_id} started")
        
        while self.running:
            try:
                # Get next message from queue
                messages = await self.queue.dequeue(batch_size=1)
                
                if not messages:
                    # No messages - wait and retry
                    await asyncio.sleep(1)
                    continue
                
                message_item = messages[0]
                message_id = message_item["message_id"]
                phone_number = message_item["phone_number"]
                message_text = message_item["message"]
                retry_count = message_item["retry_count"]
                
                logger.info(
                    f"📤 Worker {worker_id} processing: {message_id} "
                    f"(phone: {phone_number}, retry: {retry_count})"
                )
                
                # Send message via WaSender
                try:
                    await self.wasender_client.send_message(
                        phone_number=phone_number,
                        message=message_text
                    )
                    
                    # Success!
                    await self.queue.mark_success(message_id)
                    logger.info(f"✅ Worker {worker_id} sent: {message_id}")
                    
                except Exception as exc:
                    # Failed - retry logic handled by queue
                    error_msg = str(exc)
                    await self.queue.mark_failed(message_id, error_msg)
                    logger.error(f"❌ Worker {worker_id} failed: {message_id} - {error_msg}")
                
            except Exception as exc:
                logger.error(f"❌ Worker {worker_id} error: {exc}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error
        
        logger.info(f"👷 Worker {worker_id} stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check worker health."""
        stats = await self.queue.get_queue_stats()
        
        return {
            "status": "healthy" if self.running else "stopped",
            "workers": self.max_concurrent,
            "queue_stats": stats,
            "timestamp": time.time()
        }
