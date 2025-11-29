"""
Background worker process for message queue.

Run this as a separate service:
    python -m app.queue.worker

Or with PM2:
    pm2 start app/queue/worker.py --name message-worker --interpreter python

Or with systemd:
    systemctl start message-worker
"""
import asyncio
import signal
from loguru import logger
import redis.asyncio as redis

from app.config import get_settings
from app.api.wasender_client import WaSenderClient
from app.queue.message_queue import MessageQueue, QueueWorker


async def main():
    """Main worker entry point."""
    settings = get_settings()
    
    # Initialize Redis
    redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    
    # Initialize queue and worker
    message_queue = MessageQueue(redis_client)
    wasender_client = WaSenderClient()
    
    worker = QueueWorker(
        message_queue=message_queue,
        wasender_client=wasender_client,
        max_concurrent=5  # 5 concurrent workers
    )
    
    # Graceful shutdown handler
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum} - stopping worker...")
        asyncio.create_task(worker.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start worker
    logger.info("=" * 70)
    logger.info("🚀 MESSAGE QUEUE WORKER STARTING")
    logger.info("=" * 70)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt - stopping...")
    finally:
        await worker.stop()
        await redis_client.close()
        logger.info("Worker stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
