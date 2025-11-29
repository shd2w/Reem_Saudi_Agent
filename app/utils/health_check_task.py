"""
Periodic Health Check Task
===========================
Background task that periodically checks system health and logs status.
"""

import asyncio
from loguru import logger
from ..memory.session_manager import SessionManager
from ..api.agent_api import AgentApiClient


class HealthCheckTask:
    """
    Periodic health check for runtime monitoring.
    Checks Redis, Agent API, and system components.
    """
    
    def __init__(self, interval_seconds: int = 60):
        """
        Args:
            interval_seconds: Time between health checks (default: 60s)
        """
        self.interval = interval_seconds
        self.running = False
        self.task = None
    
    async def check_redis(self) -> dict:
        """Check Redis connectivity"""
        try:
            session_mgr = SessionManager()
            test_key = f"health_check_{asyncio.get_event_loop().time()}"
            
            # Write test
            await session_mgr.redis.set(test_key, "ok", ex=5)
            # Read test
            result = await session_mgr.redis.get(test_key)
            # Cleanup
            await session_mgr.redis.delete(test_key)
            
            if result == "ok":
                return {"status": "healthy", "latency_ms": 0}
            else:
                return {"status": "unhealthy", "error": "read/write mismatch"}
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}
    
    async def check_agent_api(self) -> dict:
        """Check Agent API connectivity"""
        try:
            api_client = AgentApiClient()
            # Light health check - just verify token
            token = await api_client.get_jwt()
            
            if token:
                return {"status": "healthy"}
            else:
                return {"status": "degraded", "error": "no token"}
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}
    
    async def run_health_check(self):
        """Run comprehensive health check"""
        logger.info("=" * 60)
        logger.info("🏥 PERIODIC HEALTH CHECK")
        
        # Check Redis
        redis_health = await self.check_redis()
        if redis_health["status"] == "healthy":
            logger.info("✅ Redis: HEALTHY")
        else:
            logger.error(f"🔴 Redis: {redis_health['status'].upper()} - {redis_health.get('error', 'unknown')}")
        
        # Check Agent API
        api_health = await self.check_agent_api()
        if api_health["status"] == "healthy":
            logger.info("✅ Agent API: HEALTHY")
        else:
            logger.error(f"🔴 Agent API: {api_health['status'].upper()} - {api_health.get('error', 'unknown')}")
        
        # Performance metrics (Issues #28-32)
        try:
            from .metrics import get_metrics
            metrics = get_metrics()
            
            # Record current usage
            metrics.record_memory_usage()
            metrics.record_cpu_usage()
            
            # Log metrics summary
            metrics.log_metrics_summary()
        except Exception as exc:
            logger.error(f"Failed to collect performance metrics: {exc}")
        
        # Overall status
        all_healthy = (
            redis_health["status"] == "healthy" and
            api_health["status"] == "healthy"
        )
        
        if all_healthy:
            logger.info("✅ OVERALL SYSTEM: HEALTHY")
        else:
            logger.warning("⚠️ OVERALL SYSTEM: DEGRADED")
        
        logger.info("=" * 60)
    
    async def _health_check_loop(self):
        """Background loop for periodic health checks"""
        logger.info(f"🏥 Health check task started (interval: {self.interval}s)")
        
        while self.running:
            try:
                await self.run_health_check()
            except Exception as exc:
                logger.error(f"Health check error: {exc}", exc_info=True)
            
            # Wait for next check
            await asyncio.sleep(self.interval)
        
        logger.info("🏥 Health check task stopped")
    
    def start(self):
        """Start periodic health checks"""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._health_check_loop())
            logger.info(f"🏥 Started periodic health checks (every {self.interval}s)")
    
    async def stop(self):
        """Stop periodic health checks"""
        if self.running:
            self.running = False
            if self.task:
                await self.task
            logger.info("🏥 Stopped periodic health checks")


# Global instance
_health_check_task = None

def get_health_check_task(interval_seconds: int = 60) -> HealthCheckTask:
    """Get or create global health check task"""
    global _health_check_task
    if _health_check_task is None:
        _health_check_task = HealthCheckTask(interval_seconds)
    return _health_check_task
