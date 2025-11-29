"""
Metrics Middleware
==================
Tracks request metrics including concurrency, response times, and resource usage.
"""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from loguru import logger


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking request-level metrics.
    
    Issues addressed:
    - #32: Concurrent request tracking
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Track concurrent requests (Issue #32)
        from ..utils.metrics import get_metrics
        metrics = get_metrics()
        
        # Start request tracking
        metrics.start_request(request_id)
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record successful completion
            duration = metrics.end_request(request_id)
            
            # Log request completion
            logger.info(
                f"📊 REQUEST COMPLETE: {request.method} {request.url.path} "
                f"[{response.status_code}] {duration*1000:.0f}ms "
                f"(concurrent: {metrics.concurrent_requests})"
            )
            
            return response
            
        except Exception as exc:
            # Record failed request
            duration = metrics.end_request(request_id)
            
            logger.error(
                f"📊 REQUEST FAILED: {request.method} {request.url.path} "
                f"{duration*1000:.0f}ms - {exc}"
            )
            raise
