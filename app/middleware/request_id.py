"""
Request ID Middleware - Adds unique trace ID to every request
Enables end-to-end request tracing across all components
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from loguru import logger
import contextvars

# Context variable for request ID - accessible across async calls
request_id_var = contextvars.ContextVar('request_id', default=None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Adds unique request ID to every incoming request.
    Injects ID into logs and response headers for tracing.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        
        # Check if client sent X-Request-ID header (for request chaining)
        if 'x-request-id' in request.headers:
            request_id = request.headers['x-request-id']
        
        # Set in context var (accessible to all async tasks)
        request_id_var.set(request_id)
        
        # Add to request state
        request.state.request_id = request_id
        
        # Log middleware execution order (Issue #19)
        logger.debug(f"🔧 [4/5] RequestIDMiddleware executing [{request_id}]")
        
        # Get request size (Issue #21)
        content_length = request.headers.get("content-length", "0")
        try:
            size_bytes = int(content_length)
            size_kb = size_bytes / 1024
            size_str = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
        except (ValueError, TypeError):
            size_str = "unknown"
        
        # Log request start with ID and size (Issue #21)
        logger.bind(request_id=request_id).info(
            f"📨 REQUEST START [{request_id}] {request.method} {request.url.path} (size: {size_str})"
        )
        
        # Warn on large requests (Issue #21)
        if size_bytes > 1_000_000:  # > 1MB
            logger.warning(f"⚠️ LARGE REQUEST [{request_id}]: {size_str} - potential DoS risk")
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Log request end with ID
        logger.bind(request_id=request_id).info(
            f"✅ REQUEST END [{request_id}] Status: {response.status_code}"
        )
        
        return response


def get_request_id() -> str:
    """Get current request ID from context"""
    return request_id_var.get() or "NO-ID"
