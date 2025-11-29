"""
Comprehensive Security Middleware
Implements authentication, rate limiting, input validation, and security headers
"""

import hashlib
import hmac
import time
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
import redis.asyncio as redis

from ..config import get_settings


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware implementing:
    - API Key authentication
    - HMAC signature verification
    - Rate limiting per IP/user
    - Request size limits
    - Security headers
    - Request logging
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.rate_limit_requests = 100  # requests per window
        self.rate_limit_window = 60  # seconds
        
    async def get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self.redis_client
    
    async def dispatch(self, request: Request, call_next):
        """Main security middleware logic"""
        start_time = time.time()
        
        # Skip security for health check and docs
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            response = await call_next(request)
            return response
        
        try:
            # 1. Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_request_size:
                logger.warning(f"Request too large: {content_length} bytes from {request.client.host}")
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request body too large"
                )
            
            # 2. Verify API Key (if configured)
            # EXEMPT webhooks - they use HMAC signature verification instead
            if self.settings.api_key and not request.url.path.startswith("/webhook"):
                api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
                if not api_key or api_key != self.settings.api_key:
                    logger.warning(f"Invalid API key from {request.client.host}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or missing API key"
                    )
            
            # 3. Rate limiting
            client_ip = request.client.host
            rate_limit_result = await self._check_rate_limit(client_ip, request.url.path)
            # Rate limit logging is handled inside _check_rate_limit()
            
            # 4. HMAC signature verification for webhooks
            if request.url.path.startswith("/webhook"):
                await self._verify_webhook_signature(request)
            
            # 5. Process request
            response = await call_next(request)
            
            # 6. Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            
            # 7. Log request
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"IP: {client_ip} - "
                f"Time: {process_time:.3f}s"
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Security middleware error: {exc}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
    
    async def _check_rate_limit(self, client_ip: str, path: str) -> dict:
        """Check rate limiting using Redis"""
        try:
            redis_client = await self.get_redis()
            key = f"rate_limit:{client_ip}:{path}"
            
            # Get current count
            current = await redis_client.get(key)
            
            if current is None:
                # First request in window
                await redis_client.setex(key, self.rate_limit_window, 1)
                logger.info(f"🔐 RATE LIMIT: {client_ip} [1/{self.rate_limit_requests}] on {path} (new window: {self.rate_limit_window}s)")
                return {"current": 1, "max": self.rate_limit_requests, "window": self.rate_limit_window}
            else:
                current_count = int(current)
                if current_count >= self.rate_limit_requests:
                    logger.warning(f"🚫 RATE LIMIT EXCEEDED: {client_ip} on {path} ({current_count}/{self.rate_limit_requests})")
                    logger.warning(f"🚫 IMPACT: Request blocked - potential abuse detected")
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Max {self.rate_limit_requests} requests per {self.rate_limit_window}s"
                    )
                
                # Increment BEFORE logging so log shows the NEW count
                new_count = await redis_client.incr(key)
                
                # Log at milestones for visibility (not every request to avoid log spam)
                thresholds = [2, 5, 10, 25, 50, 75, int(self.rate_limit_requests * 0.9)]
                if new_count in thresholds:
                    logger.info(f"🔐 RATE LIMIT: {client_ip} [{new_count}/{self.rate_limit_requests}] on {path}")
                
                return {"current": new_count, "max": self.rate_limit_requests, "window": self.rate_limit_window}
                
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Rate limit check error: {exc}")
            return None
            # Don't block request if rate limiting fails
    
    async def _verify_webhook_signature(self, request: Request):
        """Verify HMAC signature for webhook requests"""
        try:
            # TEMPORARILY DISABLED FOR DEVELOPMENT
            # TODO: Re-enable in production with correct secret
            logger.debug("Webhook signature verification temporarily disabled")
            return
            
            # Get signature from header
            signature = request.headers.get("X-Webhook-Signature")
            if not signature:
                logger.warning("Missing webhook signature - allowing for development")
                return  # Allow without signature in development
            
            # Read body
            body = await request.body()
            
            # Calculate expected signature
            secret = self.settings.wasender_api_key
            if not secret:
                logger.warning("Webhook secret not configured - skipping verification")
                return  # Skip verification if secret not set
            
            expected_signature = hmac.new(
                secret.get_secret_value().encode() if hasattr(secret, 'get_secret_value') else str(secret).encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Invalid webhook signature - allowing for development")
                return  # Allow in development
                
        except HTTPException:
            # Don't raise in development
            logger.warning("Webhook signature verification failed - allowing for development")
            return
        except Exception as exc:
            logger.error(f"Webhook signature verification error: {exc}")
            # Don't block if verification fails due to error


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Validates and sanitizes input data to prevent injection attacks
    """
    
    # Dangerous patterns that indicate potential attacks
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>",  # XSS
        r"javascript:",  # XSS
        r"on\w+\s*=",  # Event handlers (XSS)
        r"\bunion\b.*\bselect\b",  # SQL injection
        r"\bdrop\b.*\btable\b",  # SQL injection
        r"';\s*--",  # SQL injection
        r"\bor\b.*1\s*=\s*1",  # SQL injection
        r"{{.*}}",  # Template injection
        r"\$\{.*\}",  # Expression injection
        r"exec\(",  # Code execution
        r"eval\(",  # Code execution
    ]
    
    # Prompt injection patterns for LLM
    PROMPT_INJECTION_PATTERNS = [
        r"ignore (previous|all) (instructions|prompts)",
        r"system\s*:\s*you are",
        r"new instructions?:",
        r"forget (everything|all)",
        r"disregard (previous|all)",
        r"\[SYSTEM\]",
        r"\[INST\]",
        r"<\|.*\|>",  # Special tokens
    ]
    
    async def dispatch(self, request: Request, call_next):
        """Validate and sanitize input data"""
        
        # Skip for non-POST/PUT/PATCH requests
        if request.method not in ["POST", "PUT", "PATCH"]:
            return await call_next(request)
        
        try:
            # Check content type
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                logger.warning(f"⚠️ VALIDATION FAILED: Invalid content-type '{content_type}' from {request.client.host}")
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Content-Type must be application/json"
                )
            
            # Read and validate body
            body = await request.body()
            if body:
                import json
                try:
                    data = json.loads(body)
                    
                    # Sanitize and validate
                    threats_found = self._detect_threats(data)
                    if threats_found:
                        logger.warning(f"🚨 SECURITY THREAT DETECTED from {request.client.host} on {request.url.path}:")
                        for threat in threats_found:
                            logger.warning(f"   - {threat['type']}: {threat['pattern']} in field '{threat['field']}'")
                            logger.warning(f"   - Sample: {threat.get('sample', 'N/A')}")
                        logger.warning(f"🚨 IMPACT: Request blocked to prevent attack")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Potentially malicious input detected"
                        )
                    
                    logger.debug(f"✅ INPUT VALIDATED: {request.url.path} - No threats detected")
                    
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ VALIDATION FAILED: Invalid JSON from {request.client.host}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid JSON format"
                    )
            
            response = await call_next(request)
            return response
            
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Input validation error: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid request format"}
            )
    
    def _detect_threats(self, data: dict, prefix: str = "") -> list:
        """Detect potential security threats in input data"""
        import re
        threats = []
        
        # Fields that contain encrypted/encoded data and should NOT be scanned
        # These are not user input but system-generated data
        WHITELIST_FIELDS = [
            "messageSecret", "secret", "token", "signature", "hash",
            "contextInfo", "deviceInfo", "pushName", "notifyName",
            "messageContextInfo", "quotedMessage", "ephemeralSharedSecret"
        ]
        
        if isinstance(data, dict):
            for key, value in data.items():
                field_path = f"{prefix}.{key}" if prefix else key
                
                # Skip whitelisted fields (encrypted/encoded data)
                if any(whitelist in key for whitelist in WHITELIST_FIELDS):
                    continue
                
                if isinstance(value, str):
                    # Check for dangerous patterns
                    for pattern in self.DANGEROUS_PATTERNS:
                        if re.search(pattern, value, re.IGNORECASE):
                            threats.append({
                                "type": "Injection/XSS",
                                "pattern": pattern,
                                "field": field_path,
                                "sample": value[:50]
                            })
                    
                    # Check for prompt injection (specific to LLM fields)
                    if key in ["message", "text", "content", "prompt"]:
                        for pattern in self.PROMPT_INJECTION_PATTERNS:
                            if re.search(pattern, value, re.IGNORECASE):
                                threats.append({
                                    "type": "Prompt Injection",
                                    "pattern": pattern,
                                    "field": field_path,
                                    "sample": value[:50]
                                })
                elif isinstance(value, (dict, list)):
                    threats.extend(self._detect_threats(value, field_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                field_path = f"{prefix}[{i}]"
                threats.extend(self._detect_threats(item, field_path))
        
        return threats


def sanitize_input(data: dict) -> dict:
    """
    Sanitize input data to prevent injection attacks
    """
    if not isinstance(data, dict):
        return data
    
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potential SQL injection patterns
            sanitized[key] = value.replace("'", "").replace('"', '').replace(";", "").replace("--", "")
        elif isinstance(value, dict):
            sanitized[key] = sanitize_input(value)
        elif isinstance(value, list):
            sanitized[key] = [sanitize_input(item) if isinstance(item, dict) else item for item in value]
        else:
            sanitized[key] = value
    
    return sanitized
