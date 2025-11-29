import httpx
import asyncio
import time
import re
from typing import Dict, Any
from loguru import logger

from ..config import get_settings


class WaSenderClient:
    """Client for WaSender WhatsApp API with comprehensive rate limit handling"""
    
    # Class-level rate limiting tracking (shared across all instances)
    _last_request_time = 0.0
    _min_interval_seconds = 0.5  # Minimum 500ms between requests
    _MAX_RETRIES = 3  # Maximum retry attempts for rate limits
    _request_count_window = []  # Track requests in sliding window
    _window_size_seconds = 60  # 1 minute window
    _max_requests_per_minute = 20  # Adjust based on WaSender limits
    
    # Circuit breaker pattern (Issue #41)
    _circuit_breaker_failures = []  # Track 429 failures
    _circuit_breaker_threshold = 10  # Open circuit after 10 failures in window
    _circuit_breaker_window = 60  # 1 minute window for failures
    _circuit_breaker_open_until = 0  # Timestamp when circuit can close
    _circuit_breaker_cooldown = 300  # 5 minutes cooldown when open
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = str(self.settings.wasender_base_url)
        self.api_token = self.settings.wasender_api_key.get_secret_value() if self.settings.wasender_api_key else None

    async def send_typing_indicator(self, phone_number: str) -> Dict[str, Any]:
        """
        Send typing indicator (simulate bot is typing)
        
        Args:
            phone_number: Phone number (with country code)
            
        Returns:
            API response
            
        Note: This endpoint appears to be unavailable in WaSender API (404).
              Function disabled until correct endpoint is confirmed.
              Main message flow works fine without typing indicator.
        """
        # DISABLED: Endpoint returns 404 - appears not to exist in WaSender API
        # The typing indicator is a nice-to-have UX feature, not critical.
        # Main message sending works fine without it.
        # 
        # To re-enable:
        # 1. Check WaSender documentation for correct endpoint
        # 2. Verify it's available in your plan/subscription
        # 3. Uncomment the code below
        # 4. Test with correct endpoint URL
        
        logger.debug(f"⏭️ Typing indicator disabled (endpoint unavailable)")
        return {}
        
        # Original code (disabled):
        # try:
        #     clean_phone = phone_number.replace('+', '').replace('@s.whatsapp.net', '')
        #     
        #     headers = {
        #         "Content-Type": "application/json"
        #     }
        #     
        #     if self.api_token:
        #         headers["Authorization"] = f"Bearer {self.api_token}"
        #     
        #     payload = {
        #         "sessionId": self.api_token,
        #         "to": clean_phone,
        #         "action": "typing"  # WaSender typing indicator
        #     }
        #     
        #     logger.debug(f"⌨️ Sending typing indicator to {clean_phone}")
        #     
        #     async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
        #         resp = await client.post(
        #             f"{self.base_url}/api/send-typing",
        #             json=payload,
        #             headers=headers
        #         )
        #         # Don't raise on error - typing indicator is optional
        #         if resp.status_code == 200:
        #             logger.debug(f"✅ Typing indicator sent to {clean_phone}")
        #         return resp.json() if resp.status_code == 200 else {}
        #         
        # except Exception as exc:
        #     # Silently fail - typing indicator is not critical
        #     logger.debug(f"Typing indicator failed (non-critical): {exc}")
        #     return {}
    
    def _check_circuit_breaker(self) -> tuple[bool, str]:
        """
        Check if circuit breaker is open (too many failures).
        
        Returns:
            (is_open, reason) tuple
        """
        current_time = time.time()
        
        # If circuit is open, check if cooldown period has passed
        if current_time < WaSenderClient._circuit_breaker_open_until:
            remaining = WaSenderClient._circuit_breaker_open_until - current_time
            return (True, f"Circuit breaker OPEN - {remaining:.0f}s remaining in cooldown")
        
        # Circuit closed or cooldown expired - check failure rate
        cutoff = current_time - self._circuit_breaker_window
        WaSenderClient._circuit_breaker_failures = [
            t for t in WaSenderClient._circuit_breaker_failures if t > cutoff
        ]
        
        failure_count = len(WaSenderClient._circuit_breaker_failures)
        if failure_count >= self._circuit_breaker_threshold:
            # Too many failures - OPEN circuit
            WaSenderClient._circuit_breaker_open_until = current_time + self._circuit_breaker_cooldown
            logger.error(
                f"🚨 CIRCUIT BREAKER OPENED: {failure_count} rate limit failures in {self._circuit_breaker_window}s. "
                f"Cooling down for {self._circuit_breaker_cooldown}s"
            )
            return (True, f"Circuit breaker OPENED due to {failure_count} failures")
        
        return (False, "")
    
    def _record_circuit_breaker_failure(self):
        """Record a rate limit failure for circuit breaker."""
        WaSenderClient._circuit_breaker_failures.append(time.time())
        logger.debug(f"Circuit breaker: {len(WaSenderClient._circuit_breaker_failures)} failures in window")
    
    async def _wait_for_rate_limit(self):
        """
        Proactive rate limiting - wait if sending too fast.
        
        Implements two-tier rate limiting:
        1. Minimum interval: 500ms between any two requests
        2. Sliding window: Max 20 requests per minute
        """
        current_time = time.time()
        
        # TIER 1: Minimum interval between requests (prevents bursts)
        time_since_last = current_time - WaSenderClient._last_request_time
        if time_since_last < self._min_interval_seconds:
            wait_time = self._min_interval_seconds - time_since_last
            logger.debug(f"⏱️ Rate limit (interval): waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            current_time = time.time()  # Update after wait
        
        # TIER 2: Sliding window rate limit (prevents sustained overload)
        # Remove requests older than window
        cutoff_time = current_time - self._window_size_seconds
        WaSenderClient._request_count_window = [
            t for t in WaSenderClient._request_count_window 
            if t > cutoff_time
        ]
        
        # Check if at capacity
        if len(WaSenderClient._request_count_window) >= self._max_requests_per_minute:
            # Calculate how long to wait for oldest request to age out
            oldest_request = min(WaSenderClient._request_count_window)
            wait_time = (oldest_request + self._window_size_seconds) - current_time
            
            if wait_time > 0:
                logger.warning(
                    f"⏱️ Rate limit (window): {len(WaSenderClient._request_count_window)}/{self._max_requests_per_minute} "
                    f"requests in last {self._window_size_seconds}s - waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
                current_time = time.time()
        
        # Record this request
        WaSenderClient._request_count_window.append(current_time)
        WaSenderClient._last_request_time = current_time
    
    async def send_message(self, phone_number: str, message: str, options: dict = None, retry_count: int = 0) -> Dict[str, Any]:
        """
        Send WhatsApp message via WaSender API with intelligent rate limiting
        
        Args:
            phone_number: Phone number (with country code)
            message: Message text to send
            options: Optional message options
            retry_count: Internal retry counter (DO NOT SET MANUALLY)
            
        Returns:
            API response dict
            
        Raises:
            httpx.HTTPStatusError: If request fails after all retries
        """
        # CIRCUIT BREAKER: Check if too many recent failures (Issue #41)
        is_open, reason = self._check_circuit_breaker()
        if is_open:
            logger.error(f"🚫 Cannot send message - {reason}")
            raise Exception(f"Circuit breaker open: {reason}")
        
        # CRITICAL: Enforce maximum retries to prevent infinite loops
        if retry_count >= self._MAX_RETRIES:
            logger.error(f"❌ MAX RETRIES EXCEEDED ({self._MAX_RETRIES}) - Giving up")
            raise Exception(f"Rate limit retry exhausted after {self._MAX_RETRIES} attempts")
        
        try:
            # PROACTIVE: Wait if sending too fast (prevent 429 before it happens)
            await self._wait_for_rate_limit()
            
            # Clean phone number (remove + and @s.whatsapp.net if present)
            clean_phone = phone_number.replace('+', '').replace('@s.whatsapp.net', '')
            
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            
            payload = {
                "sessionId": self.api_token,
                "to": clean_phone,
                "text": message
            }
            
            # SAFETY: Sanitize outgoing text to avoid leaking JSON/metadata (Issue #1/#2)
            def _sanitize_outgoing_text(text: str) -> str:
                if not isinstance(text, str):
                    return "عذرًا يا الغالي 🙏 صار عندي خلل بسيط الحين. ممكن تعيد رسالتك بعدين؟"
                trimmed = text.strip()
                looks_like_json = (
                    trimmed.startswith('{') or trimmed.startswith('[') or
                    trimmed.lower().startswith('```json') or
                    re.match(r"^\s*[\[{].*[\]}]\s*$", trimmed) is not None or
                    ('"intent"' in trimmed and '"confidence"' in trimmed)
                )
                if looks_like_json:
                    logger.error("🚫 Attempted to send JSON metadata to user. Replacing with safe fallback (Issue #1/#2).")
                    return "عذرًا يا الغالي 🙏 صار عندي خلل بسيط الحين. ممكن تعيد رسالتك بعدين؟"
                return text

            message = _sanitize_outgoing_text(message)

            # Log only on first attempt to avoid spam
            if retry_count == 0:
                logger.info(f"📤 Sending WhatsApp message to {clean_phone}")
                # CRITICAL: Log actual message content for debugging
                logger.info(f"📝 MESSAGE CONTENT: {message[:200]}{'...' if len(message) > 200 else ''}")
            else:
                logger.info(f"Retry {retry_count}/{self._MAX_RETRIES}: Sending to {clean_phone}")
            
            logger.debug(f"WaSender URL: {self.base_url}/api/send-message")
            logger.debug(f"Payload: {payload}")
            
            # Track WhatsApp API performance (Issue #41)
            import time
            api_start = time.time()
            
            # CRITICAL: Reduce timeout to prevent blocking (Issue: 2.93s response times)
            # 5 seconds max to avoid tying up workers
            async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
                try:
                    resp = await client.post(
                        f"{self.base_url}/api/send-message",
                        json=payload,
                        headers=headers
                    )
                    resp.raise_for_status()
                    
                    api_duration = time.time() - api_start
                    
                    # Success! Log with performance metrics
                    if retry_count > 0:
                        logger.info(f"✅ Retry successful! Message sent to {clean_phone} (API: {api_duration:.2f}s)")
                    else:
                        logger.info(f"✅ Message sent successfully to {clean_phone} (API: {api_duration:.2f}s)")
                    
                    # Warn if WhatsApp API is slow (> 1 second is concerning)
                    if api_duration > 2.0:
                        logger.warning(f"⚠️ WhatsApp API slow response: {api_duration:.2f}s - Consider contacting WaSender support")
                    elif api_duration > 1.0:
                        logger.warning(f"⚠️ WhatsApp API degraded: {api_duration:.2f}s (should be < 1s)")
                    
                    return resp.json()
                    
                except httpx.TimeoutException as timeout_exc:
                    api_duration = time.time() - api_start
                    logger.error(f"❌ WhatsApp API TIMEOUT ({api_duration:.1f}s) for {clean_phone} - API may be down")
                    
                    # Retry with exponential backoff
                    if retry_count < self._MAX_RETRIES - 1:  # Leave one retry for potential rate limit
                        import random
                        backoff = (2 ** retry_count) + random.uniform(0, 1)
                        logger.warning(f"⏳ Retry {retry_count + 1}/{self._MAX_RETRIES} after {backoff:.1f}s...")
                        await asyncio.sleep(backoff)
                        return await self.send_message(phone_number, message, options, retry_count + 1)
                    else:
                        logger.error(f"❌ MAX TIMEOUT RETRIES EXCEEDED - Giving up")
                        raise
                
        except httpx.HTTPStatusError as exc:
            # Handle 429 rate limiting with exponential backoff
            if exc.response.status_code == 429:
                # Record failure for circuit breaker
                self._record_circuit_breaker_failure()
                
                # Record for monitoring/alerting (Issue #41-8)
                try:
                    from app.monitoring.rate_limit_monitor import RateLimitMonitor
                    # This will be injected properly in production
                    # For now, just log the event
                    logger.info(f"📊 [MONITOR] Rate limit event to be recorded")
                except ImportError:
                    pass  # Monitoring not available yet
                
                # Parse rate limit headers (Issue #41-11)
                retry_after = 5  # Default
                rate_limit_type = "unknown"
                
                # Extract all rate limit headers for logging
                rate_limit_headers = {
                    'X-RateLimit-Limit': exc.response.headers.get('X-RateLimit-Limit'),
                    'X-RateLimit-Remaining': exc.response.headers.get('X-RateLimit-Remaining'),
                    'X-RateLimit-Reset': exc.response.headers.get('X-RateLimit-Reset'),
                    'Retry-After': exc.response.headers.get('Retry-After')
                }
                
                # Log all rate limit headers for debugging
                logger.warning(
                    f"📊 [429 Headers] "
                    f"Limit: {rate_limit_headers['X-RateLimit-Limit'] or 'N/A'}, "
                    f"Remaining: {rate_limit_headers['X-RateLimit-Remaining'] or 'N/A'}, "
                    f"Reset: {rate_limit_headers['X-RateLimit-Reset'] or 'N/A'}, "
                    f"Retry-After: {rate_limit_headers['Retry-After'] or 'N/A'}"
                )
                
                # Use Retry-After header if present (Issue #41-12)
                try:
                    retry_after = int(exc.response.headers.get('Retry-After', 5))
                except:
                    retry_after = 5
                
                try:
                    error_data = exc.response.json()
                    retry_after = error_data.get('retry_after', retry_after)
                    rate_limit_type = error_data.get('error_type', rate_limit_type)
                    limit_reason = error_data.get('reason', 'rate_limit_exceeded')
                except:
                    limit_reason = 'rate_limit_exceeded'
                
                # Analyze retry-after to determine limit type (Issue #41-C)
                if retry_after > 3600:  # > 1 hour
                    limit_type = "QUOTA_EXHAUSTED"  # Daily/monthly quota
                    should_retry = False
                    logger.error(f"🚫 API quota exhausted (retry-after: {retry_after}s) - NOT retrying")
                elif retry_after > 300:  # > 5 minutes
                    limit_type = "ACCOUNT_LIMIT"  # Account-level limit
                    should_retry = retry_count < 1  # Only retry once
                    logger.warning(f"⚠️ Account limit reached (retry-after: {retry_after}s) - Limited retries")
                else:  # < 5 minutes
                    limit_type = "RATE_SPIKE"  # Temporary spike
                    should_retry = True
                    logger.warning(f"⏱️ Rate spike detected (retry-after: {retry_after}s) - Will retry")
                
                if not should_retry:
                    logger.error(
                        f"❌ RATE LIMIT ({limit_type}): Not retrying. "
                        f"Reason: {limit_reason}, Retry-after: {retry_after}s"
                    )
                    raise Exception(f"Rate limit ({limit_type}): {limit_reason}")
                
                # Apply exponential backoff with jitter (Issue #41-12)
                # Prevents thundering herd when multiple requests retry simultaneously
                import random
                base_backoff = retry_after + (2 ** retry_count)
                jitter = random.uniform(0, min(base_backoff * 0.1, 2))  # 10% jitter, max 2s
                exponential_backoff = base_backoff + jitter
                
                logger.warning(
                    f"⚠️ RATE LIMITED ({limit_type}): Attempt {retry_count + 1}/{self._MAX_RETRIES} "
                    f"- Waiting {exponential_backoff:.1f}s before retry "
                    f"(base: {base_backoff}s + jitter: {jitter:.1f}s)"
                )
                
                await asyncio.sleep(exponential_backoff)
                
                # Recursive retry with incremented counter
                return await self.send_message(phone_number, message, options, retry_count + 1)
            
            # Non-429 errors - don't retry
            logger.error(f"❌ WaSender API error {exc.response.status_code}: {exc.response.text}")
            logger.error(f"Request URL: {exc.request.url}")
            raise
            
        except httpx.ConnectError as exc:
            logger.error(f"❌ WaSender connection error: {exc}")
            logger.error(f"Target URL: {self.base_url}/api/send-message")
            logger.error(f"This might be a DNS, firewall, or network issue")
            raise
            
        except Exception as exc:
            logger.error(f"❌ WaSender send error: {exc}")
            logger.error(f"Error type: {type(exc).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


