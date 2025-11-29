"""
Circuit Breaker Pattern - Prevents cascading failures
Implements exponential backoff and automatic recovery
"""

import time
from enum import Enum
from typing import Callable, Any
from loguru import logger


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    Prevents cascading failures by failing fast when service is down.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"🔄 Circuit breaker [{self.name}] attempting recovery (HALF_OPEN)")
                self.state = CircuitState.HALF_OPEN
            else:
                time_remaining = int(self.recovery_timeout - (time.time() - self.last_failure_time))
                logger.warning(f"🚫 Circuit breaker [{self.name}] is OPEN - failing fast (retry in {time_remaining}s)")
                raise CircuitBreakerOpenError(
                    f"Circuit breaker [{self.name}] is OPEN. Service unavailable. Retry in {time_remaining}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"✅ Circuit breaker [{self.name}] recovered - state: CLOSED")
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"🔴 Circuit breaker [{self.name}] OPENED after {self.failure_count} failures - "
                f"will retry in {self.recovery_timeout}s"
            )
        else:
            logger.warning(
                f"⚠️ Circuit breaker [{self.name}] failure {self.failure_count}/{self.failure_threshold}"
            )


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Global circuit breakers for external services
_circuit_breakers = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create circuit breaker for a service"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, **kwargs)
    return _circuit_breakers[name]
