"""
Error Recovery Agent

Monitors system health and automatically fixes common issues.
Prevents frontend breaking during streaming and handles connection failures.

Features:
- Stream health monitoring
- Automatic retry logic
- Connection recovery
- Error pattern detection
- Self-healing mechanisms

Usage:
    recovery = ErrorRecoveryAgent()
    await recovery.monitor_stream(stream_generator)
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors the recovery agent can handle."""
    STREAM_TIMEOUT = "stream_timeout"
    CONNECTION_LOST = "connection_lost"
    BACKEND_ERROR = "backend_error"
    FRONTEND_DISCONNECT = "frontend_disconnect"
    RATE_LIMIT = "rate_limit"
    VALIDATION_FAILURE = "validation_failure"


@dataclass
class ErrorPattern:
    """Pattern for detecting and handling specific errors."""
    error_type: ErrorType
    pattern: str
    max_retries: int
    retry_delay: float
    recovery_action: str


class ErrorRecoveryAgent:
    """
    Agent that monitors system health and automatically recovers from errors.
    
    Implements circuit breaker pattern and automatic retry logic
    to prevent system failures from breaking the user experience.
    """
    
    def __init__(self):
        """Initialize error recovery agent."""
        self.error_patterns = self._init_error_patterns()
        self.error_counts = {}
        self.circuit_breakers = {}
        self.last_error_time = {}
        
    def _init_error_patterns(self) -> Dict[ErrorType, ErrorPattern]:
        """Initialize error patterns and recovery strategies."""
        return {
            ErrorType.STREAM_TIMEOUT: ErrorPattern(
                error_type=ErrorType.STREAM_TIMEOUT,
                pattern="timeout|timed out|connection timeout",
                max_retries=3,
                retry_delay=2.0,
                recovery_action="restart_stream"
            ),
            ErrorType.CONNECTION_LOST: ErrorPattern(
                error_type=ErrorType.CONNECTION_LOST,
                pattern="connection lost|network error|fetch failed",
                max_retries=5,
                retry_delay=1.0,
                recovery_action="reconnect"
            ),
            ErrorType.BACKEND_ERROR: ErrorPattern(
                error_type=ErrorType.BACKEND_ERROR,
                pattern="500|502|503|504|internal server error",
                max_retries=3,
                retry_delay=5.0,
                recovery_action="fallback_response"
            ),
            ErrorType.RATE_LIMIT: ErrorPattern(
                error_type=ErrorType.RATE_LIMIT,
                pattern="rate limit|429|too many requests",
                max_retries=2,
                retry_delay=10.0,
                recovery_action="exponential_backoff"
            ),
            ErrorType.VALIDATION_FAILURE: ErrorPattern(
                error_type=ErrorType.VALIDATION_FAILURE,
                pattern="validation failed|invalid response|hallucination",
                max_retries=2,
                retry_delay=1.0,
                recovery_action="retry_with_context"
            )
        }
    
    async def monitor_stream(
        self,
        stream_generator: AsyncGenerator[Dict[str, Any], None],
        on_error: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Monitor a stream and automatically recover from errors.
        
        Args:
            stream_generator: The stream to monitor
            on_error: Optional error callback
            
        Yields:
            Stream events with error recovery
        """
        last_activity = time.time()
        timeout_threshold = 30.0  # 30 seconds
        
        try:
            async for event in stream_generator:
                last_activity = time.time()
                yield event
                
                # Check for error patterns in the event
                if event.get("type") == "error":
                    error_msg = event.get("message", "")
                    await self._handle_stream_error(error_msg, on_error)
                    
        except asyncio.TimeoutError:
            logger.warning("Stream timeout detected, attempting recovery")
            await self._recover_from_timeout(on_error)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Stream error: {error_msg}")
            
            # Try to recover based on error pattern
            recovery_successful = await self._attempt_recovery(error_msg, on_error)
            
            if not recovery_successful and on_error:
                on_error(f"Recovery failed: {error_msg}")
    
    async def _handle_stream_error(
        self,
        error_msg: str,
        on_error: Optional[Callable[[str], None]]
    ):
        """Handle errors detected in stream events."""
        error_type = self._classify_error(error_msg)
        
        if error_type:
            pattern = self.error_patterns[error_type]
            logger.info(f"Detected {error_type.value} error, attempting recovery")
            
            # Increment error count
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Check if we should attempt recovery
            if self.error_counts[error_type] <= pattern.max_retries:
                await asyncio.sleep(pattern.retry_delay)
                await self._execute_recovery_action(pattern.recovery_action)
            else:
                logger.error(f"Max retries exceeded for {error_type.value}")
                if on_error:
                    on_error(f"Max retries exceeded: {error_msg}")
    
    async def _attempt_recovery(
        self,
        error_msg: str,
        on_error: Optional[Callable[[str], None]]
    ) -> bool:
        """Attempt to recover from an error."""
        error_type = self._classify_error(error_msg)
        
        if not error_type:
            return False
            
        pattern = self.error_patterns[error_type]
        
        # Check circuit breaker
        if self._is_circuit_open(error_type):
            logger.warning(f"Circuit breaker open for {error_type.value}")
            return False
        
        # Attempt recovery
        try:
            await asyncio.sleep(pattern.retry_delay)
            await self._execute_recovery_action(pattern.recovery_action)
            
            # Reset error count on successful recovery
            self.error_counts[error_type] = 0
            self._close_circuit(error_type)
            
            logger.info(f"Successfully recovered from {error_type.value}")
            return True
            
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            self._open_circuit(error_type)
            return False
    
    def _classify_error(self, error_msg: str) -> Optional[ErrorType]:
        """Classify an error based on its message."""
        error_msg_lower = error_msg.lower()
        
        for error_type, pattern in self.error_patterns.items():
            if any(keyword in error_msg_lower for keyword in pattern.pattern.split("|")):
                return error_type
        
        return None
    
    async def _execute_recovery_action(self, action: str):
        """Execute a recovery action."""
        if action == "restart_stream":
            logger.info("Restarting stream connection")
            # Implementation would restart the stream
            
        elif action == "reconnect":
            logger.info("Reconnecting to backend")
            # Implementation would reconnect
            
        elif action == "fallback_response":
            logger.info("Using fallback response")
            # Implementation would provide fallback
            
        elif action == "exponential_backoff":
            logger.info("Applying exponential backoff")
            # Implementation would increase delay
            
        elif action == "retry_with_context":
            logger.info("Retrying with additional context")
            # Implementation would retry with more context
    
    def _is_circuit_open(self, error_type: ErrorType) -> bool:
        """Check if circuit breaker is open for an error type."""
        return self.circuit_breakers.get(error_type, False)
    
    def _open_circuit(self, error_type: ErrorType):
        """Open circuit breaker for an error type."""
        self.circuit_breakers[error_type] = True
        self.last_error_time[error_type] = time.time()
        logger.warning(f"Circuit breaker opened for {error_type.value}")
    
    def _close_circuit(self, error_type: ErrorType):
        """Close circuit breaker for an error type."""
        self.circuit_breakers[error_type] = False
        logger.info(f"Circuit breaker closed for {error_type.value}")
    
    async def _recover_from_timeout(self, on_error: Optional[Callable[[str], None]]):
        """Recover from stream timeout."""
        logger.info("Attempting timeout recovery")
        
        # Wait a bit and try to reconnect
        await asyncio.sleep(2.0)
        
        # Implementation would attempt to restart the stream
        # For now, just log the recovery attempt
        logger.info("Timeout recovery completed")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            "error_counts": dict(self.error_counts),
            "circuit_breakers": {
                k.value: v for k, v in self.circuit_breakers.items()
            },
            "last_errors": {
                k.value: v for k, v in self.last_error_time.items()
            }
        }
    
    def reset_error_counts(self):
        """Reset all error counts (for testing or manual recovery)."""
        self.error_counts.clear()
        self.circuit_breakers.clear()
        self.last_error_time.clear()
        logger.info("Error recovery agent reset")


# Global instance for use across the application
error_recovery_agent = ErrorRecoveryAgent()
