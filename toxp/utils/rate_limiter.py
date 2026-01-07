"""Adaptive rate limiter for Bedrock API calls with quota-aware concurrency.

This module implements an AdaptiveRateLimiter that controls concurrent API requests
based on model quotas to prevent throttling. It uses semaphore-based concurrency
control and implements exponential backoff on throttling.

Feature: toxp-cli
Requirements: 6.3
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QuotaInfo:
    """Bedrock quota information for a model.
    
    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        tokens_per_minute: Maximum tokens allowed per minute
        model_id: Short identifier for the model type
    """
    requests_per_minute: int
    tokens_per_minute: int
    model_id: str


# Default quotas for Claude models (conservative estimates)
# These are used when we can't query Service Quotas API
DEFAULT_QUOTAS = {
    # Global cross-region Opus 4.5
    "global.anthropic.claude-opus-4-5": QuotaInfo(250, 500_000, "opus-4.5"),
    # Cross-region Opus 4.5
    "anthropic.claude-opus-4-5": QuotaInfo(125, 250_000, "opus-4.5"),
    # Global cross-region Opus 4.1
    "global.anthropic.claude-opus-4-1": QuotaInfo(50, 500_000, "opus-4.1"),
    # Sonnet models (higher limits)
    "anthropic.claude-sonnet": QuotaInfo(1000, 2_000_000, "sonnet"),
    "us.anthropic.claude-sonnet": QuotaInfo(1000, 2_000_000, "sonnet"),
    # Default fallback
    "default": QuotaInfo(50, 200_000, "default"),
}


def get_quota_for_model(model_id: str) -> QuotaInfo:
    """Get quota info for a model ID.
    
    Args:
        model_id: Full model identifier (e.g., "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        
    Returns:
        QuotaInfo with RPM and TPM limits
    """
    model_lower = model_id.lower()
    
    # Check for specific model patterns
    for pattern, quota in DEFAULT_QUOTAS.items():
        if pattern in model_lower:
            logger.debug(f"Matched quota pattern '{pattern}' for model '{model_id}'")
            return quota
    
    logger.debug(f"No specific quota found for '{model_id}', using default")
    return DEFAULT_QUOTAS["default"]


class AdaptiveRateLimiter:
    """Token-bucket rate limiter with adaptive concurrency.
    
    This limiter:
    1. Tracks both RPM and TPM quotas
    2. Calculates optimal concurrency based on expected token usage
    3. Uses semaphore to limit concurrent requests
    4. Implements exponential backoff on throttling
    
    The key insight: Bedrock reserves max_tokens from your TPM quota
    when a request starts, not when tokens are actually generated.
    
    Attributes:
        model_id: Bedrock model identifier
        max_tokens_per_request: Maximum output tokens per request
        estimated_input_tokens: Estimated input tokens per request
        safety_margin: Fraction of quota to use (0.0-1.0)
        quota: QuotaInfo for the model
        max_concurrency: Maximum concurrent requests allowed
    """
    
    def __init__(
        self,
        model_id: str,
        max_tokens_per_request: int = 8192,
        estimated_input_tokens: int = 10000,
        safety_margin: float = 0.7,  # Use 70% of quota to leave headroom
        max_concurrency_override: Optional[int] = None,
    ):
        """Initialize rate limiter.
        
        Args:
            model_id: Bedrock model identifier
            max_tokens_per_request: Max output tokens per request
            estimated_input_tokens: Estimated input tokens per request
            safety_margin: Fraction of quota to use (0.0-1.0)
            max_concurrency_override: Override calculated concurrency
        """
        self.model_id = model_id
        self.max_tokens_per_request = max_tokens_per_request
        self.estimated_input_tokens = estimated_input_tokens
        self.safety_margin = safety_margin
        
        # Get quota for this model
        self.quota = get_quota_for_model(model_id)
        
        # Calculate optimal concurrency
        self.max_concurrency = self._calculate_optimal_concurrency(
            max_concurrency_override
        )
        
        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        
        # Track request timing for rate limiting
        self._request_times: list[float] = []
        self._lock = asyncio.Lock()
        
        # Backoff state
        self._consecutive_throttles = 0
        self._last_throttle_time = 0.0
        
        # Track current in-flight requests
        self._in_flight = 0
        self._in_flight_lock = asyncio.Lock()
        
        logger.info(
            f"RateLimiter initialized: model={model_id}, "
            f"max_concurrency={self.max_concurrency}, "
            f"quota_rpm={self.quota.requests_per_minute}, "
            f"quota_tpm={self.quota.tokens_per_minute}"
        )
    
    def _calculate_optimal_concurrency(
        self, 
        override: Optional[int] = None
    ) -> int:
        """Calculate optimal concurrency based on quotas and token usage.
        
        Args:
            override: Manual override for concurrency
            
        Returns:
            Optimal number of concurrent requests
        """
        if override is not None:
            logger.info(f"Using concurrency override: {override}")
            return max(1, override)
        
        # Total tokens reserved per request (input + max output)
        tokens_per_request = self.estimated_input_tokens + self.max_tokens_per_request
        
        # Calculate concurrency limits
        # 1. RPM-based limit (requests per second with safety margin)
        rpm_limit = int(self.quota.requests_per_minute * self.safety_margin / 60)
        rpm_limit = max(1, rpm_limit)  # At least 1 per second
        
        # 2. TPM-based limit (more restrictive for large requests)
        tpm_limit = int(
            (self.quota.tokens_per_minute * self.safety_margin) / tokens_per_request
        )
        tpm_limit = max(1, tpm_limit)
        
        # Use the more restrictive limit
        optimal = min(rpm_limit, tpm_limit)
        
        # Cap at reasonable maximum
        optimal = min(optimal, 10)
        
        logger.info(
            f"Calculated concurrency: rpm_limit={rpm_limit}, "
            f"tpm_limit={tpm_limit}, optimal={optimal}, "
            f"tokens_per_request={tokens_per_request}"
        )
        
        return optimal
    
    @property
    def in_flight_count(self) -> int:
        """Return the current number of in-flight requests."""
        return self._in_flight
    
    async def acquire(self) -> None:
        """Acquire permission to make a request.
        
        This method:
        1. Waits for semaphore slot
        2. Applies rate limiting delay if needed
        3. Handles backoff from previous throttles
        """
        await self._semaphore.acquire()
        
        async with self._in_flight_lock:
            self._in_flight += 1
        
        async with self._lock:
            now = time.time()
            
            # Apply backoff if we've been throttled recently
            if self._consecutive_throttles > 0:
                backoff = min(2 ** self._consecutive_throttles, 30)
                time_since_throttle = now - self._last_throttle_time
                if time_since_throttle < backoff:
                    wait_time = backoff - time_since_throttle
                    logger.debug(f"Applying backoff: {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Clean old request times (older than 60 seconds)
            self._request_times = [
                t for t in self._request_times if now - t < 60
            ]
            
            # Rate limit: ensure we don't exceed RPM
            rpm_threshold = self.quota.requests_per_minute * self.safety_margin
            if len(self._request_times) >= rpm_threshold:
                oldest = self._request_times[0]
                wait_time = 60 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            self._request_times.append(time.time())
    
    def release(self, throttled: bool = False) -> None:
        """Release the semaphore after request completes.
        
        Args:
            throttled: Whether the request was throttled
        """
        if throttled:
            self._consecutive_throttles += 1
            self._last_throttle_time = time.time()
            logger.warning(
                f"Request throttled, consecutive={self._consecutive_throttles}"
            )
        else:
            # Reset throttle counter on success
            self._consecutive_throttles = max(0, self._consecutive_throttles - 1)
        
        # Decrement in-flight count synchronously (safe since it's just a counter)
        self._in_flight = max(0, self._in_flight - 1)
        
        self._semaphore.release()
    
    async def __aenter__(self) -> "AdaptiveRateLimiter":
        """Context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        # Check if exception was a throttling error
        throttled = False
        if exc_val is not None:
            error_str = str(exc_val).lower()
            throttled = "throttl" in error_str or "too many" in error_str
        
        self.release(throttled)
        return False  # Don't suppress exceptions


def calculate_recommended_agents(
    model_id: str,
    max_tokens: int = 8192,
    estimated_input_tokens: int = 10000,
) -> int:
    """Calculate recommended number of agents based on model quotas.
    
    Args:
        model_id: Bedrock model identifier
        max_tokens: Max output tokens per request
        estimated_input_tokens: Estimated input tokens
        
    Returns:
        Recommended number of agents
    """
    quota = get_quota_for_model(model_id)
    tokens_per_request = estimated_input_tokens + max_tokens
    
    # Calculate based on TPM with safety margin
    tpm_based = int((quota.tokens_per_minute * 0.6) / tokens_per_request)
    
    # Also consider RPM
    rpm_based = int(quota.requests_per_minute * 0.6 / 60)  # Per second
    
    recommended = min(tpm_based, rpm_based, 8)  # Cap at 8 for Opus
    recommended = max(2, recommended)  # At least 2
    
    return recommended
