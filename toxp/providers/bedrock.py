"""AWS Bedrock provider using Converse API with retry logic.

This module implements the BedrockProvider class that connects to AWS Bedrock
using the Converse API for model invocation, with support for streaming responses
and exponential backoff retry logic.
"""

import asyncio
import re
import ssl
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

from toxp.exceptions import (
    CredentialsError,
    ModelNotFoundError,
    ProviderError,
    ThrottlingError,
    TimeoutError as ToxpTimeoutError,
)
from toxp.models.conversation import Message
from .base import BaseProvider, ProviderResponse


# Keep BedrockProviderError as an alias for backward compatibility
BedrockProviderError = ProviderError

# Lock for SSL initialization to prevent race conditions in OpenSSL cert store
# This is a workaround for OpenSSL 3.x thread safety issues on macOS
_ssl_init_lock = threading.Lock()
_ssl_initialized = False


def _ensure_ssl_initialized() -> None:
    """Pre-initialize SSL context to avoid race conditions during concurrent handshakes.
    
    OpenSSL 3.x has thread safety issues with X509_STORE_CTX when multiple threads
    perform SSL handshakes simultaneously. By pre-loading the certificate store
    in a single-threaded context, we avoid the race condition.
    """
    global _ssl_initialized
    if _ssl_initialized:
        return
    
    with _ssl_init_lock:
        if _ssl_initialized:
            return
        
        # Create a default SSL context and load system certificates
        # This pre-populates the certificate store before concurrent access
        ctx = ssl.create_default_context()
        ctx.load_default_certs()
        
        _ssl_initialized = True


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider using Converse API.
    
    Implements the BaseProvider interface for AWS Bedrock, supporting both
    synchronous and streaming model invocation with exponential backoff retry
    logic for handling throttling.
    
    Attributes:
        region: AWS region for Bedrock service
        _model_id: Model identifier for invocations
        aws_profile: AWS credentials profile name
        timeout_seconds: Timeout for API calls
        max_retries: Maximum number of retry attempts for throttling
    """

    # Valid Claude model ID patterns
    # Format: anthropic.claude-{variant}-{version} OR {region}.anthropic.claude-{variant}-{version}
    MODEL_ID_PATTERN = re.compile(r"^(us\.|eu\.|ap\.|jp\.|apac\.|global\.)?anthropic\.claude-[a-z0-9\-:]+$")

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_profile: str = "default",
        timeout_seconds: int = 120,
        max_retries: int = 3,
        context_1m: bool = False,
    ):
        """Initialize Bedrock provider.
        
        Args:
            region: AWS region for Bedrock service (default: us-east-1)
            model_id: Model identifier for invocations
            aws_profile: AWS credentials profile name (default: default)
            timeout_seconds: Timeout for API calls (default: 120)
            max_retries: Maximum retry attempts for throttling (default: 3)
            context_1m: Enable 1M token context window beta (default: False)
            
        Raises:
            CredentialsError: If AWS credentials are not configured
            ValueError: If model_id format is invalid
        """
        self.region = region
        self._model_id = model_id
        self.aws_profile = aws_profile
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.context_1m = context_1m
        
        # Validate model ID format
        self.validate_model_id(model_id)
        
        # Initialize boto3 client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize boto3 Bedrock runtime client.
        
        Raises:
            CredentialsError: If AWS credentials are not configured
        """
        # Pre-initialize SSL to avoid race conditions in concurrent handshakes
        _ensure_ssl_initialized()
        
        boto_config = BotoConfig(
            read_timeout=self.timeout_seconds,
            connect_timeout=30,
            retries={"max_attempts": 0},  # We handle retries ourselves
            # Enable connection pooling to reuse SSL connections
            max_pool_connections=10,
        )
        
        try:
            session = boto3.Session(profile_name=self.aws_profile)
            self.client = session.client(
                "bedrock-runtime",
                region_name=self.region,
                config=boto_config,
            )
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise CredentialsError(
                "AWS credentials not found.",
                details=str(e),
            )

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "bedrock"

    @property
    def model_id(self) -> str:
        """Return the model ID being used."""
        return self._model_id

    def _additional_request_fields(self) -> dict:
        """Build additionalModelRequestFields for Converse API."""
        if self.context_1m:
            return {"anthropic_beta": ["context-1m-2025-08-07"]}
        return {}

    @classmethod
    def validate_model_id(cls, model_id: str) -> bool:
        """Validate model ID format.
        
        Args:
            model_id: Model identifier to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If model ID format is invalid
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("Model ID must be a non-empty string")
        
        if not cls.MODEL_ID_PATTERN.match(model_id):
            raise ValueError(
                f"Invalid model ID format: {model_id}\n"
                "Expected format: anthropic.claude-{{variant}}-{{version}} or "
                "{{region}}.anthropic.claude-{{variant}}-{{version}}\n"
                "Example: us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            )
        
        return True

    @classmethod
    def is_valid_model_id(cls, model_id: str) -> bool:
        """Check if model ID format is valid without raising.
        
        Args:
            model_id: Model identifier to check
            
        Returns:
            True if valid, False otherwise
        """
        try:
            cls.validate_model_id(model_id)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _build_bedrock_messages(
        user_message: str,
        messages: Optional[List[Message]] = None,
    ) -> List[Dict[str, Any]]:
        """Build Bedrock Converse API messages from either a single user_message or multi-turn messages.

        Args:
            user_message: Single user message (used when messages is None)
            messages: Optional multi-turn conversation messages

        Returns:
            List of Bedrock-formatted message dicts
        """
        if messages:
            return [
                {"role": m["role"], "content": [{"text": m["content"]}]}
                for m in messages
            ]
        return [{"role": "user", "content": [{"text": user_message}]}]

    async def invoke_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        messages: Optional[List[Message]] = None,
    ) -> ProviderResponse:
        """Invoke Bedrock model using Converse API with retry logic.

        Args:
            system_prompt: System prompt to set model behavior
            user_message: User's input message (used when messages is None)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            messages: Optional multi-turn conversation messages

        Returns:
            ProviderResponse containing model output and metrics

        Raises:
            CredentialsError: If authentication fails
            ModelNotFoundError: If model is not found
            ThrottlingError: If rate limiting persists after retries
            BedrockProviderError: For other API errors
            TimeoutError: If invocation exceeds timeout
        """
        bedrock_messages = self._build_bedrock_messages(user_message, messages)
        
        system_prompts = [{"text": system_prompt}]
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
        }

        retry_count = 0
        base_delay = 1.0  # Start with 1 second delay
        start_time = time.perf_counter()

        while retry_count <= self.max_retries:
            try:
                # Run synchronous boto3 call in executor to avoid blocking
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.converse(
                            modelId=self._model_id,
                            messages=bedrock_messages,
                            system=system_prompts,
                            inferenceConfig=inference_config,
                            **({
                                "additionalModelRequestFields": self._additional_request_fields()
                            } if self.context_1m else {}),
                        ),
                    ),
                    timeout=self.timeout_seconds,
                )

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Parse response
                parsed = self._parse_converse_response(response)
                
                return ProviderResponse(
                    text=parsed["text"],
                    input_tokens=parsed["usage"]["input_tokens"],
                    output_tokens=parsed["usage"]["output_tokens"],
                    latency_ms=latency_ms,
                    model_id=self._model_id,
                )

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_message = e.response.get("Error", {}).get("Message", str(e))

                # Handle throttling with exponential backoff
                if error_code == "ThrottlingException" and retry_count < self.max_retries:
                    retry_count += 1
                    delay = base_delay * (2 ** (retry_count - 1))
                    await asyncio.sleep(delay)
                    continue

                # Handle authentication errors
                if error_code in ["UnrecognizedClientException", "InvalidSignatureException"]:
                    raise CredentialsError(
                        "AWS authentication failed. Please verify your credentials.",
                        details=error_message,
                    )
                
                if error_code == "ExpiredTokenException":
                    raise CredentialsError(
                        "AWS credentials have expired.",
                        details=error_message,
                    )
                
                if error_code == "AccessDeniedException":
                    raise CredentialsError(
                        f"Access denied to model {self._model_id} in {self.region}. "
                        "Ensure you have Bedrock model access enabled in AWS console.",
                        details=error_message,
                    )
                
                if error_code == "ResourceNotFoundException":
                    raise ModelNotFoundError(
                        model_id=self._model_id,
                        region=self.region,
                    )
                
                if error_code == "ValidationException":
                    raise ProviderError(
                        f"Invalid request to Bedrock API: {error_message}",
                        provider="bedrock",
                        details=f"Model: {self._model_id}",
                    )

                # For other client errors
                raise ProviderError(
                    f"Bedrock API error ({error_code}): {error_message}",
                    provider="bedrock",
                )

            except asyncio.TimeoutError:
                raise ToxpTimeoutError(
                    f"Bedrock invocation exceeded timeout",
                    timeout_seconds=self.timeout_seconds,
                )

        # Exhausted all retries
        raise ThrottlingError(
            f"Failed to invoke model after {self.max_retries} retries due to throttling.",
            retry_count=self.max_retries,
        )

    async def invoke_model_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        messages: Optional[List[Message]] = None,
    ) -> AsyncIterator[str]:
        """Invoke Bedrock model with streaming response.

        Args:
            system_prompt: System prompt to set model behavior
            user_message: User's input message (used when messages is None)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            messages: Optional multi-turn conversation messages

        Yields:
            String tokens as they are generated

        Raises:
            CredentialsError: If authentication fails
            ModelNotFoundError: If model is not found
            BedrockProviderError: For other API errors
        """
        import queue as queue_module
        from concurrent.futures import ThreadPoolExecutor

        bedrock_messages = self._build_bedrock_messages(user_message, messages)

        system_prompts = [{"text": system_prompt}]
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
        }
        
        # Use thread-safe queue for cross-thread communication
        token_queue: queue_module.Queue[str | None] = queue_module.Queue()
        error_holder: list = []  # To propagate exceptions from thread

        def _stream_and_process():
            """Run entire stream in thread, pushing tokens to queue."""
            try:
                response = self.client.converse_stream(
                    modelId=self._model_id,
                    messages=bedrock_messages,
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    **({
                        "additionalModelRequestFields": self._additional_request_fields()
                    } if self.context_1m else {}),
                )
                for event in response.get("stream", []):
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            token_queue.put(delta["text"])
            except Exception as e:
                error_holder.append(e)
            finally:
                token_queue.put(None)  # Signal end

        try:
            # Use a dedicated executor to avoid blocking the default pool
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                stream_future = loop.run_in_executor(executor, _stream_and_process)
                
                # Yield tokens as they arrive
                while True:
                    try:
                        token = token_queue.get(timeout=0.05)
                        if token is None:
                            break
                        yield token
                    except queue_module.Empty:
                        # Check if stream task is done
                        if stream_future.done():
                            # Drain remaining tokens
                            while True:
                                try:
                                    token = token_queue.get_nowait()
                                    if token is None:
                                        break
                                    yield token
                                except queue_module.Empty:
                                    break
                            break
                        await asyncio.sleep(0)  # Yield to event loop
                
                # Wait for stream to complete
                await stream_future
            
            # Re-raise any exception from the stream thread
            if error_holder:
                raise error_holder[0]
                        
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            
            if error_code in ["UnrecognizedClientException", "InvalidSignatureException"]:
                raise CredentialsError(
                    "AWS authentication failed. Please verify your credentials.",
                    details=error_message,
                )
            
            if error_code == "ExpiredTokenException":
                raise CredentialsError(
                    "AWS credentials have expired.",
                    details=error_message,
                )
            
            if error_code == "AccessDeniedException":
                raise CredentialsError(
                    f"Access denied to model {self._model_id} in {self.region}. "
                    "Ensure you have Bedrock model access enabled in AWS console.",
                    details=error_message,
                )
            
            if error_code == "ResourceNotFoundException":
                raise ModelNotFoundError(
                    model_id=self._model_id,
                    region=self.region,
                )
            
            raise ProviderError(
                f"Bedrock streaming error ({error_code}): {error_message}",
                provider="bedrock",
            )

    def _parse_converse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Converse API response.
        
        Args:
            response: Raw response from Bedrock Converse API
            
        Returns:
            Dict with 'text' and 'usage' keys
            
        Raises:
            ValueError: If response format is invalid
        """
        try:
            output_message = response.get("output", {}).get("message", {})
            content = output_message.get("content", [])
            
            if not content:
                raise ValueError("Response contains no content")

            text_parts = []
            for block in content:
                if "text" in block:
                    text_parts.append(block["text"])

            if not text_parts:
                raise ValueError("No text content found in response")

            full_text = "".join(text_parts)

            usage = response.get("usage", {})

            return {
                "text": full_text,
                "usage": {
                    "input_tokens": usage.get("inputTokens", 0),
                    "output_tokens": usage.get("outputTokens", 0),
                },
            }

        except KeyError as e:
            raise ValueError(f"Failed to parse Converse response: {e}")
