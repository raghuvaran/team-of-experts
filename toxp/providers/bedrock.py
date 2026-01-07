"""AWS Bedrock provider using Converse API with retry logic.

This module implements the BedrockProvider class that connects to AWS Bedrock
using the Converse API for model invocation, with support for streaming responses
and exponential backoff retry logic.
"""

import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, Optional

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
from .base import BaseProvider, ProviderResponse


# Keep BedrockProviderError as an alias for backward compatibility
BedrockProviderError = ProviderError


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
    MODEL_ID_PATTERN = re.compile(r"^(us\.|eu\.|ap\.|global\.)?anthropic\.claude-[a-z0-9\-:]+$")

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_profile: str = "default",
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ):
        """Initialize Bedrock provider.
        
        Args:
            region: AWS region for Bedrock service (default: us-east-1)
            model_id: Model identifier for invocations
            aws_profile: AWS credentials profile name (default: default)
            timeout_seconds: Timeout for API calls (default: 120)
            max_retries: Maximum retry attempts for throttling (default: 3)
            
        Raises:
            CredentialsError: If AWS credentials are not configured
            ValueError: If model_id format is invalid
        """
        self.region = region
        self._model_id = model_id
        self.aws_profile = aws_profile
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Validate model ID format
        self.validate_model_id(model_id)
        
        # Initialize boto3 client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize boto3 Bedrock runtime client.
        
        Raises:
            CredentialsError: If AWS credentials are not configured
        """
        boto_config = BotoConfig(
            read_timeout=self.timeout_seconds,
            connect_timeout=30,
            retries={"max_attempts": 0},  # We handle retries ourselves
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

    async def invoke_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        """Invoke Bedrock model using Converse API with retry logic.
        
        Args:
            system_prompt: System prompt to set model behavior
            user_message: User's input message
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            ProviderResponse containing model output and metrics
            
        Raises:
            CredentialsError: If authentication fails
            ModelNotFoundError: If model is not found
            ThrottlingError: If rate limiting persists after retries
            BedrockProviderError: For other API errors
            TimeoutError: If invocation exceeds timeout
        """
        messages = [
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        ]
        
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
                            messages=messages,
                            system=system_prompts,
                            inferenceConfig=inference_config,
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
    ) -> AsyncIterator[str]:
        """Invoke Bedrock model with streaming response.
        
        Args:
            system_prompt: System prompt to set model behavior
            user_message: User's input message
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Yields:
            String tokens as they are generated
            
        Raises:
            CredentialsError: If authentication fails
            ModelNotFoundError: If model is not found
            BedrockProviderError: For other API errors
        """
        messages = [
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        ]
        
        system_prompts = [{"text": system_prompt}]
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
        }

        def _stream():
            return self.client.converse_stream(
                modelId=self._model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config,
            )

        try:
            response = await asyncio.get_event_loop().run_in_executor(None, _stream)
            
            # Process stream events
            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield delta["text"]
                        
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
