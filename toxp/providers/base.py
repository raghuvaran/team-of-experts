"""Abstract base class for LLM providers.

Defines the interface that all LLM providers must implement, including
both synchronous and streaming model invocation methods.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

from pydantic import BaseModel, Field, field_validator


class ProviderResponse(BaseModel):
    """Structured response from an LLM provider.
    
    Contains the model output along with usage metrics for monitoring
    and cost estimation.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The generated text response from the model",
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="Number of input tokens consumed",
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Number of output tokens generated",
    )
    latency_ms: float = Field(
        ...,
        gt=0,
        description="Response latency in milliseconds",
    )
    model_id: str = Field(
        ...,
        min_length=1,
        description="The model ID that generated this response",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("text must be a non-empty string")
        return v

    @field_validator("model_id")
    @classmethod
    def validate_model_id_not_empty(cls, v: str) -> str:
        """Validate that model_id is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("model_id must be a non-empty string")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderResponse":
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ProviderResponse":
        """Create from JSON string."""
        return cls.model_validate_json(json_str)


class BaseProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM provider implementations must inherit from this class and
    implement the invoke_model and invoke_model_stream methods.
    
    This enables a pluggable architecture where new providers can be
    added without modifying core orchestration logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'bedrock', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model ID being used by this provider instance."""
        pass

    @abstractmethod
    async def invoke_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        """Invoke the model and return a complete response.
        
        Args:
            system_prompt: The system prompt to set model behavior
            user_message: The user's input message
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            ProviderResponse containing the model output and metrics
            
        Raises:
            ProviderError: If the model invocation fails
        """
        pass

    @abstractmethod
    async def invoke_model_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Invoke the model with streaming response.
        
        Args:
            system_prompt: The system prompt to set model behavior
            user_message: The user's input message
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Yields:
            String tokens as they are generated
            
        Raises:
            ProviderError: If the model invocation fails
        """
        pass
