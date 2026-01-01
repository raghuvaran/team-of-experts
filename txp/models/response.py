"""Response data models for agents and coordinator."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AgentResponse(BaseModel):
    """Represents the response from a single reasoning agent."""

    agent_id: int = Field(..., description="Unique identifier for the agent")
    success: bool = Field(..., description="Whether the agent invocation succeeded")
    chain_of_thought: str = Field(
        default="",
        description="The agent's reasoning process",
    )
    final_answer: str = Field(
        default="",
        description="The agent's final answer",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if invocation failed",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Time taken for agent invocation",
    )
    token_count: int = Field(
        default=0,
        description="Number of tokens used",
    )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "AgentResponse":
        """Load from JSON string."""
        return cls.model_validate_json(json_str)


# Valid confidence levels for coordinator responses
VALID_CONFIDENCE_LEVELS = ("Low", "Medium", "High")


class CoordinatorResponse(BaseModel):
    """Represents the synthesized response from the coordinator agent."""

    synthesis: str = Field(..., description="The complete synthesis text")
    confidence: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Confidence level: Low, Medium, or High",
    )
    consensus_summary: str = Field(
        default="",
        description="Summary of agent agreements",
    )
    debates_summary: str = Field(
        default="",
        description="Summary of agent disagreements",
    )
    final_answer: str = Field(..., description="The synthesized final answer")
    duration_seconds: float = Field(
        default=0.0,
        description="Time taken for coordinator synthesis",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        """Validate that confidence is one of the allowed values."""
        if v not in VALID_CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence must be one of {VALID_CONFIDENCE_LEVELS}, got '{v}'"
            )
        return v

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "CoordinatorResponse":
        """Load from JSON string."""
        return cls.model_validate_json(json_str)
