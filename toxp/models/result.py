"""Result data model."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .query import Query
from .response import AgentResponse, CoordinatorResponse


class Result(BaseModel):
    """Represents the complete result of processing a query through TOXP."""

    query: Query = Field(..., description="The original query")
    agent_responses: List[AgentResponse] = Field(
        ...,
        description="All agent responses",
    )
    coordinator_response: CoordinatorResponse = Field(
        ...,
        description="The coordinator's synthesis",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metadata",
    )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "Result":
        """Load from JSON string."""
        return cls.model_validate_json(json_str)

    def to_display(self) -> str:
        """Format for terminal display."""
        successful_agents = [r for r in self.agent_responses if r.success]
        success_rate = (
            len(successful_agents) / len(self.agent_responses)
            if self.agent_responses
            else 0
        )

        output = []
        output.append("=" * 80)
        output.append("TOXP RESULT")
        output.append("=" * 80)
        output.append(f"\nQuery ID: {self.query.query_id}")
        output.append(f"Timestamp: {self.query.timestamp}")
        output.append(f"\nQuery: {self.query.text}")
        output.append(f"\n{'-' * 80}")
        output.append("\nAgent Statistics:")
        output.append(f"  Total Agents: {len(self.agent_responses)}")
        output.append(f"  Successful: {len(successful_agents)}")
        output.append(f"  Success Rate: {success_rate:.1%}")
        output.append(f"\n{'-' * 80}")
        output.append("\nCoordinator Synthesis:")
        output.append(f"  Confidence: {self.coordinator_response.confidence}")
        output.append(f"  Duration: {self.coordinator_response.duration_seconds:.2f}s")
        output.append(f"\n{'-' * 80}")
        output.append(f"\n{self.coordinator_response.synthesis}")
        output.append(f"\n{'-' * 80}")

        if self.metadata:
            output.append("\nMetadata:")
            for key, value in self.metadata.items():
                output.append(f"  {key}: {value}")

        output.append("=" * 80)
        return "\n".join(output)
