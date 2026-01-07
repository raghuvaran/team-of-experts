"""Query data model."""

from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field


class Query(BaseModel):
    """Represents a user query to be processed by the TOXP system."""

    text: str = Field(..., description="The query text to process")
    query_id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Unique query identifier (8 chars)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Query creation timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional query metadata",
    )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "Query":
        """Load from JSON string."""
        return cls.model_validate_json(json_str)
