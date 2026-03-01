"""Public API for toxp — the single entry point for CLI, GUI, and programmatic use.

Usage:
    from toxp import run_query, QueryResult

    result = await run_query("What is 2+2?")
    print(result.final_answer, result.confidence)

    # With overrides and callbacks:
    result = await run_query(
        "Explain recursion",
        config_overrides={"num_agents": 8, "temperature": 0.7},
        callbacks=MyCallbacks(),
        cancel_token=my_cancel_event,
    )
"""

import asyncio
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from toxp.config import ConfigManager, ToxpConfig
from toxp.exceptions import ToxpError
from toxp.models.query import Query
from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.models.result import Result
from toxp.orchestrator import Orchestrator
from toxp.providers.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# Callback Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class QueryCallbacks(Protocol):
    """Callback protocol for observing query execution.

    All methods have no-op defaults — implement only what you need.
    GUI consumers implement these to drive UI updates.
    CLI implements these to drive Rich progress + streaming output.
    """

    def on_agent_start(self, agent_id: int) -> None:
        """Called when a reasoning agent begins execution."""
        ...

    def on_agent_complete(
        self, agent_id: int, success: bool, error: Optional[str] = None
    ) -> None:
        """Called when a reasoning agent finishes (success or failure)."""
        ...

    def on_agent_token(self, agent_id: int, token: str) -> None:
        """Called for each streamed token from a reasoning agent."""
        ...

    def on_agents_done(self) -> None:
        """Called when all reasoning agents have finished, before coordinator starts."""
        ...

    def on_coordinator_token(self, token: str) -> None:
        """Called for each streamed token from the coordinator synthesis."""
        ...


class NoOpCallbacks:
    """Default no-op implementation of QueryCallbacks."""

    def on_agent_start(self, agent_id: int) -> None:
        pass

    def on_agent_complete(
        self, agent_id: int, success: bool, error: Optional[str] = None
    ) -> None:
        pass

    def on_agent_token(self, agent_id: int, token: str) -> None:
        pass

    def on_agents_done(self) -> None:
        pass

    def on_coordinator_token(self, token: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Structured Result
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """Structured result from a toxp query — the public return type.

    Wraps the internal Result model with a cleaner, consumer-friendly interface.
    """

    final_answer: str = Field(..., description="The synthesized final answer")
    confidence: str = Field(..., description="Confidence level: Low, Medium, or High")
    synthesis_markdown: str = Field(
        ..., description="Full coordinator synthesis as Markdown"
    )
    consensus_summary: str = Field(default="", description="What agents agreed on")
    debates_summary: str = Field(default="", description="Where agents disagreed")

    # Agent statistics
    total_agents: int = Field(..., description="Total agents spawned")
    successful_agents: int = Field(..., description="Agents that succeeded")
    failed_agents: int = Field(..., description="Agents that failed")

    # Usage & cost
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    total_duration_seconds: float = Field(
        default=0.0, description="Wall-clock time for entire query"
    )

    # Access to raw internals if needed
    raw_result: Optional[Result] = Field(
        default=None,
        exclude=True,
        description="Internal Result object for advanced consumers",
    )

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_result(cls, result: Result) -> "QueryResult":
        """Construct from internal Result object."""
        cr = result.coordinator_response
        meta = result.metadata
        successful = [r for r in result.agent_responses if r.success]

        return cls(
            final_answer=cr.final_answer,
            confidence=cr.confidence,
            synthesis_markdown=cr.synthesis,
            consensus_summary=cr.consensus_summary,
            debates_summary=cr.debates_summary,
            total_agents=len(result.agent_responses),
            successful_agents=len(successful),
            failed_agents=len(result.agent_responses) - len(successful),
            total_tokens=meta.get("total_agent_tokens", 0),
            total_duration_seconds=meta.get("total_duration_seconds", 0.0),
            raw_result=result,
        )


# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------

async def run_query(
    query: str,
    *,
    config_overrides: Optional[dict] = None,
    callbacks: Optional[QueryCallbacks] = None,
    cancel_token: Optional[asyncio.Event] = None,
) -> QueryResult:
    """Execute a toxp query through the full pipeline.

    This is THE entry point for all consumers (CLI, GUI, programmatic).

    Args:
        query: The question or problem to solve.
        config_overrides: Optional dict of config keys to override for this
            query only (e.g. {"num_agents": 8}). Does NOT mutate the config file.
        callbacks: Optional QueryCallbacks implementation for observing
            execution progress (agent lifecycle, streaming tokens).
        cancel_token: Optional asyncio.Event. When set, the orchestrator
            will stop spawning new agents and cancel in-flight work.

    Returns:
        QueryResult with the synthesized answer, confidence, and metadata.

    Raises:
        toxp.exceptions.CredentialsError: AWS credentials missing/expired.
        toxp.exceptions.ModelNotFoundError: Requested model not available.
        toxp.exceptions.ThrottlingError: Rate limited after retries.
        toxp.exceptions.InsufficientAgentsError: Too few agents succeeded.
        toxp.exceptions.ToxpError: Base class for all toxp errors.
        asyncio.CancelledError: If cancel_token was set during execution.
    """
    cb = callbacks or NoOpCallbacks()

    # 1. Load config with optional overrides (no file mutation)
    config_manager = ConfigManager()
    config = config_manager.load_with_overrides(config_overrides)

    # 2. Create provider
    provider_class = ProviderRegistry.get(config.provider)
    provider = provider_class(
        region=config.region,
        aws_profile=config.aws_profile,
        model_id=config.model,
        context_1m=config.context_1m,
    )

    # 3. Create orchestrator
    orchestrator = Orchestrator(
        provider=provider,
        num_agents=config.num_agents,
        temperature=config.temperature,
        coordinator_temperature=config.coordinator_temperature,
        max_tokens=config.max_tokens,
        max_concurrency=config.max_concurrency,
    )

    # 4. Execute query with callbacks wired through
    query_obj = Query(text=query)

    result = await orchestrator.process_query(
        query_obj,
        on_coordinator_token=cb.on_coordinator_token,
        on_agent_start=cb.on_agent_start,
        on_agent_complete=cb.on_agent_complete,
        on_agents_done=cb.on_agents_done,
        on_agent_token=cb.on_agent_token,
        cancel_token=cancel_token,
    )

    return QueryResult.from_result(result)


def validate_credentials(
    aws_profile: str = "default",
    region: str = "us-east-1",
) -> bool:
    """Check if AWS credentials are valid and Bedrock is accessible.

    Useful for GUI first-run wizards and health checks.

    Args:
        aws_profile: AWS profile name to test.
        region: AWS region to test.

    Returns:
        True if credentials are valid.

    Raises:
        toxp.exceptions.CredentialsError: With specific details about what's wrong.
    """
    import boto3
    from botocore.exceptions import (
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
    )

    from toxp.exceptions import CredentialsError

    try:
        session = boto3.Session(profile_name=aws_profile)
        sts = session.client("sts", region_name=region)
        sts.get_caller_identity()
        return True
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise CredentialsError(
            "AWS credentials not configured.",
            details=str(e),
        )
    except ProfileNotFound as e:
        raise CredentialsError(
            f"AWS profile '{aws_profile}' not found.",
            details=str(e),
        )
    except ClientError as e:
        raise CredentialsError(
            "AWS credentials invalid or expired.",
            details=str(e),
        )


def get_default_config() -> dict:
    """Return the current effective config as a dict.

    Useful for GUI settings panels to show current values.
    """
    return ConfigManager().load().to_dict()
