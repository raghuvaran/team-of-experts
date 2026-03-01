# TOXP v0.5.0 — API-First Architecture Design

**Version**: 1.0  
**Date**: 2026-02-23  
**Status**: Ready for Implementation  
**Target**: Senior Engineer (toxp repo owner)  
**Scope**: Backward-compatible refactor — CLI behavior unchanged, new public library API added  

---

## 1. Motivation

toxp v0.4.1 is a well-structured CLI tool. The `Orchestrator` already supports async callbacks, `ConfigManager` supports overrides, and models are Pydantic-based. However, any non-CLI consumer (Flet desktop app, web UI, VS Code extension) currently has to:

1. Import internal modules (`toxp.orchestrator`, `toxp.providers.registry`, `toxp.config`) directly
2. Manually wire up provider instantiation, config loading, and orchestrator construction — duplicating ~40 lines of boilerplate from `cli.py:handle_query_command`
3. Invent its own event/callback plumbing for UI updates
4. Handle cancellation by killing the process (no cooperative cancellation)

This design adds a clean public API layer (`toxp/api.py`) that both the CLI and any external consumer can use. The CLI becomes a thin formatting shell on top of this API.

---

## 2. What Changes

| File | Change Type | Description |
|---|---|---|
| `toxp/api.py` | **NEW** | Public async API: `run_query()`, `QueryCallbacks`, `QueryResult`, `validate_credentials()` |
| `toxp/__init__.py` | MODIFY | Re-export public API symbols |
| `toxp/cli.py` | MODIFY | Refactor `handle_query_command` to call `api.run_query()` |
| `toxp/orchestrator.py` | MODIFY | Accept `asyncio.Event` cancellation token |
| `toxp/config.py` | MODIFY | Add `ConfigManager.load_with_overrides()` convenience method |
| `pyproject.toml` | MODIFY | Bump to v0.5.0, update description |
| `README.md` | MODIFY | Add "Library Usage" section |

No existing public behavior changes. All changes are additive and backward-compatible.

---

## 3. Detailed Design

### 3.1 `toxp/api.py` — The Public API

This is the single entry point for all consumers. It encapsulates the full pipeline: config → provider → orchestrator → result.

```python
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

    class Config:
        arbitrary_types_allowed = True

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
```


```python
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
    config = config_manager.load()
    if config_overrides:
        config = config_manager.apply_overrides(config, config_overrides)

    # 2. Create provider
    provider_class = ProviderRegistry.get(config.provider)
    provider = provider_class(
        region=config.region,
        aws_profile=config.aws_profile,
        model_id=config.model,
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
    )

    try:
        session = boto3.Session(profile_name=aws_profile)
        sts = session.client("sts", region_name=region)
        sts.get_caller_identity()
        return True
    except (NoCredentialsError, PartialCredentialsError) as e:
        from toxp.exceptions import CredentialsError
        raise CredentialsError(
            "AWS credentials not configured.",
            details=str(e),
        )
    except ClientError as e:
        from toxp.exceptions import CredentialsError
        raise CredentialsError(
            "AWS credentials invalid or expired.",
            details=str(e),
        )


def get_default_config() -> dict:
    """Return the current effective config as a dict.

    Useful for GUI settings panels to show current values.
    """
    return ConfigManager().load().to_dict()
```

---

### 3.2 Cancellation Token Support in Orchestrator

The `Orchestrator.process_query()` method gains an optional `cancel_token` parameter. When set, it:
1. Stops spawning new agents
2. Cancels in-flight agent tasks via `asyncio.Task.cancel()`
3. Raises `asyncio.CancelledError` to the caller

```python
# In toxp/orchestrator.py — changes to process_query signature:

async def process_query(
    self,
    query: Query,
    on_coordinator_token: Optional[Callable[[str], None]] = None,
    on_agent_start: Optional[Callable[[int], None]] = None,
    on_agent_complete: Optional[Callable[[int, bool, Optional[str]], None]] = None,
    on_agents_done: Optional[Callable[[], None]] = None,
    on_agent_token: Optional[Callable[[int, str], None]] = None,
    cancel_token: Optional[asyncio.Event] = None,  # NEW
) -> Result:
```

Inside `_spawn_agents`, before each `run_with_rate_limit` call:
```python
if cancel_token and cancel_token.is_set():
    # Cancel remaining tasks, collect partial results
    for task in pending_tasks:
        task.cancel()
    break
```

This is a minimal, non-breaking change. When `cancel_token` is `None` (the default), behavior is identical to v0.4.1.

---

### 3.3 ConfigManager Convenience Method

```python
# In toxp/config.py — add to ConfigManager class:

def load_with_overrides(self, overrides: Optional[dict] = None) -> ToxpConfig:
    """Load config and apply overrides in one call. Does not mutate the file.

    This is the preferred method for programmatic consumers that need
    per-query config without affecting the user's saved settings.

    Args:
        overrides: Optional dict of config keys to override.

    Returns:
        ToxpConfig with overrides applied.
    """
    config = self.load()
    if overrides:
        config = self.apply_overrides(config, overrides)
    return config
```

---

### 3.4 Updated `toxp/__init__.py` Exports

```python
"""TOXP — Team of eXPerts parallel reasoning system.

Library usage:
    from toxp import run_query, QueryResult, QueryCallbacks, validate_credentials

    result = await run_query("What is 2+2?")
    print(result.final_answer, result.confidence)
"""

import sys

__version__ = "0.5.0"

# Runtime Python version check
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    sys.exit(
        f"Error: TOXP requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, "
        f"but you're running Python {sys.version_info.major}.{sys.version_info.minor}"
    )

# Public API
from toxp.api import (
    run_query,
    QueryResult,
    QueryCallbacks,
    NoOpCallbacks,
    validate_credentials,
    get_default_config,
)

# Exceptions (already well-structured)
from toxp.exceptions import (
    ToxpError,
    CredentialsError,
    ConfigurationError,
    ProviderError,
    ModelNotFoundError,
    ThrottlingError,
    InsufficientAgentsError,
    NetworkError,
    TimeoutError,
)

__all__ = [
    "__version__",
    # API
    "run_query",
    "QueryResult",
    "QueryCallbacks",
    "NoOpCallbacks",
    "validate_credentials",
    "get_default_config",
    # Exceptions
    "ToxpError",
    "CredentialsError",
    "ConfigurationError",
    "ProviderError",
    "ModelNotFoundError",
    "ThrottlingError",
    "InsufficientAgentsError",
    "NetworkError",
    "TimeoutError",
]
```

---

### 3.5 CLI Refactor (`toxp/cli.py`)

`handle_query_command` becomes a thin wrapper around `api.run_query()`:

```python
from toxp.api import run_query, QueryCallbacks


class CLICallbacks:
    """QueryCallbacks implementation that drives Rich progress + streaming."""

    def __init__(self, formatter: OutputFormatter, progress):
        self._formatter = formatter
        self._progress = progress
        self._agent_start_cb, self._agent_complete_cb, self._agent_token_cb = (
            progress.get_callbacks() if progress else (None, None, None)
        )

    def on_agent_start(self, agent_id: int) -> None:
        if self._agent_start_cb:
            self._agent_start_cb(agent_id)

    def on_agent_complete(
        self, agent_id: int, success: bool, error: str = None
    ) -> None:
        if self._agent_complete_cb:
            self._agent_complete_cb(agent_id, success, error)

    def on_agent_token(self, agent_id: int, token: str) -> None:
        if self._agent_token_cb:
            self._agent_token_cb(agent_id, token)

    def on_agents_done(self) -> None:
        if self._progress:
            self._progress.stop()

    def on_coordinator_token(self, token: str) -> None:
        self._formatter.stream_token(token)


async def handle_query_command(args, remaining, formatter) -> int:
    query_text = get_query_text(args, remaining)
    if not query_text:
        formatter.error("No query provided")
        return 1

    # Build config overrides from CLI args
    cli_overrides = {
        k: v for k, v in {
            "num_agents": getattr(args, "num_agents", None),
            "model": getattr(args, "model", None),
            "temperature": getattr(args, "temperature", None),
            "aws_profile": getattr(args, "aws_profile", None),
            "region": getattr(args, "region", None),
            "max_concurrency": getattr(args, "max_concurrency", None),
        }.items() if v is not None
    }

    progress = create_progress_display(...)
    callbacks = CLICallbacks(formatter, progress)

    if progress:
        progress.start()

    try:
        result = await run_query(
            query_text,
            config_overrides=cli_overrides or None,
            callbacks=callbacks if not args.quiet else None,
        )

        formatter.stream_end()
        formatter.agent_summary(result.successful_agents, result.total_agents)
        formatter.final_answer(result.final_answer, result.confidence)
        return 0

    except ToxpError as e:
        formatter.error(e.format_for_user(verbose=args.verbose))
        return 1
    finally:
        if progress:
            progress.stop()
```

This reduces `handle_query_command` from ~120 lines to ~40 lines. All the provider/orchestrator wiring is now in `api.run_query()`.

---

## 4. What Does NOT Change

- CLI user-facing behavior (commands, flags, output format) — identical
- Config file format and location (`~/.toxp/config.json`)
- Exception hierarchy (already well-designed)
- Provider registry and plugin architecture
- Agent/Coordinator prompt templates
- Session logging

---

## 5. What Does NOT Belong in toxp

These stay exclusively in the GUI project:

- Flet UI controls, views, themes
- Window management / "New Session" process spawning
- Query history persistence (local JSON)
- First-run setup wizard UI
- Dark/light mode, keyboard shortcuts

---

## 6. Migration Path

### For existing CLI users
Zero impact. `toxp "question"` works identically. The CLI just calls `api.run_query()` internally now.

### For GUI/library consumers
```python
# Before (v0.4.1) — reaching into internals:
from toxp.config import ConfigManager
from toxp.orchestrator import Orchestrator
from toxp.providers.registry import ProviderRegistry
# ... 40 lines of wiring ...

# After (v0.5.0) — clean public API:
from toxp import run_query, QueryResult, validate_credentials

result = await run_query("question", config_overrides={"num_agents": 8})
print(result.final_answer)
```

---

## 7. Implementation Plan

| Day | Task | Files |
|-----|------|-------|
| 1 AM | Create `toxp/api.py` with `run_query`, `QueryCallbacks`, `QueryResult`, `validate_credentials` | `api.py` |
| 1 PM | Add `cancel_token` to `Orchestrator.process_query()` | `orchestrator.py` |
| 1 PM | Add `ConfigManager.load_with_overrides()` | `config.py` |
| 2 AM | Refactor `cli.py` to use `api.run_query()` via `CLICallbacks` | `cli.py` |
| 2 AM | Update `__init__.py` exports | `__init__.py` |
| 2 PM | Tests: unit tests for `api.run_query()`, `validate_credentials()`, cancellation | `tests/` |
| 2 PM | Update README with "Library Usage" section, bump to v0.5.0 | `README.md`, `pyproject.toml` |

Estimated effort: 2 days for a senior engineer familiar with the codebase.

---

## 8. Testing Strategy

```python
# tests/test_api.py

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from toxp.api import run_query, QueryResult, NoOpCallbacks, validate_credentials


class TestRunQuery:
    """Tests for the public run_query API."""

    @pytest.mark.asyncio
    async def test_run_query_returns_query_result(self, mock_orchestrator):
        """run_query returns a QueryResult, not a raw Result."""
        result = await run_query("What is 2+2?")
        assert isinstance(result, QueryResult)
        assert result.confidence in ("Low", "Medium", "High")

    @pytest.mark.asyncio
    async def test_config_overrides_not_persisted(self, tmp_config):
        """Config overrides for a query do not mutate the config file."""
        original = get_default_config()
        await run_query("test", config_overrides={"num_agents": 2})
        after = get_default_config()
        assert original == after

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self, mock_orchestrator):
        """Callbacks are invoked during query execution."""
        cb = MagicMock(spec=NoOpCallbacks)
        await run_query("test", callbacks=cb)
        cb.on_agent_start.assert_called()
        cb.on_coordinator_token.assert_called()

    @pytest.mark.asyncio
    async def test_cancel_token_stops_execution(self, mock_orchestrator):
        """Setting cancel_token stops the query gracefully."""
        cancel = asyncio.Event()
        cancel.set()  # Cancel immediately
        with pytest.raises(asyncio.CancelledError):
            await run_query("test", cancel_token=cancel)

    @pytest.mark.asyncio
    async def test_credentials_error_propagates(self):
        """CredentialsError from provider is not swallowed."""
        with pytest.raises(CredentialsError):
            await run_query(
                "test",
                config_overrides={"aws_profile": "nonexistent-profile-xyz"},
            )


class TestValidateCredentials:
    """Tests for credential validation helper."""

    def test_valid_credentials(self, valid_aws_profile):
        assert validate_credentials(aws_profile=valid_aws_profile) is True

    def test_invalid_profile_raises(self):
        with pytest.raises(CredentialsError):
            validate_credentials(aws_profile="nonexistent-profile-xyz")
```

---

## 9. Version Bump Checklist

- [ ] `pyproject.toml`: `version = "0.5.0"`
- [ ] `toxp/__init__.py`: `__version__ = "0.5.0"`
- [ ] `AGENTS.md`: Update if needed
- [ ] Git tag: `v0.5.0`
- [ ] Commit message: `feat: add public library API (run_query, QueryCallbacks, QueryResult)`
