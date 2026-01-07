"""Integration tests for TOXP CLI.

These tests verify end-to-end functionality of the TOXP CLI tool,
including query flow, config commands, and error scenarios.

Feature: toxp-cli
Requirements: 1.1-1.4, 2.1-2.10, 3.1-3.5, 9.1-9.9, 10.1-10.8
"""

import asyncio
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import AsyncIterator, List, Optional
from unittest.mock import patch

import pytest

from toxp.cli import (
    create_parser,
    parse_args,
    get_query_text,
    handle_config_command,
    handle_query_command,
    main,
)
from toxp.config import ConfigManager, ToxpConfig
from toxp.exceptions import (
    CredentialsError,
    InsufficientAgentsError,
    ModelNotFoundError,
    ThrottlingError,
)
from toxp.models.query import Query
from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.models.result import Result
from toxp.orchestrator import Orchestrator
from toxp.output.formatter import OutputFormatter
from toxp.providers.base import BaseProvider, ProviderResponse
from toxp.providers.registry import ProviderRegistry


# =============================================================================
# Mock Provider for Integration Tests
# =============================================================================

class IntegrationMockProvider(BaseProvider):
    """Mock provider for integration testing without real API calls."""
    
    def __init__(
        self,
        model_id: str = "test-model",
        region: str = "us-east-1",
        aws_profile: str = "default",
        should_fail: bool = False,
        fail_count: int = 0,
        response_text: str = "This is a test response with reasoning.\n\nFinal Answer: 42",
        coordinator_response: str = None,
    ):
        self._model_id = model_id
        self._name = "mock"
        self.region = region
        self.aws_profile = aws_profile
        self.should_fail = should_fail
        self.fail_count = fail_count
        self._current_fail_count = 0
        self.response_text = response_text
        self.coordinator_response = coordinator_response or (
            "**Consensus Summary**: All agents agree.\n\n"
            "**Key Debates**: None.\n\n"
            "**Final Synthesized Answer**: The answer is 42.\n\n"
            "**Confidence Level**: High"
        )
        self.call_count = 0
        self.calls: List[dict] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    async def invoke_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        self.call_count += 1
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        
        if self.should_fail:
            if self._current_fail_count < self.fail_count:
                self._current_fail_count += 1
                raise RuntimeError("Mock provider failure")
        
        await asyncio.sleep(0)
        
        # Determine if this is a coordinator call (contains agent outputs)
        is_coordinator = "Agent" in system_prompt and "synthesis" in user_message.lower()
        
        text = self.coordinator_response if is_coordinator else self.response_text
        
        return ProviderResponse(
            text=text,
            input_tokens=100,
            output_tokens=50,
            latency_ms=100.0,
            model_id=self._model_id,
        )
    
    async def invoke_model_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        self.call_count += 1
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": True,
        })
        
        if self.should_fail:
            raise RuntimeError("Mock provider failure")
        
        for char in self.coordinator_response:
            yield char
            await asyncio.sleep(0)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_manager(temp_config_dir):
    """Create a ConfigManager with temporary directory."""
    return ConfigManager(
        config_dir=temp_config_dir,
        config_file=temp_config_dir / "config.json"
    )


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    return IntegrationMockProvider()


@pytest.fixture
def output_formatter():
    """Create an output formatter with captured output."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    return OutputFormatter(
        quiet=False,
        verbose=False,
        use_color=False,
        stdout=stdout,
        stderr=stderr,
    )


# =============================================================================
# Config Command Integration Tests
# =============================================================================

class TestConfigCommandFlow:
    """Integration tests for config command flow.
    
    Tests the complete config command workflow including set, get, show,
    path, and reset operations.
    """

    def test_config_set_and_get_flow(self, temp_config_dir):
        """Test setting and getting config values end-to-end."""
        config_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        
        # Set a value
        config_manager.set("num-agents", "8")
        
        # Get the value back
        value = config_manager.get("num-agents")
        assert value == 8
        
        # Verify it persists
        new_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        assert new_manager.get("num-agents") == 8

    def test_config_show_displays_all_values(self, temp_config_dir):
        """Test that config show returns all configuration values."""
        config_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        
        # Set some values
        config_manager.set("num-agents", "12")
        config_manager.set("temperature", "0.8")
        
        # Show all config
        config = config_manager.show()
        
        # Verify all expected keys are present
        expected_keys = {
            "provider", "aws_profile", "region", "model", "num_agents",
            "temperature", "coordinator_temperature", "max_tokens",
            "log_enabled", "log_retention_days", "max_concurrency"
        }
        assert set(config.keys()) == expected_keys
        
        # Verify our set values
        assert config["num_agents"] == 12
        assert config["temperature"] == 0.8

    def test_config_reset_restores_defaults(self, temp_config_dir):
        """Test that config reset restores all default values."""
        config_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        
        # Modify several values
        config_manager.set("num-agents", "4")
        config_manager.set("temperature", "0.5")
        config_manager.set("provider", "custom")
        
        # Reset
        config_manager.reset()
        
        # Verify defaults are restored
        defaults = ToxpConfig.get_defaults()
        config = config_manager.load()
        
        assert config.num_agents == defaults.num_agents
        assert config.temperature == defaults.temperature
        assert config.provider == defaults.provider

    def test_config_path_returns_correct_path(self, temp_config_dir):
        """Test that config path returns the correct file path."""
        config_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        
        assert config_manager.config_path == temp_config_dir / "config.json"

    def test_config_invalid_key_raises_error(self, temp_config_dir):
        """Test that setting an invalid key raises KeyError."""
        config_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        
        with pytest.raises(KeyError) as exc_info:
            config_manager.set("invalid-key", "value")
        
        assert "invalid-key" in str(exc_info.value)

    def test_config_invalid_value_raises_error(self, temp_config_dir):
        """Test that setting an invalid value raises ValueError."""
        config_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        
        # num-agents must be 2-32
        with pytest.raises(ValueError):
            config_manager.set("num-agents", "100")

    def test_config_precedence_cli_over_env_over_file(self, temp_config_dir):
        """Test configuration precedence: CLI > ENV > file > defaults."""
        import os
        
        config_manager = ConfigManager(
            config_dir=temp_config_dir,
            config_file=temp_config_dir / "config.json"
        )
        
        # Set file value
        config_manager.set("aws-profile", "file-profile")
        
        # Set env value
        old_env = os.environ.get("TOXP_AWS_PROFILE")
        try:
            os.environ["TOXP_AWS_PROFILE"] = "env-profile"
            
            # Load config (should have env value)
            config = config_manager.load()
            assert config.aws_profile == "env-profile"
            
            # Apply CLI override
            config = config_manager.apply_overrides(config, {"aws-profile": "cli-profile"})
            assert config.aws_profile == "cli-profile"
            
        finally:
            if old_env is None:
                os.environ.pop("TOXP_AWS_PROFILE", None)
            else:
                os.environ["TOXP_AWS_PROFILE"] = old_env


# =============================================================================
# Query Flow Integration Tests
# =============================================================================

class TestQueryFlow:
    """Integration tests for end-to-end query processing flow."""

    def test_query_from_positional_argument(self):
        """Test query input from positional argument."""
        args, remaining = parse_args(["What is 2+2?"])
        query = get_query_text(args, remaining)
        assert query == "What is 2+2?"

    def test_query_from_query_flag(self):
        """Test query input from --query flag."""
        args, remaining = parse_args(["--query", "What is 2+2?"])
        query = get_query_text(args, remaining)
        assert query == "What is 2+2?"

    def test_query_flag_short_form(self):
        """Test query input from -q flag."""
        args, remaining = parse_args(["-q", "What is 2+2?"])
        query = get_query_text(args, remaining)
        assert query == "What is 2+2?"

    def test_positional_takes_precedence_over_flag(self):
        """Test that positional argument takes precedence over --query flag."""
        args, remaining = parse_args(["positional query", "--query", "flag query"])
        query = get_query_text(args, remaining)
        assert query == "positional query"

    def test_end_to_end_query_with_mock_provider(self, temp_config_dir):
        """Test complete query flow with mocked provider."""
        # Setup
        mock_provider = IntegrationMockProvider()
        
        async def run_test():
            orchestrator = Orchestrator(
                provider=mock_provider,
                num_agents=4,  # Small number for faster tests
            )
            
            query = Query(text="What is the meaning of life?")
            result = await orchestrator.process_query(query)
            
            # Verify result structure
            assert result.query == query
            assert len(result.agent_responses) == 4
            assert result.coordinator_response is not None
            assert result.coordinator_response.confidence in ["Low", "Medium", "High"]
            
            # Verify metadata
            assert result.metadata["num_agents"] == 4
            assert result.metadata["model_id"] == "test-model"
            
            return result
        
        result = asyncio.run(run_test())
        assert result is not None

    def test_query_with_streaming_output(self, temp_config_dir):
        """Test query with streaming coordinator output."""
        mock_provider = IntegrationMockProvider()
        streamed_tokens = []
        
        def on_token(token: str):
            streamed_tokens.append(token)
        
        async def run_test():
            orchestrator = Orchestrator(
                provider=mock_provider,
                num_agents=2,
            )
            
            query = Query(text="Test query")
            result = await orchestrator.process_query(
                query,
                on_coordinator_token=on_token,
            )
            
            return result
        
        result = asyncio.run(run_test())
        
        # Verify streaming occurred
        assert len(streamed_tokens) > 0
        # Verify streamed content matches coordinator response
        streamed_content = "".join(streamed_tokens)
        assert "Confidence Level" in streamed_content

    def test_query_with_output_file(self, temp_config_dir):
        """Test writing query result to output file."""
        output_file = temp_config_dir / "output.txt"
        mock_provider = IntegrationMockProvider()
        
        async def run_test():
            orchestrator = Orchestrator(
                provider=mock_provider,
                num_agents=2,
            )
            
            query = Query(text="Test query")
            result = await orchestrator.process_query(query)
            
            # Write to file
            output_file.write_text(result.coordinator_response.final_answer)
            
            return result
        
        asyncio.run(run_test())
        
        # Verify file was written
        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0


# =============================================================================
# Error Scenario Integration Tests
# =============================================================================

class TestErrorScenarios:
    """Integration tests for error handling scenarios."""

    def test_insufficient_agents_error(self):
        """Test error when too few agents succeed."""
        # Create a provider that fails most calls
        class FailingProvider(IntegrationMockProvider):
            def __init__(self):
                super().__init__()
                self._call_count = 0
            
            async def invoke_model(self, *args, **kwargs):
                self._call_count += 1
                # Fail all but one agent
                if self._call_count <= 3:
                    raise RuntimeError("Simulated failure")
                return await super().invoke_model(*args, **kwargs)
        
        provider = FailingProvider()
        
        async def run_test():
            orchestrator = Orchestrator(
                provider=provider,
                num_agents=4,
            )
            
            query = Query(text="Test query")
            return await orchestrator.process_query(query)
        
        with pytest.raises(InsufficientAgentsError) as exc_info:
            asyncio.run(run_test())
        
        error = exc_info.value
        assert error.successful_count < error.total_count * 0.5

    def test_credentials_error_formatting(self):
        """Test that credentials error provides helpful message."""
        error = CredentialsError("AWS credentials not found")
        formatted = error.format_for_user()
        
        assert "AWS credentials not found" in formatted
        assert "aws configure" in formatted.lower() or "AWS_ACCESS_KEY_ID" in formatted

    def test_model_not_found_error_formatting(self):
        """Test that model not found error provides helpful message."""
        error = ModelNotFoundError(
            model_id="invalid-model",
            region="us-east-1",
        )
        formatted = error.format_for_user()
        
        assert "invalid-model" in formatted
        assert "us-east-1" in formatted

    def test_throttling_error_formatting(self):
        """Test that throttling error provides helpful message."""
        error = ThrottlingError(
            message="Rate limit exceeded",
            retry_count=3,
            num_agents=16,
        )
        formatted = error.format_for_user()
        
        assert "Rate limit" in formatted
        assert "num-agents" in formatted.lower() or "reduce" in formatted.lower()

    def test_no_query_provided_error(self):
        """Test error when no query is provided."""
        args, remaining = parse_args([])
        query = get_query_text(args, remaining)
        assert query is None


# =============================================================================
# CLI Argument Parsing Tests
# =============================================================================

class TestCLIArgumentParsing:
    """Integration tests for CLI argument parsing."""

    def test_verbose_flag(self):
        """Test --verbose flag is parsed correctly."""
        args, _ = parse_args(["--verbose", "test query"])
        assert args.verbose is True

    def test_quiet_flag(self):
        """Test --quiet flag is parsed correctly."""
        args, _ = parse_args(["--quiet", "test query"])
        assert args.quiet is True

    def test_no_log_flag(self):
        """Test --no-log flag is parsed correctly."""
        args, _ = parse_args(["--no-log", "test query"])
        assert args.no_log is True

    def test_num_agents_option(self):
        """Test --num-agents option is parsed correctly."""
        args, _ = parse_args(["--num-agents", "8", "test query"])
        assert args.num_agents == 8

    def test_temperature_option(self):
        """Test --temperature option is parsed correctly."""
        args, _ = parse_args(["--temperature", "0.5", "test query"])
        assert args.temperature == 0.5

    def test_model_option(self):
        """Test --model option is parsed correctly."""
        args, _ = parse_args(["--model", "custom-model", "test query"])
        assert args.model == "custom-model"

    def test_aws_profile_option(self):
        """Test --aws-profile option is parsed correctly."""
        args, _ = parse_args(["--aws-profile", "my-profile", "test query"])
        assert args.aws_profile == "my-profile"

    def test_region_option(self):
        """Test --region option is parsed correctly."""
        args, _ = parse_args(["--region", "eu-west-1", "test query"])
        assert args.region == "eu-west-1"

    def test_output_option(self):
        """Test --output option is parsed correctly."""
        args, _ = parse_args(["--output", "result.txt", "test query"])
        assert args.output_file == "result.txt"

    def test_config_subcommand_set(self):
        """Test config set subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["config", "set", "num-agents", "8"])
        assert args.command == "config"
        assert args.config_action == "set"
        assert args.key == "num-agents"
        assert args.value == "8"

    def test_config_subcommand_get(self):
        """Test config get subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["config", "get", "model"])
        assert args.command == "config"
        assert args.config_action == "get"
        assert args.key == "model"

    def test_config_subcommand_show(self):
        """Test config show subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["config", "show"])
        assert args.command == "config"
        assert args.config_action == "show"

    def test_config_subcommand_reset(self):
        """Test config reset subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["config", "reset"])
        assert args.command == "config"
        assert args.config_action == "reset"

    def test_config_subcommand_path(self):
        """Test config path subcommand parsing."""
        parser = create_parser()
        args = parser.parse_args(["config", "path"])
        assert args.command == "config"
        assert args.config_action == "path"


# =============================================================================
# Output Formatter Integration Tests
# =============================================================================

class TestOutputFormatterIntegration:
    """Integration tests for output formatting."""

    def test_quiet_mode_suppresses_info(self):
        """Test that quiet mode suppresses info messages."""
        stdout = io.StringIO()
        formatter = OutputFormatter(quiet=True, stdout=stdout, use_color=False)
        
        formatter.info("This should not appear")
        formatter.success("This should not appear either")
        
        output = stdout.getvalue()
        assert output == ""

    def test_quiet_mode_shows_final_answer(self):
        """Test that quiet mode still shows final answer."""
        stdout = io.StringIO()
        formatter = OutputFormatter(quiet=True, stdout=stdout, use_color=False)
        
        formatter.final_answer("The answer is 42")
        
        output = stdout.getvalue()
        assert "42" in output

    def test_verbose_mode_shows_debug(self):
        """Test that verbose mode shows debug messages."""
        stdout = io.StringIO()
        formatter = OutputFormatter(verbose=True, stdout=stdout, use_color=False)
        
        formatter.debug("Debug message")
        
        output = stdout.getvalue()
        assert "Debug message" in output

    def test_error_goes_to_stderr(self):
        """Test that errors go to stderr."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        formatter = OutputFormatter(stdout=stdout, stderr=stderr, use_color=False)
        
        formatter.error("Error message")
        
        assert "Error message" in stderr.getvalue()
        assert "Error message" not in stdout.getvalue()

    def test_streaming_output(self):
        """Test streaming token output."""
        stdout = io.StringIO()
        formatter = OutputFormatter(stdout=stdout, use_color=False)
        
        formatter.stream_token("Hello")
        formatter.stream_token(" ")
        formatter.stream_token("World")
        formatter.stream_end()
        
        output = stdout.getvalue()
        assert "Hello World" in output

    def test_agent_summary_all_success(self):
        """Test agent summary with all agents successful."""
        stdout = io.StringIO()
        formatter = OutputFormatter(stdout=stdout, use_color=False)
        
        formatter.agent_summary(successful=8, total=8)
        
        output = stdout.getvalue()
        assert "8" in output
        assert "success" in output.lower()

    def test_agent_summary_partial_failure(self):
        """Test agent summary with some failures."""
        stdout = io.StringIO()
        formatter = OutputFormatter(stdout=stdout, use_color=False)
        
        formatter.agent_summary(successful=6, total=8)
        
        output = stdout.getvalue()
        assert "6" in output
        assert "8" in output


# =============================================================================
# Session Logging Integration Tests
# =============================================================================

class TestSessionLoggingIntegration:
    """Integration tests for session logging."""

    def test_session_log_created_on_query(self, temp_config_dir):
        """Test that session log is created after query."""
        from toxp.logging.session_logger import SessionLogger
        
        logs_dir = temp_config_dir / "logs" / "sessions"
        logger = SessionLogger(
            enabled=True,
            retention_days=30,
            logs_dir=logs_dir,
        )
        
        # Create a mock result
        result = Result(
            query=Query(text="Test query"),
            agent_responses=[
                AgentResponse(
                    agent_id=0,
                    success=True,
                    chain_of_thought="Thinking...",
                    final_answer="42",
                    duration_seconds=1.0,
                    token_count=100,
                ),
            ],
            coordinator_response=CoordinatorResponse(
                synthesis="Synthesis",
                confidence="High",
                final_answer="42",
                duration_seconds=0.5,
            ),
            metadata={"model_id": "test-model"},
        )
        
        # Log the session
        log_path = logger.log_session(result)
        
        # Verify log was created
        assert log_path is not None
        assert log_path.exists()
        
        # Verify log content
        content = log_path.read_text()
        assert "Test query" in content
        assert "42" in content
        assert "High" in content

    def test_session_log_disabled(self, temp_config_dir):
        """Test that no log is created when logging is disabled."""
        from toxp.logging.session_logger import SessionLogger
        
        logs_dir = temp_config_dir / "logs" / "sessions"
        logger = SessionLogger(
            enabled=False,
            retention_days=30,
            logs_dir=logs_dir,
        )
        
        result = Result(
            query=Query(text="Test query"),
            agent_responses=[],
            coordinator_response=CoordinatorResponse(
                synthesis="",
                confidence="Low",
                final_answer="",
                duration_seconds=0.0,
            ),
            metadata={},
        )
        
        log_path = logger.log_session(result)
        
        assert log_path is None


# =============================================================================
# Provider Registry Integration Tests
# =============================================================================

class TestProviderRegistryIntegration:
    """Integration tests for provider registry."""

    def test_bedrock_provider_registered(self):
        """Test that Bedrock provider is registered by default."""
        # Clear and re-register
        ProviderRegistry.clear()
        from toxp.providers.bedrock import BedrockProvider
        ProviderRegistry.register("bedrock", BedrockProvider)
        
        assert ProviderRegistry.is_registered("bedrock")
        provider_class = ProviderRegistry.get("bedrock")
        assert provider_class == BedrockProvider

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        ProviderRegistry.clear()
        
        with pytest.raises(ValueError) as exc_info:
            ProviderRegistry.get("unknown-provider")
        
        assert "unknown-provider" in str(exc_info.value).lower()

    def test_list_providers(self):
        """Test listing registered providers."""
        ProviderRegistry.clear()
        ProviderRegistry.register("test1", IntegrationMockProvider)
        ProviderRegistry.register("test2", IntegrationMockProvider)
        
        providers = ProviderRegistry.list_providers()
        
        assert "test1" in providers
        assert "test2" in providers
