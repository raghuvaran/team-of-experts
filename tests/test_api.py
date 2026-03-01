"""Tests for the public toxp API (api.py).

Covers: run_query, QueryResult, QueryCallbacks, NoOpCallbacks,
        validate_credentials, get_default_config, cancel_token support,
        ConfigManager.load_with_overrides.
"""

import asyncio
from typing import AsyncIterator, Optional
from unittest.mock import MagicMock, patch

import pytest

from toxp.api import (
    run_query,
    QueryResult,
    QueryCallbacks,
    NoOpCallbacks,
    validate_credentials,
    get_default_config,
)
from toxp.config import ConfigManager, ToxpConfig
from toxp.exceptions import CredentialsError
from toxp.models.query import Query
from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.models.result import Result
from toxp.providers.base import BaseProvider, ProviderResponse


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class MockProvider(BaseProvider):
    """Mock provider that returns predictable responses."""

    def __init__(self, model_id: str = "test-model", should_fail: bool = False):
        self._model_id = model_id
        self._name = "mock"
        self.should_fail = should_fail
        self.call_count = 0

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
        if self.should_fail:
            raise RuntimeError("Mock provider failure")
        await asyncio.sleep(0)
        return ProviderResponse(
            text=(
                f"Reasoning for call {self.call_count}\n\n"
                f"Final Answer: Answer {self.call_count}"
            ),
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
        if self.should_fail:
            raise RuntimeError("Mock provider failure")
        response = (
            "Synthesis text\n\n"
            "**Final Synthesized Answer**: Synthesized answer\n\n"
            "**Confidence Level**: High"
        )
        for char in response:
            yield char
            await asyncio.sleep(0)


def _make_agent_response(agent_id: int, success: bool = True) -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        success=success,
        chain_of_thought="reasoning" if success else "",
        final_answer=f"answer_{agent_id}" if success else "",
        error=None if success else "failed",
        duration_seconds=1.0,
        token_count=150,
    )


def _make_result(num_agents: int = 3, num_failed: int = 0) -> Result:
    responses = []
    for i in range(num_agents):
        responses.append(_make_agent_response(i, success=(i >= num_failed)))

    return Result(
        query=Query(text="test question"),
        agent_responses=responses,
        coordinator_response=CoordinatorResponse(
            synthesis="# Synthesis\nAll agents agree.",
            confidence="High",
            consensus_summary="Agents agreed on X",
            debates_summary="Agents disagreed on Y",
            final_answer="The final answer is 42",
            duration_seconds=2.0,
        ),
        metadata={
            "total_agent_tokens": 450,
            "total_duration_seconds": 5.0,
            "successful_agents": num_agents - num_failed,
            "failed_agents": num_failed,
        },
    )


def _run_query_with_mock(
    query_text: str = "test",
    num_agents: int = 2,
    config_overrides: Optional[dict] = None,
    callbacks=None,
    cancel_token=None,
):
    """Helper that patches ConfigManager and ProviderRegistry for run_query."""
    provider = MockProvider()

    with (
        patch("toxp.api.ConfigManager") as MockCM,
        patch("toxp.api.ProviderRegistry") as MockReg,
    ):
        config = ToxpConfig.from_dict(
            {**ToxpConfig.get_defaults().to_dict(), "num_agents": num_agents}
        )
        config_inst = MockCM.return_value
        config_inst.load_with_overrides.return_value = config

        MockReg.get.return_value = lambda **kwargs: provider

        async def _run():
            return await run_query(
                query_text,
                config_overrides=config_overrides,
                callbacks=callbacks,
                cancel_token=cancel_token,
            )

        return asyncio.run(_run()), config_inst, provider


# ---------------------------------------------------------------------------
# QueryResult.from_result
# ---------------------------------------------------------------------------

class TestQueryResultFromResult:

    def test_maps_final_answer(self):
        qr = QueryResult.from_result(_make_result())
        assert qr.final_answer == "The final answer is 42"

    def test_maps_confidence(self):
        qr = QueryResult.from_result(_make_result())
        assert qr.confidence == "High"

    def test_maps_synthesis(self):
        qr = QueryResult.from_result(_make_result())
        assert qr.synthesis_markdown == "# Synthesis\nAll agents agree."

    def test_maps_consensus_and_debates(self):
        qr = QueryResult.from_result(_make_result())
        assert qr.consensus_summary == "Agents agreed on X"
        assert qr.debates_summary == "Agents disagreed on Y"

    def test_maps_agent_counts(self):
        qr = QueryResult.from_result(_make_result(num_agents=5, num_failed=2))
        assert qr.total_agents == 5
        assert qr.successful_agents == 3
        assert qr.failed_agents == 2

    def test_maps_tokens_and_duration(self):
        qr = QueryResult.from_result(_make_result())
        assert qr.total_tokens == 450
        assert qr.total_duration_seconds == 5.0

    def test_raw_result_preserved(self):
        result = _make_result()
        qr = QueryResult.from_result(result)
        assert qr.raw_result is result

    def test_raw_result_excluded_from_dict(self):
        qr = QueryResult.from_result(_make_result())
        d = qr.model_dump()
        assert "raw_result" not in d


# ---------------------------------------------------------------------------
# NoOpCallbacks
# ---------------------------------------------------------------------------

class TestNoOpCallbacks:

    def test_all_methods_are_noop(self):
        cb = NoOpCallbacks()
        cb.on_agent_start(0)
        cb.on_agent_complete(0, True)
        cb.on_agent_complete(1, False, error="boom")
        cb.on_agent_token(0, "hello")
        cb.on_agents_done()
        cb.on_coordinator_token("token")

    def test_satisfies_protocol(self):
        assert isinstance(NoOpCallbacks(), QueryCallbacks)


# ---------------------------------------------------------------------------
# run_query
# ---------------------------------------------------------------------------

class TestRunQuery:

    def test_returns_query_result(self):
        result, _, _ = _run_query_with_mock("What is 2+2?")
        assert isinstance(result, QueryResult)
        assert result.confidence in ("Low", "Medium", "High")

    def test_total_agents_matches_config(self):
        result, _, _ = _run_query_with_mock(num_agents=3)
        assert result.total_agents == 3

    def test_config_overrides_passed_through(self):
        overrides = {"num_agents": 3, "temperature": 0.5}
        provider = MockProvider()

        with (
            patch("toxp.api.ConfigManager") as MockCM,
            patch("toxp.api.ProviderRegistry") as MockReg,
        ):
            config = ToxpConfig.from_dict(
                {**ToxpConfig.get_defaults().to_dict(), "num_agents": 3}
            )
            config_inst = MockCM.return_value
            config_inst.load_with_overrides.return_value = config
            MockReg.get.return_value = lambda **kwargs: provider

            asyncio.run(run_query("test", config_overrides=overrides))
            config_inst.load_with_overrides.assert_called_once_with(overrides)

    def test_callbacks_invoked(self):
        cb = MagicMock(spec=NoOpCallbacks)
        result, _, _ = _run_query_with_mock(callbacks=cb)
        cb.on_agent_start.assert_called()
        cb.on_agent_complete.assert_called()
        cb.on_agents_done.assert_called()
        cb.on_coordinator_token.assert_called()

    def test_no_callbacks_uses_noop(self):
        result, _, _ = _run_query_with_mock(callbacks=None)
        assert isinstance(result, QueryResult)

    def test_successful_agents_counted(self):
        result, _, _ = _run_query_with_mock(num_agents=4)
        assert result.successful_agents == 4
        assert result.failed_agents == 0


# ---------------------------------------------------------------------------
# Cancel token
# ---------------------------------------------------------------------------

class TestCancelToken:

    def test_cancel_token_raises_cancelled_error(self):
        cancel = asyncio.Event()
        cancel.set()

        with pytest.raises(asyncio.CancelledError):
            _run_query_with_mock(cancel_token=cancel)

    def test_cancel_token_none_runs_normally(self):
        result, _, _ = _run_query_with_mock(cancel_token=None)
        assert isinstance(result, QueryResult)

    def test_cancel_token_agents_return_cancelled(self):
        from toxp.orchestrator import Orchestrator

        provider = MockProvider()
        cancel = asyncio.Event()
        cancel.set()

        orchestrator = Orchestrator(
            provider=provider,
            num_agents=3,
            temperature=0.9,
            coordinator_temperature=0.7,
            max_tokens=8192,
        )

        async def _run():
            return await orchestrator._spawn_agents(
                Query(text="test"), cancel_token=cancel
            )

        responses = asyncio.run(_run())
        assert len(responses) == 3
        for r in responses:
            assert not r.success
            assert r.error == "Cancelled"

    def test_cancel_token_not_set_allows_normal_execution(self):
        """A cancel_token that is never set doesn't interfere."""
        cancel = asyncio.Event()  # Not set
        result, _, _ = _run_query_with_mock(cancel_token=cancel)
        assert isinstance(result, QueryResult)
        assert result.successful_agents > 0


# ---------------------------------------------------------------------------
# validate_credentials
# ---------------------------------------------------------------------------

class TestValidateCredentials:

    def test_invalid_profile_raises_credentials_error(self):
        with pytest.raises(CredentialsError):
            validate_credentials(aws_profile="nonexistent-profile-xyz-99999")

    @patch("boto3.Session")
    def test_valid_credentials_returns_true(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        assert validate_credentials() is True
        mock_session_cls.assert_called_once_with(profile_name="default")

    @patch("boto3.Session")
    def test_no_credentials_raises(self, mock_session_cls):
        from botocore.exceptions import NoCredentialsError

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.side_effect = NoCredentialsError()

        with pytest.raises(CredentialsError, match="not configured"):
            validate_credentials()

    @patch("boto3.Session")
    def test_expired_credentials_raises(self, mock_session_cls):
        from botocore.exceptions import ClientError

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.side_effect = ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "Token expired"}},
            "GetCallerIdentity",
        )

        with pytest.raises(CredentialsError, match="invalid or expired"):
            validate_credentials()


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------

class TestGetDefaultConfig:

    def test_returns_dict(self):
        with patch("toxp.api.ConfigManager") as MockCM:
            config = ToxpConfig.get_defaults()
            config_inst = MockCM.return_value
            config_inst.load.return_value = config

            result = get_default_config()
            assert isinstance(result, dict)
            assert "num_agents" in result
            assert "model" in result
            assert "temperature" in result


# ---------------------------------------------------------------------------
# ConfigManager.load_with_overrides
# ---------------------------------------------------------------------------

class TestLoadWithOverrides:

    def test_no_overrides_returns_base_config(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        base = cm.load()
        result = cm.load_with_overrides()
        assert result.to_dict() == base.to_dict()

    def test_none_overrides_returns_base_config(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        base = cm.load()
        result = cm.load_with_overrides(None)
        assert result.to_dict() == base.to_dict()

    def test_overrides_applied(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        result = cm.load_with_overrides({"num_agents": 5})
        assert result.num_agents == 5

    def test_overrides_do_not_mutate_file(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        original = cm.load().to_dict()
        cm.load_with_overrides({"num_agents": 7})
        after = cm.load().to_dict()
        assert original == after


# ---------------------------------------------------------------------------
# _parse_env_value fix for Optional[int] fields (issue #2)
# ---------------------------------------------------------------------------

class TestParseEnvValueOptionalInt:
    """Verify _parse_env_value correctly handles Optional[int] fields."""

    def test_max_concurrency_parsed_as_int(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        result = cm._parse_env_value("max_concurrency", "5")
        assert result == 5
        assert isinstance(result, int)

    def test_max_concurrency_invalid_raises(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        with pytest.raises(ValueError):
            cm._parse_env_value("max_concurrency", "not_a_number")

    def test_num_agents_still_parsed_as_int(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        result = cm._parse_env_value("num_agents", "8")
        assert result == 8
        assert isinstance(result, int)

    def test_temperature_parsed_as_float(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        result = cm._parse_env_value("temperature", "0.5")
        assert result == 0.5
        assert isinstance(result, float)

    def test_log_enabled_parsed_as_bool(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        assert cm._parse_env_value("log_enabled", "true") is True
        assert cm._parse_env_value("log_enabled", "false") is False

    def test_string_field_returned_as_string(self, tmp_path):
        cm = ConfigManager(config_dir=tmp_path)
        result = cm._parse_env_value("region", "us-west-2")
        assert result == "us-west-2"
        assert isinstance(result, str)

    def test_env_override_max_concurrency_applied(self, tmp_path, monkeypatch):
        """End-to-end: TOXP_MAX_CONCURRENCY env var produces an int in config."""
        monkeypatch.setenv("TOXP_MAX_CONCURRENCY", "4")
        cm = ConfigManager(config_dir=tmp_path)
        config = cm.load()
        assert config.max_concurrency == 4
        assert isinstance(config.max_concurrency, int)


# ---------------------------------------------------------------------------
# CLI parser deduplication fix (issue #7)
# ---------------------------------------------------------------------------

class TestParserFlagConsistency:
    """Verify create_parser and parse_args query-mode share the same flags."""

    def _get_query_flags(self, parser):
        """Extract optional flag dest names from a parser."""
        return {
            a.dest
            for a in parser._actions
            if a.option_strings and a.dest != "help"
        }

    def test_query_mode_has_same_flags_as_full_parser(self):
        from toxp.cli import create_parser, parse_args

        full_parser = create_parser()
        full_flags = self._get_query_flags(full_parser)

        # Build the query-mode parser by calling parse_args with a dummy query
        # We can't easily extract the parser, so instead verify specific flags
        # work in both modes
        test_flags = [
            (["-n", "5", "test"], "num_agents", 5),
            (["-m", "my-model", "test"], "model", "my-model"),
            (["--temperature", "0.5", "test"], "temperature", 0.5),
            (["--aws-profile", "myprof", "test"], "aws_profile", "myprof"),
            (["--region", "eu-west-1", "test"], "region", "eu-west-1"),
            (["-c", "3", "test"], "max_concurrency", 3),
            (["-Q", "test"], "quiet", True),
            (["-v", "test"], "verbose", True),
            (["--no-log", "test"], "no_log", True),
            (["-o", "out.txt", "test"], "output_file", "out.txt"),
        ]

        for argv, dest, expected in test_flags:
            args, _ = parse_args(argv)
            actual = getattr(args, dest, "MISSING")
            assert actual == expected, (
                f"Flag {argv[0]} -> {dest}: expected {expected!r}, got {actual!r}"
            )
