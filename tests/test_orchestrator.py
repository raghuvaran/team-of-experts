"""Property-based tests for TOXP orchestrator.

Feature: toxp-cli
"""

import asyncio
from typing import AsyncIterator, List
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, strategies as st, settings

from toxp.models.query import Query
from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.orchestrator import Orchestrator, InsufficientAgentsError
from toxp.providers.base import BaseProvider, ProviderResponse


class MockProvider(BaseProvider):
    """Mock provider for testing orchestrator behavior."""
    
    def __init__(
        self,
        model_id: str = "test-model",
        response_text: str = "Test response",
        should_fail: bool = False,
        fail_probability: float = 0.0,
    ):
        self._model_id = model_id
        self._name = "mock"
        self.response_text = response_text
        self.should_fail = should_fail
        self.fail_probability = fail_probability
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
        
        # Simulate some async work
        await asyncio.sleep(0)
        
        return ProviderResponse(
            text=f"{self.response_text}\n\nFinal Answer: Test answer {self.call_count}",
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
        
        response = f"{self.response_text}\n\n**Final Synthesized Answer**: Test synthesis\n\n**Confidence Level**: High"
        for char in response:
            yield char
            await asyncio.sleep(0)


# Strategy for valid num_agents values (2-32)
valid_num_agents = st.integers(min_value=2, max_value=32)

# Strategy for invalid num_agents values
invalid_num_agents = st.one_of(
    st.integers(max_value=1),
    st.integers(min_value=33),
)


class TestAgentSpawnCount:
    """Property tests for agent spawn count.
    
    Property 8: Agent Spawn Count
    Validates: Requirements 6.1
    
    For any configured num_agents value N (where 2 ≤ N ≤ 32),
    the orchestrator SHALL spawn exactly N reasoning agents.
    """

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_spawns_exactly_n_agents(self, num_agents: int) -> None:
        """Property: Orchestrator spawns exactly num_agents agents.
        
        Property 8: Agent Spawn Count
        Validates: Requirements 6.1
        """
        provider = MockProvider()
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            result = await orchestrator.process_query(query)
            return result
        
        result = asyncio.run(run_test())
        
        # Verify exactly num_agents responses were collected
        assert len(result.agent_responses) == num_agents, (
            f"Expected {num_agents} agent responses, got {len(result.agent_responses)}"
        )

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_agent_ids_are_sequential(self, num_agents: int) -> None:
        """Property: Agent IDs are sequential from 0 to num_agents-1.
        
        Property 8: Agent Spawn Count
        Validates: Requirements 6.1
        """
        provider = MockProvider()
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            result = await orchestrator.process_query(query)
            return result
        
        result = asyncio.run(run_test())
        
        # Extract agent IDs and sort them
        agent_ids = sorted([r.agent_id for r in result.agent_responses])
        expected_ids = list(range(num_agents))
        
        assert agent_ids == expected_ids, (
            f"Expected agent IDs {expected_ids}, got {agent_ids}"
        )

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_orchestrator_stores_num_agents(self, num_agents: int) -> None:
        """Property: Orchestrator stores the configured num_agents value.
        
        Property 8: Agent Spawn Count
        Validates: Requirements 6.1
        """
        provider = MockProvider()
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        assert orchestrator.num_agents == num_agents

    @given(num_agents=invalid_num_agents)
    @settings(max_examples=100)
    def test_rejects_invalid_num_agents(self, num_agents: int) -> None:
        """Property: Orchestrator rejects num_agents outside valid range.
        
        Property 8: Agent Spawn Count
        Validates: Requirements 6.1
        """
        provider = MockProvider()
        
        with pytest.raises(ValueError) as exc_info:
            Orchestrator(
                provider=provider,
                num_agents=num_agents,
            )
        
        assert "num_agents must be between" in str(exc_info.value)

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_metadata_contains_num_agents(self, num_agents: int) -> None:
        """Property: Result metadata contains correct num_agents.
        
        Property 8: Agent Spawn Count
        Validates: Requirements 6.1
        """
        provider = MockProvider()
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            result = await orchestrator.process_query(query)
            return result
        
        result = asyncio.run(run_test())
        
        assert result.metadata["num_agents"] == num_agents


class PartialFailureProvider(BaseProvider):
    """Provider that fails for specific agent IDs."""
    
    def __init__(
        self,
        model_id: str = "test-model",
        fail_agent_ids: List[int] = None,
    ):
        self._model_id = model_id
        self._name = "partial-failure"
        self.fail_agent_ids = fail_agent_ids or []
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
        current_call = self.call_count
        self.call_count += 1
        
        # Fail for specific agent IDs
        if current_call in self.fail_agent_ids:
            raise RuntimeError(f"Simulated failure for agent {current_call}")
        
        await asyncio.sleep(0)
        
        return ProviderResponse(
            text=f"Response from agent {current_call}\n\nFinal Answer: Answer {current_call}",
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
        
        response = "**Final Synthesized Answer**: Synthesis\n\n**Confidence Level**: Medium"
        for char in response:
            yield char
            await asyncio.sleep(0)


class TestSuccessRateValidation:
    """Property tests for success rate validation.
    
    Property 10: Success Rate Validation
    Validates: Requirements 6.5
    
    For any set of agent responses where the success rate is below 50%,
    the orchestrator SHALL raise an InsufficientAgentsError.
    """

    @given(
        num_agents=st.integers(min_value=4, max_value=16),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_raises_error_when_below_50_percent(
        self, num_agents: int, data: st.DataObject
    ) -> None:
        """Property: Raises InsufficientAgentsError when success rate < 50%.
        
        Property 10: Success Rate Validation
        Validates: Requirements 6.5
        """
        # Calculate how many failures would put us below 50%
        # For 50% threshold, we need more than half to fail
        min_failures_for_error = (num_agents // 2) + 1
        
        # Generate a number of failures that will trigger the error
        num_failures = data.draw(
            st.integers(min_value=min_failures_for_error, max_value=num_agents)
        )
        
        # Select which agents will fail
        fail_agent_ids = list(range(num_failures))
        
        provider = PartialFailureProvider(fail_agent_ids=fail_agent_ids)
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            return await orchestrator.process_query(query)
        
        with pytest.raises(InsufficientAgentsError) as exc_info:
            asyncio.run(run_test())
        
        error = exc_info.value
        assert error.successful_count < error.total_count * 0.5
        assert error.total_count == num_agents

    @given(
        num_agents=st.integers(min_value=4, max_value=16),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_succeeds_when_at_or_above_50_percent(
        self, num_agents: int, data: st.DataObject
    ) -> None:
        """Property: Succeeds when success rate >= 50%.
        
        Property 10: Success Rate Validation
        Validates: Requirements 6.5
        """
        # Calculate max failures that still allow success (< 50% failures)
        max_failures_for_success = num_agents // 2
        
        # Generate a number of failures that won't trigger the error
        num_failures = data.draw(
            st.integers(min_value=0, max_value=max_failures_for_success)
        )
        
        # Select which agents will fail
        fail_agent_ids = list(range(num_failures))
        
        provider = PartialFailureProvider(fail_agent_ids=fail_agent_ids)
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            return await orchestrator.process_query(query)
        
        # Should not raise an error
        result = asyncio.run(run_test())
        
        # Verify success rate is at or above 50%
        successful = len([r for r in result.agent_responses if r.success])
        success_rate = successful / num_agents
        assert success_rate >= 0.5

    @given(num_agents=valid_num_agents)
    @settings(max_examples=100)
    def test_all_agents_succeed_passes_validation(self, num_agents: int) -> None:
        """Property: 100% success rate always passes validation.
        
        Property 10: Success Rate Validation
        Validates: Requirements 6.5
        """
        provider = MockProvider()
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            return await orchestrator.process_query(query)
        
        result = asyncio.run(run_test())
        
        # All agents should succeed
        successful = len([r for r in result.agent_responses if r.success])
        assert successful == num_agents
        assert result.metadata["success_rate"] == 1.0

    @given(num_agents=st.integers(min_value=4, max_value=16))
    @settings(max_examples=100)
    def test_exactly_50_percent_passes(self, num_agents: int) -> None:
        """Property: Exactly 50% success rate passes validation.
        
        Property 10: Success Rate Validation
        Validates: Requirements 6.5
        """
        # Fail exactly half the agents
        num_failures = num_agents // 2
        fail_agent_ids = list(range(num_failures))
        
        provider = PartialFailureProvider(fail_agent_ids=fail_agent_ids)
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            return await orchestrator.process_query(query)
        
        # Should not raise an error at exactly 50%
        result = asyncio.run(run_test())
        
        successful = len([r for r in result.agent_responses if r.success])
        success_rate = successful / num_agents
        assert success_rate >= 0.5

    @given(num_agents=st.integers(min_value=4, max_value=16))
    @settings(max_examples=100)
    def test_error_contains_correct_counts(self, num_agents: int) -> None:
        """Property: InsufficientAgentsError contains accurate counts.
        
        Property 10: Success Rate Validation
        Validates: Requirements 6.5
        """
        # Fail all agents to guarantee error
        fail_agent_ids = list(range(num_agents))
        
        provider = PartialFailureProvider(fail_agent_ids=fail_agent_ids)
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            return await orchestrator.process_query(query)
        
        with pytest.raises(InsufficientAgentsError) as exc_info:
            asyncio.run(run_test())
        
        error = exc_info.value
        assert error.successful_count == 0
        assert error.total_count == num_agents
        assert error.min_required == num_agents // 2  # 50% of total

    @given(
        num_agents=st.integers(min_value=4, max_value=16),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_metadata_tracks_success_rate(
        self, num_agents: int, data: st.DataObject
    ) -> None:
        """Property: Result metadata accurately tracks success rate.
        
        Property 10: Success Rate Validation
        Validates: Requirements 6.5
        """
        # Generate some failures but stay above 50%
        max_failures = num_agents // 2
        num_failures = data.draw(st.integers(min_value=0, max_value=max_failures))
        fail_agent_ids = list(range(num_failures))
        
        provider = PartialFailureProvider(fail_agent_ids=fail_agent_ids)
        orchestrator = Orchestrator(
            provider=provider,
            num_agents=num_agents,
        )
        
        query = Query(text="Test query")
        
        async def run_test():
            return await orchestrator.process_query(query)
        
        result = asyncio.run(run_test())
        
        expected_successful = num_agents - num_failures
        expected_rate = expected_successful / num_agents
        
        assert result.metadata["successful_agents"] == expected_successful
        assert result.metadata["failed_agents"] == num_failures
        assert abs(result.metadata["success_rate"] - expected_rate) < 0.001
