"""Tests for multi-turn conversation support.

Verifies that conversation_history flows correctly from run_query() down to
the Bedrock Converse API, that agents and coordinator receive the right
context, and that the feature is fully backward-compatible with single-shot
usage.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import patch

import pytest

from toxp.api import run_query, QueryResult
from toxp.agents.coordinator import CoordinatorAgent
from toxp.agents.prompts import (
    COORDINATOR_SYSTEM_PROMPT,
    COORDINATOR_SYSTEM_PROMPT_MULTITURN,
    format_coordinator_prompt,
    _format_conversation_context,
)
from toxp.agents.reasoning import ReasoningAgent
from toxp.config import ConfigManager, ToxpConfig
from toxp.models.conversation import Message
from toxp.models.query import Query
from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.models.result import Result
from toxp.orchestrator import Orchestrator
from toxp.providers.base import BaseProvider, ProviderResponse
from toxp.providers.bedrock import BedrockProvider


# ---------------------------------------------------------------------------
# Capturing mock provider — records every call with full arguments
# ---------------------------------------------------------------------------

class CapturingProvider(BaseProvider):
    """Provider that records all calls so tests can inspect what was sent."""

    def __init__(self, model_id: str = "test-model"):
        self._model_id = model_id
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "capturing"

    @property
    def model_id(self) -> str:
        return self._model_id

    async def invoke_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        messages: Optional[List[Message]] = None,
    ) -> ProviderResponse:
        self.calls.append({
            "method": "invoke_model",
            "system_prompt": system_prompt,
            "user_message": user_message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        })
        await asyncio.sleep(0)
        return ProviderResponse(
            text="Reasoning\n\nFinal Answer: test answer",
            input_tokens=100,
            output_tokens=50,
            latency_ms=50.0,
            model_id=self._model_id,
        )

    async def invoke_model_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        messages: Optional[List[Message]] = None,
    ) -> AsyncIterator[str]:
        self.calls.append({
            "method": "invoke_model_stream",
            "system_prompt": system_prompt,
            "user_message": user_message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        })
        # Return reasoning-style or coordinator-style based on the system prompt
        if "referee" in system_prompt.lower() or "study group" in system_prompt.lower():
            response = (
                "**Consensus Summary**: Agreed.\n\n"
                "**Key Debates**: None.\n\n"
                "**Final Synthesized Answer**: test synthesis\n\n"
                "**Confidence Level**: High"
            )
        else:
            response = "Step-by-step reasoning here.\n\nFinal Answer: test answer"
        for char in response:
            yield char
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _is_reasoning_call(call: Dict[str, Any]) -> bool:
    """Return True if a captured provider call came from a reasoning agent (not coordinator)."""
    prompt = call["system_prompt"].lower()
    return "expert reasoning ai" in prompt and "referee" not in prompt


def _is_coordinator_call(call: Dict[str, Any]) -> bool:
    """Return True if a captured provider call came from the coordinator."""
    return "referee" in call["system_prompt"].lower()


TWO_TURN_HISTORY: List[Message] = [
    {"role": "user", "content": "What causes inflation?"},
    {"role": "assistant", "content": "Inflation is caused by demand-pull and cost-push factors."},
]

FOLLOW_UP_QUERY = "Elaborate on the demand-pull factors"

THREE_TURN_HISTORY: List[Message] = [
    {"role": "user", "content": "What causes inflation?"},
    {"role": "assistant", "content": "Demand-pull and cost-push factors."},
    {"role": "user", "content": "Elaborate on demand-pull."},
    {"role": "assistant", "content": "Demand-pull occurs when aggregate demand exceeds supply."},
]


# ============================================================================
# BedrockProvider._build_bedrock_messages
# ============================================================================

class TestBuildBedrockMessages:
    """Tests for the static message builder on BedrockProvider."""

    def test_single_user_message_when_no_messages(self):
        result = BedrockProvider._build_bedrock_messages("hello")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"text": "hello"}]

    def test_none_messages_falls_back_to_user_message(self):
        result = BedrockProvider._build_bedrock_messages("hello", messages=None)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_multi_turn_messages_converted(self):
        messages: List[Message] = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "follow-up"},
        ]
        result = BedrockProvider._build_bedrock_messages("ignored", messages=messages)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": [{"text": "first"}]}
        assert result[1] == {"role": "assistant", "content": [{"text": "response"}]}
        assert result[2] == {"role": "user", "content": [{"text": "follow-up"}]}

    def test_user_message_ignored_when_messages_provided(self):
        """When messages list is given, user_message param is not used."""
        messages: List[Message] = [{"role": "user", "content": "from messages"}]
        result = BedrockProvider._build_bedrock_messages("from param", messages=messages)

        assert len(result) == 1
        assert result[0]["content"] == [{"text": "from messages"}]

    def test_preserves_message_order(self):
        messages: List[Message] = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ]
        result = BedrockProvider._build_bedrock_messages("ignored", messages=messages)

        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        texts = [m["content"][0]["text"] for m in result]
        assert texts == ["q1", "a1", "q2", "a2", "q3"]

    def test_empty_messages_list_falls_back_to_user_message(self):
        """Empty list is falsy, should fall back to user_message."""
        result = BedrockProvider._build_bedrock_messages("fallback", messages=[])
        assert len(result) == 1
        assert result[0]["content"] == [{"text": "fallback"}]


# ============================================================================
# ReasoningAgent with conversation history
# ============================================================================

class TestReasoningAgentMultiTurn:
    """Verify ReasoningAgent builds the correct messages for the provider."""

    def test_no_history_passes_none_messages(self):
        """Without history, messages should be None (single-turn fallback)."""
        provider = CapturingProvider()
        agent = ReasoningAgent(agent_id=0, provider=provider)

        asyncio.run(agent.reason("What is 2+2?"))

        assert len(provider.calls) == 1
        assert provider.calls[0]["messages"] is None
        assert provider.calls[0]["user_message"] == "What is 2+2?"

    def test_with_history_builds_messages_list(self):
        """With history, messages should be history + current query."""
        provider = CapturingProvider()
        agent = ReasoningAgent(agent_id=0, provider=provider)

        asyncio.run(agent.reason(
            FOLLOW_UP_QUERY,
            conversation_history=TWO_TURN_HISTORY,
        ))

        call = provider.calls[0]
        messages = call["messages"]
        assert messages is not None
        assert len(messages) == 3  # 2 history + 1 current

        # History preserved in order
        assert messages[0] == TWO_TURN_HISTORY[0]
        assert messages[1] == TWO_TURN_HISTORY[1]

        # Current query appended as last user message
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == FOLLOW_UP_QUERY

    def test_history_not_mutated(self):
        """Passing history should not modify the original list."""
        provider = CapturingProvider()
        agent = ReasoningAgent(agent_id=0, provider=provider)

        history = list(TWO_TURN_HISTORY)  # copy
        original_len = len(history)

        asyncio.run(agent.reason("follow up", conversation_history=history))

        assert len(history) == original_len

    def test_empty_history_treated_as_none(self):
        """Empty list is falsy — should behave like no history."""
        provider = CapturingProvider()
        agent = ReasoningAgent(agent_id=0, provider=provider)

        asyncio.run(agent.reason("query", conversation_history=[]))

        assert provider.calls[0]["messages"] is None

    def test_agent_still_parses_response_with_history(self):
        """Agent should parse chain-of-thought and final answer normally."""
        provider = CapturingProvider()
        agent = ReasoningAgent(agent_id=0, provider=provider)

        response = asyncio.run(agent.reason(
            FOLLOW_UP_QUERY,
            conversation_history=TWO_TURN_HISTORY,
        ))

        assert response.success is True
        assert response.final_answer == "test answer"

    def test_long_history_all_turns_included(self):
        """A 4-turn history should result in 5 messages (4 + current)."""
        provider = CapturingProvider()
        agent = ReasoningAgent(agent_id=0, provider=provider)

        asyncio.run(agent.reason("latest question", conversation_history=THREE_TURN_HISTORY))

        messages = provider.calls[0]["messages"]
        assert len(messages) == 5  # 4 history + 1 current
        assert messages[-1]["content"] == "latest question"


# ============================================================================
# Coordinator prompt formatting
# ============================================================================

class TestCoordinatorPromptFormatting:

    def _make_agent_responses(self, n: int = 2) -> List[AgentResponse]:
        return [
            AgentResponse(
                agent_id=i,
                success=True,
                chain_of_thought=f"Agent {i} reasoning",
                final_answer=f"Answer {i}",
                duration_seconds=1.0,
                token_count=100,
            )
            for i in range(n)
        ]

    def test_no_history_uses_single_turn_template(self):
        """Without history, uses the original COORDINATOR_SYSTEM_PROMPT."""
        prompt = format_coordinator_prompt("What is 2+2?", self._make_agent_responses())

        assert "Conversation so far" not in prompt
        assert "follow-up" not in prompt.lower()
        assert "Question: What is 2+2?" in prompt
        assert "Agent 0" in prompt
        assert "Agent 1" in prompt

    def test_none_history_uses_single_turn_template(self):
        prompt = format_coordinator_prompt(
            "query", self._make_agent_responses(), conversation_history=None,
        )
        assert "Conversation so far" not in prompt

    def test_with_history_uses_multi_turn_template(self):
        prompt = format_coordinator_prompt(
            FOLLOW_UP_QUERY,
            self._make_agent_responses(),
            conversation_history=TWO_TURN_HISTORY,
        )

        assert "Conversation so far" in prompt
        assert "follow-up" in prompt.lower()
        assert "Current question: Elaborate on the demand-pull factors" in prompt

    def test_history_labels_user_and_toxp(self):
        """History should label user messages as [User] and assistant as [TOXP]."""
        prompt = format_coordinator_prompt(
            "q", self._make_agent_responses(), conversation_history=TWO_TURN_HISTORY,
        )

        assert "[User]: What causes inflation?" in prompt
        assert "[TOXP]: Inflation is caused by" in prompt

    def test_agent_outputs_still_present_with_history(self):
        """Agent outputs should be in the prompt regardless of history."""
        prompt = format_coordinator_prompt(
            "q", self._make_agent_responses(3), conversation_history=TWO_TURN_HISTORY,
        )

        assert "Agent 0" in prompt
        assert "Agent 1" in prompt
        assert "Agent 2" in prompt
        assert "Answer 0" in prompt

    def test_empty_history_uses_single_turn_template(self):
        prompt = format_coordinator_prompt(
            "query", self._make_agent_responses(), conversation_history=[],
        )
        assert "Conversation so far" not in prompt


class TestFormatConversationContext:

    def test_formats_user_messages(self):
        result = _format_conversation_context([{"role": "user", "content": "hello"}])
        assert result == "[User]: hello"

    def test_formats_assistant_messages(self):
        result = _format_conversation_context([{"role": "assistant", "content": "hi"}])
        assert result == "[TOXP]: hi"

    def test_multi_turn_separated_by_blank_lines(self):
        result = _format_conversation_context(TWO_TURN_HISTORY)
        lines = result.split("\n\n")
        assert len(lines) == 2
        assert lines[0].startswith("[User]")
        assert lines[1].startswith("[TOXP]")


# ============================================================================
# CoordinatorAgent with conversation history
# ============================================================================

class TestCoordinatorAgentMultiTurn:

    def _make_agent_responses(self) -> List[AgentResponse]:
        return [
            AgentResponse(
                agent_id=0, success=True,
                chain_of_thought="Reasoning", final_answer="42",
                duration_seconds=1.0, token_count=100,
            ),
        ]

    def test_synthesize_stream_with_history_uses_multi_turn_prompt(self):
        """Coordinator should use the multi-turn template when history is given."""
        provider = CapturingProvider()
        coordinator = CoordinatorAgent(provider=provider)

        tokens = []

        asyncio.run(coordinator.synthesize_stream(
            Query(text=FOLLOW_UP_QUERY),
            self._make_agent_responses(),
            on_token=lambda t: tokens.append(t),
            conversation_history=TWO_TURN_HISTORY,
        ))

        # The system_prompt sent to provider should be the multi-turn one
        call = provider.calls[0]
        assert "Conversation so far" in call["system_prompt"]
        assert "follow-up" in call["system_prompt"].lower()

    def test_synthesize_stream_without_history_uses_original_prompt(self):
        provider = CapturingProvider()
        coordinator = CoordinatorAgent(provider=provider)

        asyncio.run(coordinator.synthesize_stream(
            Query(text="simple query"),
            self._make_agent_responses(),
        ))

        call = provider.calls[0]
        assert "Conversation so far" not in call["system_prompt"]

    def test_synthesize_nonstream_with_history(self):
        """Non-streaming synthesize also supports history."""
        provider = CapturingProvider()
        coordinator = CoordinatorAgent(provider=provider)

        asyncio.run(coordinator.synthesize(
            Query(text=FOLLOW_UP_QUERY),
            self._make_agent_responses(),
            conversation_history=TWO_TURN_HISTORY,
        ))

        call = provider.calls[0]
        assert "Conversation so far" in call["system_prompt"]


# ============================================================================
# Orchestrator with conversation history
# ============================================================================

class TestOrchestratorMultiTurn:

    def test_history_reaches_all_agents(self):
        """Every reasoning agent should receive the conversation history."""
        provider = CapturingProvider()
        orchestrator = Orchestrator(provider=provider, num_agents=3)

        asyncio.run(orchestrator.process_query(
            Query(text=FOLLOW_UP_QUERY),
            conversation_history=TWO_TURN_HISTORY,
        ))

        # 3 agent calls + 1 coordinator call = 4 total
        agent_calls = [c for c in provider.calls if c["method"] == "invoke_model_stream"]

        # First 3 are agent calls (streaming), last is coordinator (also streaming here)
        # Agent calls have the reasoning system prompt
        reasoning_calls = [
            c for c in agent_calls
            if _is_reasoning_call(c)
        ]
        assert len(reasoning_calls) == 3

        for call in reasoning_calls:
            messages = call["messages"]
            assert messages is not None
            assert len(messages) == 3  # 2 history + 1 current query
            assert messages[-1]["content"] == FOLLOW_UP_QUERY

    def test_history_reaches_coordinator(self):
        """Coordinator should receive conversation context in its prompt."""
        provider = CapturingProvider()
        orchestrator = Orchestrator(provider=provider, num_agents=2)

        asyncio.run(orchestrator.process_query(
            Query(text=FOLLOW_UP_QUERY),
            conversation_history=TWO_TURN_HISTORY,
        ))

        # Find the coordinator call (has "referee" or "study group" in system prompt)
        coordinator_calls = [
            c for c in provider.calls
            if _is_coordinator_call(c)
        ]
        assert len(coordinator_calls) == 1
        assert "Conversation so far" in coordinator_calls[0]["system_prompt"]

    def test_no_history_agents_get_none_messages(self):
        """Without history, agents should get messages=None (backward compat)."""
        provider = CapturingProvider()
        orchestrator = Orchestrator(provider=provider, num_agents=2)

        asyncio.run(orchestrator.process_query(Query(text="simple query")))

        reasoning_calls = [
            c for c in provider.calls
            if _is_reasoning_call(c)
        ]
        for call in reasoning_calls:
            assert call["messages"] is None

    def test_result_structure_unchanged_with_history(self):
        """Result should have the same structure regardless of history."""
        provider = CapturingProvider()
        orchestrator = Orchestrator(provider=provider, num_agents=2)

        result = asyncio.run(orchestrator.process_query(
            Query(text=FOLLOW_UP_QUERY),
            conversation_history=TWO_TURN_HISTORY,
        ))

        assert len(result.agent_responses) == 2
        assert result.coordinator_response is not None
        assert result.coordinator_response.confidence in ("Low", "Medium", "High")
        assert result.metadata["num_agents"] == 2

    def test_callbacks_work_with_history(self):
        """Agent lifecycle callbacks should fire normally with conversation history."""
        provider = CapturingProvider()
        orchestrator = Orchestrator(provider=provider, num_agents=2)

        started = []
        completed = []
        coord_tokens = []

        result = asyncio.run(orchestrator.process_query(
            Query(text=FOLLOW_UP_QUERY),
            on_agent_start=lambda aid: started.append(aid),
            on_agent_complete=lambda aid, ok, err: completed.append((aid, ok)),
            on_coordinator_token=lambda t: coord_tokens.append(t),
            conversation_history=TWO_TURN_HISTORY,
        ))

        assert len(started) == 2
        assert len(completed) == 2
        assert all(ok for _, ok in completed)
        assert len(coord_tokens) > 0


# ============================================================================
# Full pipeline: run_query with conversation_history
# ============================================================================

def _run_pipeline(
    query: str,
    conversation_history: Optional[List[Message]] = None,
    num_agents: int = 2,
) -> tuple[QueryResult, CapturingProvider]:
    """Run run_query() with a capturing provider and return result + provider."""
    provider = CapturingProvider()

    with (
        patch("toxp.api.ConfigManager") as MockCM,
        patch("toxp.api.ProviderRegistry") as MockReg,
    ):
        config = ToxpConfig.from_dict(
            {**ToxpConfig.get_defaults().to_dict(), "num_agents": num_agents}
        )
        MockCM.return_value.load_with_overrides.return_value = config
        MockReg.get.return_value = lambda **kwargs: provider

        result = asyncio.run(run_query(
            query,
            conversation_history=conversation_history,
        ))

    return result, provider


class TestRunQueryMultiTurn:
    """End-to-end tests for run_query with conversation_history."""

    def test_first_turn_no_history(self):
        """First turn: no history, agents get messages=None."""
        result, provider = _run_pipeline("What causes inflation?")

        assert isinstance(result, QueryResult)
        assert result.confidence in ("Low", "Medium", "High")

        # Agents should NOT have received messages
        reasoning_calls = [
            c for c in provider.calls
            if _is_reasoning_call(c)
        ]
        for call in reasoning_calls:
            assert call["messages"] is None

    def test_second_turn_with_history(self):
        """Second turn: history is threaded to agents and coordinator."""
        result, provider = _run_pipeline(
            FOLLOW_UP_QUERY,
            conversation_history=TWO_TURN_HISTORY,
        )

        assert isinstance(result, QueryResult)

        # Agents got multi-turn messages
        reasoning_calls = [
            c for c in provider.calls
            if _is_reasoning_call(c)
        ]
        for call in reasoning_calls:
            assert call["messages"] is not None
            assert len(call["messages"]) == 3
            assert call["messages"][0]["content"] == "What causes inflation?"
            assert call["messages"][-1]["content"] == FOLLOW_UP_QUERY

        # Coordinator got multi-turn prompt
        coordinator_calls = [
            c for c in provider.calls
            if _is_coordinator_call(c)
        ]
        assert len(coordinator_calls) == 1
        assert "Conversation so far" in coordinator_calls[0]["system_prompt"]

    def test_third_turn_deeper_history(self):
        """Third turn: longer history all flows through."""
        result, provider = _run_pipeline(
            "What are the monetary policy responses?",
            conversation_history=THREE_TURN_HISTORY,
        )

        reasoning_calls = [
            c for c in provider.calls
            if _is_reasoning_call(c)
        ]
        for call in reasoning_calls:
            messages = call["messages"]
            assert len(messages) == 5  # 4 history turns + 1 current query
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[3]["role"] == "assistant"
            assert messages[4]["role"] == "user"
            assert messages[4]["content"] == "What are the monetary policy responses?"

    def test_none_history_is_same_as_no_history(self):
        """Explicitly passing None should behave identically to omitting it."""
        result_none, provider_none = _run_pipeline("query", conversation_history=None)
        result_omit, provider_omit = _run_pipeline("query")

        # Both should have agents with messages=None
        for provider in [provider_none, provider_omit]:
            for call in provider.calls:
                if _is_reasoning_call(call):
                    assert call["messages"] is None

    def test_empty_list_history_is_same_as_no_history(self):
        """Empty list should behave like no history."""
        result, provider = _run_pipeline("query", conversation_history=[])

        for call in provider.calls:
            if _is_reasoning_call(call):
                assert call["messages"] is None


# ============================================================================
# Simulated real conversation flow
# ============================================================================

class TestSimulatedConversation:
    """Simulate a realistic multi-turn conversation as a UI would drive it."""

    def test_three_turn_conversation_flow(self):
        """Simulate: question → follow-up → clarification, building history each turn."""

        # Turn 1: fresh question
        result1, prov1 = _run_pipeline("What is quantum entanglement?")
        turn1_answer = result1.synthesis_markdown

        # Verify turn 1 was single-shot
        for call in prov1.calls:
            if _is_reasoning_call(call):
                assert call["messages"] is None

        # Turn 2: follow-up (UI builds history from turn 1)
        history_for_turn2: List[Message] = [
            {"role": "user", "content": "What is quantum entanglement?"},
            {"role": "assistant", "content": turn1_answer},
        ]
        result2, prov2 = _run_pipeline(
            "How does it relate to quantum computing?",
            conversation_history=history_for_turn2,
        )
        turn2_answer = result2.synthesis_markdown

        # Verify turn 2 agents got 3 messages
        for call in prov2.calls:
            if _is_reasoning_call(call):
                assert call["messages"] is not None
                assert len(call["messages"]) == 3
                assert call["messages"][0]["content"] == "What is quantum entanglement?"
                assert call["messages"][2]["content"] == "How does it relate to quantum computing?"

        # Turn 3: deeper follow-up (UI appends turn 2 to history)
        history_for_turn3: List[Message] = [
            {"role": "user", "content": "What is quantum entanglement?"},
            {"role": "assistant", "content": turn1_answer},
            {"role": "user", "content": "How does it relate to quantum computing?"},
            {"role": "assistant", "content": turn2_answer},
        ]
        result3, prov3 = _run_pipeline(
            "What are the current hardware challenges?",
            conversation_history=history_for_turn3,
        )

        # Verify turn 3 agents got 5 messages
        for call in prov3.calls:
            if _is_reasoning_call(call):
                assert call["messages"] is not None
                assert len(call["messages"]) == 5
                # Correct ordering
                assert call["messages"][0]["role"] == "user"
                assert call["messages"][1]["role"] == "assistant"
                assert call["messages"][2]["role"] == "user"
                assert call["messages"][3]["role"] == "assistant"
                assert call["messages"][4]["role"] == "user"
                assert call["messages"][4]["content"] == "What are the current hardware challenges?"

        # All three turns produced valid results
        for result in [result1, result2, result3]:
            assert isinstance(result, QueryResult)
            assert result.confidence in ("Low", "Medium", "High")
            assert result.successful_agents == 2

    def test_num_agents_configurable_with_history(self):
        """num_agents should work correctly alongside conversation_history."""
        for n in [2, 5, 8]:
            result, provider = _run_pipeline(
                "follow-up",
                conversation_history=TWO_TURN_HISTORY,
                num_agents=n,
            )
            assert result.total_agents == n
            assert result.successful_agents == n

            reasoning_calls = [
                c for c in provider.calls
                if _is_reasoning_call(c)
            ]
            assert len(reasoning_calls) == n


# ============================================================================
# Edge cases
# ============================================================================

class TestMultiTurnEdgeCases:

    def test_single_user_message_history(self):
        """History with just one user message (no assistant response yet)."""
        history: List[Message] = [{"role": "user", "content": "first attempt"}]
        result, provider = _run_pipeline("retry my question", conversation_history=history)

        for call in provider.calls:
            if _is_reasoning_call(call):
                messages = call["messages"]
                assert messages is not None
                assert len(messages) == 2
                assert messages[0] == {"role": "user", "content": "first attempt"}
                assert messages[1] == {"role": "user", "content": "retry my question"}

    def test_history_with_long_content(self):
        """History with very long messages should pass through without truncation."""
        long_content = "x" * 10000
        history: List[Message] = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": long_content},
        ]
        result, provider = _run_pipeline("follow up", conversation_history=history)

        for call in provider.calls:
            if _is_reasoning_call(call):
                messages = call["messages"]
                assert messages[1]["content"] == long_content  # not truncated

    def test_cancel_token_works_with_history(self):
        """Cancel token should work normally when history is provided."""
        provider = CapturingProvider()
        cancel = asyncio.Event()
        cancel.set()

        orchestrator = Orchestrator(provider=provider, num_agents=3)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(orchestrator.process_query(
                Query(text="query"),
                cancel_token=cancel,
                conversation_history=TWO_TURN_HISTORY,
            ))
