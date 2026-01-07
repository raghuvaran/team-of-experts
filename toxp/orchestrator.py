"""Orchestrator for parallel agent execution and synthesis.

This module implements the main orchestration logic for TOXP, coordinating
parallel reasoning agent execution and coordinator synthesis.

Feature: toxp-cli
Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.5
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from toxp.agents.coordinator import CoordinatorAgent
from toxp.agents.reasoning import ReasoningAgent
from toxp.exceptions import InsufficientAgentsError
from toxp.models.query import Query
from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.models.result import Result
from toxp.providers.base import BaseProvider
from toxp.utils.rate_limiter import AdaptiveRateLimiter


logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates parallel agent execution and synthesis.
    
    The Orchestrator is the main entry point for processing queries through TOXP.
    It:
    1. Spawns N reasoning agents concurrently with rate limiting
    2. Validates that enough agents succeeded (≥50%)
    3. Invokes the coordinator to synthesize agent outputs
    4. Returns a complete Result with all metadata
    
    Attributes:
        provider: LLM provider for model invocation
        num_agents: Number of reasoning agents to spawn (2-32)
        temperature: Sampling temperature for reasoning agents
        coordinator_temperature: Sampling temperature for coordinator
        max_tokens: Maximum tokens per agent response
        min_success_rate: Minimum fraction of agents that must succeed
    """
    
    # Valid range for num_agents
    MIN_AGENTS = 2
    MAX_AGENTS = 32
    
    def __init__(
        self,
        provider: BaseProvider,
        num_agents: int = 16,
        temperature: float = 0.9,
        coordinator_temperature: float = 0.7,
        max_tokens: int = 8192,
        min_success_rate: float = 0.5,
        max_concurrency: Optional[int] = None,
    ):
        """Initialize the orchestrator.
        
        Args:
            provider: LLM provider for model invocation
            num_agents: Number of reasoning agents (2-32, default: 16)
            temperature: Agent sampling temperature (default: 0.9 for diversity)
            coordinator_temperature: Coordinator temperature (default: 0.7)
            max_tokens: Maximum tokens per response (default: 8192)
            min_success_rate: Minimum success rate required (default: 0.5)
            max_concurrency: Maximum concurrent API requests (None = auto-calculate)
            
        Raises:
            ValueError: If num_agents is outside valid range
        """
        if not self.MIN_AGENTS <= num_agents <= self.MAX_AGENTS:
            raise ValueError(
                f"num_agents must be between {self.MIN_AGENTS} and {self.MAX_AGENTS}, "
                f"got {num_agents}"
            )
        
        self.provider = provider
        self.num_agents = num_agents
        self.temperature = temperature
        self.coordinator_temperature = coordinator_temperature
        self.max_tokens = max_tokens
        self.min_success_rate = min_success_rate
        self.max_concurrency = max_concurrency
        
        # Initialize rate limiter for this provider/model
        self.rate_limiter = AdaptiveRateLimiter(
            model_id=provider.model_id,
            max_tokens_per_request=max_tokens,
            max_concurrency_override=max_concurrency,
        )
        
        logger.info(
            f"Orchestrator initialized: num_agents={num_agents}, "
            f"temperature={temperature}, coordinator_temp={coordinator_temperature}, "
            f"max_concurrency={self.rate_limiter.max_concurrency}"
        )

    async def process_query(
        self,
        query: Query,
        on_coordinator_token: Optional[Callable[[str], None]] = None,
        on_agent_start: Optional[Callable[[int], None]] = None,
        on_agent_complete: Optional[Callable[[int, bool, Optional[str]], None]] = None,
        on_agents_done: Optional[Callable[[], None]] = None,
    ) -> Result:
        """Process a query through the full TOXP pipeline.
        
        This is the main entry point for query processing. It:
        1. Spawns N reasoning agents concurrently
        2. Validates success rate (≥50%)
        3. Invokes coordinator for synthesis
        4. Returns complete result with metadata
        
        Args:
            query: The user query to process
            on_coordinator_token: Optional callback for streaming coordinator output
            on_agent_start: Optional callback when an agent starts (receives agent_id)
            on_agent_complete: Optional callback when agent completes (agent_id, success, error)
            on_agents_done: Optional callback when all agents finish (before coordinator)
            
        Returns:
            Result containing all agent responses and coordinator synthesis
            
        Raises:
            InsufficientAgentsError: If fewer than 50% of agents succeed
        """
        start_time = time.time()
        
        logger.info(
            f"Processing query '{query.query_id}': "
            f"spawning {self.num_agents} agents"
        )
        
        # Step 1: Spawn reasoning agents concurrently
        agent_responses = await self._spawn_agents(
            query,
            on_agent_start=on_agent_start,
            on_agent_complete=on_agent_complete,
        )
        
        # Signal that all agents are done (before coordinator starts)
        if on_agents_done:
            on_agents_done()
        
        # Step 2: Validate success rate
        if not self._validate_responses(agent_responses):
            successful = [r for r in agent_responses if r.success]
            failed = [r for r in agent_responses if not r.success]
            min_required = int(len(agent_responses) * self.min_success_rate)
            
            raise InsufficientAgentsError(
                successful_count=len(successful),
                total_count=len(agent_responses),
                min_required=min_required,
                agent_errors=[r.error for r in failed if r.error],
            )
        
        # Step 3: Invoke coordinator for synthesis
        try:
            coordinator_response = await self._synthesize(
                query, agent_responses, on_coordinator_token
            )
        except Exception as e:
            # Fallback: return partial results if coordinator fails
            logger.error(f"Coordinator synthesis failed: {e}")
            coordinator_response = self._create_fallback_response(
                agent_responses, str(e)
            )
        
        # Step 4: Build and return result
        total_duration = time.time() - start_time
        
        result = Result(
            query=query,
            agent_responses=agent_responses,
            coordinator_response=coordinator_response,
            metadata=self._build_metadata(
                agent_responses, coordinator_response, total_duration
            ),
        )
        
        logger.info(
            f"Query '{query.query_id}' completed in {total_duration:.2f}s: "
            f"{len([r for r in agent_responses if r.success])}/{len(agent_responses)} "
            f"agents succeeded, confidence={coordinator_response.confidence}"
        )
        
        return result

    async def _spawn_agents(
        self,
        query: Query,
        on_agent_start: Optional[Callable[[int], None]] = None,
        on_agent_complete: Optional[Callable[[int, bool, Optional[str]], None]] = None,
    ) -> List[AgentResponse]:
        """Spawn reasoning agents with rate-limited concurrency.
        
        Creates N reasoning agents and executes them concurrently, using
        the rate limiter to prevent API throttling.
        
        Args:
            query: The query for agents to process
            on_agent_start: Optional callback when agent starts
            on_agent_complete: Optional callback when agent completes
            
        Returns:
            List of AgentResponse objects (both successful and failed)
        """
        # Create reasoning agents
        agents = [
            ReasoningAgent(
                agent_id=i,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            for i in range(self.num_agents)
        ]
        
        async def run_with_rate_limit(agent: ReasoningAgent) -> AgentResponse:
            """Execute agent with rate limiting and progress callbacks."""
            try:
                async with self.rate_limiter:
                    # Signal start AFTER acquiring semaphore (actually running now)
                    if on_agent_start:
                        on_agent_start(agent.agent_id)
                    result = await agent.reason(query.text)
                
                if on_agent_complete:
                    error_msg = result.error if not result.success else None
                    on_agent_complete(agent.agent_id, result.success, error_msg)
                
                return result
            except Exception as e:
                if on_agent_complete:
                    on_agent_complete(agent.agent_id, False, str(e))
                raise
        
        logger.debug(f"Spawning {len(agents)} agents with rate limiting")
        
        # Execute all agents concurrently with rate limiting
        # return_exceptions=True ensures we get all results even if some fail
        results = await asyncio.gather(
            *[run_with_rate_limit(agent) for agent in agents],
            return_exceptions=True,
        )
        
        # Convert any exceptions to failed AgentResponses
        agent_responses: List[AgentResponse] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed response
                agent_responses.append(
                    AgentResponse(
                        agent_id=i,
                        success=False,
                        chain_of_thought="",
                        final_answer="",
                        error=f"{type(result).__name__}: {str(result)}",
                        duration_seconds=0.0,
                        token_count=0,
                    )
                )
            else:
                agent_responses.append(result)
        
        successful = len([r for r in agent_responses if r.success])
        logger.info(f"Agent execution complete: {successful}/{len(agent_responses)} succeeded")
        
        return agent_responses

    def _validate_responses(self, agent_responses: List[AgentResponse]) -> bool:
        """Validate that enough agents succeeded.
        
        Args:
            agent_responses: List of all agent responses
            
        Returns:
            True if success rate meets minimum threshold
        """
        if not agent_responses:
            return False
        
        successful_count = len([r for r in agent_responses if r.success])
        success_rate = successful_count / len(agent_responses)
        
        is_valid = success_rate >= self.min_success_rate
        
        logger.debug(
            f"Success rate validation: {successful_count}/{len(agent_responses)} "
            f"({success_rate:.0%}) >= {self.min_success_rate:.0%} = {is_valid}"
        )
        
        return is_valid

    async def _synthesize(
        self,
        query: Query,
        agent_responses: List[AgentResponse],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> CoordinatorResponse:
        """Invoke coordinator to synthesize agent outputs.
        
        Args:
            query: The original query
            agent_responses: All agent responses
            on_token: Optional callback for streaming output
            
        Returns:
            CoordinatorResponse with synthesis
        """
        coordinator = CoordinatorAgent(
            provider=self.provider,
            temperature=self.coordinator_temperature,
            max_tokens=self.max_tokens,
        )
        
        if on_token:
            # Use streaming synthesis
            return await coordinator.synthesize_stream(
                query, agent_responses, on_token
            )
        else:
            # Use non-streaming synthesis
            return await coordinator.synthesize(query, agent_responses)

    def _create_fallback_response(
        self,
        agent_responses: List[AgentResponse],
        error_message: str,
    ) -> CoordinatorResponse:
        """Create a fallback coordinator response when synthesis fails.
        
        This provides partial results to the user when the coordinator
        cannot complete synthesis.
        
        Args:
            agent_responses: The agent responses that were collected
            error_message: The error that caused synthesis to fail
            
        Returns:
            CoordinatorResponse with fallback content
        """
        successful = [r for r in agent_responses if r.success]
        
        # Build a simple summary from successful agents
        if successful:
            answers = [r.final_answer for r in successful if r.final_answer]
            summary = "\n\n".join([
                f"**Agent {r.agent_id}**: {r.final_answer[:500]}..."
                if len(r.final_answer) > 500 else f"**Agent {r.agent_id}**: {r.final_answer}"
                for r in successful[:5]  # Limit to first 5
            ])
            
            synthesis = (
                f"**Note**: Coordinator synthesis failed ({error_message}). "
                f"Showing individual agent responses:\n\n{summary}"
            )
            final_answer = answers[0] if answers else "No answer available"
        else:
            synthesis = f"Coordinator synthesis failed: {error_message}"
            final_answer = "No answer available"
        
        return CoordinatorResponse(
            synthesis=synthesis,
            confidence="Low",
            consensus_summary="",
            debates_summary="",
            final_answer=final_answer,
            duration_seconds=0.0,
        )

    def _build_metadata(
        self,
        agent_responses: List[AgentResponse],
        coordinator_response: CoordinatorResponse,
        total_duration: float,
    ) -> Dict[str, Any]:
        """Build metadata dictionary for the result.
        
        Args:
            agent_responses: All agent responses
            coordinator_response: The coordinator synthesis
            total_duration: Total processing time in seconds
            
        Returns:
            Dictionary with processing metadata
        """
        successful = [r for r in agent_responses if r.success]
        
        total_agent_tokens = sum(r.token_count for r in agent_responses)
        total_agent_duration = sum(r.duration_seconds for r in agent_responses)
        
        return {
            "model_id": self.provider.model_id,
            "provider": self.provider.name,
            "num_agents": self.num_agents,
            "successful_agents": len(successful),
            "failed_agents": len(agent_responses) - len(successful),
            "success_rate": len(successful) / len(agent_responses) if agent_responses else 0,
            "total_agent_tokens": total_agent_tokens,
            "total_agent_duration_seconds": total_agent_duration,
            "coordinator_duration_seconds": coordinator_response.duration_seconds,
            "total_duration_seconds": total_duration,
            "temperature": self.temperature,
            "coordinator_temperature": self.coordinator_temperature,
            "max_tokens": self.max_tokens,
        }
