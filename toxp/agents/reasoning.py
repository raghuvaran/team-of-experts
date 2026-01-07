"""Reasoning agent implementation for parallel test-time compute."""

import logging
import re
import time
from typing import Optional, Tuple

from toxp.models.response import AgentResponse
from toxp.providers.base import BaseProvider
from toxp.agents.prompts import REASONING_AGENT_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class ReasoningAgent:
    """
    Independent reasoning agent with chain-of-thought output.
    
    Each agent operates independently with high temperature sampling to
    ensure diversity in reasoning approaches across the agent pool.
    """

    def __init__(
        self,
        agent_id: int,
        provider: BaseProvider,
        temperature: float = 0.9,
        max_tokens: int = 8192,
    ):
        """
        Initialize reasoning agent.
        
        Args:
            agent_id: Unique identifier for this agent
            provider: LLM provider for model invocation
            temperature: Sampling temperature (0.8-1.0 for diversity)
            max_tokens: Maximum tokens for chain-of-thought output
        """
        self.agent_id = agent_id
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def reason(self, query: str) -> AgentResponse:
        """
        Execute reasoning on the given query.
        
        This method:
        1. Invokes the provider with the reasoning system prompt
        2. Parses the response to extract chain-of-thought and final answer
        3. Handles errors gracefully without blocking other agents
        
        Args:
            query: The user's question or problem to solve
            
        Returns:
            AgentResponse with reasoning results or error information
        """
        start_time = time.time()
        
        try:
            logger.info(f"Agent {self.agent_id}: Starting reasoning")
            
            # Invoke provider with reasoning prompt
            response = await self.provider.invoke_model(
                system_prompt=REASONING_AGENT_SYSTEM_PROMPT,
                user_message=query,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract chain-of-thought and final answer
            chain_of_thought, final_answer = self._parse_response(response.text)
            
            duration = time.time() - start_time
            token_count = response.input_tokens + response.output_tokens
            
            logger.info(
                f"Agent {self.agent_id}: Completed successfully "
                f"({duration:.2f}s, {token_count} tokens)"
            )
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                chain_of_thought=chain_of_thought,
                final_answer=final_answer,
                error=None,
                duration_seconds=duration,
                token_count=token_count,
            )
            
        except TimeoutError as e:
            duration = time.time() - start_time
            error_msg = f"Timeout after {duration:.1f}s: {str(e)}"
            
            logger.warning(
                f"Agent {self.agent_id}: Timed out after {duration:.2f}s"
            )
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                chain_of_thought="",
                final_answer="",
                error=error_msg,
                duration_seconds=duration,
                token_count=0,
            )
            
        except ValueError as e:
            duration = time.time() - start_time
            error_msg = f"Invalid response: {str(e)}"
            
            logger.warning(
                f"Agent {self.agent_id}: Invalid response after {duration:.2f}s - {error_msg}"
            )
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                chain_of_thought="",
                final_answer="",
                error=error_msg,
                duration_seconds=duration,
                token_count=0,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            logger.warning(
                f"Agent {self.agent_id}: Failed after {duration:.2f}s - {error_msg}"
            )
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                chain_of_thought="",
                final_answer="",
                error=error_msg,
                duration_seconds=duration,
                token_count=0,
            )

    def _parse_response(self, text: str) -> Tuple[str, str]:
        """
        Parse agent response to extract chain-of-thought and final answer.
        
        The agent is instructed to use \\boxed{...} notation or clear formatting
        for the final answer. This method attempts to extract that.
        
        Args:
            text: Raw text response from the model
            
        Returns:
            Tuple of (chain_of_thought, final_answer)
        """
        # Try to find boxed answer using LaTeX notation
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        boxed_match = re.search(boxed_pattern, text)
        
        if boxed_match:
            final_answer = boxed_match.group(1)
            # Everything is chain-of-thought
            chain_of_thought = text
            return chain_of_thought, final_answer
        
        # Try to find "Final Answer:" section
        final_answer_pattern = r"(?:Final Answer|Answer):\s*(.+?)(?:\n\n|\Z)"
        final_match = re.search(final_answer_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if final_match:
            final_answer = final_match.group(1).strip()
            chain_of_thought = text
            return chain_of_thought, final_answer
        
        # If no clear final answer delimiter, treat last paragraph as answer
        paragraphs = text.strip().split("\n\n")
        if len(paragraphs) > 1:
            final_answer = paragraphs[-1].strip()
            chain_of_thought = text
        else:
            # Entire response is both CoT and answer
            chain_of_thought = text
            final_answer = text.strip()
        
        return chain_of_thought, final_answer
