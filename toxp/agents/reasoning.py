"""Reasoning agent implementation for parallel test-time compute."""

import logging
import re
import time
from typing import Callable, List, Optional, Tuple

from toxp.models.conversation import Message
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

    async def reason(
        self,
        query: str,
        on_token: Optional[Callable[[str], None]] = None,
        conversation_history: Optional[List[Message]] = None,
    ) -> AgentResponse:
        """
        Execute reasoning on the given query using streaming.

        This method:
        1. Invokes the provider with streaming to avoid timeout issues
        2. Optionally calls on_token callback for each streamed token
        3. Parses the response to extract chain-of-thought and final answer
        4. Handles errors gracefully without blocking other agents

        Args:
            query: The user's question or problem to solve
            on_token: Optional callback invoked with each streamed token
            conversation_history: Optional prior conversation turns for multi-turn
                context. When provided, the agent sees the full conversation as
                native multi-turn messages via the Bedrock Converse API.

        Returns:
            AgentResponse with reasoning results or error information
        """
        start_time = time.time()
        
        try:
            logger.info(f"Agent {self.agent_id}: Starting reasoning (streaming)")

            # Build multi-turn messages when conversation history is available
            multi_turn_messages: Optional[List[Message]] = None
            if conversation_history:
                multi_turn_messages = list(conversation_history) + [
                    {"role": "user", "content": query}
                ]

            # Use streaming to avoid timeout issues with large prompts
            response_text = ""
            async for token in self.provider.invoke_model_stream(
                system_prompt=REASONING_AGENT_SYSTEM_PROMPT,
                user_message=query,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=multi_turn_messages,
            ):
                response_text += token
                if on_token:
                    on_token(token)
            
            # Extract chain-of-thought and final answer
            chain_of_thought, final_answer = self._parse_response(response_text)
            
            duration = time.time() - start_time
            # Estimate token count from response length (streaming doesn't provide exact count)
            estimated_tokens = len(response_text) // 4
            
            logger.info(
                f"Agent {self.agent_id}: Completed successfully "
                f"({duration:.2f}s, ~{estimated_tokens} tokens)"
            )
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                chain_of_thought=chain_of_thought,
                final_answer=final_answer,
                error=None,
                duration_seconds=duration,
                token_count=estimated_tokens,
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
