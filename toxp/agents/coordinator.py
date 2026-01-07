"""Coordinator agent for synthesizing multiple reasoning agent outputs."""

import logging
import re
import time
from typing import Callable, List, Literal, Optional

from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.models.query import Query
from toxp.providers.base import BaseProvider
from toxp.agents.prompts import format_coordinator_prompt


logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    Coordinator agent that synthesizes outputs from multiple reasoning agents.
    
    The coordinator acts as an impartial referee, identifying agreements and
    contradictions, critiquing logical errors, and synthesizing the best answer
    from multiple independent expert reasoning outputs.
    """

    def __init__(
        self,
        provider: BaseProvider,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ):
        """
        Initialize coordinator agent.
        
        Args:
            provider: LLM provider for model invocation
            temperature: Sampling temperature (lower for focused synthesis)
            max_tokens: Maximum tokens for synthesis output
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def synthesize(
        self,
        query: Query,
        agent_responses: List[AgentResponse],
    ) -> CoordinatorResponse:
        """
        Synthesize multiple agent responses into a single coordinated answer.
        
        This method:
        1. Builds coordinator prompt with all successful agent outputs
        2. Invokes the provider
        3. Parses the synthesis to extract structured components
        4. Returns a CoordinatorResponse with all synthesis details
        
        Args:
            query: The original user query
            agent_responses: List of responses from reasoning agents
            
        Returns:
            CoordinatorResponse containing synthesis, confidence, and final answer
            
        Raises:
            ValueError: If no successful agent responses to synthesize
        """
        # Filter to only successful agent responses
        successful_responses = [r for r in agent_responses if r.success]
        
        if not successful_responses:
            raise ValueError("No successful agent responses to synthesize")
        
        # Build coordinator prompt with all agent outputs
        coordinator_prompt = format_coordinator_prompt(query.text, successful_responses)
        
        logger.debug(
            f"Coordinator prompt length: {len(coordinator_prompt):,} chars "
            f"(~{len(coordinator_prompt) // 4:,} tokens)"
        )
        
        start_time = time.time()
        
        # Invoke provider with coordinator prompt
        response = await self.provider.invoke_model(
            system_prompt=coordinator_prompt,
            user_message="Please provide your synthesis following the output format specified.",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        duration = time.time() - start_time
        
        # Parse the coordinator response
        return self._parse_synthesis(response.text, duration)

    async def synthesize_stream(
        self,
        query: Query,
        agent_responses: List[AgentResponse],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> CoordinatorResponse:
        """
        Synthesize with streaming output - tokens are passed to on_token callback as they arrive.
        
        Args:
            query: The original user query
            agent_responses: List of responses from reasoning agents
            on_token: Callback function called with each text chunk
            
        Returns:
            CoordinatorResponse containing synthesis, confidence, and final answer
            
        Raises:
            ValueError: If no successful agent responses to synthesize
        """
        successful_responses = [r for r in agent_responses if r.success]
        
        if not successful_responses:
            raise ValueError("No successful agent responses to synthesize")
        
        coordinator_prompt = format_coordinator_prompt(query.text, successful_responses)
        
        logger.debug(
            f"Coordinator prompt length: {len(coordinator_prompt):,} chars "
            f"(~{len(coordinator_prompt) // 4:,} tokens)"
        )
        
        start_time = time.time()
        synthesis_text = ""
        
        async for token in self.provider.invoke_model_stream(
            system_prompt=coordinator_prompt,
            user_message="Please provide your synthesis following the output format specified.",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            synthesis_text += token
            if on_token:
                on_token(token)
        
        duration = time.time() - start_time
        return self._parse_synthesis(synthesis_text, duration)

    def _parse_synthesis(self, synthesis_text: str, duration: float) -> CoordinatorResponse:
        """
        Parse coordinator synthesis text to extract structured components.
        
        Expected format from coordinator:
        - **Consensus Summary**: ...
        - **Key Debates**: ...
        - **Critique**: ...
        - **Final Synthesized Answer**: ...
        - **Confidence Level**: Low/Medium/High ...
        
        Args:
            synthesis_text: Raw text output from coordinator
            duration: Time taken for synthesis in seconds
            
        Returns:
            CoordinatorResponse with parsed components
        """
        # Extract sections using regex patterns
        # Pattern looks for section headers (newline + ** + capital letter or ### header)
        # to avoid stopping at inline bold text like **Paris**
        section_end = r"(?=\n\*\*[A-Z]|\n###|\Z)"
        
        consensus_summary = self._extract_section(
            synthesis_text,
            rf"\*\*Consensus Summary\*\*:?\s*(.*?){section_end}"
        )
        
        debates_summary = self._extract_section(
            synthesis_text,
            rf"\*\*Key Debates\*\*:?\s*(.*?){section_end}"
        )
        
        final_answer = self._extract_section(
            synthesis_text,
            rf"\*\*Final Synthesized Answer\*\*:?\s*(.*?){section_end}"
        )
        
        confidence_text = self._extract_section(
            synthesis_text,
            rf"\*\*Confidence Level\*\*:?\s*(.*?){section_end}"
        )
        
        # Extract confidence level (Low/Medium/High)
        confidence = self._extract_confidence_level(confidence_text)
        
        return CoordinatorResponse(
            synthesis=synthesis_text,
            confidence=confidence,
            consensus_summary=consensus_summary,
            debates_summary=debates_summary,
            final_answer=final_answer or synthesis_text,  # Fallback to full text
            duration_seconds=duration,
        )

    def _extract_section(self, text: str, pattern: str) -> str:
        """
        Extract a section from synthesis text using regex pattern.
        
        Args:
            text: Full synthesis text
            pattern: Regex pattern to match section
            
        Returns:
            Extracted section text, or empty string if not found
        """
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_confidence_level(
        self, confidence_text: str
    ) -> Literal["Low", "Medium", "High"]:
        """
        Extract confidence level from confidence text.
        
        Looks for "Low", "Medium", or "High" in the text.
        
        Args:
            confidence_text: Text containing confidence level
            
        Returns:
            "Low", "Medium", or "High" (defaults to "Medium" if not found)
        """
        confidence_lower = confidence_text.lower()
        
        if "high" in confidence_lower:
            return "High"
        elif "low" in confidence_lower:
            return "Low"
        elif "medium" in confidence_lower:
            return "Medium"
        
        # Default to Medium if not explicitly stated
        return "Medium"
