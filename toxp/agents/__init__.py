"""Agents module for TOXP CLI.

Contains reasoning agents and coordinator agent for parallel test-time compute.
"""

from .prompts import (
    REASONING_AGENT_SYSTEM_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    format_coordinator_prompt,
)
from .reasoning import ReasoningAgent
from .coordinator import CoordinatorAgent

__all__ = [
    "REASONING_AGENT_SYSTEM_PROMPT",
    "COORDINATOR_SYSTEM_PROMPT",
    "format_coordinator_prompt",
    "ReasoningAgent",
    "CoordinatorAgent",
]
