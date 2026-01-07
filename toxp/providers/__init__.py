"""Pluggable LLM provider architecture for TOXP CLI.

This module provides an extensible provider system that allows connecting
to different LLM backends (Bedrock, Anthropic API, OpenAI, etc.) without
modifying core logic.
"""

from .base import BaseProvider, ProviderResponse
from .registry import ProviderRegistry
from .bedrock import (
    BedrockProvider,
    BedrockProviderError,
    CredentialsError,
    ModelNotFoundError,
    ThrottlingError,
)

# Register built-in providers
ProviderRegistry.register("bedrock", BedrockProvider)

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "ProviderRegistry",
    "BedrockProvider",
    "BedrockProviderError",
    "CredentialsError",
    "ModelNotFoundError",
    "ThrottlingError",
]
