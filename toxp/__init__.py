"""TOXP — Team of eXPerts parallel reasoning system.

Library usage:
    from toxp import run_query, QueryResult, QueryCallbacks, validate_credentials

    result = await run_query("What is 2+2?")
    print(result.final_answer, result.confidence)
"""

import sys

__version__ = "0.7.1"

# Runtime Python version check
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    sys.exit(
        f"Error: TOXP requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, "
        f"but you're running Python {sys.version_info.major}.{sys.version_info.minor}\n\n"
        f"Fix: Use uvx with --python flag:\n"
        f"  uvx --python 3.12 --from team-of-experts txp \"your question\"\n\n"
        f"Or install Python 3.10+ and reinstall TOXP."
    )

# Public API
from toxp.api import (
    run_query,
    QueryResult,
    QueryCallbacks,
    NoOpCallbacks,
    validate_credentials,
    get_default_config,
)

# Conversation types
from toxp.models.conversation import Message

# Exceptions (already well-structured)
from toxp.exceptions import (
    ToxpError,
    CredentialsError,
    ConfigurationError,
    ProviderError,
    ModelNotFoundError,
    ThrottlingError,
    InsufficientAgentsError,
    NetworkError,
    TimeoutError,
)

__all__ = [
    "__version__",
    # API
    "run_query",
    "QueryResult",
    "QueryCallbacks",
    "NoOpCallbacks",
    "validate_credentials",
    "get_default_config",
    # Conversation types
    "Message",
    # Exceptions
    "ToxpError",
    "CredentialsError",
    "ConfigurationError",
    "ProviderError",
    "ModelNotFoundError",
    "ThrottlingError",
    "InsufficientAgentsError",
    "NetworkError",
    "TimeoutError",
]
