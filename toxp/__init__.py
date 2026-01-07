"""TOXP CLI - Team of eXPerts parallel reasoning system."""

import sys

__version__ = "0.2.0"
__all__ = ["__version__"]

# Runtime Python version check - catches cases where wrong package was installed
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    sys.exit(
        f"Error: TOXP requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, "
        f"but you're running Python {sys.version_info.major}.{sys.version_info.minor}\n\n"
        f"Fix: Use uvx with --python flag:\n"
        f"  uvx --python 3.12 --from team-of-experts txp \"your question\"\n\n"
        f"Or install Python 3.10+ and reinstall TOXP."
    )
