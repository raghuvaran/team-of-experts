"""Live smoke tests against real AWS Bedrock.

These tests make REAL API calls and incur costs. They are skipped by default.

Run:
    pytest -m live                          # all live tests
    pytest -m live -k opus_46               # just opus 4.6 tests

Configuration via environment variables:
    TOXP_LIVE_PROFILE   AWS profile name  (default: "default")
    TOXP_LIVE_REGION    AWS region        (default: "us-east-1")
"""

import asyncio
import os

import pytest

from toxp.providers.bedrock import BedrockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _profile():
    return _env("TOXP_LIVE_PROFILE", "default")


def _region():
    return _env("TOXP_LIVE_REGION", "us-east-1")


def _provider(model_id: str, context_1m: bool = False):
    return BedrockProvider(
        region=_region(),
        model_id=model_id,
        aws_profile=_profile(),
        context_1m=context_1m,
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestBedrockLiveSmoke:
    """Minimal smoke tests — tiny prompts, max_tokens=1, just verify 200 OK."""

    def test_default_sonnet_responds(self):
        """Regression: the default Sonnet model still works."""
        provider = _provider("us.anthropic.claude-sonnet-4-5-20250929-v1:0")

        async def _run():
            return await provider.invoke_model(
                system_prompt="Reply with one word.",
                user_message="Hi",
                temperature=0.0,
                max_tokens=1,
            )

        resp = asyncio.run(_run())
        assert resp.text
        assert resp.input_tokens > 0

    def test_opus_46_global_responds(self):
        """Opus 4.6 global inference profile accepts a converse() call."""
        provider = _provider("global.anthropic.claude-opus-4-6-v1")

        async def _run():
            return await provider.invoke_model(
                system_prompt="Reply with one word.",
                user_message="Hi",
                temperature=0.0,
                max_tokens=1,
            )

        resp = asyncio.run(_run())
        assert resp.text
        assert resp.model_id == "global.anthropic.claude-opus-4-6-v1"

    def test_opus_46_us_responds(self):
        """Opus 4.6 US cross-region profile accepts a converse() call."""
        provider = _provider("us.anthropic.claude-opus-4-6-v1")

        async def _run():
            return await provider.invoke_model(
                system_prompt="Reply with one word.",
                user_message="Hi",
                temperature=0.0,
                max_tokens=1,
            )

        resp = asyncio.run(_run())
        assert resp.text

    def test_opus_46_with_1m_context_header(self):
        """Bedrock accepts the context-1m beta header without error."""
        provider = _provider("global.anthropic.claude-opus-4-6-v1", context_1m=True)

        async def _run():
            return await provider.invoke_model(
                system_prompt="Reply with one word.",
                user_message="Hi",
                temperature=0.0,
                max_tokens=1,
            )

        resp = asyncio.run(_run())
        assert resp.text

    def test_opus_46_streaming_responds(self):
        """Opus 4.6 streaming via converse_stream() works."""
        provider = _provider("global.anthropic.claude-opus-4-6-v1")

        async def _run():
            tokens = []
            async for token in provider.invoke_model_stream(
                system_prompt="Reply with one word.",
                user_message="Hi",
                temperature=0.0,
                max_tokens=5,
            ):
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_run())
        assert len(tokens) > 0
        assert "".join(tokens).strip()
