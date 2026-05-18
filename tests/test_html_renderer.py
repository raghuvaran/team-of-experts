"""Tests for toxp.output.html_renderer."""

import asyncio
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator, List, Optional

import pytest

from toxp.api import QueryResult
from toxp.models.query import Query
from toxp.models.response import AgentResponse, CoordinatorResponse
from toxp.models.result import Result
from toxp.output.html_renderer import (
    HTML_AUTHOR_SYSTEM_PROMPT,
    HtmlGenerationError,
    build_html_author_user_message,
    cleanup_old_html,
    default_html_path,
    generate_html_artifact,
    open_in_browser,
    safe_generate,
    slugify_query,
    strip_code_fences,
    write_html,
)
from toxp.providers.base import BaseProvider, ProviderResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(
    *,
    query_text: str = "What is the capital of France?",
    query_id: str = "abcd1234",
    timestamp: Optional[datetime] = None,
    final_answer: str = "Paris.",
    confidence: str = "High",
    synthesis: str = "Synthesis text.",
    consensus: str = "Agreed it's Paris.",
    debates: str = "No major debates.",
    successful: int = 12,
    total: int = 15,
    tokens: int = 47238,
    duration: float = 38.4,
    with_raw: bool = True,
) -> QueryResult:
    """Build a QueryResult with optional raw_result for tests."""
    raw: Optional[Result] = None
    if with_raw:
        ts = timestamp or datetime(2026, 5, 15, 9, 30, 45)
        query = Query(text=query_text, query_id=query_id, timestamp=ts)
        coord = CoordinatorResponse(
            synthesis=synthesis,
            confidence=confidence,
            consensus_summary=consensus,
            debates_summary=debates,
            final_answer=final_answer,
            duration_seconds=2.5,
        )
        agents: List[AgentResponse] = [
            AgentResponse(agent_id=i, success=i < successful, final_answer="x")
            for i in range(total)
        ]
        raw = Result(
            query=query,
            agent_responses=agents,
            coordinator_response=coord,
            metadata={
                "total_agent_tokens": tokens,
                "total_duration_seconds": duration,
            },
        )

    return QueryResult(
        final_answer=final_answer,
        confidence=confidence,
        synthesis_markdown=synthesis,
        consensus_summary=consensus,
        debates_summary=debates,
        total_agents=total,
        successful_agents=successful,
        failed_agents=total - successful,
        total_tokens=tokens,
        total_duration_seconds=duration,
        raw_result=raw,
    )


class _StubStreamProvider(BaseProvider):
    """Provider that yields predetermined chunks from invoke_model_stream."""

    def __init__(self, chunks: List[str]):
        self._chunks = chunks
        self.captured_system_prompt: Optional[str] = None
        self.captured_user_message: Optional[str] = None
        self.captured_temperature: Optional[float] = None
        self.captured_max_tokens: Optional[int] = None

    @property
    def name(self) -> str:
        return "stub"

    @property
    def model_id(self) -> str:
        return "stub-model"

    async def invoke_model(
        self, system_prompt, user_message, temperature, max_tokens, messages=None,
    ) -> ProviderResponse:
        raise NotImplementedError

    async def invoke_model_stream(
        self, system_prompt, user_message, temperature, max_tokens, messages=None,
    ) -> AsyncIterator[str]:
        self.captured_system_prompt = system_prompt
        self.captured_user_message = user_message
        self.captured_temperature = temperature
        self.captured_max_tokens = max_tokens
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


# ---------------------------------------------------------------------------
# strip_code_fences
# ---------------------------------------------------------------------------

class TestStripCodeFences:
    def test_no_fence_passthrough(self):
        text = "<!DOCTYPE html><html></html>"
        assert strip_code_fences(text) == text

    def test_html_fence(self):
        wrapped = "```html\n<!DOCTYPE html><html></html>\n```"
        assert strip_code_fences(wrapped) == "<!DOCTYPE html><html></html>"

    def test_bare_fence(self):
        wrapped = "```\n<!DOCTYPE html><html></html>\n```"
        assert strip_code_fences(wrapped) == "<!DOCTYPE html><html></html>"

    def test_fence_with_leading_whitespace(self):
        wrapped = "   ```html\n<!DOCTYPE html>\n```"
        assert strip_code_fences(wrapped) == "<!DOCTYPE html>"

    def test_no_closing_fence_keeps_content(self):
        # Open fence but no close — open is stripped, content remains
        partial = "```html\n<!DOCTYPE html>"
        assert strip_code_fences(partial) == "<!DOCTYPE html>"


# ---------------------------------------------------------------------------
# slugify_query
# ---------------------------------------------------------------------------

class TestSlugifyQuery:
    def test_ascii(self):
        assert slugify_query("Hello World") == "hello-world"

    def test_unicode_normalized(self):
        # Accented chars get stripped, not raw-passed
        assert slugify_query("café résumé") == "cafe-resume"

    def test_punctuation_only_returns_query(self):
        assert slugify_query("!!!???") == "query"

    def test_empty_returns_query(self):
        assert slugify_query("") == "query"

    def test_long_query_truncates_cleanly(self):
        text = "a" * 100
        slug = slugify_query(text, max_len=10)
        assert len(slug) <= 10
        assert not slug.endswith("-")

    def test_cjk_falls_back_to_query(self):
        # CJK strips to nothing under ASCII fold → 'query'
        assert slugify_query("日本語") == "query"

    def test_internal_punctuation_collapses(self):
        assert slugify_query("foo!!!bar...baz") == "foo-bar-baz"


# ---------------------------------------------------------------------------
# default_html_path
# ---------------------------------------------------------------------------

class TestDefaultHtmlPath:
    def test_uses_timestamp_and_slug_from_raw_result(self, tmp_path: Path):
        result = make_result(
            query_text="Compare PostgreSQL vs MongoDB",
            query_id="aabbccdd",
            timestamp=datetime(2026, 5, 15, 9, 30, 45),
        )
        path = default_html_path(result, base_dir=tmp_path)
        assert path.parent == tmp_path
        assert path.name == "2026-05-15_093045-compare-postgresql-vs-mongodb.html"

    def test_falls_back_when_raw_result_missing(self, tmp_path: Path):
        result = make_result(with_raw=False)
        before = datetime.now() - timedelta(seconds=2)
        path = default_html_path(result, base_dir=tmp_path)
        after = datetime.now() + timedelta(seconds=2)

        assert path.parent == tmp_path
        assert path.suffix == ".html"
        assert path.name.endswith("-query.html")

        ts_str = path.name.split("-query.html")[0]
        parsed = datetime.strptime(ts_str, "%Y-%m-%d_%H%M%S")
        assert before <= parsed <= after


# ---------------------------------------------------------------------------
# write_html
# ---------------------------------------------------------------------------

class TestWriteHtml:
    def test_creates_parent_dirs_and_writes(self, tmp_path: Path):
        dest = tmp_path / "deep" / "nested" / "out.html"
        write_html("<!DOCTYPE html><body/></html>", dest)
        assert dest.exists()
        assert dest.read_text() == "<!DOCTYPE html><body/></html>"
        # No leftover .tmp file
        assert not (dest.parent / "out.html.tmp").exists()

    def test_overwrites_existing_atomically(self, tmp_path: Path):
        dest = tmp_path / "x.html"
        dest.write_text("OLD")
        write_html("NEW", dest)
        assert dest.read_text() == "NEW"
        assert not (tmp_path / "x.html.tmp").exists()


# ---------------------------------------------------------------------------
# build_html_author_user_message
# ---------------------------------------------------------------------------

class TestBuildHtmlAuthorUserMessage:
    def test_includes_all_fields(self):
        result = make_result(
            query_text="My query",
            final_answer="My final answer",
            confidence="Medium",
            synthesis="Full synthesis text",
            consensus="What agents agreed on",
            debates="Where they disagreed",
            successful=10, total=12, tokens=12345, duration=4.5,
        )
        msg = build_html_author_user_message(result)

        assert "My query" in msg
        assert "Medium" in msg
        assert "10/12" in msg
        assert "12,345" in msg
        assert "4.5s" in msg
        assert "My final answer" in msg
        assert "What agents agreed on" in msg
        assert "Where they disagreed" in msg
        assert "Full synthesis text" in msg

    def test_handles_missing_raw_result(self):
        result = make_result(with_raw=False)
        msg = build_html_author_user_message(result)
        # Query appears as empty in QUERY block but message must still be valid
        assert msg.startswith("QUERY:\n")
        assert result.final_answer in msg


# ---------------------------------------------------------------------------
# generate_html_artifact
# ---------------------------------------------------------------------------

class TestGenerateHtmlArtifact:
    def test_streams_and_strips_fences(self):
        chunks = ["```html\n", "<!DOCTYPE html>", "<html><body>hi</body>", "</html>", "\n```"]
        provider = _StubStreamProvider(chunks)

        seen: List[str] = []
        html = asyncio.run(
            generate_html_artifact(
                make_result(),
                provider=provider,
                on_token=lambda t: seen.append(t),
                temperature=0.5,
                max_tokens=12000,
            )
        )

        assert html.startswith("<!DOCTYPE html>")
        assert html.endswith("</html>")
        assert seen == chunks
        # Temperature/max_tokens propagated
        assert provider.captured_temperature == 0.5
        assert provider.captured_max_tokens == 12000

    def test_raises_on_non_html_output(self):
        provider = _StubStreamProvider(["plain text only, no doctype"])
        with pytest.raises(HtmlGenerationError):
            asyncio.run(generate_html_artifact(make_result(), provider=provider))

    def test_raises_on_empty_output(self):
        provider = _StubStreamProvider([])
        with pytest.raises(HtmlGenerationError):
            asyncio.run(generate_html_artifact(make_result(), provider=provider))

    def test_style_hint_appended_to_system_prompt(self):
        provider = _StubStreamProvider(["<!DOCTYPE html><html></html>"])
        asyncio.run(
            generate_html_artifact(
                make_result(),
                provider=provider,
                style_hint="minimal academic paper",
            )
        )
        sp = provider.captured_system_prompt or ""
        assert HTML_AUTHOR_SYSTEM_PROMPT in sp
        assert "STYLE HINT FROM USER: minimal academic paper" in sp

    def test_no_style_hint_uses_base_prompt(self):
        provider = _StubStreamProvider(["<!DOCTYPE html><html></html>"])
        asyncio.run(generate_html_artifact(make_result(), provider=provider))
        assert provider.captured_system_prompt == HTML_AUTHOR_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# safe_generate
# ---------------------------------------------------------------------------

class TestSafeGenerate:
    def test_returns_html_on_success(self):
        provider = _StubStreamProvider(["<!DOCTYPE html><html></html>"])
        html, err = asyncio.run(safe_generate(make_result(), provider=provider))
        assert err is None
        assert html and html.startswith("<!DOCTYPE html>")

    def test_returns_error_on_failure(self):
        provider = _StubStreamProvider(["nope"])
        html, err = asyncio.run(safe_generate(make_result(), provider=provider))
        assert html is None
        assert isinstance(err, HtmlGenerationError)


# ---------------------------------------------------------------------------
# cleanup_old_html
# ---------------------------------------------------------------------------

class TestCleanupOldHtml:
    def test_deletes_files_older_than_retention(self, tmp_path: Path):
        old = tmp_path / "old.html"
        new = tmp_path / "new.html"
        old.write_text("<html/>")
        new.write_text("<html/>")

        old_time = (datetime.now() - timedelta(days=40)).timestamp()
        os.utime(old, (old_time, old_time))

        deleted = cleanup_old_html(tmp_path, retention_days=30)

        assert deleted == 1
        assert not old.exists()
        assert new.exists()

    def test_returns_zero_when_dir_missing(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        assert cleanup_old_html(missing, retention_days=30) == 0

    def test_ignores_non_html_files(self, tmp_path: Path):
        other = tmp_path / "old.md"
        other.write_text("md")
        old_time = (datetime.now() - timedelta(days=40)).timestamp()
        os.utime(other, (old_time, old_time))

        cleanup_old_html(tmp_path, retention_days=30)
        assert other.exists()


# ---------------------------------------------------------------------------
# open_in_browser — patched so no real browser launches
# ---------------------------------------------------------------------------

class TestOpenInBrowser:
    def test_calls_webbrowser_open_with_file_url(self, tmp_path: Path, monkeypatch):
        captured: dict = {}

        def fake_open(url, *args, **kwargs):
            captured["url"] = url
            return True

        monkeypatch.setattr("toxp.output.html_renderer.webbrowser.open", fake_open)

        f = tmp_path / "x.html"
        f.write_text("<html/>")
        assert open_in_browser(f) is True
        assert captured["url"].startswith("file://")
        assert captured["url"].endswith(str(f.resolve()))
