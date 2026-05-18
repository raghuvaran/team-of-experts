# TOXP `--html` Rich HTML Output

**Status**: Implemented
**Date**: 2026-05-15
**Target version**: TBD (no version bump yet)
**Scope**: Backward-compatible — CLI default behavior unchanged, new `--html` flag and `render_html()` library function added

---

## 1. Motivation

TOXP today streams the coordinator's synthesis to the terminal as Markdown
and prints the final answer at the end. As syntheses get longer and more
structured, plain Markdown-in-a-terminal becomes a poor reading surface — no
diagrams, no color, no tables, hard to scan, hard to share.

Inspired by the article "Using Claude Code: The Unreasonable Effectiveness
of HTML" (Thariq, May 2026), this feature lets the user opt into a rich,
**self-contained HTML artifact** that opens in the browser. Crucially, the
artifact is *authored* by the model (with SVG diagrams, tables, color-coded
sections, hero blocks) — not produced by mechanically rendering the existing
Markdown into HTML. That distinction is the whole point: HTML is only "more
effective" when the model actually uses its expressive range.

### Goals

- Single new CLI flag `--html` that **replaces** the Markdown stream and
  final-answer print with a self-contained HTML artifact opened in the
  default browser.
- Optional `--html-style TEXT` for free-form style guidance (e.g.
  "minimal academic paper", "dashboard").
- New public library function `render_html(result, ...)` so non-CLI
  consumers (Open WebUI, desktop apps, scripts) can produce the same artifact.
- 100% backward compatible: without `--html`, behavior is byte-identical
  to today. No new top-level dependencies.
- Robust fallback: if the HTML pass fails for any reason, fall back to the
  existing Markdown final-answer print so the user never loses their result.

### Non-Goals

- Markdown-to-HTML mechanical rendering. (Renders as flat Markdown-in-HTML;
  misses the article's whole point.)
- Persistence of HTML artifacts as a first-class concept (no "history"
  command). They're written to `~/.toxp/html/` and aged out by retention.
- A `/html` skill, prompt template marketplace, or CSS-theme system.
- Streaming the HTML to the terminal token-by-token. The artifact is
  written all at once after the LLM call completes.

---

## 2. Architecture Overview

```
                  ┌────────────────────────────────────┐
                  │   handle_query_command (cli.py)    │
                  └─────────────────┬──────────────────┘
                                    │
                          run_query(query, ...)
                                    │
                                    ▼
                          QueryResult returned
                                    │
                       ┌────────────┴────────────┐
                  no --html                  --html
                       │                         │
                       ▼                         ▼
              existing markdown          render_html(result, ...)
              final-answer path                  │
                                          ┌──────┴──────┐
                                       success       failure
                                          │             │
                                          ▼             ▼
                                   write_html    fall back to
                                   open_in_browser  markdown print
```

The `--html` path is a strict superset of the no-flag path: same
`run_query`, same `QueryResult`, same orchestrator/agents/providers. The
only thing that changes is **what we do with the result after it's
synthesized**.

### Why a second LLM call (and not, say, markdown→HTML conversion)?

The article's argument — and our experience — is that the value of HTML
output comes from the model authoring real visual structure: SVG diagrams,
color-coded callouts, tables, hero blocks for the final answer. Mechanically
rendering Markdown produces Markdown-in-HTML, which has none of that. The
extra cost (one streaming call, ~5–15 seconds, a few cents) is worth it.

### Why not stream the HTML to terminal?

HTML is not a terminal-readable format. Streaming partial HTML tokens
provides no signal during generation and just clutters the terminal. We show
a Rich `Status` spinner ("Generating HTML artifact...") while the call
runs, then write+open at the end.

---

## 3. What Changes

| File | Change Type | Description |
|---|---|---|
| `toxp/output/html_renderer.py` | **NEW** | All HTML logic: prompt constants, `generate_html_artifact`, `default_html_path`, `slugify_query`, `write_html`, `open_in_browser`, `cleanup_old_html`, `strip_code_fences`, `safe_generate`, `HtmlGenerationError`. |
| `toxp/api.py` | MODIFY | Add `async def render_html(result, *, config_overrides, style_hint, on_token)`. |
| `toxp/__init__.py` | MODIFY | Re-export `render_html` and add to `__all__`. |
| `toxp/cli.py` | MODIFY | Add `--html` and `--html-style` flags inside `_add_query_flags` (single source of truth). Add `suppress_coordinator_stream` kwarg to `CLICallbacks`. Branch in `handle_query_command` after `run_query` returns. New helper `_render_and_open_html` with try/except fallback. |
| `tests/test_html_renderer.py` | **NEW** | 29 unit tests covering all renderer functions. |
| `README.md` | MODIFY | Add the two flags to the CLI Reference table; add Quick Start examples; add "HTML artifacts" subsection. |

No new top-level dependencies. `webbrowser` is stdlib; `rich` is already a
required dep.

Files explicitly **not** modified: `pyproject.toml`, `toxp/orchestrator.py`,
`toxp/agents/*.py`, `toxp/providers/*.py`, `toxp/output/formatter.py`,
`toxp/output/progress.py`, `toxp/logging/session_logger.py`, `ui/toxp_pipe.py`,
`ui/lobe/toxp_api.py`.

---

## 4. Detailed Design

### 4.1 `toxp/output/html_renderer.py`

Pure functions where possible; one async function for the LLM call.

```python
HTML_AUTHOR_SYSTEM_PROMPT: str
HTML_DIR = Path.home() / ".toxp" / "html"

class HtmlGenerationError(RuntimeError): ...

def build_html_author_user_message(
    result: QueryResult, *, style_hint: Optional[str] = None,
) -> str

async def generate_html_artifact(
    result: QueryResult, *,
    provider: BaseProvider,
    style_hint: Optional[str] = None,
    on_token: Optional[Callable[[str], None]] = None,
    temperature: float = 0.7,
    max_tokens: int = 16384,
) -> str
    # Streams provider.invoke_model_stream(...), accumulates chunks,
    # strip_code_fences, validates output starts with <!DOCTYPE / <html.
    # Raises HtmlGenerationError on empty / non-HTML output.

def strip_code_fences(text: str) -> str
def slugify_query(text: str, *, max_len: int = 48) -> str
def default_html_path(result, base_dir=None) -> Path
    # ~/.toxp/html/<YYYY-MM-DD_HHMMSS>-<slug>.html
def write_html(html: str, dest_path: Path) -> Path
    # Atomic: write to <dest>.tmp then os.replace
def open_in_browser(path: Path) -> bool
def cleanup_old_html(base_dir: Path, retention_days: int) -> int
async def safe_generate(...) -> tuple[Optional[str], Optional[Exception]]
```

#### HTML-author system prompt (excerpt)

The prompt is opinionated about output rules to keep generation reliable:

- Output ONLY a complete HTML5 document starting with `<!DOCTYPE html>`.
- No prose, no code fences, no leading/trailing text.
- Self-contained: all CSS in a single `<style>` tag, JS inline, **no
  external URLs** (no CDN fonts/images, no `<link rel="stylesheet">`).
- Mobile-responsive (~880px container, viewport meta tag).
- System font stack, line-height 1.6, semantic HTML, prominent "Final
  Answer" hero block, color-coded confidence pill (green/amber/red),
  agent-stats line, color-coded callout cards for Consensus vs Debates,
  inline `<svg>` for processes/comparisons/architectures, real `<table>`
  for tabular data, `prefers-color-scheme` for light/dark.

`{style_hint_block}` is appended at the end of the system prompt only if
`--html-style` was given:

```
STYLE HINT FROM USER: <verbatim hint text>
```

#### Defensive fence-stripping

Despite "No markdown code fences" being explicit in the prompt, models
occasionally wrap the output in ` ```html ... ``` `. `strip_code_fences`
removes a leading ` ```html ` / ` ``` ` and trailing ` ``` ` if present.
After stripping, if the result doesn't start with `<!DOCTYPE` or `<html`
(case-insensitive), `HtmlGenerationError` is raised so the CLI fallback
path triggers.

#### Filename scheme

`<YYYY-MM-DD_HHMMSS>-<slug>.html` where `<slug>` is the first 48 ASCII
characters of the query, lowercased, hyphen-separated. Examples:

- `2026-05-15_093045-compare-postgresql-vs-mongodb.html`
- `2026-05-15_093045-explain-crdts.html`
- `2026-05-15_093045-query.html` (CJK / punctuation-only / empty queries)

This is a **sibling scheme** to `SessionLogger`'s
`<YYYY-MM-DD_HHMMSS>_<query_id>.md` — close enough that both files for
the same query sort adjacent in directory listings, but human-readable.
Collisions are effectively impossible at second resolution + slug-of-query.

### 4.2 `toxp/api.py` — `render_html()`

```python
HTML_MIN_MAX_TOKENS = 16384  # rich HTML+SVG can exceed default 8192 floor

async def render_html(
    result: QueryResult, *,
    config_overrides: Optional[dict] = None,
    style_hint: Optional[str] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Generate a self-contained HTML artifact for an existing QueryResult."""
```

Internals mirror the provider-construction block of `run_query`:

1. `ConfigManager().load_with_overrides(config_overrides)` → `config`.
2. `ProviderRegistry.get(config.provider)(...)` → fresh provider.
3. `await generate_html_artifact(result, provider=provider,
   temperature=config.coordinator_temperature,
   max_tokens=max(config.max_tokens, HTML_MIN_MAX_TOKENS), ...)`.

Reusing the same provider construction keeps the HTML pass on the same
model, region, and AWS profile as the synthesis. Building a fresh provider
is cheap (no API call — just a boto3 client).

`render_html` does **not** touch the filesystem or open a browser. Those
are CLI concerns. Library users decide what to do with the returned HTML
string.

### 4.3 `toxp/cli.py` integration

#### New flags (in `_add_query_flags` — single source of truth)

```python
parser.add_argument("--html", action="store_true",
                    help="Generate a rich self-contained HTML artifact and "
                    "open it in the browser instead of streaming markdown")
parser.add_argument("--html-style", dest="html_style", metavar="TEXT",
                    help="Optional style hint for the HTML artifact "
                    "(e.g. 'minimal academic paper', 'dashboard'). "
                    "Only meaningful with --html.")
```

#### `CLICallbacks` change

A new kwarg-only parameter `suppress_coordinator_stream: bool = False`. In
HTML mode this flips on, making `on_coordinator_token` a no-op so the
synthesis Markdown does not stream to the terminal. All existing call
sites remain valid (default value preserves current behavior).

#### `handle_query_command` flow

After `run_query(...)` returns:

- **No `--html`**: existing path (`stream_end`, `agent_summary`,
  `final_answer`, `-o` plain-text write, session log).
- **With `--html`**: `await _render_and_open_html(...)`, then session log.

`_render_and_open_html` wraps the HTML pass in try/except:

```python
try:
    with console.status("Generating HTML artifact..."):  # skipped if --quiet
        html = await render_html(result, config_overrides=cli_overrides,
                                 style_hint=html_style)
    dest = Path(args.output_file) if args.output_file else default_html_path(result)
    write_html(html, dest)
    if used_default_dir:
        cleanup_old_html(HTML_DIR, config.log_retention_days)
    opened = open_in_browser(dest)
    # print path / message
except Exception as e:
    formatter.warning(f"HTML generation failed: {e}. Falling back to text answer.")
    formatter.final_answer(result.final_answer, confidence_level=result.confidence)
    if args.output_file:
        Path(args.output_file).write_text(result.final_answer)
```

Exit code is 0 in both success and fallback paths — the user got their
answer either way.

### 4.4 Interactions with existing flags

| Existing flag | With `--html` |
|---|---|
| `-o/--output FILE` | Writes HTML to FILE instead of plain text. Auto-open still happens. If extension is not `.html`/`.htm`, emit a warning but proceed. |
| `--quiet` | Suppress all chatter and the spinner. Print only the final file path on stdout. Auto-open still happens. |
| `--no-log` | Honored normally. Session log is independent of HTML artifact. |
| `-v/--verbose` | On HTML failure, prints traceback inside the fallback. |
| `--num-agents`, `--temperature`, etc. | Passed through to `run_query` and to `render_html` via `config_overrides`. |

### 4.5 Retention / cleanup

The default HTML directory `~/.toxp/html/` reuses
`config.log_retention_days` (default 30) for cleanup. This avoids adding
a new config key for what is essentially the same concern as session-log
retention. Cleanup runs only when the destination is the default directory
— if the user passes `-o`, they own the file's lifetime.

---

## 5. What Does NOT Change

- `run_query()` signature, behavior, return shape.
- `QueryResult` shape.
- `Orchestrator`, agents, providers, prompt templates.
- `OutputFormatter`, `progress.py`, `SessionLogger`.
- `ui/toxp_pipe.py`, `ui/lobe/toxp_api.py` (Open WebUI integrations).
- All existing CLI flags and their behavior when `--html` is absent.
- Default exit codes (0 success, 1 error, 130 SIGINT).
- All existing tests pass without modification (none pass `--html`).

---

## 6. Library Usage

```python
import asyncio
from toxp import run_query, render_html

async def main():
    result = await run_query("Compare PostgreSQL vs MongoDB for time-series")
    html = await render_html(result, style_hint="dashboard")
    # Caller decides what to do with the HTML
    Path("report.html").write_text(html)

asyncio.run(main())
```

`render_html` is independent of the CLI — UI consumers (Open WebUI Pipe,
desktop apps) can use it without depending on `toxp.cli` or `argparse`.

---

## 7. Edge Cases & Failure Modes

| Case | Handling |
|---|---|
| Provider/network failure during HTML pass | try/except wrapper → `formatter.warning(...)` → fallback to existing markdown print. Exit 0. |
| HTML output doesn't start with `<!DOCTYPE` after fence stripping | `HtmlGenerationError` → fallback path. |
| HTML output looks complete but is actually truncated mid-stream | Currently written as-is (often still renders). Future enhancement: detect missing `</html>` and surface "(possibly truncated)". |
| `-o file.txt` with `--html` | Write HTML to that path. Warn once that extension doesn't match. Don't fail. |
| `--html-style` without `--html` | Warn and ignore. |
| `~/.toxp/html/` doesn't exist | `write_html` mkdirs parents (also creates the html dir). |
| Same-second filename collision | Slug includes the query text; `query_id` would be a tiebreaker. Atomic `.tmp` → `os.replace` rules out partial reads either way. |
| `webbrowser.open()` returns False (no registered handler) | Treat as not-opened; the file URL is printed so the user can copy/paste. Not an error. |
| Non-TTY stdout (piped output) | Auto-open is still attempted. Path is printed regardless. |
| Coordinator synthesis is huge | Provider already handles. Users can pass `--context-1m` if needed. `max_tokens` floor of 16384 reduces truncation risk for the HTML pass itself. |
| `--quiet --html` | Suppress all chatter and spinner; print only the path. Browser still opens. |

---

## 8. Testing

`tests/test_html_renderer.py` (new) — 29 unit tests covering:

- `strip_code_fences` × {no fence, ```html fence, bare ``` fence, leading whitespace, partial}.
- `slugify_query` × {ascii, unicode (NFKD), empty, punctuation only, very long, CJK, internal punctuation collapse}.
- `default_html_path` × {with raw_result, fallback when raw_result is None}.
- `write_html` atomic with parent-creation; no leftover `.tmp`.
- `build_html_author_user_message` includes all `QueryResult` fields and handles missing `raw_result`.
- `generate_html_artifact` streams chunks, strips fences, propagates temperature/max_tokens; raises on non-HTML and on empty output; appends style hint to system prompt when given.
- `safe_generate` returns `(html, None)` on success and `(None, error)` on failure.
- `cleanup_old_html` deletes files older than retention; ignores non-html; returns 0 when dir missing.
- `open_in_browser` invokes `webbrowser.open` with a `file://` URL (test patches `webbrowser.open`).

Full suite (`pytest -m "not live"`): 274 tests passing — no regressions
introduced by this change.

---

## 9. Future Enhancements

- **Truncation detection**: If the streamed HTML doesn't end with `</html>`,
  surface a "(possibly truncated)" warning and consider a retry.
- **Style presets**: Persist named `--html-style` aliases in
  `~/.toxp/config.json` (e.g. `toxp config set html-style dashboard`).
- **Inline images**: When the model wants to reference a specific image
  (e.g. an architecture diagram from the user's repo), allow embedding via
  data URIs.
- **Server-rendered share links**: Optional `--share` flag that uploads the
  HTML to a user-configured object store and prints a URL.
- **Continued conversations from HTML artifacts**: An "Ask follow-up"
  control in the artifact that copies a prompt back into the user's
  clipboard for the next `toxp` invocation. (Pure HTML/JS; no server.)
