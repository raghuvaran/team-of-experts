# TOXP - Team Of eXPerts

Parallel reasoning CLI and Python library using multiple LLM agents. TOXP spawns N independent reasoning agents to tackle complex queries, then synthesizes their outputs through a coordinator agent into a coherent, high-confidence answer.

## Features

- **Parallel Reasoning**: Spawn 2-32 independent reasoning agents
- **Intelligent Synthesis**: Coordinator analyzes agreements, contradictions, synthesizes best answer
- **Confidence Levels**: Low/Medium/High confidence ratings
- **Streaming Output**: Real-time coordinator synthesis
- **Session Logging**: Markdown logs with token counts and cost estimates
- **Python API**: Use `run_query()` directly from your own code
- **1M Context**: Optional 1M token context window for supported models (Opus 4.6)

## Prerequisites

- **Python 3.10+**
- **AWS credentials** with Bedrock access:
  ```bash
  aws configure --profile your-profile
  ```
- **Claude model access** in AWS Bedrock console

## Installation

```bash
# Using uvx (no install needed)
uvx toxp "Your question"

# Or install permanently
uv tool install toxp    # recommended
pipx install toxp       # alternative
pip install toxp        # in current env
```

## Upgrading

```bash
uv tool upgrade toxp    # if installed with uv
pipx upgrade toxp       # if installed with pipx
pip install -U toxp     # if installed with pip
```

## Setup

```bash
toxp config set aws-profile your-profile
toxp config show
```

## Quick Start

```bash
toxp "Solve: x^2 + 5x + 6 = 0"
echo "Explain recursion" | toxp
toxp -v "Analyze quicksort"        # verbose
toxp --quiet "What is 2 + 2?"      # only answer
toxp --output answer.txt "Question"
toxp --context-1m "Summarize this very long document..."
toxp --html "Compare PostgreSQL vs MongoDB for time-series data"
toxp --html --html-style "minimal academic paper" "Explain CRDTs"
```

## Python API

```python
from toxp import run_query, QueryResult, validate_credentials

# Simple usage
result = await run_query("What is 2+2?")
print(result.final_answer, result.confidence)

# With overrides and callbacks
result = await run_query(
    "Explain recursion",
    config_overrides={"num_agents": 8, "temperature": 0.7},
    callbacks=MyCallbacks(),
    cancel_token=my_cancel_event,
)

# Check credentials before running
validate_credentials()
```

## Configuration

Stored at `~/.toxp/config.json`:

```bash
toxp config show                    # view all
toxp config get model               # get value
toxp config set num-agents 24       # set value
toxp config reset                   # reset defaults
```

| Key | Default | Description |
|-----|---------|-------------|
| `aws-profile` | `default` | AWS profile |
| `region` | `us-east-1` | AWS region |
| `num-agents` | `15` | Parallel agents (2-32) |
| `temperature` | `0.9` | Agent temperature |
| `coordinator-temperature` | `0.7` | Coordinator temperature |
| `model` | `claude-opus-4-7` | Model ID (supports Sonnet 4.5, Opus 4.6, Opus 4.7) |
| `max-tokens` | `8192` | Max tokens per response |
| `max-concurrency` | `auto` | Max concurrent API requests |
| `context-1m` | `false` | Enable 1M token context window beta |
| `log-enabled` | `true` | Enable session logging |
| `log-retention-days` | `30` | Days to keep logs |

Environment variables: `TOXP_AWS_PROFILE`, `TOXP_REGION`, `TOXP_NUM_AGENTS`

## CLI Reference

```
toxp [OPTIONS] [QUERY]

Options:
  -q, --query TEXT         Query string
  -n, --num-agents INT     Agents (2-32)
  -t, --temperature FLOAT
  -c, --max-concurrency N  Max concurrent API requests
  --aws-profile TEXT
  --region TEXT
  --context-1m             Enable 1M token context window beta
  --html                   Generate a rich self-contained HTML artifact and open it
  --html-style TEXT        Style hint for the HTML artifact (e.g. "dashboard")
  -o, --output FILE
  -v, --verbose
  --quiet
  --help
```

### HTML artifacts

`--html` swaps the terminal markdown output for a rich, self-contained HTML
file that opens in your browser. The HTML is authored by a single additional
LLM call (so it costs ~one extra request and a few extra seconds) and is
asked to use SVG diagrams, tables, color-coded sections, and a prominent
"Final Answer" hero block.

- **Where it lands**: `~/.toxp/html/<timestamp>-<query-slug>.html` by
  default. Pass `-o path/to/file.html` to override.
- **Self-contained**: all CSS and JS are inlined, no external requests —
  shareable by just sending the file.
- **Retention**: old HTML files are aged out using the same
  `log-retention-days` config as session logs.
- **`--html-style TEXT`**: optional free-form hint (e.g. `"dashboard"`,
  `"academic paper"`, `"playful"`) appended to the HTML-author prompt.
- **Fallback**: if the HTML pass fails for any reason, toxp falls back to
  printing the markdown answer so you never lose your result.

## How It Works

1. Query sent to N reasoning agents (T=0.9)
2. At least 50% must succeed
3. Coordinator synthesizes responses
4. Final answer with confidence level

## Troubleshooting

```bash
# Credentials expired
aws sso login --profile your-profile

# Rate limiting
toxp config set num-agents 8

# Debug
toxp -v "Your question"
```

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

## License

MIT
