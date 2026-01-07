# TOXP - Team Of eXPerts

Parallel reasoning CLI using multiple LLM agents. TOXP spawns N independent reasoning agents to tackle complex queries, then synthesizes their outputs through a coordinator agent into a coherent, high-confidence answer.

## Features

- **Parallel Reasoning**: Spawn 2-32 independent reasoning agents
- **Intelligent Synthesis**: Coordinator analyzes agreements, contradictions, synthesizes best answer
- **Confidence Levels**: Low/Medium/High confidence ratings
- **Streaming Output**: Real-time coordinator synthesis
- **Session Logging**: Markdown logs with token counts and cost estimates

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
| `num-agents` | `16` | Parallel agents (2-32) |
| `temperature` | `0.9` | Agent temperature |
| `model` | `claude-sonnet-4-5` | Model ID |

Environment variables: `TOXP_AWS_PROFILE`, `TOXP_REGION`, `TOXP_NUM_AGENTS`

## CLI Reference

```
toxp [OPTIONS] [QUERY]

Options:
  -q, --query TEXT       Query string
  -n, --num-agents INT   Agents (2-32)
  -t, --temperature FLOAT
  --aws-profile TEXT
  --region TEXT
  -o, --output FILE
  -v, --verbose
  --quiet
  --help
```

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
