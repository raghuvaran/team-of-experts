# TXP CLI - Team of eXPerts

A CLI tool that implements parallel test-time compute using multiple LLM reasoning agents. TXP spawns N independent reasoning agents to tackle complex queries, then synthesizes their outputs through a coordinator agent into a coherent, high-confidence answer.

## Features

- **Parallel Reasoning**: Spawn 2-32 independent reasoning agents for diverse perspectives
- **Intelligent Synthesis**: Coordinator agent analyzes agreements, contradictions, and synthesizes the best answer
- **Confidence Levels**: Get Low/Medium/High confidence ratings with your answers
- **Streaming Output**: Real-time streaming of coordinator synthesis
- **Session Logging**: Automatic markdown logs with token counts and cost estimates
- **Flexible Input**: Query via arguments, flags, or stdin piping
- **Configurable**: Persistent configuration with CLI, environment, and file-based settings

## Installation

### Using uvx (recommended)

```bash
uvx txp "Your complex question here"
```

### Using pipx

```bash
pipx install team-of-experts
txp "Your complex question here"
```

### Using pip

```bash
pip install team-of-experts
```

## Prerequisites

- Python 3.10+
- AWS credentials configured with Bedrock access
- Claude model access in your AWS region (us-east-1 by default)

## Quick Start

```bash
# Basic usage - ask a complex question
txp "Solve: x^2 + 5x + 6 = 0"

# Pipe input from stdin
echo "Explain the implications of Gödel's incompleteness theorems" | txp

# Use explicit query flag
txp --query "What are the trade-offs between microservices and monolithic architectures?"

# Verbose output with agent details
txp -v "Analyze the time complexity of quicksort"

# Quiet mode - only show final answer
txp --quiet "What is 2 + 2?"

# Save output to file
txp --output answer.txt "Explain quantum entanglement"

# Skip session logging for this query
txp --no-log "Quick question"
```

## Usage Examples

### Mathematical Problems

```bash
# Algebra
txp "Solve the system of equations: 2x + 3y = 7, x - y = 1"

# Calculus
txp "Find the derivative of f(x) = x^3 * sin(x)"

# Probability
txp "What is the probability of getting exactly 3 heads in 5 coin flips?"
```

### Programming Questions

```bash
# Algorithm analysis
txp "Explain the difference between BFS and DFS, and when to use each"

# Code review
cat code.py | txp "Review this code for potential bugs and improvements"

# Architecture decisions
txp "Compare REST vs GraphQL for a mobile app backend"
```

### Complex Reasoning

```bash
# Multi-step reasoning
txp "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?"

# Analysis
txp "Analyze the economic implications of universal basic income"

# Synthesis
txp "Compare and contrast the philosophies of Kant and Nietzsche on morality"
```

### Using with Other Tools

```bash
# Pipe from clipboard (macOS)
pbpaste | txp

# Pipe from file
cat question.txt | txp

# Save to file and view
txp "Explain quantum computing" --output answer.md && cat answer.md

# Use with fewer agents for faster response
txp --num-agents 4 "Quick question about Python"

# Use with more agents for complex problems
txp --num-agents 24 "Prove the Pythagorean theorem using three different methods"
```

## Configuration

TXP stores configuration at `~/.txp/config.json`. Manage it via CLI commands:

```bash
# View all configuration
txp config show

# Get a specific value
txp config get model

# Set a value
txp config set num-agents 24
txp config set aws-profile my-profile
txp config set temperature 0.85

# Reset to defaults
txp config reset

# Show config file path
txp config path
```

### Configuration Reference

| Key | Default | Range/Values | Description |
|-----|---------|--------------|-------------|
| `provider` | `bedrock` | `bedrock` | LLM provider (currently only Bedrock supported) |
| `aws-profile` | `default` | Any valid AWS profile | AWS profile for credentials |
| `region` | `us-east-1` | Any AWS region | AWS region for Bedrock API calls |
| `model` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Valid Bedrock model ID | Model ID for reasoning and coordination |
| `num-agents` | `16` | `2-32` | Number of parallel reasoning agents |
| `temperature` | `0.9` | `0.0-1.0` | Sampling temperature for reasoning agents (higher = more diverse) |
| `coordinator-temperature` | `0.7` | `0.0-1.0` | Sampling temperature for coordinator (lower = more focused) |
| `max-tokens` | `8192` | `1-100000` | Maximum tokens per agent response |
| `log-enabled` | `true` | `true/false` | Enable/disable session logging |
| `log-retention-days` | `30` | `1-365` | Days to retain session logs before auto-cleanup |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TXP_AWS_PROFILE` | Override AWS profile (takes precedence over config file) |
| `TXP_REGION` | Override AWS region |
| `TXP_NUM_AGENTS` | Override number of agents |
| `TXP_TEMPERATURE` | Override temperature |
| `TXP_LOG_ENABLED` | Override log enabled setting |

### Configuration Precedence

Configuration values are resolved in this order (highest to lowest priority):

1. **CLI arguments** (e.g., `--num-agents 8`)
2. **Environment variables** (e.g., `TXP_AWS_PROFILE`)
3. **Config file** (`~/.txp/config.json`)
4. **Default values**

## CLI Reference

```
Usage: txp [OPTIONS] [QUERY]

Arguments:
  QUERY                 Query string to process (positional)

Options:
  -q, --query TEXT      Query string (alternative to positional argument)
  -n, --num-agents INT  Number of parallel reasoning agents (2-32)
  -t, --temperature FLOAT
                        Sampling temperature (0.0-1.0)
  -m, --model TEXT      Model ID to use
  --aws-profile TEXT    AWS profile name
  --region TEXT         AWS region
  -o, --output FILE     Write final answer to file
  --no-log              Skip session logging for this execution
  -v, --verbose         Show detailed debug information and stack traces
  --quiet               Only show final answer (no progress or status)
  --version             Show version and exit
  --help                Show this message and exit

Config Commands:
  txp config show       Show all configuration values
  txp config get KEY    Get a configuration value
  txp config set KEY VALUE
                        Set a configuration value
  txp config reset      Reset configuration to defaults
  txp config path       Show configuration file path
```

## How It Works

1. **Query Input** - Your question is sent to N independent reasoning agents
2. **Parallel Reasoning** - Each agent processes the query with chain-of-thought reasoning using high temperature (0.9) for diverse perspectives
3. **Validation** - At least 50% of agents must succeed for synthesis to proceed
4. **Synthesis** - A coordinator agent analyzes all responses:
   - Identifies agreements and contradictions
   - Critiques logical errors and weak reasoning
   - Ranks solutions by correctness and rigor
   - Synthesizes the best answer
5. **Output** - Final answer with confidence level (Low/Medium/High) is displayed

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Orchestrator                          │
│  - Spawns N agents concurrently with rate limiting          │
│  - Validates success rate (≥50%)                            │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Agent 1 │     │ Agent 2 │ ... │ Agent N │
        │ (T=0.9) │     │ (T=0.9) │     │ (T=0.9) │
        └─────────┘     └─────────┘     └─────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Coordinator Agent                        │
│  - Analyzes agreements/contradictions                       │
│  - Critiques reasoning                                      │
│  - Synthesizes final answer with confidence                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Final Answer                           │
│  - Synthesized response                                     │
│  - Confidence level (Low/Medium/High)                       │
└─────────────────────────────────────────────────────────────┘
```

## Session Logs

When logging is enabled, sessions are saved to `~/.txp/logs/sessions/` in markdown format:

```
~/.txp/logs/sessions/2024-12-31_143022_abc12345.md
```

Each log includes:
- Original query
- Final synthesized answer with confidence level
- Agent response summary (success/failure counts, token usage)
- Metadata (timestamps, model versions, cost estimates)

### Example Log Structure

```markdown
# TXP Session Log

**Query ID:** abc12345
**Timestamp:** 2024-12-31T14:30:22
**Model:** us.anthropic.claude-sonnet-4-5-20250929-v1:0
**Agents:** 14/16 successful

## Query
[Your original question]

## Final Answer
**Confidence:** High
[Synthesized answer]

## Agent Summary
| Agent | Status | Tokens | Duration |
|-------|--------|--------|----------|
| 0     | ✓      | 2,341  | 3.2s     |
| 1     | ✓      | 2,156  | 2.8s     |
...

## Metadata
- **Total Tokens:** 45,234
- **Estimated Cost:** $0.41
- **Total Duration:** 12.5s
```

## Troubleshooting

### AWS Credentials Not Found

**Error:** `AWS credentials not found` or `Unable to locate credentials`

**Solutions:**

1. Configure AWS credentials using AWS CLI:
   ```bash
   aws configure --profile your-profile
   ```

2. Set the profile in TXP:
   ```bash
   txp config set aws-profile your-profile
   # Or use environment variable
   export TXP_AWS_PROFILE=your-profile
   ```

3. Verify credentials are working:
   ```bash
   aws sts get-caller-identity --profile your-profile
   ```

### Model Not Available

**Error:** `Model not found: <model-id> in region <region>`

**Solutions:**

1. Check available models in your region:
   ```bash
   aws bedrock list-foundation-models --region us-east-1 \
     --query "modelSummaries[?contains(modelId, 'claude')]"
   ```

2. Ensure you have model access enabled in AWS Bedrock console

3. Try a different region:
   ```bash
   txp config set region us-west-2
   ```

4. Use a different model:
   ```bash
   txp config set model anthropic.claude-3-sonnet-20240229-v1:0
   ```

### Rate Limiting / Throttling

**Error:** `Rate limit exceeded` or `ThrottlingException`

**Solutions:**

1. Reduce number of agents:
   ```bash
   txp config set num-agents 8
   # Or for a single query
   txp --num-agents 4 "Your question"
   ```

2. Wait a few minutes before retrying

3. Check your AWS Bedrock quota limits in the AWS console

4. Request a quota increase if needed

### Insufficient Agents Error

**Error:** `Insufficient agents succeeded: X/Y`

This means fewer than 50% of agents completed successfully.

**Solutions:**

1. Check for rate limiting (see above)

2. Reduce number of agents:
   ```bash
   txp config set num-agents 8
   ```

3. Check your network connectivity

4. Run with verbose mode to see agent errors:
   ```bash
   txp -v "Your question"
   ```

### Network / Connectivity Issues

**Error:** `Network error` or `Connection timeout`

**Solutions:**

1. Check your internet connection

2. Verify AWS services are accessible:
   ```bash
   aws bedrock list-foundation-models --region us-east-1
   ```

3. Check proxy/firewall settings

4. Try again in a few moments

### Timeout Errors

**Error:** `Operation timed out`

**Solutions:**

1. Try with fewer agents for faster response:
   ```bash
   txp --num-agents 4 "Your question"
   ```

2. Simplify your query

3. Check if AWS Bedrock is experiencing issues

### Configuration Issues

**Error:** `Invalid configuration value`

**Solutions:**

1. Reset to defaults:
   ```bash
   txp config reset
   ```

2. Check valid ranges:
   - `num-agents`: 2-32
   - `temperature`: 0.0-1.0
   - `log-retention-days`: 1-365

3. View current configuration:
   ```bash
   txp config show
   ```

### Debug Mode

For detailed debugging information, use verbose mode:

```bash
txp -v "Your question"
```

This will show:
- Configuration being used
- Agent execution progress
- Detailed error messages with stack traces

## Performance Tips

1. **Faster responses**: Use fewer agents (4-8) for simpler questions
2. **Better quality**: Use more agents (16-24) for complex reasoning
3. **Cost optimization**: Lower `num-agents` reduces API costs
4. **Diverse perspectives**: Higher temperature (0.9) gives more varied agent responses
5. **Focused synthesis**: Lower coordinator temperature (0.7) gives more consistent final answers

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/team-of-experts/txp-cli.git
cd txp-cli

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=txp
```

### Code Quality

```bash
# Format code
black txp tests

# Lint
ruff check txp tests

# Type check
mypy txp
```

### Testing Changes

After making code changes, they take effect immediately (editable install). Test manually:

```bash
txp "your test query"
```

## License

MIT
