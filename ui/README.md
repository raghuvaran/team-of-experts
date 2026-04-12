# TOXP Web UI

Chat interface for TOXP multi-agent reasoning, powered by [Open WebUI](https://github.com/open-webui/open-webui).

## Quick Start

```bash
cd ui/
./start.sh
```

That's it. The script will:
1. Build a container with Open WebUI + TOXP
2. Start it on http://localhost:3000
3. Create an admin account
4. Register the TOXP pipe function
5. Open your browser

Login: `admin@toxp.local` / `toxp-admin`

Select **"TOXP - Team of Experts"** from the model dropdown and start chatting.

### Prerequisites

- **finch** or **docker** (the script auto-detects which you have)
- **AWS credentials** configured (`~/.aws/credentials` or environment variables)

### Commands

```bash
./start.sh              # start (builds on first run)
./start.sh --rebuild    # force rebuild after code changes
./start.sh --stop       # stop the container
```

## Configuration

### Per-chat (any user)

Click **Controls** (top-right) > expand **Valves**:

| Setting | Default | Description |
|---------|---------|-------------|
| Num Agents | 15 | Parallel reasoning agents (2-32) |
| Max Concurrency | 5 | Concurrent Bedrock API calls |
| Temperature | 0.9 | Agent diversity (higher = more varied) |
| Coordinator Temperature | 0.7 | Synthesis focus |
| Max Tokens | 8192 | Max response length |

### Admin (infrastructure)

Go to **Workspace > Functions > TOXP** (gear icon):

| Setting | Default | Description |
|---------|---------|-------------|
| AWS Profile | *(empty)* | Named profile from `~/.aws` (empty = env vars / toxp config) |
| Region | *(empty)* | AWS region (empty = inherit from toxp config) |
| Model | *(empty)* | Bedrock model ID (empty = inherit from toxp config) |
| Context 1M | *(none)* | Enable 1M token context for Opus 4.6 |

Empty settings inherit from your existing `~/.toxp/config.json` (set via `toxp config set ...`).

## How It Works

```
Browser  -->  Open WebUI  -->  TOXP Pipe  -->  N agents (Bedrock)
                                                     |
                                              Coordinator
                                                     |
                                         Streamed to browser
```

The **Pipe function** (`toxp_pipe.py`) bridges Open WebUI to TOXP's `run_query()` API.
It extracts conversation history from the chat, calls TOXP with the current question + prior turns, and streams the coordinator's synthesis back. Progress updates appear in the status bar.

## Files

```
ui/
  start.sh            # One-command setup (build, start, register, open browser)
  toxp_pipe.py        # Open WebUI Pipe function
  Dockerfile          # Open WebUI + TOXP image
  docker-compose.yml  # Container configuration
  .dockerignore       # Build exclusions
```
