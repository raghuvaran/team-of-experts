#!/usr/bin/env bash
#
# TOXP Web UI — one-command setup
#
# Usage:
#   ./start.sh              # build + start + register pipe + open browser
#   ./start.sh --rebuild    # force rebuild the image
#   ./start.sh --stop       # stop the container
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${TOXP_PORT:-3000}"

# Write .env for docker-compose.yml volume mounts
# (finch compose doesn't inherit shell exports reliably)
echo "HOST_HOME=${HOME}" > "${SCRIPT_DIR}/.env"
URL="http://localhost:${PORT}"
ADMIN_EMAIL="admin@toxp.local"
ADMIN_PASSWORD="toxp-admin"
COMPOSE_CMD=""

# --- Detect container runtime ---
if command -v finch &>/dev/null; then
    COMPOSE_CMD="finch compose"
elif command -v docker &>/dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "Error: Neither finch nor docker found. Install one of them first."
    exit 1
fi

# --- Handle --stop ---
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping TOXP Web UI..."
    cd "$SCRIPT_DIR" && $COMPOSE_CMD down
    echo "Stopped."
    exit 0
fi

# --- Handle --rebuild ---
BUILD_FLAG=""
if [[ "${1:-}" == "--rebuild" ]]; then
    BUILD_FLAG="--build"
fi

# --- Step 1: Build and start ---
echo "========================================"
echo "  TOXP Web UI"
echo "========================================"
echo ""

cd "$SCRIPT_DIR"

# Generate compose override for host-specific volume mounts
OVERRIDE_FILE="${SCRIPT_DIR}/docker-compose.override.yml"
EXTRA_VOLUMES=""

# Mount ~/.aws if it exists
if [ -d "${HOME}/.aws" ]; then
    EXTRA_VOLUMES="      - ${HOME}/.aws:/root/.aws:ro"
fi

# Mount credential_process binary if it exists
CRED_FETCH="${HOME}/.local/bin/cred-fetch"
if [ -f "$CRED_FETCH" ]; then
    EXTRA_VOLUMES="${EXTRA_VOLUMES}
      - ${CRED_FETCH}:${CRED_FETCH}:ro"
    echo "       credential_process binary found — mounting into container."
fi

if [ -n "$EXTRA_VOLUMES" ]; then
    cat > "$OVERRIDE_FILE" <<YAML
services:
  open-webui:
    volumes:
${EXTRA_VOLUMES}
YAML
else
    rm -f "$OVERRIDE_FILE"
fi

echo "[1/4] Starting container..."
$COMPOSE_CMD up -d $BUILD_FLAG 2>&1 | grep -v "^$"

# --- Step 2: Wait for healthy ---
echo "[2/4] Waiting for Open WebUI to be ready..."
for i in $(seq 1 60); do
    if curl -sf "${URL}/api/version" >/dev/null 2>&1; then
        echo "       Ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "       Timed out waiting for Open WebUI. Check logs:"
        echo "       $COMPOSE_CMD logs"
        exit 1
    fi
    sleep 3
done

# --- Step 3: Create admin user (idempotent — fails silently if exists) ---
echo "[3/4] Setting up admin account..."

SIGNUP_RESP=$(curl -sf -X POST "${URL}/api/v1/auths/signup" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"Admin\",\"email\":\"${ADMIN_EMAIL}\",\"password\":\"${ADMIN_PASSWORD}\"}" 2>/dev/null || echo '{}')

TOKEN=$(echo "$SIGNUP_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")

# If signup failed (user exists), sign in instead
if [ -z "$TOKEN" ]; then
    SIGNIN_RESP=$(curl -sf -X POST "${URL}/api/v1/auths/signin" \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"${ADMIN_EMAIL}\",\"password\":\"${ADMIN_PASSWORD}\"}" 2>/dev/null || echo '{}')
    TOKEN=$(echo "$SIGNIN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")
fi

if [ -z "$TOKEN" ]; then
    echo "       Warning: Could not get auth token. You'll need to register the pipe manually."
    echo "       Go to ${URL}, create an account, then Workspace > Functions > import toxp_pipe.py"
else
    echo "       Admin account ready."

    # --- Step 4: Register TOXP pipe function (idempotent) ---
    echo "[4/4] Registering TOXP pipe function..."

    # Read pipe source from the container (or local file)
    PIPE_CONTENT=$(cat "${SCRIPT_DIR}/toxp_pipe.py")

    FUNC_PAYLOAD=$(python3 -c "
import json
content = open('${SCRIPT_DIR}/toxp_pipe.py').read()
print(json.dumps({
    'id': 'toxp',
    'name': 'TOXP - Team of Experts',
    'type': 'pipe',
    'meta': {'description': 'Parallel multi-agent reasoning via AWS Bedrock'},
    'content': content
}))
")

    # Try create, fall back to update if exists
    CREATE_RESP=$(curl -sf -X POST "${URL}/api/v1/functions/create" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$FUNC_PAYLOAD" 2>/dev/null || echo '{"detail":"exists"}')

    if echo "$CREATE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('id') else 1)" 2>/dev/null; then
        echo "       Pipe function created."
    else
        # Already exists — update it
        curl -sf -X POST "${URL}/api/v1/functions/id/toxp/update" \
            -H "Authorization: Bearer ${TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$FUNC_PAYLOAD" >/dev/null 2>&1
        echo "       Pipe function updated."
    fi

    # Ensure the function is enabled (only toggle if currently disabled)
    IS_ACTIVE=$(curl -sf "${URL}/api/v1/functions/id/toxp" \
        -H "Authorization: Bearer ${TOKEN}" 2>/dev/null | \
        python3 -c "import sys,json; print(json.load(sys.stdin).get('is_active',False))" 2>/dev/null)
    if [ "$IS_ACTIVE" != "True" ]; then
        curl -sf -X POST "${URL}/api/v1/functions/id/toxp/toggle" \
            -H "Authorization: Bearer ${TOKEN}" >/dev/null 2>&1
    fi
    echo "       Pipe function enabled."
fi

# --- Done ---
echo ""
echo "========================================"
echo "  TOXP Web UI is running at: ${URL}"
echo ""
echo "  Login: ${ADMIN_EMAIL} / ${ADMIN_PASSWORD}"
echo "  Model: Select 'TOXP - Team of Experts'"
echo ""
echo "  Configure per-chat: Controls > Valves"
echo "  Stop: ./start.sh --stop"
echo "========================================"

# Open browser
if command -v open &>/dev/null; then
    open "$URL"
elif command -v xdg-open &>/dev/null; then
    xdg-open "$URL"
fi
