#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$PROJECT_DIR/.jupyter-local"
mkdir -p "$LOCAL_DIR/data" "$LOCAL_DIR/runtime" "$LOCAL_DIR/config" "$LOCAL_DIR/ipython"

export JUPYTER_PATH="$PROJECT_DIR/.venv/share/jupyter"
export JUPYTER_DATA_DIR="$LOCAL_DIR/data"
export JUPYTER_RUNTIME_DIR="$LOCAL_DIR/runtime"
export JUPYTER_CONFIG_DIR="$LOCAL_DIR/config"
export IPYTHONDIR="$LOCAL_DIR/ipython"

cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/jupyter" lab \
    --notebook-dir="$PROJECT_DIR" \
    --ServerApp.ip=127.0.0.1 \
    --ServerApp.allow_remote_access=False \
    "$@"
