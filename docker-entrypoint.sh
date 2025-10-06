#!/usr/bin/env bash
set -euo pipefail

cd /opt/llm-studio

APP=${UVICORN_APP:-base_model_server:app}
HOST=${UVICORN_HOST:-0.0.0.0}
PORT=${UVICORN_PORT:-8001}
WORKERS=${UVICORN_WORKERS:-1}
EXTRA=${UVICORN_EXTRA_ARGS:-}

MODELS_DIR=${MODELS_DIR:-/opt/llm-studio/Models}
mkdir -p "${MODELS_DIR}"

export MODELS_DIR

cmd=(
  python3 -m uvicorn
  "${APP}"
  --host "${HOST}"
  --port "${PORT}"
  --workers "${WORKERS}"
)

if [[ -n "${EXTRA}" ]]; then
  read -r -a extra_args <<<"${EXTRA}"
  cmd+=("${extra_args[@]}")
fi

exec "${cmd[@]}"
