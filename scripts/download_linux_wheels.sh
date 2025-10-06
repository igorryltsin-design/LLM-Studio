#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON:-python3}
PYTHON_VERSION=${PYTHON_VERSION:-311}
TARGET_PLATFORM=${TARGET_PLATFORM:-manylinux2014_x86_64}
TARGET_IMPL=${TARGET_IMPL:-cp}
OUTPUT_DIR=${OUTPUT_DIR:-vendor/pip}

mkdir -p "$OUTPUT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Не найден интерпретатор Python (используется переменная PYTHON=$PYTHON_BIN)." >&2
  exit 1
fi

echo "[info] Загружаю Linux-колёса зависимостей в $OUTPUT_DIR" >&2
"$PYTHON_BIN" -m pip download \
  --only-binary=:all: \
  --platform "$TARGET_PLATFORM" \
  --implementation "$TARGET_IMPL" \
  --python-version "$PYTHON_VERSION" \
  --no-deps \
  -r requirements.txt \
  -d "$OUTPUT_DIR"

echo "[info] Готово. Колёса сохранены в $OUTPUT_DIR" >&2
