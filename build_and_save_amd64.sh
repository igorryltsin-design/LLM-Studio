#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME=${IMAGE_NAME:-llm-studio:cuda-amd64}
TAR_PATH=${TAR_PATH:-llm-studio-cuda-amd64.tar}
DOCKERFILE=${DOCKERFILE:-Dockerfile}
PLATFORM=${PLATFORM:-linux/amd64}
SKIP_FRONTEND_BUILD=${SKIP_FRONTEND_BUILD:-0}
DOCKER_BUILD_ARGS=${DOCKER_BUILD_ARGS:-}

if ! command -v docker >/dev/null 2>&1; then
  echo "[error] Docker не найден в PATH." >&2
  exit 1
fi

if ! docker buildx inspect >/dev/null 2>&1; then
  echo "[info] Создаю builder buildx..."
  docker buildx create --use >/dev/null
fi

if [[ "$SKIP_FRONTEND_BUILD" != "1" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "[error] npm требуется для сборки фронтенда. Установите Node.js или запустите со SKIP_FRONTEND_BUILD=1." >&2
    exit 1
  fi

  echo "[info] Устанавливаю npm-зависимости..."
  npm install

  echo "[info] Собираю production-бандл..."
  npm run build
else
  echo "[info] Пропускаю сборку фронтенда (SKIP_FRONTEND_BUILD=1)."
fi

shopt -s nullglob
bnb_wheels=(vendor/pip/bitsandbytes-*.whl)
shopt -u nullglob

if (( ${#bnb_wheels[@]} == 0 )); then
  echo "[info] Локальный wheel bitsandbytes не найден — pip установит пакет во время сборки." >&2
else
  echo "[info] Найден локальный wheel: ${bnb_wheels[0]}" >&2
fi

read -r -a build_args <<<"${DOCKER_BUILD_ARGS:-}"

echo "[info] Собираю образ ${IMAGE_NAME} для ${PLATFORM}..."
build_cmd=(
  docker buildx build
  --platform "$PLATFORM"
  -f "$DOCKERFILE"
  -t "$IMAGE_NAME"
  --load
)
if (( ${#build_args[@]} > 0 )); then
  build_cmd+=("${build_args[@]}")
fi
build_cmd+=("$ROOT_DIR")

"${build_cmd[@]}"

mkdir -p "$(dirname "$TAR_PATH")"

echo "[info] Сохраняю образ в ${TAR_PATH}..."
docker save "$IMAGE_NAME" -o "$TAR_PATH"

echo "[info] Готово. Перенесите файл ${TAR_PATH} на целевую машину и загрузите:"
echo "       docker load -i ${TAR_PATH}"
