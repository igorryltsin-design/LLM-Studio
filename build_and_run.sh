#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="preview"
INSTALL=0
START_BASE=1
PYTHON_BIN=${PYTHON:-python3}
BASE_HOST=${BASE_HOST:-127.0.0.1}
BASE_PORT=${BASE_PORT:-8001}
BASE_PID=0

install_python_runtime() {
  local torch_version
  local torch_index
  local extra_packages

  torch_version=${TORCH_VERSION:-2.4.0}
  torch_index=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

  echo "[Инфо] Обновляю pip..."
  "$PYTHON_BIN" -m pip install --upgrade pip

  echo "[Инфо] Устанавливаю CUDA-сборку torch ${torch_version}..."
  "$PYTHON_BIN" -m pip install --upgrade "torch==${torch_version}" --index-url "${torch_index}"

  extra_packages=(fastapi uvicorn transformers psutil accelerate peft)

  if [[ "$(uname -s)" == "Linux" ]]; then
    extra_packages+=(bitsandbytes)
  else
    echo "[Предупреждение] bitsandbytes поддерживается только на Linux. Пропускаю установку." >&2
  fi

  echo "[Инфо] Устанавливаю Python-зависимости: ${extra_packages[*]}"
  "$PYTHON_BIN" -m pip install --upgrade "${extra_packages[@]}"
}

usage() {
  cat <<'USAGE'
Использование: ./build_and_run.sh [опции]
  --dev           Запустить фронтенд в режиме разработки (vite dev)
  --preview       Запустить сборку и предпросмотр (по умолчанию)
  --install       Установить/обновить зависимости npm и Python
  --skip-base     Не запускать локальный сервер базовой модели
  --python PATH   Использовать указанный интерпретатор Python
  --base-host H   Хост для сервера базовой модели (по умолчанию 127.0.0.1)
  --base-port P   Порт для сервера базовой модели (по умолчанию 8001)
  -h, --help      Показать эту справку
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      MODE="dev"
      ;;
    --preview)
      MODE="preview"
      ;;
    --install)
      INSTALL=1
      ;;
    --skip-base)
      START_BASE=0
      ;;
    --python)
      shift || { echo "[Ошибка] Для --python требуется путь до интерпретатора." >&2; exit 1; }
      PYTHON_BIN="$1"
      ;;
    --base-host)
      shift || { echo "[Ошибка] Для --base-host требуется значение." >&2; exit 1; }
      BASE_HOST="$1"
      ;;
    --base-port)
      shift || { echo "[Ошибка] Для --base-port требуется значение." >&2; exit 1; }
      BASE_PORT="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[Ошибка] Неизвестный аргумент: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift || true
done

require_cmd() {
  local cmd="$1"
  local message="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[Ошибка] ${message}" >&2
    exit 1
  fi
}

cleanup() {
  if [[ ${BASE_PID:-0} -ne 0 ]]; then
    if kill -0 "$BASE_PID" >/dev/null 2>&1; then
      echo "[Инфо] Останавливаю сервер базовой модели (PID=$BASE_PID)..."
      kill "$BASE_PID" >/dev/null 2>&1 || true
      wait "$BASE_PID" 2>/dev/null || true
    fi
  fi
}

trap cleanup EXIT

require_cmd npm "npm не найден. Установите Node.js и npm либо добавьте их в PATH."

if [[ $START_BASE -eq 1 ]]; then
  require_cmd "$PYTHON_BIN" "Интерпретатор Python не найден (ожидается >=3.9)."
fi

if [[ $INSTALL -eq 1 || ! -d node_modules ]]; then
  echo "[Инфо] Устанавливаю npm зависимости..."
  npm install
fi

if [[ $START_BASE -eq 1 && $INSTALL -eq 1 ]]; then
  echo "[Инфо] Устанавливаю зависимости Python для базовой модели..."
  install_python_runtime
fi

check_python_deps() {
  "$PYTHON_BIN" - <<'PY'
import importlib, platform, sys

modules = ["fastapi", "uvicorn", "transformers", "torch", "accelerate", "peft"]
if platform.system() == "Linux":
    modules.append("bitsandbytes")

missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    print(','.join(missing))
    sys.exit(1)
PY
}

if [[ $START_BASE -eq 1 ]]; then
  if ! missing=$(check_python_deps 2>/dev/null); then
    missing=${missing//,/ }
    echo "[Ошибка] Не найдены Python-библиотеки: ${missing:-fastapi uvicorn transformers torch}" >&2
    echo "        Запустите скрипт с опцией --install или установите пакеты вручную." >&2
    exit 1
  fi
fi

if [[ "$MODE" != "dev" ]]; then
  echo "[Инфо] Собираю production-бандл фронтенда..."
  npm run build
else
  echo "[Инфо] Запуск в dev-режиме — сборка не требуется."
fi

start_base_server() {
  if [[ $START_BASE -ne 1 ]]; then
    return
  fi

  local log_prefix="[Инфо]"
  echo "${log_prefix} Запускаю сервер базовой модели на http://${BASE_HOST}:${BASE_PORT}..."
  "$PYTHON_BIN" -m uvicorn base_model_server:app --host "$BASE_HOST" --port "$BASE_PORT" &
  BASE_PID=$!
  sleep 2
  if ! kill -0 "$BASE_PID" >/dev/null 2>&1; then
    echo "[Ошибка] Не удалось запустить сервер базовой модели. Проверьте логи выше." >&2
    exit 1
  fi
}

start_base_server

if [[ "$MODE" == "dev" ]]; then
  echo "[Инфо] Запускаю фронтенд Vite в dev-режиме..."
  npm run dev
else
  echo "[Инфо] Запускаю фронтенд Vite в режиме предпросмотра..."
  npm run preview -- --host
fi
