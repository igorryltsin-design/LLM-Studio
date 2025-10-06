#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN=${PYTHON:-python3}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}

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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[Ошибка] Не удалось найти интерпретатор Python (ожидается python3)." >&2
  exit 1
fi

if [[ "${1:-}" == "--install" ]]; then
  install_python_runtime
fi

if command -v lsof >/dev/null 2>&1; then
  # Предыдущие запуски иногда оставляют процесс, который продолжает занимать порт.
  mapfile -t _busy_pids < <(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
  if ((${#_busy_pids[@]} > 0)); then
    echo "[Предупреждение] Найдены процессы, слушающие ${HOST}:${PORT}. Пытаюсь завершить их..."
    for _pid in "${_busy_pids[@]}"; do
      if ! kill "${_pid}" 2>/dev/null; then
        echo "  [Внимание] Не удалось послать SIGTERM процессу ${_pid}." >&2
        continue
      fi

      _deadline=$((SECONDS + 5))
      while kill -0 "${_pid}" 2>/dev/null && ((SECONDS < _deadline)); do
        sleep 0.2
      done

      if kill -0 "${_pid}" 2>/dev/null; then
        echo "  [Внимание] Процесс ${_pid} не завершился вовремя, посылаю SIGKILL." >&2
        kill -9 "${_pid}" 2>/dev/null || true
      fi
    done
  fi
else
  echo "[Предупреждение] Утилита lsof не найдена. Не могу освободить порт ${PORT} автоматически." >&2
fi

echo "[Инфо] Запускаю локальный сервер базовой модели Gemma на ${HOST}:${PORT}"
exec "$PYTHON_BIN" -m uvicorn base_model_server:app --host "${HOST}" --port "${PORT}" --reload
