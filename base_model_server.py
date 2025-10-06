#!/usr/bin/env python3
"""Простой сервер для чата с локальной моделью Gemma.

Запуск:
    UVICORN_CMD="uvicorn base_model_server:app --host 127.0.0.1 --port 8001".
Перед запуском установите зависимости:
    pip install fastapi uvicorn transformers torch psutil
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple
from typing import Literal
from urllib.parse import urljoin


_VENDOR_DIR = (Path(__file__).parent / "python_libs").resolve()
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import requests
from pydantic import AnyHttpUrl, BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # pragma: no cover - optional dependency
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - gracefully handle missing bitsandbytes
    BitsAndBytesConfig = None  # type: ignore[assignment]

try:  # Optional import for LoRA adapters
    from peft import PeftModel
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    PeftModel = None  # type: ignore[assignment]

from fine_tune_manager import (
    FineTuneConfig,
    FineTuneError,
    FineTuneManager,
    build_dataset,
)

try:  # Optional dependency for system metrics
    import psutil
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    psutil = None  # type: ignore[assignment]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_path: str = Field(..., description="Путь до локальной модели Gemma")
    messages: List[ChatMessage]
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    adapter_path: Optional[str] = Field(
        default=None,
        description="Необязательный путь до адаптера LoRA/дообученной модели",
    )
    quantization: Literal['none', '4bit', '8bit'] = 'none'
    device: Optional[Literal['auto', 'cpu', 'cuda', 'mps']] = Field(
        default=None,
        description="Предпочитаемое устройство выполнения (по умолчанию авто)",
    )


class FineTuneDatasetItemModel(BaseModel):
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    source: Optional[str] = Field(default=None, max_length=256)


class FineTuneConfigModel(BaseModel):
    method: Literal['lora', 'qlora', 'full'] = 'lora'
    quantization: Literal['none', '4bit', '8bit'] = 'none'
    lora_rank: int = Field(16, ge=1, le=512)
    lora_alpha: int = Field(32, ge=1, le=1024)
    learning_rate: float = Field(2e-4, gt=0, lt=1)
    batch_size: int = Field(4, ge=1, le=64)
    epochs: int = Field(3, ge=1, le=20)
    max_length: int = Field(512, ge=64, le=4096)
    warmup_steps: int = Field(0, ge=0, le=1000)
    target_modules: Optional[List[str]] = Field(default=None)
    initial_adapter_path: Optional[str] = Field(
        default=None,
        description="Путь до адаптера LoRA, с которого следует продолжить обучение",
    )


class FineTuneRequestModel(BaseModel):
    base_model_path: str = Field(..., description="Путь до базовой модели для дообучения")
    output_dir: Optional[str] = Field(
        default=None,
        description="Каталог для сохранения адаптера (по умолчанию Models/finetune-<timestamp>)",
    )
    dataset: List[FineTuneDatasetItemModel] = Field(..., min_items=1)
    config: FineTuneConfigModel


class AgregatorRequestModel(BaseModel):
    base_url: AnyHttpUrl = Field(..., description="Базовый URL сервиса Agregator")
    endpoint: str = Field(
        "/export/csv",
        description="REST-эндпоинт Agregator для экспорта данных (например, /export/csv)",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query-параметры для фильтрации выгрузки (collection, tags, status и т.д.)",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Дополнительные HTTP-заголовки (например, Authorization)",
    )
    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Таймаут запроса к Agregator, секунды")

    def resolved_endpoint(self) -> str:
        path = self.endpoint.strip() or "/export/csv"
        if not path.startswith("/"):
            path = f"/{path}"
        return path


class AutoFineTuneRequestModel(BaseModel):
    aggregator: AgregatorRequestModel
    base_model_path: str = Field(..., description="Путь до базовой модели для дообучения")
    output_dir: Optional[str] = Field(
        default=None,
        description="Каталог для сохранения результата (по умолчанию Models/finetune-<timestamp>)",
    )
    config: FineTuneConfigModel
    include_previous_dataset: bool = Field(
        default=True,
        description="Если указан previous_fine_tune_path, добавить старый датасет к новым примерам",
    )
    previous_fine_tune_path: Optional[str] = Field(
        default=None,
        description="Каталог с прошлым адаптером (dataset.jsonl будет объединён, а адаптер можно использовать как стартовый)",
    )
    deduplicate: bool = Field(
        default=True,
        description="Удалять дубликаты (одинаковые input/output) после объединения датасетов",
    )
    min_examples: int = Field(
        default=1,
        ge=1,
        le=100_000,
        description="Минимальное количество примеров для запуска обучения",
    )


class FineTuneEventModel(BaseModel):
    timestamp: float
    level: str
    message: str


class FineTuneJobConfigResponse(BaseModel):
    method: Literal['lora', 'qlora', 'full']
    quantization: Literal['none', '4bit', '8bit']
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int
    warmup_steps: int
    target_modules: Optional[List[str]]
    initial_adapter_path: Optional[str] = None


class FineTuneJobResponse(BaseModel):
    id: str
    status: str
    progress: float
    message: str
    metrics: Dict[str, Any]
    error: Optional[str]
    createdAt: float
    updatedAt: float
    startedAt: Optional[float]
    finishedAt: Optional[float]
    datasetSize: int
    outputDir: str
    baseModelPath: str
    config: FineTuneJobConfigResponse
    events: Optional[List[FineTuneEventModel]] = None
    resumeCheckpoint: Optional[str] = None


class FineTunedModelInfo(BaseModel):
    id: str
    name: str
    path: str
    base_model_path: Optional[str]
    method: Optional[str]
    dataset_size: Optional[int]
    created_at: Optional[float]
    finished_at: Optional[float]

_DEFAULT_CACHE_TTL_SECONDS = 300
try:
    _CACHE_TTL_SECONDS = max(0, int(os.environ.get("LLM_STUDIO_MODEL_CACHE_TTL", str(_DEFAULT_CACHE_TTL_SECONDS))))
except ValueError:
    _CACHE_TTL_SECONDS = _DEFAULT_CACHE_TTL_SECONDS

_CACHE_SWEEP_INTERVAL_SECONDS = 60 if _CACHE_TTL_SECONDS >= 120 else max(10, _CACHE_TTL_SECONDS // 2 or 10)

try:
    _CACHE_MAX_ENTRIES = max(1, int(os.environ.get("LLM_STUDIO_MODEL_CACHE_LIMIT", "2")))
except ValueError:
    _CACHE_MAX_ENTRIES = 2


@dataclass
class _CachedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    last_used: float
    model_path: str
    adapter_path: Optional[str]
    quantization: str
    device: str


ModelCache = Dict[str, _CachedModel]
_model_cache: ModelCache = {}
_MODEL_CACHE_LOCK = threading.Lock()
_cache_gc_task: Optional[asyncio.Task] = None
_MODELS_SIZE_CACHE: dict[str, float] = {"value": 0.0, "timestamp": 0.0}
_MODELS_SIZE_TTL_SECONDS = 300

logger = logging.getLogger("llm_studio.base_model_server")
if not logger.handlers:
    logger.setLevel(logging.INFO)


app = FastAPI(title="Локальный чат Gemma", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DIST_DIR = (Path(__file__).parent / "dist").resolve()
_ASSETS_DIR = _DIST_DIR / "assets"
_INDEX_FILE = _DIST_DIR / "index.html"

_FINE_TUNE_MANAGER = FineTuneManager(root_dir=Path(__file__).parent)

try:
    _INDEX_HTML = _INDEX_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    _INDEX_HTML = None

if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


def _is_api_path(path: str) -> bool:
    normalized = path.lstrip("/")
    return normalized.startswith("v1/") or normalized.startswith("system/")


def _safe_static_file(path: str) -> Optional[Path]:
    if not path:
        return None
    try:
        candidate = (_DIST_DIR / path).resolve()
    except (OSError, RuntimeError):  # pragma: no cover - defensive
        return None
    try:
        candidate.relative_to(_DIST_DIR)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None

if psutil is not None:
    # Prime CPU counters so the first measurement is meaningful.
    try:
        psutil.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001 - best effort only
        pass

    try:
        _PROCESS = psutil.Process()
        _PROCESS.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001 - process metrics are optional
        _PROCESS = None
else:
    _PROCESS = None

_BYTES_IN_GB = 1024 ** 3


def _extract_llm_pair(row: Dict[str, Any]) -> Optional[Dict[str, Optional[str]]]:
    metadata_raw: Any = row.get("metadata") or row.get("meta")
    llm_meta: Dict[str, Any] = {}
    if isinstance(metadata_raw, str) and metadata_raw.strip():
        with suppress(json.JSONDecodeError):
            decoded = json.loads(metadata_raw)
            if isinstance(decoded, dict):
                candidate = decoded.get("llm") if isinstance(decoded.get("llm"), dict) else decoded
                if isinstance(candidate, dict):
                    llm_meta = candidate
    elif isinstance(metadata_raw, dict):
        candidate = metadata_raw.get("llm") if isinstance(metadata_raw.get("llm"), dict) else metadata_raw
        if isinstance(candidate, dict):
            llm_meta = candidate

    prompt = row.get("prompt") or llm_meta.get("prompt") or row.get("title") or row.get("question")
    reference = (
        row.get("reference")
        or llm_meta.get("reference")
        or row.get("answer")
        or row.get("expected")
        or row.get("text")
        or row.get("content")
    )

    if not prompt or not reference:
        return None

    source = (
        llm_meta.get("source")
        or row.get("source")
        or row.get("collection")
        or row.get("path")
        or row.get("id")
    )

    prompt_text = str(prompt).strip()
    reference_text = str(reference).strip()
    if not prompt_text or not reference_text:
        return None

    source_text = str(source).strip() if source else None
    return {"input": prompt_text, "output": reference_text, "source": source_text}


def _fetch_agregator_examples(config: AgregatorRequestModel) -> List[Dict[str, Optional[str]]]:
    url = urljoin(str(config.base_url), config.resolved_endpoint())
    try:
        response = requests.get(
            url,
            params=config.params,
            headers=config.headers if config.headers else None,
            timeout=config.timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - network dependency
        raise FineTuneError(f"Не удалось обратиться к Agregator ({exc})") from exc

    if response.status_code >= 400:
        excerpt = response.text[:200]
        raise FineTuneError(
            f"Agregator вернул ошибку {response.status_code}: {excerpt}"
        )

    text = response.text.strip()
    if not text:
        return []

    content_type = response.headers.get("content-type", "").lower()
    rows: List[Dict[str, Any]] = []
    if "application/json" in content_type:
        with suppress(json.JSONDecodeError):
            payload = response.json()
            if isinstance(payload, list):
                rows = [item for item in payload if isinstance(item, dict)]
            elif isinstance(payload, dict):
                items = payload.get("items")
                if isinstance(items, list):
                    rows = [item for item in items if isinstance(item, dict)]
    if not rows:
        stream = io.StringIO(text)
        reader = csv.DictReader(stream)
        rows = [dict(item) for item in reader]

    examples: List[Dict[str, Optional[str]]] = []
    for row in rows:
        item = _extract_llm_pair(row)
        if item is not None:
            examples.append(item)
    return examples


def _load_previous_dataset(previous_dir: Path) -> List[Dict[str, Optional[str]]]:
    dataset_path = previous_dir / "dataset.jsonl"
    if not dataset_path.exists():
        raise FineTuneError(
            f"Не найден snapshot датасета: {dataset_path}. Нечего объединять с новыми примерами."
        )

    examples: List[Dict[str, Optional[str]]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            payload = raw_line.strip()
            if not payload:
                continue
            with suppress(json.JSONDecodeError):
                data = json.loads(payload)
                if not isinstance(data, dict):
                    continue
                prompt = str(data.get("input", "")).strip()
                reference = str(data.get("output", "")).strip()
                if not prompt or not reference:
                    continue
                source = data.get("source")
                examples.append(
                    {
                        "input": prompt,
                        "output": reference,
                        "source": str(source).strip() if source else None,
                    }
                )
    return examples


def _merge_examples(
    new_examples: List[Dict[str, Optional[str]]],
    previous_examples: Iterable[Dict[str, Optional[str]]],
    *,
    deduplicate: bool,
) -> List[Dict[str, Optional[str]]]:
    merged = list(new_examples)
    merged.extend(previous_examples)
    if not deduplicate:
        return merged

    seen: set[Tuple[str, str]] = set()
    unique: List[Dict[str, Optional[str]]] = []
    for item in merged:
        input_text = (item.get("input") or "").strip()
        output_text = (item.get("output") or "").strip()
        if not input_text or not output_text:
            continue
        key = (input_text, output_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append({
            "input": input_text,
            "output": output_text,
            "source": item.get("source"),
        })
    return unique


def _bytes_to_gb(value: float) -> float:
    return round(value / _BYTES_IN_GB, 2)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_cache_key(
    model_path: Path,
    adapter_path: Optional[Path],
    quantization: str,
    device: str,
) -> str:
    adapter_part = str(adapter_path) if adapter_path else ""
    return f"{model_path}::{adapter_part}::{quantization}::{device}"


def _compute_directory_size_gb(path: Path) -> float:
    total_bytes = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            try:
                total_bytes += (Path(root) / file_name).stat().st_size
            except OSError:  # noqa: PERF203 - file could disappear between walk and stat
                continue
    return round(total_bytes / _BYTES_IN_GB, 2)


def _get_models_dir_size_gb() -> float:
    models_root = _FINE_TUNE_MANAGER.models_root
    if not models_root.exists():
        return 0.0

    now = time.time()
    cache_age = now - _MODELS_SIZE_CACHE["timestamp"]
    if cache_age < _MODELS_SIZE_TTL_SECONDS and _MODELS_SIZE_CACHE["value"] >= 0:
        return _MODELS_SIZE_CACHE["value"]

    size_gb = _compute_directory_size_gb(models_root)
    _MODELS_SIZE_CACHE["value"] = size_gb
    _MODELS_SIZE_CACHE["timestamp"] = now
    return size_gb


def _open_in_file_manager(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return

    if sys.platform == "darwin":
        command = ["open", str(path)]
    else:
        command = ["xdg-open", str(path)]

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Команда {command[0]} недоступна в системе") from exc

    if result.returncode != 0:
        raise RuntimeError(f"Не удалось открыть каталог с помощью {command[0]}")


def _get_cached_entry(cache_key: str) -> Optional[_CachedModel]:
    if _CACHE_TTL_SECONDS == 0:
        return None
    with _MODEL_CACHE_LOCK:
        entry = _model_cache.get(cache_key)
        if entry is not None:
            entry.last_used = time.time()
            return entry
    return None


def _store_cache_entry(cache_key: str, entry: _CachedModel) -> None:
    if _CACHE_TTL_SECONDS == 0:
        return
    victims: List[_CachedModel] = []
    with _MODEL_CACHE_LOCK:
        _model_cache[cache_key] = entry
        if len(_model_cache) > _CACHE_MAX_ENTRIES:
            overflow = len(_model_cache) - _CACHE_MAX_ENTRIES
            # Select least recently used entries (excluding the one we just stored when possible)
            candidates = sorted(
                _model_cache.items(),
                key=lambda item: item[1].last_used,
            )
            for key, candidate in candidates:
                if overflow <= 0:
                    break
                if key == cache_key:
                    continue
                victims.append(candidate)
                _model_cache.pop(key, None)
                overflow -= 1

    if victims:
        _release_models(victims)


def _release_models(models: Iterable[_CachedModel]) -> None:
    freed_gpu = False
    for entry in models:
        try:
            entry.model.to("cpu")  # type: ignore[call-arg]
            freed_gpu = True
        except Exception:  # noqa: BLE001 - best effort
            pass
    if freed_gpu and torch.cuda.is_available():  # pragma: no cover - depends on hardware
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - optional cleanup
            pass


def _find_non_finite_adapter_tensor(model: AutoModelForCausalLM) -> Optional[Tuple[str, str]]:
    if not hasattr(model, "get_peft_model_state_dict"):
        return None

    try:
        adapter_state = model.get_peft_model_state_dict()  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - defensive
        return None

    with torch.no_grad():
        for name, tensor in adapter_state.items():
            if tensor is None:
                continue
            detached = tensor.detach()
            if torch.isfinite(detached).all():
                continue
            if torch.isnan(detached).any():
                return name, "NaN"
            if torch.isinf(detached).any():
                return name, "Inf"
            return name, "non-finite"
    return None


def _evict_idle_models(*, force: bool = False) -> None:
    victims: List[_CachedModel] = []
    if force or _CACHE_TTL_SECONDS > 0:
        now = time.time()
        with _MODEL_CACHE_LOCK:
            for key, entry in list(_model_cache.items()):
                if force or now - entry.last_used >= _CACHE_TTL_SECONDS:
                    victims.append(entry)
                    _model_cache.pop(key, None)

    if victims:
        _release_models(victims)


async def _cache_gc_worker() -> None:
    try:
        while True:
            await asyncio.sleep(_CACHE_SWEEP_INTERVAL_SECONDS)
            _evict_idle_models()
    except asyncio.CancelledError:  # pragma: no cover - lifecycle management
        raise


@app.on_event("startup")
async def _start_cache_gc() -> None:  # pragma: no cover - integration hook
    global _cache_gc_task
    if _CACHE_TTL_SECONDS == 0:
        _evict_idle_models(force=True)
        return
    loop = asyncio.get_running_loop()
    _cache_gc_task = loop.create_task(_cache_gc_worker())


@app.on_event("shutdown")
async def _stop_cache_gc() -> None:  # pragma: no cover - integration hook
    global _cache_gc_task
    if _cache_gc_task is not None:
        _cache_gc_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cache_gc_task
        _cache_gc_task = None
    _evict_idle_models(force=True)


def _collect_gpu_stats() -> dict[str, object]:
    backend = _detect_device()

    devices: list[Dict[str, Optional[object]]] = []

    if backend == "cuda":
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_bytes = float(props.total_memory)
            reserved = float(torch.cuda.memory_reserved(index))
            allocated = float(torch.cuda.memory_allocated(index))
            used_bytes = max(reserved, allocated)
            memory_percent = (used_bytes / total_bytes * 100.0) if total_bytes else 0.0

            utilization: Optional[float] = None
            if hasattr(torch.cuda, "utilization"):
                try:
                    util_value = torch.cuda.utilization(index)
                except Exception:  # noqa: BLE001 - utilization is best effort
                    util_value = None
                if isinstance(util_value, (int, float)):
                    utilization = float(util_value)

            devices.append(
                {
                    "id": index,
                    "name": props.name,
                    "memory_total_gb": _bytes_to_gb(total_bytes),
                    "memory_used_gb": _bytes_to_gb(used_bytes),
                    "memory_percent": round(memory_percent, 1),
                    "utilization_percent": round(utilization, 1) if utilization is not None else None,
                }
            )
    elif backend == "mps":
        # PyTorch does not expose precise utilization counters for MPS yet.
        devices.append(
            {
                "id": 0,
                "name": "Apple MPS",
                "memory_total_gb": None,
                "memory_used_gb": None,
                "memory_percent": None,
                "utilization_percent": None,
            }
        )

    return {
        "available": backend in {"cuda", "mps"},
        "backend": backend,
        "devices": devices,
    }


@app.get("/system/status")
async def system_status():
    if psutil is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "psutil_not_installed",
                "message": "Пакет psutil не установлен. Запустите start_base_model.sh --install",
            },
        )

    cpu_percent = psutil.cpu_percent(interval=None)
    try:
        cpu_freq = psutil.cpu_freq()
    except Exception:  # noqa: BLE001 - not available on some systems
        cpu_freq = None
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    uptime_seconds = time.time() - psutil.boot_time()

    process_cpu: Optional[float] = None
    process_rss_gb: Optional[float] = None
    if _PROCESS is not None:
        try:
            process_cpu = _PROCESS.cpu_percent(interval=None)
            process_rss_gb = _bytes_to_gb(float(_PROCESS.memory_info().rss))
        except Exception:  # noqa: BLE001 - optional process details
            process_cpu = None
            process_rss_gb = None

    response_payload = {
        "status": "ok",
        "timestamp": time.time(),
        "cpu": {
            "percent": round(float(cpu_percent), 1),
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
            "frequency_mhz": int(cpu_freq.current) if cpu_freq else None,
            "process_percent": round(process_cpu, 1) if process_cpu is not None else None,
        },
        "memory": {
            "percent": round(float(memory.percent), 1),
            "used_gb": _bytes_to_gb(float(memory.used)),
            "available_gb": _bytes_to_gb(float(memory.available)),
            "total_gb": _bytes_to_gb(float(memory.total)),
            "models_dir_gb": _get_models_dir_size_gb(),
        },
        "swap": {
            "percent": round(float(swap.percent), 1) if swap.total else None,
            "used_gb": _bytes_to_gb(float(swap.used)) if swap.total else None,
            "total_gb": _bytes_to_gb(float(swap.total)) if swap.total else None,
        },
        "gpu": _collect_gpu_stats(),
        "uptime_seconds": int(uptime_seconds),
    }

    return response_payload


@app.post("/system/open-models-directory")
async def open_models_directory() -> Dict[str, str]:
    try:
        _open_in_file_manager(_FINE_TUNE_MANAGER.models_root)
    except RuntimeError as exc:
        logger.warning("Не удалось открыть каталог моделей: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - log unexpected issues
        logger.exception("Непредвиденная ошибка при открытии каталога моделей")
        raise HTTPException(status_code=500, detail="Не удалось открыть каталог моделей") from exc

    return {"status": "ok"}


def _resolve_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Модель не найдена по пути: {path}")
    return path


def _load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    *,
    quantization: str = "none",
    device_preference: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    resolved_path = _resolve_path(model_path)
    resolved_adapter: Optional[Path] = None
    if adapter_path:
        try:
            resolved_adapter = _resolve_path(adapter_path)
        except HTTPException:
            # Re-raise with clearer message
            raise HTTPException(status_code=404, detail=f"Адаптер не найден по пути: {Path(adapter_path).resolve()}")

    quantization_mode = (quantization or "none").lower()
    if quantization_mode not in {"none", "4bit", "8bit"}:
        raise HTTPException(status_code=400, detail=f"Неизвестный режим квантизации: {quantization}")

    preferred_device = (device_preference or "").strip().lower() or None
    if preferred_device in {"auto", ""}:
        preferred_device = None

    if preferred_device == "cuda":
        if not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="CUDA недоступна на данной системе")
        runtime_device = "cuda"
    elif preferred_device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise HTTPException(status_code=400, detail="MPS недоступен на данной системе")
        runtime_device = "mps"
    elif preferred_device == "cpu":
        runtime_device = "cpu"
    elif preferred_device is None:
        runtime_device = _detect_device()
    else:
        raise HTTPException(status_code=400, detail=f"Неизвестное устройство выполнения: {device_preference}")

    if quantization_mode in {"4bit", "8bit"}:
        if runtime_device != "cuda":
            raise HTTPException(
                status_code=400,
                detail="Квантизация 4bit/8bit поддерживается только на GPU с CUDA",
            )
        if BitsAndBytesConfig is None:
            raise HTTPException(
                status_code=500,
                detail="Для квантизации требуется пакет 'bitsandbytes'. Установите его: pip install bitsandbytes",
            )

    cache_key = _build_cache_key(resolved_path, resolved_adapter, quantization_mode, runtime_device)
    cached_entry = _get_cached_entry(cache_key)
    if cached_entry is not None:
        logger.info(
            "Использую модель из кеша: model=%s adapter=%s quant=%s device=%s",
            resolved_path,
            resolved_adapter,
            quantization_mode,
            runtime_device,
        )
        return cached_entry.tokenizer, cached_entry.model

    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось загрузить токенизатор %s", resolved_path)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить токенизатор: {exc}") from exc

    load_kwargs: Dict[str, Any] = {}
    if quantization_mode in {"4bit", "8bit"}:
        if quantization_mode == "4bit":
            quant_config = BitsAndBytesConfig(  # type: ignore[call-arg]
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            quant_config = BitsAndBytesConfig(  # type: ignore[call-arg]
                load_in_8bit=True,
            )
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
    else:
        torch_dtype = torch.float16 if runtime_device in {"cuda", "mps"} else torch.float32
        load_kwargs["torch_dtype"] = torch_dtype

    logger.info(
        "Загружаю модель: path=%s adapter=%s quant=%s device=%s",
        resolved_path,
        resolved_adapter,
        quantization_mode,
        runtime_device,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            **load_kwargs,
        )
        if quantization_mode == "none":
            if runtime_device == "mps":
                model = model.to("mps")
            elif runtime_device == "cuda":
                model = model.to("cuda")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось загрузить модель %s", resolved_path)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить модель: {exc}") from exc

    if resolved_adapter is not None:
        if PeftModel is None:  # pragma: no cover - runtime dependency
            raise HTTPException(
                status_code=500,
                detail="Поддержка LoRA не активна: пакет 'peft' не установлен",
            )
        try:
            model = PeftModel.from_pretrained(model, resolved_adapter, is_trainable=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось загрузить адаптер %s", resolved_adapter)
            raise HTTPException(status_code=500, detail=f"Не удалось загрузить адаптер LoRA: {exc}") from exc

        invalid_tensor = _find_non_finite_adapter_tensor(model)
        if invalid_tensor is not None:
            tensor_name, tensor_kind = invalid_tensor
            logger.error(
                "Адаптер содержит нечисловые значения: adapter=%s tensor=%s kind=%s",
                resolved_adapter,
                tensor_name,
                tensor_kind,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "Дообученный адаптер содержит недопустимые значения (%s) в тензоре %s. "
                    "Удалите адаптер и повторите обучение с меньшим learning_rate или большим warmup_steps."
                )
                % (tensor_kind, tensor_name),
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    _store_cache_entry(
        cache_key,
        _CachedModel(
            tokenizer=tokenizer,
            model=model,
            last_used=time.time(),
            model_path=str(resolved_path),
            adapter_path=str(resolved_adapter) if resolved_adapter else None,
            quantization=quantization_mode,
            device=runtime_device,
        ),
    )
    return tokenizer, model


def _apply_chat_template(tokenizer: AutoTokenizer, messages: List[ChatMessage]) -> str:
    conversation = [message.dict() for message in messages]
    try:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Не удалось подготовить промпт: {exc}") from exc


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    tokenizer, model = _load_model(
        request.model_path,
        request.adapter_path,
        quantization=request.quantization,
        device_preference=request.device,
    )

    prompt = _apply_chat_template(tokenizer, request.messages)

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Не удалось токенизировать запрос: {exc}") from exc

    try:
        param_device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive
        param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(param_device, "type", None) == "meta":  # pragma: no cover - defensive
        param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {k: v.to(param_device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": request.max_tokens,
        "temperature": max(request.temperature, 1e-6),
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Ошибка генерации: model=%s adapter=%s", request.model_path, request.adapter_path
        )
        message = str(exc)
        if "probability tensor contains" in message:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Дообученный адаптер вернул недопустимые вероятности. "
                    "Попробуйте временно установить temperature=0 (greedy) и перепройти обучение с меньшим learning_rate, большим warmup_steps или более крупным датасетом."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {exc}") from exc

    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_length:]
    completion_tokens = generated_tokens.shape[-1]
    total_tokens = outputs[0].shape[-1]

    try:
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Не удалось декодировать ответ: {exc}") from exc

    model_name = (
        Path(request.adapter_path).name
        if request.adapter_path
        else Path(request.model_path).name
    )

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_length),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        },
    }


def _fine_tune_config_from_request(request_config: FineTuneConfigModel) -> FineTuneConfig:
    return FineTuneConfig(
        method=request_config.method,
        quantization=request_config.quantization,
        lora_rank=request_config.lora_rank,
        lora_alpha=request_config.lora_alpha,
        learning_rate=request_config.learning_rate,
        batch_size=request_config.batch_size,
        epochs=request_config.epochs,
        max_length=request_config.max_length,
        warmup_steps=request_config.warmup_steps,
        target_modules=request_config.target_modules,
        initial_adapter_path=request_config.initial_adapter_path,
    )


@app.post("/v1/fine-tunes", response_model=FineTuneJobResponse)
async def create_fine_tune(request: FineTuneRequestModel):
    try:
        dataset = build_dataset(item.dict() for item in request.dataset)
        config = _fine_tune_config_from_request(request.config)
        job = _FINE_TUNE_MANAGER.start_job(
            base_model_path=request.base_model_path,
            output_dir=request.output_dir,
            dataset=dataset,
            config=config,
        )
    except FineTuneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@app.post("/v1/fine-tunes/from-agregator", response_model=FineTuneJobResponse)
async def create_fine_tune_from_agregator(request: AutoFineTuneRequestModel):
    try:
        new_examples = await asyncio.to_thread(_fetch_agregator_examples, request.aggregator)
        logger.info(
            "Agregator вернул %s новых примеров (endpoint=%s)",
            len(new_examples),
            request.aggregator.resolved_endpoint(),
        )

        previous_examples: List[Dict[str, Optional[str]]] = []
        previous_dir: Optional[Path] = None
        if request.previous_fine_tune_path:
            previous_dir = Path(request.previous_fine_tune_path).expanduser().resolve()
            if not previous_dir.exists() or not previous_dir.is_dir():
                raise FineTuneError(
                    f"Каталог с предыдущим адаптером не найден: {previous_dir}"
                )
            if request.include_previous_dataset:
                previous_examples = await asyncio.to_thread(_load_previous_dataset, previous_dir)
                logger.info(
                    "Добавлено %s примеров из предыдущего датасета %s",
                    len(previous_examples),
                    previous_dir,
                )

        combined_examples = _merge_examples(
            new_examples,
            previous_examples,
            deduplicate=request.deduplicate,
        )

        if len(combined_examples) < request.min_examples:
            raise FineTuneError(
                "Недостаточно примеров для обучения: %s (< %s)"
                % (len(combined_examples), request.min_examples)
            )

        dataset = build_dataset(combined_examples)
        config = _fine_tune_config_from_request(request.config)

        if (
            previous_dir is not None
            and config.initial_adapter_path is None
            and request.config.method in {"lora", "qlora"}
        ):
            config = replace(config, initial_adapter_path=str(previous_dir))

        job = _FINE_TUNE_MANAGER.start_job(
            base_model_path=request.base_model_path,
            output_dir=request.output_dir,
            dataset=dataset,
            config=config,
        )
    except FineTuneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@app.get("/v1/fine-tunes", response_model=List[FineTuneJobResponse])
async def list_fine_tunes():
    return _FINE_TUNE_MANAGER.list_jobs()


@app.get("/v1/fine-tunes/available", response_model=List[FineTunedModelInfo])
async def list_available_fine_tunes():
    models = _FINE_TUNE_MANAGER.list_available_models()
    return [
        {
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "path": item.get("path", ""),
            "base_model_path": item.get("base_model_path"),
            "method": item.get("method"),
            "dataset_size": item.get("dataset_size"),
            "created_at": item.get("created_at"),
            "finished_at": item.get("finished_at"),
        }
        for item in models
    ]


@app.get("/v1/fine-tunes/{job_id}", response_model=FineTuneJobResponse)
async def get_fine_tune(job_id: str):
    job = _FINE_TUNE_MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача дообучения не найдена")
    return job.to_dict()


@app.post("/v1/fine-tunes/{job_id}/cancel", response_model=FineTuneJobResponse)
async def cancel_fine_tune(job_id: str):
    try:
        job = _FINE_TUNE_MANAGER.cancel_job(job_id)
    except FineTuneError as exc:
        status_code = 404 if "не найдена" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return job.to_dict()


@app.post("/v1/fine-tunes/{job_id}/pause", response_model=FineTuneJobResponse)
async def pause_fine_tune(job_id: str):
    try:
        job = _FINE_TUNE_MANAGER.pause_job(job_id)
    except FineTuneError as exc:
        status_code = 404 if "не найдена" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return job.to_dict()


@app.post("/v1/fine-tunes/{job_id}/resume", response_model=FineTuneJobResponse)
async def resume_fine_tune(job_id: str):
    try:
        job = _FINE_TUNE_MANAGER.resume_job(job_id)
    except FineTuneError as exc:
        status_code = 404 if "не найдена" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return job.to_dict()


@app.get("/", response_class=HTMLResponse)
async def _serve_index():  # pragma: no cover - runtime integration
    if _INDEX_HTML is None:
        raise HTTPException(status_code=503, detail="UI bundle is not available")
    return HTMLResponse(content=_INDEX_HTML)


@app.get("/{full_path:path}", response_class=HTMLResponse)
async def _serve_spa(full_path: str):  # pragma: no cover - runtime integration
    if _is_api_path(full_path):
        raise HTTPException(status_code=404)
    static_file = _safe_static_file(full_path)
    if static_file is not None:
        return FileResponse(static_file)
    if _INDEX_HTML is None:
        raise HTTPException(status_code=404)
    return HTMLResponse(content=_INDEX_HTML)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("base_model_server:app", host="127.0.0.1", port=8001, reload=False)
