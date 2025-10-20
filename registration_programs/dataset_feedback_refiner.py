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
from fastapi.responses import JSONResponse
import requests
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:                                          
    from transformers import BitsAndBytesConfig
except ImportError:                                                             
    BitsAndBytesConfig = None                            

try:                                     
    from peft import PeftModel
except ModuleNotFoundError:                                          
    PeftModel = None                            

from adaptive_resource_trainer import (
    AdaptiveTrainingConfig as FeedbackRefineConfig,
    AdaptiveTrainingError as FeedbackRefineError,
    AdaptiveResourceTrainer as FeedbackRefineManager,
    build_dataset,
)

try:                                          
    import psutil
except ModuleNotFoundError:                                         
    psutil = None                            


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
        description="Необязательный путь до адаптера LoRA/адаптированной модели",
    )
    quantization: Literal['none', '4bit', '8bit'] = 'none'
    device: Optional[Literal['auto', 'cpu', 'cuda', 'mps']] = Field(
        default=None,
        description="Предпочитаемое устройство выполнения (по умолчанию авто)",
    )


class FeedbackRefineDatasetItemModel(BaseModel):
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    source: Optional[str] = Field(default=None, max_length=256)


class FeedbackRefineConfigModel(BaseModel):
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


class FeedbackRefineRequestModel(BaseModel):
    base_model_path: str = Field(..., description="Путь до базовой модели для дообучения")
    output_dir: Optional[str] = Field(
        default=None,
        description="Каталог для сохранения адаптера (по умолчанию Models/feedback_refine-<timestamp>)",
    )
    dataset: List[FeedbackRefineDatasetItemModel] = Field(..., min_items=1)
    config: FeedbackRefineConfigModel



class AgregatorRequestModel(BaseModel):
    base_url: str = Field(..., description="Базовый URL сервиса Agregator")
    endpoint: str = Field(
        "/export/csv",
        description="REST-эндпоинт Agregator для экспорта обучающих данных",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query-параметры для фильтрации выгрузки",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Дополнительные HTTP-заголовки",
    )
    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Таймаут запроса к Agregator, секунды")

    def resolved_endpoint(self) -> str:
        path = self.endpoint.strip() or "/export/csv"
        if not path.startswith("/"):
            path = f"/{path}"
        return path


class AutoFeedbackRefineRequestModel(BaseModel):
    aggregator: AgregatorRequestModel
    base_model_path: str = Field(..., description="Путь до модели, для которой запускается дообучение")
    output_dir: Optional[str] = Field(
        default=None,
        description="Каталог для сохранения результатов (по умолчанию Models/feedback_refine-<timestamp>)",
    )
    config: FeedbackRefineConfigModel
    include_previous_dataset: bool = Field(
        default=True,
        description="Если указан previous_feedback_refine_path, объединить старые примеры с новыми",
    )
    previous_feedback_refine_path: Optional[str] = Field(
        default=None,
        description="Путь к каталогу с предыдущим запуском, чтобы переиспользовать данные",
    )
    deduplicate: bool = Field(
        default=True,
        description="Удалять дубликаты после объединения датасетов",
    )
    min_examples: int = Field(
        default=1,
        ge=1,
        le=100_000,
        description="Минимальное количество примеров для запуска дообучения",
    )

class FeedbackRefineEventModel(BaseModel):
    timestamp: float
    level: str
    message: str


class FeedbackRefineJobConfigResponse(BaseModel):
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


class FeedbackRefineJobResponse(BaseModel):
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
    config: FeedbackRefineJobConfigResponse
    events: Optional[List[FeedbackRefineEventModel]] = None
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

_FEEDBACK_REFINE_MANAGER = FeedbackRefineManager(root_dir=Path(__file__).parent)

if psutil is not None:
                                                                
    try:
        psutil.cpu_percent(interval=None)
    except Exception:                                   
        pass

    try:
        _PROCESS = psutil.Process()
        _PROCESS.cpu_percent(interval=None)
    except Exception:                                               
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
    base_url = config.base_url.rstrip('/')
    endpoint = config.resolved_endpoint()
    url = urljoin(base_url + '/', endpoint.lstrip('/'))
    try:
        response = requests.get(url, params=config.params, headers=config.headers, timeout=config.timeout)
    except requests.RequestException as exc:
        raise FeedbackRefineError(f"Не удалось запросить обучающие примеры из Agregator ({url}): {exc}") from exc
    if response.status_code >= 400:
        raise FeedbackRefineError(f"Agregator вернул статус {response.status_code}: {response.text[:200]}")

    text_payload = response.text.strip()
    if not text_payload:
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
        reader = csv.DictReader(io.StringIO(text_payload))
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
        raise FeedbackRefineError(f"Не найден сохранённый обучающий датасет: {dataset_path}")

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
        unique.append(
            {
                "input": input_text,
                "output": output_text,
                "source": item.get("source"),
            }
        )
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
    except Exception as exc:                
        logger.exception("Не удалось загрузить токенизатор %s", resolved_path)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить токенизатор: {exc}") from exc

    load_kwargs: Dict[str, Any] = {}
    if quantization_mode in {"4bit", "8bit"}:
        if quantization_mode == "4bit":
            quant_config = BitsAndBytesConfig(                          
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            quant_config = BitsAndBytesConfig(                          
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
    except Exception as exc:                
        logger.exception("Не удалось загрузить модель %s", resolved_path)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить модель: {exc}") from exc

    if resolved_adapter is not None:
        if PeftModel is None:                                         
            raise HTTPException(
                status_code=500,
                detail="Поддержка LoRA не активна: пакет 'peft' не установлен",
            )
        try:
            model = PeftModel.from_pretrained(model, resolved_adapter, is_trainable=False)
        except Exception as exc:                
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
    except Exception as exc:                
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
    except Exception as exc:                
        raise HTTPException(status_code=500, detail=f"Не удалось токенизировать запрос: {exc}") from exc

    try:
        param_device = next(model.parameters()).device
    except StopIteration:                                
        param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(param_device, "type", None) == "meta":                                
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
    except Exception as exc:                
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
    except Exception as exc:                
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


def _feedback_refine_config_from_request(request_config: FeedbackRefineConfigModel) -> FeedbackRefineConfig:
    return FeedbackRefineConfig(
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


@app.post("/v1/feedback_refines", response_model=FeedbackRefineJobResponse)
async def create_feedback_refine(request: FeedbackRefineRequestModel):
    try:
        dataset = build_dataset(item.dict() for item in request.dataset)
        config = _feedback_refine_config_from_request(request.config)
        job = _FEEDBACK_REFINE_MANAGER.start_job(
            base_model_path=request.base_model_path,
            output_dir=request.output_dir,
            dataset=dataset,
            config=config,
        )
    except FeedbackRefineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()





@app.post("/v1/feedback_refines/from-agregator", response_model=FeedbackRefineJobResponse)
async def create_feedback_refine_from_agregator(request: AutoFeedbackRefineRequestModel):
    try:
        new_examples = await asyncio.to_thread(_fetch_agregator_examples, request.aggregator)
        logger.info(
            "Agregator вернул %s новых примеров (endpoint=%s)",
            len(new_examples),
            request.aggregator.resolved_endpoint(),
        )

        previous_examples: List[Dict[str, Optional[str]]] = []
        previous_dir: Optional[Path] = None
        if request.previous_feedback_refine_path:
            previous_dir = Path(request.previous_feedback_refine_path).expanduser().resolve()
            if not previous_dir.exists() or not previous_dir.is_dir():
                raise FeedbackRefineError(f"Каталог с предыдущим запуском дообучения не найден: {previous_dir}")
            if request.include_previous_dataset:
                previous_examples = await asyncio.to_thread(_load_previous_dataset, previous_dir)
                logger.info(
                    "Добавлено %s примеров из предыдущего набора %s",
                    len(previous_examples),
                    previous_dir,
                )

        combined_examples = _merge_examples(
            new_examples,
            previous_examples,
            deduplicate=request.deduplicate,
        )

        if len(combined_examples) < request.min_examples:
            raise FeedbackRefineError(
                "Недостаточно примеров для дообучения: %s (< %s)"
                % (len(combined_examples), request.min_examples)
            )

        dataset = build_dataset(combined_examples)
        config = _feedback_refine_config_from_request(request.config)

        if (
            previous_dir is not None
            and config.initial_adapter_path is None
            and request.config.method in {"lora", "qlora"}
        ):
            config = replace(config, initial_adapter_path=str(previous_dir))

        job = _FEEDBACK_REFINE_MANAGER.start_job(
            base_model_path=request.base_model_path,
            output_dir=request.output_dir,
            dataset=dataset,
            config=config,
        )
    except FeedbackRefineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@app.get("/v1/feedback_refines", response_model=List[FeedbackRefineJobResponse])
async def list_feedback_refines():
    return _FEEDBACK_REFINE_MANAGER.list_jobs()




@app.get("/v1/feedback_refines/{job_id}", response_model=FeedbackRefineJobResponse)
async def get_feedback_refine(job_id: str):
    job = _FEEDBACK_REFINE_MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача дообучения не найдена")
    return job.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "registration_programs.dataset_feedback_refiner:app",
        host="127.0.0.1",
        port=8013,
        reload=False,
    )
