from __future__ import annotations

import inspect
import json
import math
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import types

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

try:  # pragma: no cover - optional dependency
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - gracefully handle missing bitsandbytes
    BitsAndBytesConfig = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ModuleNotFoundError as exc:  # pragma: no cover - validated at runtime
    raise RuntimeError(
        "Package 'peft' is required for fine-tuning but is not installed. "
        "Install it with `pip install peft`."
    ) from exc


class FineTuneError(Exception):
    """Raised when a fine-tune job cannot be started or executed."""


@dataclass(frozen=True)
class DatasetItem:
    """Single training pair."""

    input: str
    output: str
    source: Optional[str] = None


@dataclass(frozen=True)
class FineTuneConfig:
    """Parameters describing a fine-tune run."""

    method: str
    quantization: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int
    warmup_steps: int = 0
    target_modules: Optional[List[str]] = None
    initial_adapter_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationDataset(Dataset):
    """Tiny dataset wrapper that applies the chat template on the fly."""

    def __init__(
        self,
        items: Sequence[DatasetItem],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        if not items:
            raise FineTuneError("Dataset must contain at least one example")
        self._items = list(items)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self._items[index]
        user_content = example.input.strip()
        assistant_content = example.output.strip()
        if not user_content or not assistant_content:
            raise FineTuneError("Training examples must not be empty")

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self._tokenizer(
            prompt,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0]
        labels = input_ids.clone()

        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels,
        }


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_non_finite_tensor(state: Mapping[str, torch.Tensor]) -> Optional[Tuple[str, str]]:
    """Return the first tensor with NaN/Inf values from the provided state dict."""

    with torch.no_grad():
        for name, tensor in state.items():
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


class FineTuneJob:
    """Tracks the lifecycle of a single fine-tune invocation."""

    def __init__(
        self,
        *,
        job_id: str,
        base_model_path: Path,
        output_dir: Path,
        config: FineTuneConfig,
        dataset_size: int,
    ) -> None:
        self.id = job_id
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.config = config
        self.dataset_size = dataset_size
        self.status: str = "queued"
        self.progress: float = 0.0
        self.message: str = "Ожидает запуска"
        self.metrics: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self.events: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.resume_from_checkpoint: Optional[Path] = None
        self._lock = threading.Lock()

    def _touch(self) -> None:
        self.updated_at = time.time()

    def update(
        self,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if status is not None:
                self.status = status
            if progress is not None:
                self.progress = max(0.0, min(100.0, float(progress)))
            if message is not None:
                self.message = message
            if metrics:
                self.metrics.update(metrics)
            if error is not None:
                self.error = error
            self._touch()

    def log(self, message: str, *, level: str = "info") -> None:
        event = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
        }
        with self._lock:
            self.events.append(event)
            # Keep only latest 200 records to avoid unbounded growth
            if len(self.events) > 200:
                self.events = self.events[-200:]
            self.message = message
            self._touch()

    def to_dict(self, *, include_events: bool = True) -> Dict[str, Any]:
        with self._lock:
            payload: Dict[str, Any] = {
                "id": self.id,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "metrics": self.metrics,
                "error": self.error,
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "startedAt": self.started_at,
                "finishedAt": self.finished_at,
                "datasetSize": self.dataset_size,
                "outputDir": str(self.output_dir),
                "baseModelPath": str(self.base_model_path),
                "config": self.config.to_dict(),
                "resumeCheckpoint": str(self.resume_from_checkpoint) if self.resume_from_checkpoint else None,
            }
            if include_events:
                payload["events"] = list(self.events)
            return payload

    def mark_started(self) -> None:
        now = time.time()
        with self._lock:
            self.started_at = now
            self.status = "running"
            self.message = "Запуск обучения"
            self.updated_at = now

    def mark_completed(self) -> None:
        now = time.time()
        with self._lock:
            self.finished_at = now
            self.status = "completed"
            self.progress = 100.0
            self.message = "Обучение завершено"
            self.resume_from_checkpoint = None
            self.updated_at = now

    def mark_failed(self, message: str) -> None:
        now = time.time()
        with self._lock:
            self.finished_at = now
            self.status = "failed"
            self.error = message
            self.message = message
            self.resume_from_checkpoint = None
            self.updated_at = now

    def mark_cancelled(self) -> None:
        now = time.time()
        with self._lock:
            self.finished_at = now
            self.status = "cancelled"
            self.message = "Обучение отменено"
            self.resume_from_checkpoint = None
            self.updated_at = now

    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.update(message="Остановка обучения", status=self.status)

    def request_pause(self) -> None:
        self.pause_event.set()
        self.update(message="Пауза обучения", status="pausing")

    def mark_paused(self, checkpoint: Optional[Path]) -> None:
        with self._lock:
            self.status = "paused"
            self.pause_event.clear()
            self.cancel_event.clear()
            self.message = "Обучение на паузе"
            self.resume_from_checkpoint = checkpoint
            self.updated_at = time.time()


class _JobProgressCallback(TrainerCallback):
    _TRACKED_KEYS = {
        "loss",
        "train_loss",
        "learning_rate",
        "epoch",
        "grad_norm",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_tokens_per_second",
        "eval_loss",
        "eval_runtime",
        "eval_samples_per_second",
    }

    def __init__(self, job: FineTuneJob, total_steps: int) -> None:
        self._job = job
        self._total_steps = max(1, total_steps)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:  # pragma: no cover - executed during training
        if state.global_step is not None and state.global_step >= 0:
            progress = state.global_step / self._total_steps * 100.0
            self._job.update(progress=progress)
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:  # pragma: no cover - executed during training
        if not logs:
            return

        numeric_logs = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }

        metrics: Dict[str, Any] = {
            key: numeric_logs[key]
            for key in self._TRACKED_KEYS.intersection(numeric_logs)
        }

        loss_value = numeric_logs.get("loss")
        if loss_value is not None and loss_value < 20:  # guard against overflow
            try:
                metrics.setdefault("perplexity", math.exp(loss_value))
            except OverflowError:  # pragma: no cover - defensive
                pass

        if metrics:
            self._job.update(metrics=metrics)


class _JobCancellationCallback(TrainerCallback):
    def __init__(self, job: FineTuneJob) -> None:
        self._job = job

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:  # pragma: no cover - executed during training
        if self._job.pause_event.is_set():
            control.should_training_stop = True
            control.should_save = True
            self._job.update(status="pausing", message="Приостанавливаю обучение по запросу пользователя")
            return control
        if self._job.cancel_event.is_set():
            control.should_training_stop = True
            control.should_save = True
            self._job.update(message="Остановка обучения пользователем")
        return control


class FineTuneManager:
    """Encapsulates fine-tune job scheduling and execution."""

    def __init__(self, *, root_dir: Path, models_subdir: str = "Models") -> None:
        self._root_dir = root_dir.resolve()
        self._models_root = (self._root_dir / models_subdir).resolve()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="finetune")
        self._jobs: Dict[str, FineTuneJob] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_lora_target_modules(model: AutoModelForCausalLM, config: FineTuneConfig) -> List[str]:
        if config.target_modules:
            targets = [module.strip() for module in config.target_modules if module and module.strip()]
            if not targets:
                raise FineTuneError("Список target_modules пуст — укажите валидные имена модулей")
            return list(dict.fromkeys(targets))

        config_obj = getattr(model, "config", None)
        model_type = str(getattr(config_obj, "model_type", ""))
        default_map: Dict[str, List[str]] = {
            "gemma": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "llama": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "mistral": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "qwen": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "phi": ["q_proj", "k_proj", "v_proj", "dense"] ,
            "gpt_neox": [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            "falcon": [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
        }

        guessed = default_map.get(model_type.lower())
        if guessed:
            return guessed

        # Heuristic: search for common projection module names among Linear layers.
        candidate_names = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                suffix = name.split(".")[-1]
                if suffix.startswith("lora_"):
                    continue
                if suffix not in {"lm_head", "embed_tokens"}:
                    candidate_names.add(suffix)

        preferred_patterns = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["gate_proj", "up_proj", "down_proj"],
            ["query_key_value", "dense"],
        ]
        collected: List[str] = []
        for pattern in preferred_patterns:
            if set(pattern).issubset(candidate_names):
                collected.extend(pattern)

        if collected:
            # Preserve order from appearance in pattern list and remove duplicates
            ordered = list(dict.fromkeys(collected))
            return ordered

        raise FineTuneError(
            "Не удалось автоматически определить target_modules для LoRA. "
            "Укажите их вручную в настройках."
        )

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [job.to_dict(include_events=False) for job in self._jobs.values()]

    def get_job(self, job_id: str) -> Optional[FineTuneJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start_job(
        self,
        *,
        base_model_path: str,
        output_dir: Optional[str],
        dataset: Sequence[DatasetItem],
        config: FineTuneConfig,
    ) -> FineTuneJob:
        base_path = Path(base_model_path).expanduser().resolve()
        if not base_path.exists():
            raise FineTuneError(f"Базовая модель не найдена: {base_path}")

        resolved_output, adjusted = self._resolve_output_dir(output_dir)
        if resolved_output.exists() and any(resolved_output.iterdir()):
            raise FineTuneError(
                f"Целевая директория {resolved_output} не пуста. Укажите другой путь или очистите каталог."
            )
        resolved_output.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job = FineTuneJob(
            job_id=job_id,
            base_model_path=base_path,
            output_dir=resolved_output,
            config=config,
            dataset_size=len(dataset),
        )

        if adjusted:
            job.log(
                "Путь вывода скорректирован на %s, так как запрошенный каталог находился вне Models" % resolved_output
            )

        with self._lock:
            if any(existing.status in {"queued", "running"} for existing in self._jobs.values()):
                raise FineTuneError("Дождитесь завершения текущей задачи дообучения")
            self._jobs[job_id] = job

        # Persist dataset snapshot for reproducibility
        self._dump_dataset(resolved_output, dataset)

        self._executor.submit(self._run_job, job, list(dataset))
        return job

    def cancel_job(self, job_id: str) -> FineTuneJob:
        job = self.get_job(job_id)
        if job is None:
            raise FineTuneError("Задача не найдена")
        if job.status not in {"queued", "running"}:
            raise FineTuneError("Задача уже завершена")
        job.request_cancel()
        return job

    def pause_job(self, job_id: str) -> FineTuneJob:
        job = self.get_job(job_id)
        if job is None:
            raise FineTuneError("Задача не найдена")
        if job.status != "running":
            raise FineTuneError("Поставить на паузу можно только выполняющуюся задачу")
        job.request_pause()
        return job

    def resume_job(self, job_id: str) -> FineTuneJob:
        job = self.get_job(job_id)
        if job is None:
            raise FineTuneError("Задача не найдена")
        if job.status != "paused":
            raise FineTuneError("Задача не находится на паузе")

        with self._lock:
            for existing in self._jobs.values():
                if existing.id == job_id:
                    continue
                if existing.status in {"queued", "running"}:
                    raise FineTuneError("Сначала завершите текущую задачу дообучения")

        dataset_items = self._load_dataset_snapshot(job)

        resume_checkpoint: Optional[Path]
        stored_checkpoint = job.resume_from_checkpoint
        if stored_checkpoint is None:
            resume_checkpoint = self._latest_checkpoint(job.output_dir)
        else:
            resume_checkpoint = stored_checkpoint if isinstance(stored_checkpoint, Path) else Path(stored_checkpoint)
            if not resume_checkpoint.exists():
                resume_checkpoint = self._latest_checkpoint(job.output_dir)

        job.pause_event.clear()
        job.cancel_event.clear()
        job.update(status="queued", message="Возобновление обучения")

        self._executor.submit(self._run_job, job, dataset_items, resume_checkpoint)
        return job

    def _resolve_output_dir(self, provided: Optional[str]) -> Tuple[Path, bool]:
        adjusted = False

        if provided:
            candidate = Path(provided).expanduser()
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            candidate = Path("Models") / f"finetune-{timestamp}"

        if candidate.is_absolute():
            requested_path = candidate.resolve()
        else:
            requested_path = (self._root_dir / candidate).resolve()

        candidate = requested_path

        try:
            candidate.relative_to(self._models_root)
        except ValueError:
            # Remap any path outside Models into the Models directory, preserving relative parts
            adjusted = True
            try:
                relative_to_root = candidate.relative_to(self._root_dir)
            except ValueError:
                candidate = (self._models_root / candidate.name).resolve()
            else:
                candidate = (self._models_root / relative_to_root).resolve()

        try:
            candidate.relative_to(self._models_root)
        except ValueError as exc:
            raise FineTuneError(
                f"Путь вывода {candidate} должен находиться внутри {self._models_root}"
            ) from exc

        return candidate, adjusted

    def _dump_dataset(self, output_dir: Path, dataset: Sequence[DatasetItem]) -> None:
        target = output_dir / "dataset.jsonl"
        with target.open("w", encoding="utf-8") as f:
            for item in dataset:
                payload = {
                    "input": item.input,
                    "output": item.output,
                }
                if item.source:
                    payload["source"] = item.source
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _run_job(
        self,
        job: FineTuneJob,
        dataset_items: List[DatasetItem],
        resume_from: Optional[Path] = None,
    ) -> None:
        try:
            job.mark_started()
            job.log("Определяю доступное устройство")
            device = _detect_device()
            job.log(f"Используется устройство: {device}")

            job.log("Загружаю токенизатор")
            tokenizer = AutoTokenizer.from_pretrained(job.base_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dataset = ConversationDataset(dataset_items, tokenizer, job.config.max_length)

            requested_batch = max(1, job.config.batch_size)
            if device == "cpu":
                per_device_batch = 1
            elif device == "mps":
                per_device_batch = min(requested_batch, 2)
            else:
                per_device_batch = min(requested_batch, 4)
            gradient_accumulation = max(1, math.ceil(requested_batch / per_device_batch))
            effective_batch = per_device_batch * gradient_accumulation

            steps_per_epoch = max(1, math.ceil(len(dataset_items) / effective_batch))
            total_steps = steps_per_epoch * max(1, job.config.epochs)

            requested_method = job.config.method.lower()
            requested_quant = (job.config.quantization or "none").lower()

            if requested_method == "full" and requested_quant != "none":
                raise FineTuneError("Полное дообучение не поддерживает квантизацию — отключите её")

            if requested_method == "qlora":
                if device != "cuda":
                    raise FineTuneError("QLoRA поддерживается только на GPU с CUDA")
                if requested_quant != "4bit":
                    raise FineTuneError("QLoRA требует квантизацию '4bit'")

            quantization_config: Optional[BitsAndBytesConfig]
            load_kwargs: Dict[str, Any] = {}
            is_quantized = False

            if requested_quant in {"4bit", "8bit"}:
                if device != "cuda":
                    raise FineTuneError(
                        "Квантизация %s требует GPU с поддержкой CUDA" % requested_quant
                    )
                if BitsAndBytesConfig is None:
                    raise FineTuneError(
                        "Для квантизации требуется пакет 'bitsandbytes'. Установите его: pip install bitsandbytes"
                    )
                if requested_quant == "4bit":
                    job.log("Активирована 4-битная квантизация (NF4)")
                    quantization_config = BitsAndBytesConfig(  # type: ignore[call-arg]
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                else:
                    job.log("Активирована 8-битная квантизация")
                    quantization_config = BitsAndBytesConfig(  # type: ignore[call-arg]
                        load_in_8bit=True,
                    )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
                is_quantized = True
            else:
                if requested_quant not in {"none", ""}:
                    raise FineTuneError(f"Неизвестный режим квантизации: {requested_quant}")
                torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
                load_kwargs["torch_dtype"] = torch_dtype

            job.log("Загружаю базовую модель")
            model = AutoModelForCausalLM.from_pretrained(
                job.base_model_path,
                **load_kwargs,
            )

            if not is_quantized:
                if device == "cuda":
                    model = model.to("cuda")
                elif device == "mps":
                    model = model.to("mps")

            use_cache_original = getattr(model.config, "use_cache", None)

            if is_quantized:
                job.log("Подготавливаю модель к k-битному обучению")
                model = prepare_model_for_kbit_training(model)

            initial_adapter_path: Optional[Path] = None
            if job.config.initial_adapter_path:
                initial_adapter_path = Path(job.config.initial_adapter_path).expanduser().resolve()
                if not initial_adapter_path.exists():
                    job.log(
                        "Предупреждение: стартовый адаптер %s не найден, обучение начнётся с нулевых весов"
                        % initial_adapter_path,
                        level="warning",
                    )
                    initial_adapter_path = None
                elif requested_method not in {"lora", "qlora"}:
                    raise FineTuneError(
                        "Начальный адаптер поддерживается только для методов LoRA/QLoRA"
                    )

            if requested_method in {"lora", "qlora"}:
                target_modules = self._resolve_lora_target_modules(model, job.config)
                job.log(
                    "Настраиваю %s адаптер (rank=%s, alpha=%s, modules=%s)"
                    % (
                        "QLoRA" if requested_method == "qlora" else "LoRA",
                        job.config.lora_rank,
                        job.config.lora_alpha,
                        ",".join(target_modules),
                    )
                )
                if initial_adapter_path is not None:
                    try:
                        model = PeftModel.from_pretrained(  # type: ignore[arg-type]
                            model,
                            str(initial_adapter_path),
                            is_trainable=True,
                        )
                        active_adapter = getattr(model, "active_adapter", None)
                        cfg: Optional[PeftConfig] = None
                        if active_adapter and isinstance(model.peft_config, dict):
                            cfg = model.peft_config.get(active_adapter)
                        if cfg is not None:
                            adapter_modules = cfg.target_modules or []
                            if set(adapter_modules) != set(target_modules):
                                job.log(
                                    "Предупреждение: список модулей адаптера (%s) отличается от текущей конфигурации (%s)"
                                    % (",".join(adapter_modules), ",".join(target_modules)),
                                    level="warning",
                                )
                        job.log("Стартовые веса адаптера загружены из %s" % initial_adapter_path)
                    except Exception as exc:
                        raise FineTuneError(
                            "Не удалось загрузить стартовый адаптер из %s: %s"
                            % (initial_adapter_path, exc)
                        ) from exc
                else:
                    lora_config = LoraConfig(
                        r=job.config.lora_rank,
                        lora_alpha=job.config.lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    model = get_peft_model(model, lora_config)
            elif requested_method == "full":
                job.log("Полное дообучение без LoRA — может потребоваться много памяти")
            else:
                raise FineTuneError(
                    "Метод дообучения '%s' пока не поддерживается" % job.config.method
                )

            use_gradient_checkpointing = device != "cpu"
            if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

            if torch.cuda.is_available():  # pragma: no cover - depends on hardware
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 - best effort
                    pass

            training_output = job.output_dir / "trainer_runs"
            fp16_training = device == "cuda" and not is_quantized
            optim_name = "paged_adamw_8bit" if is_quantized else "adamw_torch"

            training_args = TrainingArguments(
                output_dir=str(training_output),
                num_train_epochs=max(1, job.config.epochs),
                per_device_train_batch_size=per_device_batch,
                gradient_accumulation_steps=gradient_accumulation,
                learning_rate=job.config.learning_rate,
                warmup_steps=max(0, job.config.warmup_steps),
                logging_steps=max(1, min(steps_per_epoch, 10)),
                save_strategy="no",
                report_to=[],
                fp16=fp16_training,
                bf16=False,
                gradient_checkpointing=use_gradient_checkpointing,
                remove_unused_columns=False,
                dataloader_pin_memory=device == "cuda",
                optim=optim_name,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )

            accelerator = getattr(trainer, "accelerator", None)
            if accelerator is not None:
                unwrap_method = getattr(accelerator, "unwrap_model", None)
                if unwrap_method is not None:
                    signature = inspect.signature(unwrap_method)
                    if "keep_torch_compile" not in signature.parameters:
                        original_func = getattr(unwrap_method, "__func__", None)
                        if original_func is None:
                            original_func = unwrap_method

                        def _compatible_unwrap(self, *args: Any, **kwargs: Any):
                            kwargs.pop("keep_torch_compile", None)
                            return original_func(self, *args, **kwargs)

                        accelerator.unwrap_model = types.MethodType(
                            _compatible_unwrap,
                            accelerator,
                        )

            trainer.add_callback(_JobProgressCallback(job, total_steps))
            trainer.add_callback(_JobCancellationCallback(job))

            job.log(
                "Начинаю обучение: epochs=%d, requested_batch=%d, per_device_batch=%d, grad_accum=%d, шагов=%d"
                % (
                    job.config.epochs,
                    requested_batch,
                    per_device_batch,
                    gradient_accumulation,
                    total_steps,
                )
            )
            trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

            if job.pause_event.is_set():
                latest_checkpoint = self._latest_checkpoint(job.output_dir)
                job.log(
                    "Обучение приостановлено, последняя точка сохранения: %s"
                    % (latest_checkpoint.name if latest_checkpoint else "не найдена"),
                )
                job.mark_paused(latest_checkpoint)
                self._write_metadata(job)
                return

            if job.cancel_event.is_set():
                job.log("Задача отменена, сохраняю текущее состояние")
                job.mark_cancelled()
                return

            model_to_save = trainer.model
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module  # unwrap DDP

            invalid_tensor: Optional[Tuple[str, str]] = None
            if requested_method in {"lora", "qlora"} and hasattr(model_to_save, "get_peft_model_state_dict"):
                adapter_state = model_to_save.get_peft_model_state_dict()  # type: ignore[call-arg]
                invalid_tensor = _find_non_finite_tensor(adapter_state)

            if invalid_tensor is not None:
                tensor_name, tensor_kind = invalid_tensor
                message = (
                    "Веса адаптера содержат недопустимые значения (%s) в тензоре %s. "
                    "Снизьте learning_rate, увеличьте warmup или перепроверьте датасет."
                ) % (tensor_kind, tensor_name)
                job.log(message, level="error")
                job.mark_failed(message)
                return

            job.log("Сохраняю адаптер в %s" % job.output_dir)

            if hasattr(model.config, "use_cache"):
                model.config.use_cache = use_cache_original

            model_to_save.save_pretrained(job.output_dir)
            tokenizer.save_pretrained(job.output_dir)

            job.mark_completed()
            self._write_metadata(job)
        except FineTuneError as err:
            job.log(str(err), level="error")
            job.mark_failed(str(err))
        except Exception as err:  # noqa: BLE001 - surface unexpected failures
            message = f"Не удалось завершить обучение: {err}"
            job.log(message, level="error")
            job.mark_failed(message)
        finally:
            if torch.cuda.is_available():  # pragma: no cover - depends on hardware
                torch.cuda.empty_cache()


    def _write_metadata(self, job: FineTuneJob) -> None:
        metadata = {
            "id": job.id,
            "name": job.output_dir.name,
            "path": str(job.output_dir),
            "base_model_path": str(job.base_model_path),
            "dataset_size": job.dataset_size,
            "method": job.config.method,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "metrics": job.metrics,
            "config": job.config.to_dict(),
            "resume_checkpoint": str(job.resume_from_checkpoint) if job.resume_from_checkpoint else None,
        }

        metadata_path = job.output_dir / "finetune_metadata.json"
        try:
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as err:  # noqa: BLE001 - non-critical
            job.log(f"Не удалось сохранить метаданные дообучения: {err}", level="error")

    @staticmethod
    def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
        if not output_dir.exists():
            return None
        checkpoints = [
            candidate
            for candidate in output_dir.iterdir()
            if candidate.is_dir() and candidate.name.startswith("checkpoint-")
        ]
        if not checkpoints:
            return None
        try:
            return max(checkpoints, key=lambda item: item.stat().st_mtime)
        except OSError:
            return checkpoints[-1]

    def _load_dataset_snapshot(self, job: FineTuneJob) -> List[DatasetItem]:
        dataset_path = job.output_dir / "dataset.jsonl"
        if not dataset_path.exists():
            raise FineTuneError(
                f"Не найден snapshot датасета для {job.id} в {dataset_path}. Невозможно возобновить обучение."
            )

        items: List[DatasetItem] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:  # noqa: PERF203 - rare path
                    raise FineTuneError(f"Повреждён snapshot датасета: {exc}") from exc
                input_text = str(payload.get("input", "")).strip()
                output_text = str(payload.get("output", "")).strip()
                if not input_text or not output_text:
                    continue
                items.append(DatasetItem(input=input_text, output=output_text, source=payload.get("source")))

        if not items:
            raise FineTuneError("Snapshot датасета пуст — возобновление невозможно")
        return items

    def list_available_models(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        if not self._models_root.exists():
            return entries

        for child in self._models_root.iterdir():
            if not child.is_dir():
                continue

            metadata_path = child / "finetune_metadata.json"
            dataset_path = child / "dataset.jsonl"
            adapter_path = child / "adapter_config.json"
            full_model_path = child / "config.json"

            metadata: Dict[str, Any]
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - fallback to heuristic
                    metadata = {}
            else:
                metadata = {}

            if not metadata:
                if not dataset_path.exists():
                    continue
                if not adapter_path.exists() and not full_model_path.exists():
                    continue
                metadata = {
                    "id": child.name,
                    "name": child.name,
                    "dataset_size": None,
                    "created_at": child.stat().st_ctime,
                    "finished_at": child.stat().st_mtime,
                    "status": "completed",
                    "method": "lora" if adapter_path.exists() else "full",
                }

            metadata.setdefault("id", child.name)
            metadata.setdefault("name", child.name)
            metadata["path"] = str(child)
            metadata.setdefault("status", "completed")
            if metadata.get("status") != "completed":
                continue
            if "base_model_path" not in metadata:
                metadata["base_model_path"] = None
            entries.append(metadata)

        entries.sort(key=lambda item: item.get("finished_at") or item.get("created_at") or 0, reverse=True)
        return entries

    @property
    def models_root(self) -> Path:
        return self._models_root


def build_dataset(items: Iterable[Dict[str, Any]]) -> List[DatasetItem]:
    dataset: List[DatasetItem] = []
    for raw in items:
        if not raw:
            continue
        input_text = str(raw.get("input", "")).strip()
        output_text = str(raw.get("output", "")).strip()
        if not input_text or not output_text:
            raise FineTuneError("Каждый пример должен содержать поля 'input' и 'output'")
        source = raw.get("source")
        dataset.append(DatasetItem(input=input_text, output=output_text, source=str(source) if source else None))
    if not dataset:
        raise FineTuneError("Датасет пуст — добавьте примеры для обучения")
    return dataset
