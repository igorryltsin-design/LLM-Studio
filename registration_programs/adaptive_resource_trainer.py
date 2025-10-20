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

try:                                          
    from transformers import BitsAndBytesConfig
except ImportError:                                                             
    BitsAndBytesConfig = None                            

try:
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ModuleNotFoundError as exc:                                           
    raise RuntimeError(
        "Package 'peft' is required for fine-tuning but is not installed. "
        "Install it with `pip install peft`."
    ) from exc

class AdaptiveTrainingError(Exception):
@dataclass(frozen=True)
class DatasetItem:

    input: str
    output: str
    source: Optional[str] = None

@dataclass(frozen=True)
class AdaptiveTrainingConfig:

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

    def __init__(
        self,
        items: Sequence[DatasetItem],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        if not items:
            raise AdaptiveTrainingError("Dataset must contain at least one example")
        self._items = list(items)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:                              
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self._items[index]
        user_content = example.input.strip()
        assistant_content = example.output.strip()
        if not user_content or not assistant_content:
            raise AdaptiveTrainingError("Training examples must not be empty")

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

_BYTES_IN_GB = 1024 ** 3

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _bytes_to_gb(value: float) -> float:
    return round(value / _BYTES_IN_GB, 2)

def _collect_gpu_stats() -> Dict[str, object]:
    backend = _detect_device()
    devices: List[Dict[str, Optional[object]]] = []
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
                except Exception:
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

def _find_non_finite_tensor(state: Mapping[str, torch.Tensor]) -> Optional[Tuple[str, str]]:

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

class AdaptiveTrainingJob:

    def __init__(
        self,
        *,
        job_id: str,
        base_model_path: Path,
        output_dir: Path,
        config: AdaptiveTrainingConfig,
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
            self.updated_at = now

    def mark_failed(self, message: str) -> None:
        now = time.time()
        with self._lock:
            self.finished_at = now
            self.status = "failed"
            self.error = message
            self.message = message
            self.updated_at = now

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

    def __init__(self, job: AdaptiveTrainingJob, total_steps: int) -> None:
        self._job = job
        self._total_steps = max(1, total_steps)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:                                               
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
    ) -> None:                                               
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
        if loss_value is not None and loss_value < 20:                          
            try:
                metrics.setdefault("perplexity", math.exp(loss_value))
            except OverflowError:                                
                pass

        if metrics:
            self._job.update(metrics=metrics)

class AdaptiveResourceTrainer:

    def __init__(self, *, root_dir: Path, models_subdir: str = "Models") -> None:
        self._root_dir = root_dir.resolve()
        self._models_root = (self._root_dir / models_subdir).resolve()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adaptive_training")
        self._jobs: Dict[str, AdaptiveTrainingJob] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_lora_target_modules(model: AutoModelForCausalLM, config: AdaptiveTrainingConfig) -> List[str]:
        if config.target_modules:
            targets = [module.strip() for module in config.target_modules if module and module.strip()]
            if not targets:
                raise AdaptiveTrainingError("Список target_modules пуст — укажите валидные имена модулей")
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
                                                                                  
            ordered = list(dict.fromkeys(collected))
            return ordered

        raise AdaptiveTrainingError(
            "Не удалось автоматически определить target_modules для LoRA. "
            "Укажите их вручную в настройках."
        )

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [job.to_dict(include_events=False) for job in self._jobs.values()]

    def describe_hardware(self) -> Dict[str, object]:
        return _collect_gpu_stats()

    def get_job(self, job_id: str) -> Optional[AdaptiveTrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start_job(
        self,
        *,
        base_model_path: str,
        output_dir: Optional[str],
        dataset: Sequence[DatasetItem],
        config: AdaptiveTrainingConfig,
    ) -> AdaptiveTrainingJob:
        base_path = Path(base_model_path).expanduser().resolve()
        if not base_path.exists():
            raise AdaptiveTrainingError(f"Базовая модель не найдена: {base_path}")

        resolved_output, adjusted = self._resolve_output_dir(output_dir)
        if resolved_output.exists() and any(resolved_output.iterdir()):
            raise AdaptiveTrainingError(
                f"Целевая директория {resolved_output} не пуста. Укажите другой путь или очистите каталог."
            )
        resolved_output.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job = AdaptiveTrainingJob(
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
                raise AdaptiveTrainingError("Дождитесь завершения текущей задачи адаптивного обучения")
            self._jobs[job_id] = job

        self._dump_dataset(resolved_output, dataset)

        self._executor.submit(self._run_job, job, list(dataset))
        return job

    def _resolve_output_dir(self, provided: Optional[str]) -> Tuple[Path, bool]:
        adjusted = False

        if provided:
            candidate = Path(provided).expanduser()
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            candidate = Path("Models") / f"adaptive_training-{timestamp}"

        if candidate.is_absolute():
            requested_path = candidate.resolve()
        else:
            requested_path = (self._root_dir / candidate).resolve()

        candidate = requested_path

        try:
            candidate.relative_to(self._models_root)
        except ValueError:
                                                                                                
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
            raise AdaptiveTrainingError(
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
        job: AdaptiveTrainingJob,
        dataset_items: List[DatasetItem],
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
                raise AdaptiveTrainingError("Полное адаптивное обучение не поддерживает квантизацию — отключите её")

            if requested_method == "qlora":
                if device != "cuda":
                    raise AdaptiveTrainingError("QLoRA поддерживается только на GPU с CUDA")
                if requested_quant != "4bit":
                    raise AdaptiveTrainingError("QLoRA требует квантизацию '4bit'")

            quantization_config: Optional[BitsAndBytesConfig]
            load_kwargs: Dict[str, Any] = {}
            is_quantized = False

            if requested_quant in {"4bit", "8bit"}:
                if device != "cuda":
                    raise AdaptiveTrainingError(
                        "Квантизация %s требует GPU с поддержкой CUDA" % requested_quant
                    )
                if BitsAndBytesConfig is None:
                    raise AdaptiveTrainingError(
                        "Для квантизации требуется пакет 'bitsandbytes'. Установите его: pip install bitsandbytes"
                    )
                if requested_quant == "4bit":
                    job.log("Активирована 4-битная квантизация (NF4)")
                    quantization_config = BitsAndBytesConfig(                          
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                else:
                    job.log("Активирована 8-битная квантизация")
                    quantization_config = BitsAndBytesConfig(                          
                        load_in_8bit=True,
                    )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
                is_quantized = True
            else:
                if requested_quant not in {"none", ""}:
                    raise AdaptiveTrainingError(f"Неизвестный режим квантизации: {requested_quant}")
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
                    raise AdaptiveTrainingError(
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
                        model = PeftModel.from_pretrained(                          
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
                        raise AdaptiveTrainingError(
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
                job.log("Полное адаптивное обучение без LoRA — может потребоваться много памяти")
            else:
                raise AdaptiveTrainingError(
                    "Метод адаптивного обучения '%s' пока не поддерживается" % job.config.method
                )

            use_gradient_checkpointing = device != "cpu"
            if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

            if torch.cuda.is_available():                                          
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True                              
                except Exception:                              
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
            trainer.train()

            model_to_save = trainer.model
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module              

            invalid_tensor: Optional[Tuple[str, str]] = None
            if requested_method in {"lora", "qlora"} and hasattr(model_to_save, "get_peft_model_state_dict"):
                adapter_state = model_to_save.get_peft_model_state_dict()                          
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
        except AdaptiveTrainingError as err:
            job.log(str(err), level="error")
            job.mark_failed(str(err))
        except Exception as err:                                              
            message = f"Не удалось завершить обучение: {err}"
            job.log(message, level="error")
            job.mark_failed(message)
        finally:
            if torch.cuda.is_available():                                          
                torch.cuda.empty_cache()

def build_dataset(items: Iterable[Dict[str, Any]]) -> List[DatasetItem]:
    dataset: List[DatasetItem] = []
    for raw in items:
        if not raw:
            continue
        input_text = str(raw.get("input", "")).strip()
        output_text = str(raw.get("output", "")).strip()
        if not input_text or not output_text:
            raise AdaptiveTrainingError("Каждый пример должен содержать поля 'input' и 'output'")
        source = raw.get("source")
        dataset.append(DatasetItem(input=input_text, output=output_text, source=str(source) if source else None))
    if not dataset:
        raise AdaptiveTrainingError("Датасет пуст — добавьте примеры для обучения")
    return dataset
