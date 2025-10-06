import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Play,
  Square,
  Upload,
  Plus,
  Trash2,
  BarChart3,
  CheckCircle2,
  Circle,
  AlertCircle,
  Pause,
} from 'lucide-react';
import TrainingMetricsPanel from './TrainingMetricsPanel';
import { useHistory } from '../contexts/HistoryContext';
import { useStatus } from '../contexts/StatusContext';
import { useTraining } from '../contexts/TrainingContext';
import { useSettings } from '../contexts/SettingsContext';
import type { DatasetItem, TrainingConfig, TrainingMethod, TrainingQuantization } from '../types/training';
import {
  cancelFineTuneJob,
  createFineTuneJob,
  FineTuneError,
  FineTuneJob,
  FineTuneJobStatus,
  FineTuneConfigPayload,
  getFineTuneJob,
  listFineTuneJobs,
  pauseFineTuneJob,
  resumeFineTuneJob,
} from '../services/fineTune';
import type { SystemStats } from '../services/systemMonitor';
import { useLazyList } from '../hooks/useLazyList';

interface TrainingTabProps {
  systemStats?: SystemStats;
}

interface RecommendationResult {
  canApply: boolean;
  summary: string;
  details: string[];
  warnings: string[];
  parameterHints: ParameterHint[];
  config?: TrainingConfig;
}

interface ParameterHint {
  title: string;
  suggestions: string[];
}

interface LabelWithHintProps {
  label: string;
  hint: string;
}

const LabelWithHint = ({ label, hint }: LabelWithHintProps) => (
  <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">
    <span className="flex items-center gap-1">
      {label}
      <span
        className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 bg-slate-100 text-[11px] leading-none text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
        title={hint}
      >
        ?
      </span>
    </span>
  </label>
);

const generateId = (prefix: string) => {
  const uniquePart = typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(16).slice(2, 10)}`;
  return `${prefix}-${uniquePart}`;
};

type ProcessStepStatus = 'pending' | 'current' | 'done';

interface ProcessStepDescriptor {
  id: string;
  title: string;
  description: string;
  status: ProcessStepStatus;
}

const RoadmapStep = ({ step }: { step: ProcessStepDescriptor }) => {
  const palette: Record<ProcessStepStatus, { circle: string; text: string; bar: string }> = {
    done: {
      circle: 'bg-emerald-500 text-white border-emerald-400',
      text: 'text-emerald-700 dark:text-emerald-100',
      bar: 'bg-emerald-100 dark:bg-emerald-500/60',
    },
    current: {
      circle: 'bg-blue-500 text-white border-blue-400',
      text: 'text-blue-700 dark:text-blue-100',
      bar: 'bg-blue-100 dark:bg-blue-500/40',
    },
    pending: {
      circle: 'bg-slate-200 text-slate-500 border-slate-300 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700',
      text: 'text-slate-500 dark:text-slate-400',
      bar: 'bg-slate-200 dark:bg-slate-800',
    },
  };

  const paletteEntry = palette[step.status];

  return (
    <div className="flex min-w-[160px] flex-1 flex-col items-center">
      <span className={`flex h-10 w-10 items-center justify-center rounded-full border text-sm font-semibold shadow-sm ${paletteEntry.circle}`}>
        {step.status === 'done' ? <CheckCircle2 className="h-5 w-5" /> : <Circle className="h-5 w-5" />}
      </span>
      <div className={`mt-3 text-center ${paletteEntry.text}`}>
        <p className="text-sm font-semibold">{step.title}</p>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{step.description}</p>
      </div>
      <div className={`mt-4 h-1 w-full rounded-full ${paletteEntry.bar}`} aria-hidden="true" />
    </div>
  );
};

const InlineNotice = ({
  tone = 'info',
  children,
}: {
  tone?: 'info' | 'warning' | 'error';
  children: ReactNode;
}) => {
  const palette = tone === 'error'
    ? 'border-rose-500/40 bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-100'
    : tone === 'warning'
      ? 'border-amber-400/40 bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-100'
      : 'border-sky-400/40 bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-100';
  return (
    <div className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-xs shadow-sm ${palette}`}>
      <AlertCircle className="mt-0.5 h-4 w-4 flex-none" />
      <span className="leading-relaxed">{children}</span>
    </div>
  );
};

const parseCsvLine = (line: string): string[] => {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      result.push(current);
      current = '';
      continue;
    }

    current += char;
  }

  result.push(current);
  return result.map(value => value.trim());
};

interface ParsedDatasetRecord {
  input: string;
  output: string;
  source?: string;
}

const normaliseDatasetRecord = (record: Record<string, unknown>): ParsedDatasetRecord | null => {
  const rawInput = record.input ?? record.question ?? record.prompt;
  const rawOutput = record.output ?? record.answer ?? record.completion;

  const input = typeof rawInput === 'string' ? rawInput.trim() : '';
  const output = typeof rawOutput === 'string' ? rawOutput.trim() : '';
  const source = typeof record.source === 'string' && record.source.trim().length > 0
    ? record.source.trim()
    : undefined;

  if (!input || !output) {
    return null;
  }

  return { input, output, source };
};

const parseCsvDataset = (csv: string): ParsedDatasetRecord[] => {
  const sanitized = csv.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
  if (!sanitized) {
    return [];
  }

  const [headerLine, ...rows] = sanitized.split(/\n+/).filter(Boolean);
  if (!headerLine) {
    return [];
  }

  const headers = parseCsvLine(headerLine).map(header => header.toLowerCase());
  const headerMap = new Map<number, string>();
  headers.forEach((header, index) => headerMap.set(index, header));

  return rows
    .map((row) => {
      const values = parseCsvLine(row);
      const record: Record<string, unknown> = {};
      values.forEach((value, index) => {
        const header = headerMap.get(index) ?? `column_${index}`;
        record[header] = value;
      });
      return normaliseDatasetRecord(record);
    })
    .filter((item): item is ParsedDatasetRecord => Boolean(item));
};

const parseJsonDataset = (jsonText: string): ParsedDatasetRecord[] => {
  if (!jsonText.trim()) {
    return [];
  }

  const parsed = JSON.parse(jsonText) as unknown;
  const payload = Array.isArray(parsed)
    ? parsed
    : typeof parsed === 'object' && parsed !== null && Array.isArray((parsed as Record<string, unknown>).data)
      ? (parsed as Record<string, unknown>).data
      : null;

  if (!Array.isArray(payload)) {
    throw new Error('Файл JSON должен содержать массив объектов с полями input/output.');
  }

  return payload
    .map((entry) => (entry && typeof entry === 'object' ? normaliseDatasetRecord(entry as Record<string, unknown>) : null))
    .filter((item): item is ParsedDatasetRecord => Boolean(item));
};

const detectFileFormat = (fileName: string, mimeType: string | undefined): 'csv' | 'json' | null => {
  const lowerName = fileName.toLowerCase();
  if (lowerName.endsWith('.json') || mimeType === 'application/json') {
    return 'json';
  }
  if (lowerName.endsWith('.csv') || mimeType === 'text/csv' || mimeType === 'application/vnd.ms-excel') {
    return 'csv';
  }
  return null;
};

const TOKENS_PER_CHAR_ESTIMATE = 0.25;
const PRESET_STORAGE_KEY = 'llm-studio-training-presets';

interface TrainingPreset {
  id: string;
  label: string;
  createdAt: number;
  datasetSize: number;
  avgTokens: number;
  config: TrainingConfig;
}

const mapPayloadToConfig = (
  payload: FineTuneConfigPayload,
  fallback: TrainingConfig,
  outputDir: string,
): TrainingConfig => ({
  method: payload.method,
  quantization: payload.quantization as TrainingQuantization,
  loraRank: payload.lora_rank,
  loraAlpha: payload.lora_alpha,
  learningRate: payload.learning_rate,
  batchSize: payload.batch_size,
  epochs: payload.epochs,
  maxLength: payload.max_length,
  warmupSteps: payload.warmup_steps,
  outputDir,
  targetModules: payload.target_modules?.join(',') ?? fallback.targetModules,
  initialAdapterPath: payload.initial_adapter_path ?? fallback.initialAdapterPath,
});

const TrainingTab = ({ systemStats }: TrainingTabProps) => {
  const {
    dataset,
    addDatasetItems,
    removeDatasetItem,
    clearDataset,
    config,
    setConfig,
    isTraining,
    setIsTraining,
    trainingProgress,
    setTrainingProgress,
    activeJob,
    setActiveJob,
    activeJobId,
    lastError,
    setLastError,
    metricsHistory,
    appendMetrics,
    resetMetrics,
    trainingTasks,
    setTrainingTasks,
  } = useTraining();
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [newItem, setNewItem] = useState({ input: '', output: '' });
  const pollingIntervalRef = useRef<number | null>(null);
  const trainingProgressRef = useRef(trainingProgress);
  const startPollingRef = useRef<((jobId: string) => void) | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const [isPausePending, setIsPausePending] = useState(false);
  const [isResumePending, setIsResumePending] = useState(false);

  const { settings, updateSettings } = useSettings();
  const { addTrainingSession } = useHistory();
  const { setActivity, updateActivity, clearActivity } = useStatus();

  const [presets, setPresets] = useState<TrainingPreset[]>(() => {
    if (typeof window === 'undefined') {
      return [];
    }
    try {
      const raw = window.localStorage.getItem(PRESET_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw) as TrainingPreset[];
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed
        .filter((item): item is TrainingPreset => Boolean(item && item.config))
        .slice(0, 6);
    } catch (error) {
      console.warn('Failed to load training presets from storage', error);
      return [];
    }
  });

  const persistPresets = useCallback((updater: (current: TrainingPreset[]) => TrainingPreset[]) => {
    setPresets((prev) => {
      const next = updater(prev);
      if (typeof window !== 'undefined') {
        try {
          window.localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(next.slice(0, 6)));
        } catch (error) {
          console.warn('Failed to persist training presets', error);
        }
      }
      return next.slice(0, 6);
    });
  }, []);

  useEffect(() => {
    if (activeJob?.config?.target_modules?.length) {
      setConfig(prev => ({
        ...prev,
        targetModules: activeJob.config.target_modules!.join(','),
      }));
    }
  }, [activeJob?.config?.target_modules, setConfig]);

  useEffect(() => {
    trainingProgressRef.current = trainingProgress;
  }, [trainingProgress]);

  const activeMetricsHistory = useMemo(() => (
    activeJobId ? metricsHistory[activeJobId] ?? [] : []
  ), [activeJobId, metricsHistory]);

  const datasetSize = dataset.length;

  const datasetStats = useMemo(() => {
    if (datasetSize === 0) {
      return {
        avgInputChars: 0,
        avgOutputChars: 0,
        avgTotalTokens: 0,
        maxTotalTokens: 0,
      };
    }

    let totalInput = 0;
    let totalOutput = 0;
    let maxTokens = 0;

    dataset.forEach((item) => {
      const inputChars = item.input.length;
      const outputChars = item.output.length;
      totalInput += inputChars;
      totalOutput += outputChars;
      const estimatedTokens = Math.round((inputChars + outputChars) * TOKENS_PER_CHAR_ESTIMATE);
      if (estimatedTokens > maxTokens) {
        maxTokens = estimatedTokens;
      }
    });

    const avgInputChars = totalInput / datasetSize;
    const avgOutputChars = totalOutput / datasetSize;
    const avgTotalTokens = Math.round((avgInputChars + avgOutputChars) * TOKENS_PER_CHAR_ESTIMATE);

    return {
      avgInputChars,
      avgOutputChars,
      avgTotalTokens,
      maxTotalTokens: Math.max(maxTokens, avgTotalTokens),
    };
  }, [dataset, datasetSize]);

  const hasHistoricalMetrics = useMemo(
    () => Object.values(metricsHistory).some(points => Array.isArray(points) && points.length > 0),
    [metricsHistory],
  );

  const validationMessages = useMemo(() => {
    const messages: Array<{ tone: 'warning' | 'error'; text: string }> = [];

    if (datasetSize === 0) {
      messages.push({ tone: 'warning', text: 'Добавьте минимум одну QA-пару, чтобы запустить обучение.' });
    }

    if (datasetStats.maxTotalTokens > 0 && config.maxLength < datasetStats.maxTotalTokens) {
      messages.push({
        tone: 'warning',
        text: `Максимальная длина генерации (${config.maxLength} токенов) меньше максимальной длины примеров (${datasetStats.maxTotalTokens}). Ответы могут обрываться.`,
      });
    }

    if (config.learningRate <= 0) {
      messages.push({ tone: 'error', text: 'Learning rate должен быть больше нуля.' });
    } else if (config.learningRate > 0.01) {
      messages.push({ tone: 'warning', text: 'Learning rate выглядит подозрительно высоким. Попробуйте значения 0.0001–0.001.' });
    }

    if (datasetSize > 0 && config.batchSize > datasetSize) {
      messages.push({ tone: 'warning', text: 'Batch size больше размера датасета — обучение будет использовать одни и те же примеры.' });
    }

    if (config.method === 'full' && config.quantization !== 'none') {
      messages.push({ tone: 'error', text: 'Полное дообучение не поддерживает квантизацию. Установите «Без квантизации».' });
    }

    if (config.method === 'qlora' && config.quantization !== '4bit') {
      messages.push({ tone: 'warning', text: 'Для QLoRA выберите квантизацию 4-bit — иначе экономия памяти не сработает.' });
    }

    if (config.initialAdapterPath && config.method === 'full') {
      messages.push({ tone: 'warning', text: 'Стартовый адаптер используется только для LoRA/QLoRA и будет проигнорирован при полном обучении.' });
    }

    return messages;
  }, [config.batchSize, config.initialAdapterPath, config.learningRate, config.maxLength, config.method, config.quantization, datasetSize, datasetStats.maxTotalTokens]);

  const hasDataset = datasetSize > 0;
  const hasActiveRun = Boolean(activeJobId) || isTraining;
  const hasCompletedJob = hasHistoricalMetrics || Boolean(activeJob && activeJob.status === 'completed');
  const hasMonitoring = activeMetricsHistory.length > 0 || hasHistoricalMetrics;

  const jobStatusLabels: Record<FineTuneJobStatus, string> = {
    queued: 'в очереди',
    running: 'в работе',
    completed: 'завершено',
    failed: 'ошибка',
    cancelled: 'отменено',
    paused: 'на паузе',
    pausing: 'ставим на паузу',
  };

  const upsertTask = useCallback(
    (job: FineTuneJob) => {
      setTrainingTasks((prev) => {
        const others = prev.filter(item => item.id !== job.id);
        const next = [job, ...others];
        next.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
        return next;
      });
    },
    [setTrainingTasks],
  );

  const refreshTasks = useCallback(async () => {
    if (!settings.baseModelServerUrl) {
      setTrainingTasks([]);
      return;
    }
    try {
      const items = await listFineTuneJobs(settings.baseModelServerUrl);
      items.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
      setTrainingTasks(items);
    } catch (error) {
      console.error('Не удалось получить список задач дообучения:', error);
    }
  }, [settings.baseModelServerUrl, setTrainingTasks]);

  useEffect(() => {
    let cancelled = false;
    if (!settings.baseModelServerUrl) {
      setTrainingTasks([]);
      return () => {
        cancelled = true;
      };
    }
    const scheduleRefresh = async () => {
      if (cancelled) {
        return;
      }
      await refreshTasks();
    };

    void scheduleRefresh();
    const interval = window.setInterval(scheduleRefresh, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [refreshTasks, settings.baseModelServerUrl, setTrainingTasks]);

  useEffect(() => {
    const runningStatuses: FineTuneJobStatus[] = ['queued', 'running', 'pausing'];
    const candidate = trainingTasks.find(task => runningStatuses.includes(task.status));
    if (!candidate) {
      return;
    }

    const activeIsRunning = activeJob ? runningStatuses.includes(activeJob.status) : false;
    if (activeIsRunning && activeJob?.id === candidate.id) {
      return;
    }

    if (!activeJob || !activeIsRunning) {
      setActiveJob(candidate);
      setTrainingProgress(candidate.progress);
      setIsTraining(candidate.status === 'queued' || candidate.status === 'running');
      appendMetrics(candidate.id, candidate.metrics ?? {});
      startPollingRef.current?.(candidate.id);
    }
  }, [
    activeJob,
    trainingTasks,
    setActiveJob,
    setTrainingProgress,
    setIsTraining,
    appendMetrics,
    startPollingRef,
  ]);

  const processSteps = useMemo<ProcessStepDescriptor[]>(() => {
    const raw: Array<Omit<ProcessStepDescriptor, 'status'>> = [
      {
        id: 'collect',
        title: 'Сбор датасета',
        description: 'Импортируйте файлы или создайте пары вручную',
      },
      {
        id: 'preview',
        title: 'Предпросмотр',
        description: 'Проверьте и почистите данные перед запуском',
      },
      {
        id: 'launch',
        title: 'Запуск',
        description: 'Настройте гиперпараметры и запустите задачу',
      },
      {
        id: 'monitor',
        title: 'Мониторинг',
        description: 'Следите за метриками и статусом обучения',
      },
    ];

    const completion: Record<string, boolean> = {
      collect: hasDataset,
      preview: hasDataset,
      launch: hasActiveRun || hasCompletedJob,
      monitor: hasMonitoring || hasCompletedJob,
    };

    let currentAssigned = false;
    return raw.map((item, index) => {
      const done = completion[item.id] ?? false;
      let status: ProcessStepStatus = 'pending';
      if (done) {
        status = 'done';
      } else if (!currentAssigned) {
        status = 'current';
        currentAssigned = true;
      }
      return { ...item, status, description: item.description + (index === 0 && !hasDataset ? ' Добавьте минимум одну пару.' : '') };
    });
  }, [hasActiveRun, hasCompletedJob, hasDataset, hasMonitoring]);

  const {
    containerRef: datasetListRef,
    handleScroll: handleDatasetScroll,
    visibleItems: visibleDatasetItems,
    hasMore: hasMoreDatasetItems,
    loadMore: loadMoreDatasetItems,
  } = useLazyList(dataset, {
    initialBatchSize: 15,
    batchSize: 15,
    resetKey: dataset.length,
  });

  const applyPreset = useCallback((preset: TrainingPreset) => {
    setConfig(() => ({ ...preset.config }));
  }, [setConfig]);

  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current !== null) {
      window.clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }, []);

  const recommendation = useMemo<RecommendationResult>(() => {
    if (!systemStats) {
      return {
        canApply: false,
        summary: 'Автонастройка недоступна — нет данных о системе',
        details: [],
        warnings: ['Проверьте, что сервер базовой модели запущен и доступен по указанному URL.'],
        parameterHints: [],
      };
    }

    if (systemStats.status !== 'ok') {
      return {
        canApply: false,
        summary: 'Автонастройка недоступна — сервер вернул ошибку',
        details: [],
        warnings: [systemStats.message ?? 'Сервер системного статуса недоступен.'],
        parameterHints: [],
      };
    }

    const details: string[] = [];
    const warnings: string[] = [];
    const parameterHints: ParameterHint[] = [];

    const backend = systemStats.gpuBackend;
    const gpuName = systemStats.gpuName ?? (backend === 'cuda' ? 'CUDA GPU' : backend === 'mps' ? 'Apple MPS' : 'CPU');
    const gpuMemory = systemStats.gpuMemoryTotalGb ?? null;

    const recommended: TrainingConfig = {
      ...config,
    };

    if (datasetSize > 0) {
      const avgTotalTokens = datasetStats.avgTotalTokens;
      details.push(
        `В датасете ${datasetSize} примеров (≈${avgTotalTokens} токенов в среднем). Параметры рассчитаны так, чтобы учесть текущий объём и среднюю длину примеров.`,
      );

      if (datasetSize < 20) {
        warnings.push('Примеров < 20 — дообучение может переобучиться на отдельных ответах. Попробуйте собрать больше данных или добавьте искусственно сгенерированные пары.');
      } else if (datasetSize < 75) {
        warnings.push('Малый датасет (< 75 примеров). Рекомендуется увеличить объём или включить регуляризацию (dropout, label smoothing) при экспорте.');
      } else if (datasetSize > 1000) {
        details.push('Большой датасет — можно постепенно повышать batch size и уменьшать число эпох, чтобы ускорить обучение.');
      }

      if (datasetSize < 75) {
        parameterHints.push({
          title: 'Генерация',
          suggestions: [
            'Для небольших датасетов используйте temperature 0–0.3. Это снижает риск появления некорректных вероятностей при семплировании.',
            'Если во время чата появляется ошибка про probability tensor, временно установите temperature = 0 (greedy) и перезапустите генерацию.',
          ],
        });
      }

    } else {
      warnings.push('Датасет пуст — добавьте примеры перед запуском обучения.');
    }

    let recommendedMethod: TrainingMethod = config.method;
    let recommendedQuant: TrainingQuantization = config.quantization;
    let recommendedBatch = config.batchSize;
    let recommendedRank = config.loraRank;
    let recommendedAlpha = config.loraAlpha;
    let recommendedLr = config.learningRate;
    let recommendedEpochs = config.epochs;
    let recommendedMaxLen = config.maxLength;
    let recommendedWarmup = config.warmupSteps;

    const avgTotalTokens = datasetStats.avgTotalTokens || 256;
    const maxObservedTokens = datasetStats.maxTotalTokens || avgTotalTokens;

    const isTinyDataset = datasetSize > 0 && datasetSize < 50;
    const isLargeDataset = datasetSize >= 1000;

    if (backend === 'cuda') {
      const vram = gpuMemory ?? 12;
      details.push(`Обнаружен GPU ${gpuName}${gpuMemory ? ` (${gpuMemory.toFixed(1)} GB VRAM)` : ''}.`);

      if (vram >= 22) {
        recommendedMethod = 'full';
        recommendedQuant = 'none';
        recommendedBatch = vram >= 32 ? 6 : 4;
        recommendedLr = isTinyDataset ? 0.000008 : isLargeDataset ? 0.000008 : 0.00001;
        recommendedEpochs = datasetSize >= 1500 ? 2 : isTinyDataset ? 4 : 3;
        recommendedMaxLen = Math.max(1024, Math.min(2048, Math.max(config.maxLength, maxObservedTokens + 256)));
        recommendedWarmup = datasetSize > 0 ? Math.min(200, Math.max(10, Math.round(datasetSize * 0.05))) : 0;
        details.push('VRAM ≥ 22 GB позволяет выполнить полное дообучение без квантизации.');
      } else if (vram >= 10) {
        recommendedMethod = 'qlora';
        recommendedQuant = '4bit';
        recommendedRank = vram >= 16 ? 64 : 48;
        recommendedAlpha = recommendedRank * 2;
        recommendedBatch = vram >= 16 ? 4 : 2;
        recommendedLr = isTinyDataset ? 0.00015 : 0.0002;
        recommendedEpochs = datasetSize >= 800 ? 3 : isTinyDataset ? 5 : 4;
        recommendedMaxLen = Math.min(Math.max(768, maxObservedTokens + 128), 1536);
        recommendedWarmup = datasetSize > 0 ? Math.min(200, Math.max(10, Math.round(datasetSize * 0.1))) : 0;
        details.push('QLoRA с 4-битной квантизацией экономит память и стабильно работает на доступной VRAM.');
      } else {
        recommendedMethod = 'lora';
        recommendedQuant = vram >= 8 ? '8bit' : '4bit';
        recommendedRank = 32;
        recommendedAlpha = 64;
        recommendedBatch = vram >= 8 ? 2 : 1;
        recommendedLr = isTinyDataset ? 0.00018 : 0.00022;
        recommendedEpochs = datasetSize >= 400 ? 3 : isTinyDataset ? 5 : 4;
        recommendedMaxLen = Math.min(Math.max(640, maxObservedTokens + 96), 1024);
        recommendedWarmup = datasetSize > 0 ? Math.min(120, Math.max(5, Math.round(datasetSize * 0.1))) : 0;
        details.push('Недостаточно VRAM для полного обучения — предлагаем лёгкую LoRA с квантизацией для экономии памяти.');
      }

      if (settings.deviceType !== 'cuda') {
        warnings.push(`В настройках выбран режим ${settings.deviceType.toUpperCase()} — переключитесь на CUDA, чтобы использовать обнаруженный GPU.`);
      }
    } else if (backend === 'mps') {
      recommendedMethod = 'lora';
      recommendedQuant = 'none';
      recommendedRank = 32;
      recommendedAlpha = 64;
      recommendedBatch = 1;
      recommendedLr = isTinyDataset ? 0.00018 : 0.00022;
      recommendedEpochs = datasetSize >= 350 ? 3 : isTinyDataset ? 5 : 4;
      recommendedMaxLen = Math.min(Math.max(640, maxObservedTokens + 96), 1024);
      recommendedWarmup = datasetSize > 0 ? Math.min(100, Math.max(5, Math.round(datasetSize * 0.12))) : 0;
      details.push('Обнаружен Apple MPS — используем LoRA без квантизации, потому что bitsandbytes недоступен на этой платформе.');
      warnings.push('Для QLoRA и инференса на CUDA потребуется видеокарта NVIDIA.');
    } else {
      recommendedMethod = 'lora';
      recommendedQuant = 'none';
      recommendedRank = 16;
      recommendedAlpha = 32;
      recommendedBatch = 1;
      recommendedLr = isTinyDataset ? 0.00018 : 0.00022;
      recommendedEpochs = datasetSize >= 350 ? 3 : 5;
      recommendedMaxLen = Math.min(Math.max(512, maxObservedTokens + 64), 768);
      recommendedWarmup = datasetSize > 0 ? Math.min(80, Math.max(0, Math.round(datasetSize * 0.12))) : 0;
      details.push('GPU не обнаружен — LoRA на CPU позволит быстро получить результат, но обучение будет медленным.');
      warnings.push('Для QLoRA и быстрой генерации подключите CUDA-совместимый GPU.');
    }

    recommended.method = recommendedMethod;
    recommended.quantization = recommendedQuant;
    recommended.loraRank = recommendedRank;
    recommended.loraAlpha = recommendedAlpha;
    recommended.batchSize = Math.max(1, recommendedBatch);
    recommended.learningRate = Number(recommendedLr.toFixed(6));
    recommended.epochs = Math.max(1, recommendedEpochs);
    recommended.maxLength = Math.max(256, recommendedMaxLen);
    recommended.warmupSteps = Math.max(0, recommendedWarmup);

    if (datasetSize > 0) {
      const warmupPercent = Math.round((recommended.warmupSteps / Math.max(datasetSize, 1)) * 100);
      const tokensComment = recommended.maxLength < maxObservedTokens
        ? 'Рекомендуется сократить длину примеров или увеличить max length вручную.'
        : `Макс. длина охватывает до ≈${Math.round(recommended.maxLength * 0.8)} токенов содержимого.`;
      details.push(`Warmup: ${recommended.warmupSteps} шагов (~${warmupPercent}% от датасета) для стабильного старта обучения.`);
      details.push(`Эпохи: ${recommended.epochs} — компромисс между обобщением и риском переобучения на ${datasetSize} примерах.`);
      details.push(tokensComment);
      if (recommended.maxLength < maxObservedTokens) {
        warnings.push('Часть примеров длиннее допустимого контекста — укоротите ответы или увеличьте max length.');
      }

      if (datasetSize >= 200 && datasetSize < 500) {
        details.push('Средний датасет — можно экспериментировать с смешиванием prompt-ов и moderate augmentation без риска серьёзного переобучения.');
      } else if (datasetSize >= 500 && datasetSize <= 1000) {
        details.push('Достаточный объём для стабильного обучения. Рассмотрите увеличение batch size и снижение learning rate для более плавной сходимости.');
      }
    }

    const quantLabel = recommendedQuant === 'none' ? 'без квантизации' : `${recommendedQuant}`;
    const summary = `Метод: ${recommendedMethod.toUpperCase()}, квантизация: ${quantLabel}, batch size: ${recommended.batchSize}, lr: ${recommended.learningRate}`;

    const diffRatio = (current: number, target: number) => {
      if (target === 0) {
        return Infinity;
      }
      return Math.abs(current - target) / Math.abs(target);
    };

    if (datasetSize > 0) {
      const lrDiff = diffRatio(config.learningRate, recommended.learningRate);
      if (lrDiff >= 0.25) {
        parameterHints.push({
          title: 'Learning rate',
          suggestions: [
            `Текущее значение ${config.learningRate.toExponential()} отличается от рекомендуемого ${recommended.learningRate.toExponential()}. При размере датасета ${datasetSize} лучше держаться ближе к рекомендациям, чтобы избежать NaN или переобучения.`,
            'Если требуется более агрессивное обучение, увеличивайте lr постепенно по 1e-5 и обязательно следите за лоссом после каждой эпохи.',
          ],
        });
      } else {
        parameterHints.push({
          title: 'Learning rate',
          suggestions: [
            `Диапазон lr выглядит разумно для ${datasetSize} примеров. Если появятся NaN, попробуйте снизить его до ${Math.max(1e-6, recommended.learningRate * 0.5).toExponential()}.`,
          ],
        });
      }

      if (config.epochs !== recommended.epochs) {
        parameterHints.push({
          title: 'Epochs',
          suggestions: [
            `Рекомендовано ${recommended.epochs} эпох(и), у вас установлено ${config.epochs}. При небольшом датасете больше эпох быстро ведёт к переобучению, а меньше — может не успеть адаптироваться.`,
          ],
        });
      }

      if (config.warmupSteps !== recommended.warmupSteps) {
        parameterHints.push({
          title: 'Warmup steps',
          suggestions: [
            `Warmup сейчас ${config.warmupSteps}, рекомендовано ${recommended.warmupSteps}. Низкий warmup на маленьком датасете легко вызывает «скачок» лосса и ошибки генерации.`,
          ],
        });
      }

      if (config.batchSize !== recommended.batchSize) {
        parameterHints.push({
          title: 'Batch size',
          suggestions: [
            `Текущее batch size = ${config.batchSize}. Для ${datasetSize} примеров советуем ${recommended.batchSize}: слишком большой batch уменьшает количество шагов и ухудшает обобщение, слишком маленький растягивает время.`,
          ],
        });
      }

      if (config.maxLength < maxObservedTokens) {
        parameterHints.push({
          title: 'Max length',
          suggestions: [
            `В датасете есть примеры до ≈${maxObservedTokens} токенов. Текущее ограничение (${config.maxLength}) их обрежет — увеличьте до ${Math.min(2048, Math.max(config.maxLength + 128, maxObservedTokens + 32))}.`,
          ],
        });
      }
    }

    if (config.method === 'lora') {
      if (config.loraAlpha !== config.loraRank * 2) {
        parameterHints.push({
          title: 'LoRA alpha',
          suggestions: [
            `Обычно alpha ≈ 2 × rank для стабильности. Сейчас rank=${config.loraRank}, alpha=${config.loraAlpha}.`,
          ],
        });
      }
    }

    if (config.method === 'qlora' && config.quantization !== '4bit') {
      parameterHints.push({
        title: 'Quantization',
        suggestions: ['QLoRA требует 4-bit квантизацию. Выберите режим 4bit, чтобы избежать ошибок запуска.'],
      });
    } else if (config.method === 'full' && config.quantization !== 'none') {
      parameterHints.push({
        title: 'Quantization',
        suggestions: ['Полное дообучение не поддерживает квантизацию — оставьте «Без квантизации».'],
      });
    }

    return {
      canApply: true,
      config: recommended,
      summary,
      details,
      warnings,
      parameterHints,
    };
  }, [systemStats, config, datasetSize, settings.deviceType, datasetStats]);

  const isConfigAlignedWithRecommendation = useMemo(() => {
    if (!recommendation.config) {
      return false;
    }
    const keys: (keyof TrainingConfig)[] = [
      'method',
      'quantization',
      'loraRank',
      'loraAlpha',
      'learningRate',
      'batchSize',
      'epochs',
      'maxLength',
      'warmupSteps',
    ];
    return keys.every((key) => recommendation.config?.[key] === config[key]);
  }, [recommendation.config, config]);

  const handleApplyRecommendation = useCallback(() => {
    if (!recommendation.canApply || !recommendation.config) {
      return;
    }
    const nextConfig = recommendation.config;
    setConfig(() => ({ ...nextConfig }));
  }, [recommendation, setConfig]);

  const applyButtonDisabled = !recommendation.canApply || !recommendation.config || isConfigAlignedWithRecommendation;
  const applyButtonLabel = isConfigAlignedWithRecommendation ? 'Рекомендации применены' : 'Принять рекомендации';

  const finalizeJob = useCallback((job: FineTuneJob) => {
    stopPolling();
    setActiveJob(job);
    setIsTraining(false);
    setTrainingProgress(job.progress);
    appendMetrics(job.id, job.metrics ?? {});
    upsertTask(job);

    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      const historyStatus = job.status === 'completed' ? 'completed' : job.status === 'failed' ? 'failed' : 'failed';

      addTrainingSession({
        id: job.id,
        timestamp: Math.round(job.finishedAt ?? Date.now()),
        modelName: job.outputDir,
        datasetSize: job.datasetSize,
        status: historyStatus,
        metrics: job.metrics,
      });
    }

    if (job.status === 'completed') {
      const presetConfig = mapPayloadToConfig(job.config, config, job.outputDir);
      const preset: TrainingPreset = {
        id: job.id,
        label: job.outputDir?.split('/').pop() || job.id,
        createdAt: Date.now(),
        datasetSize: job.datasetSize,
        avgTokens: datasetStats.avgTotalTokens,
        config: presetConfig,
      };

      persistPresets((prev) => {
        const filtered = prev.filter(item => item.label !== preset.label && item.id !== preset.id);
        return [preset, ...filtered];
      });

      updateActivity('training', {
        status: 'success',
        message: job.message ?? 'Обучение завершено',
        progress: 100,
      });
      updateSettings({
        fineTunedModelPath: job.outputDir,
        fineTunedBaseModelPath: job.baseModelPath ?? settings.baseModelPath,
        fineTunedMethod: job.config.method === 'full' ? 'full' : 'lora',
      });
    } else if (job.status === 'cancelled') {
      const message = job.message ?? 'Обучение отменено пользователем';
      setLastError(message);
      updateActivity('training', {
        status: 'error',
        message,
        progress: job.progress,
      });
    } else if (job.status === 'paused') {
      updateActivity('training', {
        status: 'running',
        message: job.message ?? 'Обучение приостановлено',
        progress: job.progress,
      });
    } else {
      const message = job.error ?? job.message ?? 'Обучение завершилось с ошибкой';
      setLastError(message);
      updateActivity('training', {
        status: 'error',
        message,
        progress: job.progress,
      });
    }

    void refreshTasks();

    window.setTimeout(() => {
      clearActivity('training');
    }, job.status === 'completed' ? 3000 : 5000);
  }, [
    stopPolling,
    setActiveJob,
    setIsTraining,
    setTrainingProgress,
    appendMetrics,
    addTrainingSession,
    updateActivity,
    updateSettings,
    settings.baseModelPath,
    setLastError,
    clearActivity,
    persistPresets,
    config,
    datasetStats.avgTotalTokens,
    upsertTask,
    refreshTasks,
  ]);

  const startPolling = useCallback((jobId: string) => {
    if (!settings.baseModelServerUrl) {
      return;
    }
    const poll = async () => {
      try {
        const job = await getFineTuneJob(settings.baseModelServerUrl, jobId);
        setActiveJob(job);
        setTrainingProgress(job.progress);
        setIsTraining(job.status === 'queued' || job.status === 'running');
        appendMetrics(job.id, job.metrics ?? {});
        upsertTask(job);

        if (job.status === 'queued' || job.status === 'running') {
          updateActivity('training', {
            message: job.message ?? 'Обучение запущено',
            progress: job.progress,
          });
        } else {
          finalizeJob(job);
        }
      } catch (error) {
        stopPolling();
        const message = error instanceof FineTuneError
          ? error.message
          : 'Не удалось получить статус дообучения';
        setLastError(message);
        updateActivity('training', {
          status: 'error',
          message,
          progress: trainingProgressRef.current,
        });
        window.setTimeout(() => clearActivity('training'), 5000);
      }
    };

    stopPolling();
    void poll();
    pollingIntervalRef.current = window.setInterval(() => {
      void poll();
    }, 2000);
  }, [
    settings.baseModelServerUrl,
    setActiveJob,
    setTrainingProgress,
    setIsTraining,
    appendMetrics,
    updateActivity,
    finalizeJob,
    stopPolling,
    setLastError,
    clearActivity,
    upsertTask,
  ]);

  useEffect(() => {
    startPollingRef.current = startPolling;
    return () => {
      if (startPollingRef.current === startPolling) {
        startPollingRef.current = null;
      }
    };
  }, [startPolling]);

  const applyLaunchedJob = useCallback((job: FineTuneJob) => {
    upsertTask(job);
    resetMetrics(job.id);
    setActiveJob(job);
    setTrainingProgress(job.progress);
    setIsTraining(job.status === 'queued' || job.status === 'running');
    appendMetrics(job.id, job.metrics ?? {});
    if (job.status === 'queued' || job.status === 'running') {
      updateActivity('training', {
        message: job.message ?? 'Задача поставлена в очередь',
        progress: job.progress,
      });
      startPolling(job.id);
    } else {
      finalizeJob(job);
    }
  }, [
    appendMetrics,
    finalizeJob,
    resetMetrics,
    setActiveJob,
    setIsTraining,
    setTrainingProgress,
    startPolling,
    upsertTask,
    updateActivity,
  ]);

  useEffect(() => {
    if (!settings.baseModelServerUrl || !activeJobId) {
      return;
    }

    const isRunning = (status?: FineTuneJobStatus) => status === 'queued' || status === 'running' || status === 'pausing';

    if (!activeJob) {
      let cancelled = false;
      const restoreJob = async () => {
        try {
          const job = await getFineTuneJob(settings.baseModelServerUrl, activeJobId);
          if (cancelled) {
            return;
          }
          setActiveJob(job);
          setTrainingProgress(job.progress);
          setIsTraining(isRunning(job.status));
          appendMetrics(job.id, job.metrics ?? {});
          upsertTask(job);
          if (isRunning(job.status)) {
            startPolling(job.id);
          } else {
            finalizeJob(job);
          }
        } catch (error) {
          console.error('Failed to restore fine-tune job state', error);
        }
      };

      void restoreJob();
      return () => {
        cancelled = true;
      };
    }

    if (activeJob && isRunning(activeJob.status) && pollingIntervalRef.current === null) {
      startPolling(activeJob.id);
    }
  }, [activeJobId, activeJob, settings.baseModelServerUrl, startPolling, finalizeJob, appendMetrics, setActiveJob, setTrainingProgress, setIsTraining, upsertTask]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const handleAddDatasetItem = () => {
    if (newItem.input.trim() && newItem.output.trim()) {
      const item: DatasetItem = {
        id: generateId('manual'),
        input: newItem.input.trim(),
        output: newItem.output.trim(),
        source: 'manual',
      };
      addDatasetItems([item]);
      setNewItem({ input: '', output: '' });
    }
  };

  const handleRemoveDatasetItem = (id: string) => {
    removeDatasetItem(id);
  };

  const handleClearDataset = useCallback(() => {
    if (dataset.length === 0) {
      return;
    }
    if (window.confirm('Очистить все примеры из датасета?')) {
      clearDataset();
    }
  }, [clearDataset, dataset.length]);

  const handleImportDataset = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.target;
    const files = input.files ? Array.from(input.files) : [];
    input.value = '';

    if (files.length === 0) {
      return;
    }

    const aggregated: DatasetItem[] = [];
    const issues: string[] = [];

    await Promise.all(
      files.map(async (file) => {
        const formatHint = detectFileFormat(file.name, file.type);
        let content: string;
        try {
          content = await file.text();
        } catch (error) {
          issues.push(`${file.name}: не удалось прочитать файл (${error instanceof Error ? error.message : 'неизвестная ошибка'})`);
          return;
        }

        const parseCandidates = formatHint === 'json'
          ? [parseJsonDataset, parseCsvDataset]
          : formatHint === 'csv'
            ? [parseCsvDataset, parseJsonDataset]
            : [parseJsonDataset, parseCsvDataset];

        let records: ParsedDatasetRecord[] = [];
        for (const parser of parseCandidates) {
          try {
            records = parser(content);
            if (records.length > 0) {
              break;
            }
          } catch (error) {
            issues.push(`${file.name}: ${error instanceof Error ? error.message : 'ошибка разбора'}`);
          }
        }

        if (records.length === 0) {
          issues.push(`${file.name}: валидные записи не найдены`);
          return;
        }

        records.forEach((record) => {
          aggregated.push({
            id: generateId('imported'),
            input: record.input,
            output: record.output,
            source: record.source ?? file.name,
          });
        });
      }),
    );

    if (aggregated.length === 0) {
      if (issues.length > 0) {
        alert(`Не удалось импортировать данные:\n${issues.join('\n')}`);
      }
      return;
    }

    addDatasetItems(aggregated);
    if (issues.length > 0) {
      alert(`Импорт завершён с предупреждениями:\n${issues.join('\n')}`);
    }
  };

  const handleMethodChange = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    const nextMethod = event.target.value as TrainingMethod;
    setConfig(prev => {
      const nextConfig = { ...prev, method: nextMethod };
      if (nextMethod === 'qlora') {
        nextConfig.quantization = '4bit';
      } else if (nextMethod === 'full') {
        nextConfig.quantization = 'none';
      }
      return nextConfig;
    });
  }, [setConfig]);

  const handleQuantizationChange = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    const nextQuant = event.target.value as TrainingQuantization;
    setConfig(prev => {
      if (prev.method === 'full' && nextQuant !== 'none') {
        alert('Полное обучение не поддерживает квантизацию.');
        return prev;
      }
      if (prev.method === 'qlora' && nextQuant !== '4bit') {
        alert('QLoRA поддерживает только режим 4-bit.');
        return prev;
      }
      return {
        ...prev,
        quantization: nextQuant,
      };
    });
  }, [setConfig]);

  const handleStartTraining = async () => {
    if (!settings.baseModelPath) {
      alert('Укажите путь к базовой модели на вкладке «Настройки».');
      return;
    }

    if (!settings.baseModelServerUrl) {
      alert('Укажите URL сервера базовой модели на вкладке «Настройки».');
      return;
    }

    if (dataset.length === 0) {
      alert('Добавьте хотя бы один пример для датасета.');
      return;
    }

    if (config.method === 'qlora' && config.quantization !== '4bit') {
      alert('Для QLoRA необходимо выбрать квантизацию 4-bit.');
      return;
    }

    if (config.method === 'full' && config.quantization !== 'none') {
      alert('Полное обучение не поддерживает квантизацию.');
      return;
    }

    const datasetPayload = dataset.map(({ input, output, source }) => ({ input, output, source }));
    const configPayload: FineTuneConfigPayload = {
      method: config.method,
      quantization: config.quantization,
      lora_rank: config.loraRank,
      lora_alpha: config.loraAlpha,
      learning_rate: config.learningRate,
      batch_size: config.batchSize,
      epochs: config.epochs,
      max_length: config.maxLength,
      warmup_steps: config.warmupSteps,
    };

    const targetModules = config.targetModules
      .split(',')
      .map(token => token.trim())
      .filter(Boolean);
    if (targetModules.length > 0) {
      configPayload.target_modules = targetModules;
    }

    if (config.initialAdapterPath.trim()) {
      configPayload.initial_adapter_path = config.initialAdapterPath.trim();
    }

    stopPolling();
    setLastError(null);
    setIsTraining(true);
    setTrainingProgress(0);
    setActivity('training', { message: 'Подготовка задачи дообучения', progress: 0 });

    try {
      const job = await createFineTuneJob(settings.baseModelServerUrl, {
        base_model_path: settings.baseModelPath,
        output_dir: config.outputDir.trim() || undefined,
        dataset: datasetPayload,
        config: configPayload,
      });

      applyLaunchedJob(job);
    } catch (error) {
      const message = error instanceof FineTuneError
        ? error.message
        : 'Не удалось запустить дообучение';
      setLastError(message);
      setIsTraining(false);
      updateActivity('training', { status: 'error', message, progress: 0 });
      window.setTimeout(() => {
      clearActivity('training');
    }, 5000);
  }
  };

  const handleStopTraining = async () => {
    if (!activeJob) {
      return;
    }

    try {
      const job = await cancelFineTuneJob(settings.baseModelServerUrl, activeJob.id);
      setActiveJob(job);
      setLastError(null);
      updateActivity('training', {
        message: job.message ?? 'Останавливаю обучение',
        progress: job.progress,
      });
      upsertTask(job);
      void refreshTasks();
    } catch (error) {
      const message = error instanceof FineTuneError
        ? error.message
        : 'Не удалось отменить задачу';
      setLastError(message);
      updateActivity('training', { status: 'error', message, progress: trainingProgress });
      window.setTimeout(() => {
        clearActivity('training');
      }, 5000);
    }
  };

  const handlePauseTraining = useCallback(async () => {
    if (!activeJob || !settings.baseModelServerUrl) {
      return;
    }
    setIsPausePending(true);
    try {
      const job = await pauseFineTuneJob(settings.baseModelServerUrl, activeJob.id);
      upsertTask(job);
      setActiveJob(job);
      setIsTraining(false);
      updateActivity('training', {
        message: job.message ?? 'Приостанавливаю обучение',
        progress: job.progress,
      });
      void refreshTasks();
    } catch (error) {
      const message = error instanceof FineTuneError
        ? error.message
        : 'Не удалось поставить задачу на паузу';
      setLastError(message);
      alert(message);
    } finally {
      setIsPausePending(false);
    }
  }, [activeJob, settings.baseModelServerUrl, upsertTask, setActiveJob, setIsTraining, updateActivity, setLastError, refreshTasks]);

  const handleResumeTraining = useCallback(
    async (jobId?: string) => {
      const targetJobId = jobId ?? activeJob?.id;
      if (!targetJobId || !settings.baseModelServerUrl) {
        return;
      }
      const toggleLocalState = !jobId || (activeJob && activeJob.id === jobId);
      if (toggleLocalState) {
        setIsResumePending(true);
      }
      try {
        const job = await resumeFineTuneJob(settings.baseModelServerUrl, targetJobId);
        upsertTask(job);
        setActiveJob(job);
        setIsTraining(true);
        updateActivity('training', {
          message: job.message ?? 'Возобновление обучения',
          progress: job.progress,
        });
        startPolling(job.id);
        void refreshTasks();
      } catch (error) {
        const message = error instanceof FineTuneError
          ? error.message
          : 'Не удалось возобновить обучение';
        setLastError(message);
        alert(message);
      } finally {
        if (toggleLocalState) {
          setIsResumePending(false);
        }
      }
    },
    [activeJob, settings.baseModelServerUrl, upsertTask, setActiveJob, setIsTraining, updateActivity, startPolling, setLastError, refreshTasks],
  );

  const handleFocusTask = useCallback(
    async (task: FineTuneJob) => {
      if (!settings.baseModelServerUrl) {
        setActiveJob(task);
        setTrainingProgress(task.progress);
        setIsTraining(task.status === 'queued' || task.status === 'running');
        return;
      }

      try {
        const detailed = await getFineTuneJob(settings.baseModelServerUrl, task.id);
        setActiveJob(detailed);
        setTrainingProgress(detailed.progress);
        setIsTraining(detailed.status === 'queued' || detailed.status === 'running');
        appendMetrics(detailed.id, detailed.metrics ?? {});
        if (detailed.status === 'queued' || detailed.status === 'running') {
          startPolling(detailed.id);
        } else {
          stopPolling();
        }
      } catch (error) {
        console.error('Не удалось загрузить задачу дообучения', error);
      }
    },
    [
      settings.baseModelServerUrl,
      setActiveJob,
      setTrainingProgress,
      setIsTraining,
      appendMetrics,
      startPolling,
      stopPolling,
    ],
  );

  useEffect(() => () => {
    stopPolling();
  }, [stopPolling]);

  return (
    <div className="h-full flex">
      {/* Main Content */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Обучение моделей</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">Импортируйте примеры, настройте параметры и запустите адаптацию моделей.</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => setShowMetrics(!showMetrics)}
                className="flex items-center gap-2 rounded-lg bg-purple-600 px-4 py-2 text-white shadow-sm hover:bg-purple-700 transition-colors"
                title={showMetrics ? 'Скрыть панель метрик обучения' : 'Открыть панель метрик обучения'}
              >
                <BarChart3 className="h-4 w-4" />
                Метрики
              </button>
              <button
                onClick={() => setShowDatasetModal(true)}
                className="flex items-center gap-2 rounded-lg bg-emerald-500 px-4 py-2 text-white shadow-sm hover:bg-emerald-600 transition-colors"
                title="Добавить один пример вручную"
              >
                <Plus className="h-4 w-4" />
                Добавить свой пример
              </button>
              <button
                onClick={() => importInputRef.current?.click()}
                className="flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-slate-700 shadow-sm hover:border-blue-400 hover:text-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500/40 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                title="Импортировать набор примеров из CSV или JSON"
              >
                <Upload className="h-4 w-4" />
                Импорт CSV/JSON
              </button>
              <button
                onClick={handleClearDataset}
                disabled={dataset.length === 0}
                className="flex items-center gap-2 rounded-lg border border-rose-200 bg-white px-4 py-2 text-rose-600 shadow-sm transition-colors hover:border-rose-400 hover:text-rose-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-rose-400/40 dark:bg-slate-900 dark:text-rose-300"
                title="Удалить все примеры"
              >
                <Trash2 className="h-4 w-4" />
                Очистить датасет
              </button>
            </div>
          </div>

          <input
            ref={importInputRef}
            type="file"
            accept=".csv,.json"
            multiple
            onChange={handleImportDataset}
            className="hidden"
          />

          <div className="space-y-4">
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60">
              <div className="flex flex-col gap-4 md:flex-row md:items-stretch md:gap-6">
                {processSteps.map((step, index) => (
                  <div key={step.id} className="flex flex-1 items-center">
                    <RoadmapStep step={step} />
                    {index < processSteps.length - 1 && (
                      <div className="hidden h-1 flex-1 rounded-full bg-slate-200 dark:bg-slate-800 md:block" aria-hidden="true" />
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              {validationMessages.map((item, index) => (
                <InlineNotice key={`${item.text}-${index}`} tone={item.tone}>{item.text}</InlineNotice>
              ))}

              {presets.length > 0 && (
                <div className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Последние успешные конфигурации</h3>
                    <span className="text-xs text-slate-500 dark:text-slate-400">До 6 пресетов</span>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {presets.map(preset => (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => applyPreset(preset)}
                        className="rounded-xl border border-slate-200 bg-white p-3 text-left transition-transform hover:-translate-y-0.5 hover:border-blue-400 hover:shadow-lg hover:shadow-blue-900/10 dark:border-slate-700 dark:bg-slate-900/70"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-sm font-semibold text-slate-700 dark:text-slate-100">{preset.label}</span>
                          <span className="text-[11px] text-slate-500 dark:text-slate-400">{new Date(preset.createdAt).toLocaleDateString()}</span>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                          <span className="rounded bg-slate-100 px-2 py-1 dark:bg-slate-800/60">Метод: {preset.config.method}</span>
                          <span className="rounded bg-slate-100 px-2 py-1 dark:bg-slate-800/60">Эпохи: {preset.config.epochs}</span>
                          <span className="rounded bg-slate-100 px-2 py-1 dark:bg-slate-800/60">Batch: {preset.config.batchSize}</span>
                          <span className="rounded bg-slate-100 px-2 py-1 dark:bg-slate-800/60">Датасет: {preset.datasetSize}</span>
                        </div>
                        <p className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">Средняя длина: ≈{preset.avgTokens} токенов</p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>




          {/* Auto Recommendations */}
          <div className="rounded-2xl border border-blue-200 bg-blue-50 p-6 shadow-sm shadow-blue-900/10 dark:border-blue-900 dark:bg-blue-950/30">
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-semibold text-blue-700 dark:text-blue-200">Автоматический подбор параметров</h3>
                </div>
                <p className="mt-2 text-sm text-blue-700 dark:text-blue-200">{recommendation.summary}</p>
                {recommendation.details.length > 0 && (
                  <ul className="mt-3 list-disc list-inside space-y-1 text-sm text-blue-700 dark:text-blue-200">
                    {recommendation.details.map((item, index) => (
                      <li key={`rec-detail-${index}`}>{item}</li>
                    ))}
                  </ul>
                )}
                {recommendation.warnings.length > 0 && (
                  <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-400/40 dark:bg-amber-500/10 dark:text-amber-100">
                    <div className="font-medium">Что проверить:</div>
                    <ul className="mt-1 list-disc list-inside space-y-1">
                      {recommendation.warnings.map((item, index) => (
                        <li key={`rec-warning-${index}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {recommendation.parameterHints.length > 0 && (
                  <div className="mt-4 space-y-3">
                    {recommendation.parameterHints.map((hint, index) => (
                      <div
                        key={`rec-parameter-${index}`}
                        className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-200"
                      >
                        <div className="font-semibold text-slate-800 dark:text-slate-100">{hint.title}</div>
                        <ul className="mt-1 list-disc list-inside space-y-1">
                          {hint.suggestions.map((tip, idx) => (
                            <li key={`rec-parameter-${index}-${idx}`}>{tip}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div className="w-full md:w-72 flex flex-col gap-3">
                <button
                  type="button"
                  onClick={handleApplyRecommendation}
                  disabled={applyButtonDisabled}
                  className={`rounded-lg px-4 py-2 transition-colors ${
                    applyButtonDisabled
                      ? 'cursor-not-allowed bg-blue-200 text-blue-500 dark:bg-blue-900/40 dark:text-blue-300'
                      : 'bg-blue-500 text-white hover:bg-blue-600'
                  }`}
                >
                  {applyButtonLabel}
                </button>
                <div className="space-y-1 rounded-lg border border-blue-200 bg-white px-3 py-2 text-xs text-blue-700 shadow-sm dark:border-blue-900 dark:bg-slate-950/60 dark:text-blue-200">
                  <div className="font-medium">Обнаруженное окружение</div>
                  <div>Устройство: {systemStats?.gpuBackend ? systemStats.gpuBackend.toUpperCase() : '—'}</div>
                  <div>GPU: {systemStats?.gpuName ?? '—'}</div>
                  <div>VRAM: {systemStats?.gpuMemoryTotalGb ? `${systemStats.gpuMemoryTotalGb.toFixed(1)} GB` : '—'}</div>
                  <div>RAM: {systemStats?.ramTotalGb ? `${systemStats.ramTotalGb.toFixed(1)} GB` : '—'}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Configuration */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60">
            <h3 className="mb-4 text-lg font-semibold text-slate-800 dark:text-slate-100">Конфигурация обучения</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <LabelWithHint
                  label="Метод"
                  hint="Определяет стратегию обучения: LoRA добавляет адаптеры, QLoRA сочетает их с 4-битной квантизацией, а Полное обучение обновляет все веса."
                />
                <select
                  value={config.method}
                  onChange={handleMethodChange}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                >
                  <option value="lora">LoRA</option>
                  <option value="qlora">QLoRA</option>
                  <option value="full">Полное обучение</option>
                </select>
              </div>

              <div>
                <LabelWithHint
                  label="Квантизация"
                  hint="Определяет разрядность весов при загрузке модели. Низкая разрядность экономит видеопамять, но может немного снижать качество."
                />
                <select
                  value={config.quantization}
                  onChange={handleQuantizationChange}
                  disabled={config.method === 'full' || config.method === 'qlora'}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                >
                  <option value="none" disabled={config.method === 'qlora'}>Без квантизации</option>
                  <option value="8bit" disabled={config.method !== 'lora'}>8-bit</option>
                  <option value="4bit" disabled={config.method === 'full'}>4-bit</option>
                </select>
                {config.method === 'qlora' ? (
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">QLoRA использует фиксированную квантизацию 4-bit.</p>
                ) : config.method === 'full' ? (
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Полное обучение выполняется без квантизации.</p>
                ) : (
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Режимы 4-bit и 8-bit требуют GPU с поддержкой CUDA.</p>
                )}
              </div>

              <div>
                <LabelWithHint
                  label="LoRA Rank"
                  hint="Размер узкого слоя адаптера. Бóльший ранг повышает качество, но требует больше памяти и может переобучаться."
                />
                <input
                  type="number"
                  value={config.loraRank}
                  onChange={(e) => setConfig(prev => ({ ...prev, loraRank: parseInt(e.target.value) }))}
                  disabled={config.method === 'full'}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
              </div>

              <div>
                <LabelWithHint
                  label="LoRA Alpha"
                  hint="Коэффициент масштабирования LoRA. Большие значения усиливают вклад адаптера, но могут сделать обучение менее стабильным."
                />
                <input
                  type="number"
                  value={config.loraAlpha}
                  onChange={(e) => setConfig(prev => ({ ...prev, loraAlpha: parseInt(e.target.value) }))}
                  disabled={config.method === 'full'}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
              </div>

              <div className="md:col-span-2">
                <LabelWithHint
                  label="LoRA target modules"
                  hint="Слои модели, в которые вставляются LoRA-адаптеры. Оставьте поле пустым, чтобы использовать типичный набор для выбранной архитектуры."
                />
                <input
                  type="text"
                  value={config.targetModules}
                  onChange={(e) => setConfig(prev => ({ ...prev, targetModules: e.target.value }))}
                  placeholder="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  Оставьте поле пустым, чтобы система подобрала типовые модули автоматически (например, для Gemma и LLaMA).
                </p>
              </div>

              <div>
                <LabelWithHint
                  label="Learning Rate"
                  hint="Скорость обучения. Чем выше, тем быстрее модель адаптируется, но возрастает риск нестабильности и переобучения."
                />
                <input
                  type="number"
                  step="0.0001"
                  value={config.learningRate}
                  onChange={(e) => setConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
              </div>

              <div>
                <LabelWithHint
                  label="Batch Size"
                  hint="Количество примеров, обрабатываемых за шаг. Большие значения улучшают стабильность градиента, но требуют больше памяти."
                />
                <input
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => setConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
              </div>

              <div>
                <LabelWithHint
                  label="Эпохи"
                  hint="Сколько раз модель проходит весь датасет. Больше эпох повышают качество, но увеличивают время и риск переобучения."
                />
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
              </div>

              <div>
                <LabelWithHint
                  label="Warmup steps"
                  hint="Количество шагов постепенного разгона learning rate. Помогает стабилизировать первые итерации и избежать резких скачков."
                />
                <input
                  type="number"
                  min={0}
                  value={config.warmupSteps}
                  onChange={(e) => setConfig(prev => ({ ...prev, warmupSteps: parseInt(e.target.value) || 0 }))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
              </div>

              <div>
                <LabelWithHint
                  label="Max Length"
                  hint="Максимальное количество токенов в обучающем примере. Большое значение даёт длинный контекст, но заметно увеличивает расход памяти."
                />
                <input
                  type="number"
                  value={config.maxLength}
                  onChange={(e) => setConfig(prev => ({ ...prev, maxLength: parseInt(e.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
              </div>

              <div className="md:col-span-2">
                <LabelWithHint
                  label="Каталог вывода"
                  hint="Папка, куда будет сохранён результат обучения. Оставьте пустым, чтобы путь сформировался автоматически."
                />
                <input
                  type="text"
                  value={config.outputDir}
                  onChange={(e) => setConfig(prev => ({ ...prev, outputDir: e.target.value }))}
                  placeholder="Models/finetune-my-model"
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Если оставить поле пустым, директория будет создана автоматически внутри каталога Models.</p>
              </div>

              <div className="md:col-span-2">
                <LabelWithHint
                  label="Стартовый адаптер"
                  hint="Путь к каталогу с LoRA/QLoRA адаптером, от которого нужно продолжить обучение."
                />
                <input
                  type="text"
                  value={config.initialAdapterPath}
                  onChange={(e) => setConfig(prev => ({ ...prev, initialAdapterPath: e.target.value }))}
                  placeholder="Models/finetune-2025-01-12"
                  disabled={config.method === 'full'}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                />
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Для повторного дообучения LoRA адаптера укажите его папку (содержит adapter_config.json и dataset.jsonl). Полное обучение игнорирует это поле.</p>
              </div>
            </div>
          </div>

          {/* Training Controls */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Управление обучением</h3>
              <div className="text-sm text-slate-500 dark:text-slate-400">
                {dataset.length > 0 ? `${dataset.length} примеров в датасете` : 'Датасет пуст'}
                {activeJob && (
                  <span className={`ml-3 font-medium ${activeJob.status === 'failed' ? 'text-rose-600 dark:text-rose-400' : activeJob.status === 'paused' ? 'text-blue-600 dark:text-blue-300' : 'text-slate-500 dark:text-slate-400'}`}>
                    Статус: {jobStatusLabels[activeJob.status]}
                  </span>
                )}
              </div>
            </div>

            <div className="mb-4 flex flex-wrap gap-3">
              <button
                onClick={handleStartTraining}
                disabled={isTraining || dataset.length === 0}
                className="flex items-center gap-2 rounded-lg bg-blue-500 px-4 py-2 text-white shadow-sm transition-colors hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
                title="Запустить процесс дообучения"
              >
                <Play className="h-4 w-4" />
                Начать обучение
              </button>
              <button
                onClick={handlePauseTraining}
                disabled={!activeJob || activeJob.status !== 'running' || isPausePending}
                className="flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-slate-700 shadow-sm transition-colors hover:border-blue-400 hover:text-blue-600 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                title="Поставить обучение на паузу"
              >
                <Pause className="h-4 w-4" />
                Пауза
              </button>
              <button
                onClick={() => handleResumeTraining()}
                disabled={!activeJob || activeJob.status !== 'paused' || isResumePending}
                className="flex items-center gap-2 rounded-lg border border-emerald-300 bg-white px-4 py-2 text-emerald-600 shadow-sm transition-colors hover:border-emerald-400 hover:text-emerald-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-emerald-500/40 dark:bg-slate-900 dark:text-emerald-300"
                title="Возобновить приостановленную задачу"
              >
                <Play className="h-4 w-4" />
                Продолжить
              </button>
              <button
                onClick={handleStopTraining}
                disabled={!isTraining}
                className="flex items-center gap-2 rounded-lg bg-rose-500 px-4 py-2 text-white shadow-sm transition-colors hover:bg-rose-600 disabled:cursor-not-allowed disabled:opacity-50"
                title="Остановить текущее обучение"
              >
                <Square className="h-4 w-4" />
                Остановить
              </button>
            </div>

            {/* Progress Bar */}
            {(isTraining || trainingProgress > 0) && (
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-1">
                  <span>Прогресс обучения</span>
                  <span>{Math.round(trainingProgress)}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${trainingProgress}%` }}
                  ></div>
                </div>
              </div>
            )}

        {activeJob && (
          <div className="mb-4 text-sm text-slate-600 dark:text-slate-200">
            <div className="font-medium text-slate-700 dark:text-slate-100">{activeJob.message}</div>
            <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Выходной каталог: <span className="font-mono break-all">{activeJob.outputDir}</span>
            </div>
            {activeJob.config?.target_modules?.length ? (
              <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                LoRA модули: <span className="font-mono">{activeJob.config.target_modules.join(', ')}</span>
              </div>
            ) : null}
          </div>
        )}

            {lastError && (
              <div className="mb-4 text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2">
                {lastError}
              </div>
            )}
          </div>

      {activeJob?.events && activeJob.events.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60">
          <h3 className="mb-3 text-lg font-semibold text-slate-800 dark:text-slate-100">Журнал обучения</h3>
          <div className="max-h-60 space-y-2 overflow-y-auto">
            {activeJob.events
              .slice(-20)
              .reverse()
              .map((event, index) => (
                <div key={`${event.timestamp}-${index}`} className="flex justify-between gap-4 text-xs">
                  <span className="w-28 shrink-0 text-slate-500 dark:text-slate-400">
                    {new Date(event.timestamp * 1000).toLocaleTimeString()}
                  </span>
                  <span className={event.level === 'error' ? 'text-rose-600 dark:text-rose-400' : 'text-slate-600 dark:text-slate-200'}>
                    {event.message}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Training Tasks */}
      {trainingTasks.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Задачи обучения</h3>
            <button
              type="button"
              onClick={() => void refreshTasks()}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700 transition-colors hover:border-blue-400 hover:text-blue-600 dark:border-slate-700 dark:text-slate-200"
            >
              Обновить
            </button>
          </div>
          <div className="space-y-3">
            {trainingTasks.map((task) => (
              <div
                key={task.id}
                className="rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900/70"
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-700 dark:text-slate-100">
                      {task.outputDir.split('/').pop() || task.id}
                    </div>
                    <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      Статус: {jobStatusLabels[task.status]}
                    </div>
                    <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      Создано: {new Date(task.createdAt).toLocaleString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {Math.round(task.progress)}%
                    </div>
                    <div className="h-2 w-32 rounded-full bg-slate-200 dark:bg-slate-800">
                      <div
                        className="h-2 rounded-full bg-blue-500 transition-all"
                        style={{ width: `${Math.min(100, Math.max(3, Math.round(task.progress)))}%` }}
                      />
                    </div>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500 dark:text-slate-400">
                  <span>Метод: {task.config.method}</span>
                  <span>Batch: {task.config.batch_size}</span>
                  <span>Эпох: {task.config.epochs}</span>
                  <span>Токенов: {task.config.max_length}</span>
                  {task.resumeCheckpoint && <span>Чекпоинт: {task.resumeCheckpoint.split('/').pop()}</span>}
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => handleFocusTask(task)}
                    className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700 transition-colors hover:border-blue-400 hover:text-blue-600 dark:border-slate-600 dark:text-slate-200"
                  >
                    Показать
                  </button>
                  {task.status === 'paused' && (
                    <button
                      type="button"
                      onClick={() => handleResumeTraining(task.id)}
                      className="rounded-lg border border-emerald-300 px-3 py-1.5 text-sm text-emerald-600 transition-colors hover:border-emerald-400 hover:text-emerald-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-emerald-400/40 dark:text-emerald-300"
                      disabled={isResumePending && activeJob?.id === task.id}
                    >
                      Продолжить
                    </button>
                  )}
                  {task.status === 'running' && activeJob?.id === task.id && (
                    <button
                      type="button"
                      onClick={handlePauseTraining}
                      className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700 transition-colors hover:border-blue-400 hover:text-blue-600 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-200"
                      disabled={isPausePending}
                    >
                      Пауза
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
          {/* Dataset Preview */}
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Датасет</h3>
              <span className="text-sm text-slate-500 dark:text-slate-400">
                {dataset.length > 0 ? `${dataset.length} ${dataset.length === 1 ? 'пример' : 'примеров'}` : 'Нет примеров'}
              </span>
            </div>

            <div
              ref={datasetListRef}
              onScroll={handleDatasetScroll}
              className="max-h-96 space-y-3 overflow-y-auto pr-1"
            >
              {visibleDatasetItems.map((item) => (
                <div key={item.id} className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-900/70">
                  <div className="flex-1">
                    <div className="mb-1 text-sm font-semibold text-slate-700 dark:text-slate-200">Вход:</div>
                    <div className="mb-2 text-sm text-slate-700 dark:text-slate-100 whitespace-pre-wrap">{item.input}</div>
                    <div className="mb-1 text-sm font-semibold text-slate-700 dark:text-slate-200">Выход:</div>
                    <div className="text-sm text-slate-700 dark:text-slate-100 whitespace-pre-wrap">{item.output}</div>
                  </div>
                  <button
                    onClick={() => handleRemoveDatasetItem(item.id)}
                    className="self-start rounded-lg p-2 text-rose-600 transition-colors hover:bg-rose-50 dark:text-rose-300 dark:hover:bg-rose-500/20"
                    title="Удалить пример из датасета"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))}

              {hasMoreDatasetItems && (
                <div className="flex justify-center py-2">
                  <button
                    type="button"
                    onClick={loadMoreDatasetItems}
                    className="rounded-lg border border-slate-300 px-4 py-1.5 text-sm text-slate-700 transition-colors hover:border-blue-400 hover:text-blue-600 dark:border-slate-600 dark:text-slate-200"
                  >
                    Показать ещё
                  </button>
                </div>
              )}

              {dataset.length === 0 && (
                <div className="py-8 text-center text-slate-500 dark:text-slate-400">
                  Датасет пуст. Добавьте примеры для обучения.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Modal */}
      {showDatasetModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="w-full max-w-2xl rounded-2xl border border-slate-200 bg-white p-6 shadow-xl shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900">
            <h3 className="mb-4 text-lg font-semibold text-slate-800 dark:text-slate-100">Добавить пример в датасет</h3>

            <div className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">Входной текст</label>
                <textarea
                  value={newItem.input}
                  onChange={(e) => setNewItem(prev => ({ ...prev, input: e.target.value }))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 h-24 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                  placeholder="Введите входной текст..."
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">Ожидаемый выход</label>
                <textarea
                  value={newItem.output}
                  onChange={(e) => setNewItem(prev => ({ ...prev, output: e.target.value }))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 h-24 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
                  placeholder="Введите ожидаемый ответ..."
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowDatasetModal(false)}
                className="rounded-lg border border-slate-300 px-4 py-2 text-slate-700 transition-colors hover:border-slate-400 hover:text-slate-900 dark:border-slate-600 dark:text-slate-300 dark:hover:border-slate-500"
                title="Закрыть окно без сохранения"
              >
                Отмена
              </button>
              <button
                onClick={() => {
                  handleAddDatasetItem();
                  setShowDatasetModal(false);
                }}
                disabled={!newItem.input.trim() || !newItem.output.trim()}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title="Добавить пример в датасет"
              >
                Добавить
              </button>
            </div>
          </div>
        </div>
      )}

      {showMetrics && (
        <TrainingMetricsPanel
          history={activeMetricsHistory}
          job={activeJob}
          onClose={() => setShowMetrics(false)}
        />
      )}
    </div>
  );
};

export default TrainingTab;
