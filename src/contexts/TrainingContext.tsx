import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import type { FineTuneJob } from '../services/fineTune';
import type { DatasetItem, TrainingConfig, TrainingMetricPoint } from '../types/training';

type MetricsHistory = Record<string, TrainingMetricPoint[]>;

type TrainingContextValue = {
  dataset: DatasetItem[];
  addDatasetItems: (items: DatasetItem[]) => void;
  removeDatasetItem: (id: string) => void;
  clearDataset: () => void;
  config: TrainingConfig;
  setConfig: Dispatch<SetStateAction<TrainingConfig>>;
  updateConfig: (update: Partial<TrainingConfig>) => void;
  isTraining: boolean;
  setIsTraining: Dispatch<SetStateAction<boolean>>;
  trainingProgress: number;
  setTrainingProgress: Dispatch<SetStateAction<number>>;
  activeJob: FineTuneJob | null;
  setActiveJob: (job: FineTuneJob | null) => void;
  activeJobId: string | null;
  lastError: string | null;
  setLastError: Dispatch<SetStateAction<string | null>>;
  metricsHistory: MetricsHistory;
  appendMetrics: (jobId: string, metrics: Record<string, number>) => void;
  resetMetrics: (jobId?: string) => void;
  trainingTasks: FineTuneJob[];
  setTrainingTasks: Dispatch<SetStateAction<FineTuneJob[]>>;
};

const DATASET_STORAGE_KEY = 'llm-studio-training-dataset';
const CONFIG_STORAGE_KEY = 'llm-studio-training-config';
const ACTIVE_JOB_KEY = 'llm-studio-training-active-job';

const DEFAULT_CONFIG: TrainingConfig = {
  method: 'lora',
  quantization: 'none',
  loraRank: 16,
  loraAlpha: 32,
  learningRate: 0.0002,
  batchSize: 4,
  epochs: 3,
  maxLength: 512,
  warmupSteps: 0,
  outputDir: '',
  targetModules: '',
  initialAdapterPath: '',
};

const loadDataset = (): DatasetItem[] => {
  if (typeof window === 'undefined') {
    return [];
  }
  try {
    const raw = window.localStorage.getItem(DATASET_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as DatasetItem[];
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((item) => Boolean(item && item.input && item.output));
  } catch (error) {
    console.warn('Failed to load training dataset from storage', error);
    return [];
  }
};

const loadConfig = (): TrainingConfig => {
  if (typeof window === 'undefined') {
    return DEFAULT_CONFIG;
  }
  try {
    const raw = window.localStorage.getItem(CONFIG_STORAGE_KEY);
    if (!raw) {
      return DEFAULT_CONFIG;
    }
    const parsed = JSON.parse(raw) as Partial<TrainingConfig>;
    return {
      ...DEFAULT_CONFIG,
      ...parsed,
    };
  } catch (error) {
    console.warn('Failed to load training config from storage', error);
    return DEFAULT_CONFIG;
  }
};

const loadActiveJobId = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(ACTIVE_JOB_KEY);
    return raw?.trim() ? raw : null;
  } catch (error) {
    console.warn('Failed to load active job id from storage', error);
    return null;
  }
};

const TrainingContext = createContext<TrainingContextValue | undefined>(undefined);

export function TrainingProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<DatasetItem[]>(() => loadDataset());
  const [config, setConfig] = useState<TrainingConfig>(() => loadConfig());
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [activeJob, internalSetActiveJob] = useState<FineTuneJob | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(() => loadActiveJobId());
  const [lastError, setLastError] = useState<string | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<MetricsHistory>({});
  const [trainingTasks, setTrainingTasks] = useState<FineTuneJob[]>([]);
  const quotaWarningShownRef = useRef(false);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.setItem(DATASET_STORAGE_KEY, JSON.stringify(dataset));
    } catch (error) {
      const isQuotaError = error instanceof DOMException
        && (
          error.name === 'QuotaExceededError'
          || error.code === 22
          || error.code === 1014
        );
      console.warn('Failed to persist training dataset', error);
      if (isQuotaError && !quotaWarningShownRef.current) {
        quotaWarningShownRef.current = true;
        window.setTimeout(() => {
          alert('Не удалось сохранить датасет в локальном хранилище: лимит браузера исчерпан. Экспортируйте данные, очистите хранилище или уменьшите размер набора.');
        }, 0);
      }
    }
  }, [dataset]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config));
    } catch (error) {
      console.warn('Failed to persist training config', error);
    }
  }, [config]);

  const addDatasetItems = useCallback((items: DatasetItem[]) => {
    if (!items.length) {
      return;
    }
    setDataset((prev) => {
      const next = [...prev, ...items];
      const seen = new Set<string>();
      return next.filter((item) => {
        if (!item.input || !item.output) {
          return false;
        }
        if (seen.has(item.id)) {
          return false;
        }
        seen.add(item.id);
        return true;
      });
    });
  }, []);

  const removeDatasetItem = useCallback((id: string) => {
    setDataset((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearDataset = useCallback(() => {
    setDataset([]);
    setIsTraining(false);
    setTrainingProgress(0);
    internalSetActiveJob(null);
    setActiveJobId(null);
    setLastError(null);
    setMetricsHistory({});
    setTrainingTasks([]);

    if (typeof window !== 'undefined') {
      try {
        window.localStorage.removeItem(DATASET_STORAGE_KEY);
        window.localStorage.removeItem(ACTIVE_JOB_KEY);
      } catch (error) {
        console.warn('Failed to clear persisted training state', error);
      }
    }
  }, [
    setDataset,
    setIsTraining,
    setTrainingProgress,
    internalSetActiveJob,
    setActiveJobId,
    setLastError,
    setMetricsHistory,
    setTrainingTasks,
  ]);

  const updateConfig = useCallback((update: Partial<TrainingConfig>) => {
    setConfig((prev) => ({
      ...prev,
      ...update,
    }));
  }, []);

  const setActiveJob = useCallback((job: FineTuneJob | null) => {
    internalSetActiveJob(job);
    const nextId = job?.id ?? null;
    setActiveJobId(nextId);
    if (typeof window !== 'undefined') {
      try {
        if (nextId) {
          window.localStorage.setItem(ACTIVE_JOB_KEY, nextId);
        } else {
          window.localStorage.removeItem(ACTIVE_JOB_KEY);
        }
      } catch (error) {
        console.warn('Failed to persist active job id', error);
      }
    }
  }, []);

  const appendMetrics = useCallback((jobId: string, metrics: Record<string, number>) => {
    if (!jobId || !metrics) {
      return;
    }

    const sanitizedEntries = Object.entries(metrics).filter((entry): entry is [string, number] => {
      const [, value] = entry;
      return typeof value === 'number' && Number.isFinite(value);
    });

    if (sanitizedEntries.length === 0) {
      return;
    }

    setMetricsHistory((prev) => {
      const history = prev[jobId] ?? [];
      const lastPoint = history[history.length - 1];
      const nextMetrics = Object.fromEntries(sanitizedEntries);

      if (lastPoint) {
        const isSame = Object.keys(nextMetrics).every((key) => lastPoint.metrics[key] === nextMetrics[key]);
        if (isSame) {
          return prev;
        }
      }

      const nextPoint: TrainingMetricPoint = {
        timestamp: Date.now(),
        metrics: nextMetrics,
      };

      const nextHistory = [...history, nextPoint];
      const trimmedHistory = nextHistory.length > 300 ? nextHistory.slice(-300) : nextHistory;

      return {
        ...prev,
        [jobId]: trimmedHistory,
      };
    });
  }, []);

  const resetMetrics = useCallback((jobId?: string) => {
    if (!jobId) {
      setMetricsHistory({});
      return;
    }
    setMetricsHistory((prev) => {
      if (!(jobId in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[jobId];
      return next;
    });
  }, []);

  const value = useMemo<TrainingContextValue>(() => ({
    dataset,
    addDatasetItems,
    removeDatasetItem,
    clearDataset,
    config,
    setConfig,
    updateConfig,
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
  }), [
    dataset,
    addDatasetItems,
    removeDatasetItem,
    clearDataset,
    config,
    isTraining,
    setIsTraining,
    trainingProgress,
    activeJob,
    setActiveJob,
    activeJobId,
    lastError,
    setLastError,
    metricsHistory,
    appendMetrics,
    resetMetrics,
    updateConfig,
    trainingTasks,
    setTrainingTasks,
  ]);

  return <TrainingContext.Provider value={value}>{children}</TrainingContext.Provider>;
}

export function useTraining(): TrainingContextValue {
  const context = useContext(TrainingContext);
  if (!context) {
    throw new Error('useTraining must be used within a TrainingProvider');
  }
  return context;
}
