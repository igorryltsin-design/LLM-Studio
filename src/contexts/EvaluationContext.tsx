import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import type {
  DomainMetricSummary,
  EvaluationDataset,
  EvaluationDatasetItem,
  EvaluationMetrics,
  EvaluationReviewStatus,
  EvaluationRun,
  EvaluationRunConfig,
  EvaluationRunItem,
  EvaluationRunStatus,
  EvaluationReviewTask,
  EvaluationDifficulty,
} from '../types/evaluation';
import { useHistory } from './HistoryContext';

interface EvaluationPreferences {
  autoPollIntervalMs: number;
  autoAssignBaseline: boolean;
  defaultReviewer?: string | null;
  qualityGate?: number | null;
  compactMode: boolean;
}

interface StartRunOptions {
  datasetId: string;
  name?: string;
  config: EvaluationRunConfig;
  sampleSize?: number | null;
  domainFilter?: string[] | null;
  difficultyFilter?: EvaluationDifficulty[] | null;
  tagFilter?: string[] | null;
  manualSelection?: string[] | null;
  baselineRunId?: string | null;
}

interface UpdateRunItemOptions extends Partial<EvaluationRunItem> {
  autoScore?: number | null;
  remoteScore?: number | null;
  humanScore?: number | null;
}

interface SubmitReviewPayload {
  status: EvaluationReviewStatus;
  reviewer?: string | null;
  notes?: string | null;
  scoreOverride?: number | null;
}

interface EvaluationContextValue {
  datasets: EvaluationDataset[];
  activeDatasetId: string | null;
  activeDataset: EvaluationDataset | null;
  runs: EvaluationRun[];
  activeRunId: string | null;
  activeRun: EvaluationRun | null;
  preferences: EvaluationPreferences;
  reviewTasks: EvaluationReviewTask[];
  setActiveDatasetId: (datasetId: string | null) => void;
  setActiveRunId: (runId: string | null) => void;
  updatePreferences: (update: Partial<EvaluationPreferences>) => void;
  createDataset: (name: string, description?: string) => EvaluationDataset;
  updateDataset: (datasetId: string, update: Partial<Omit<EvaluationDataset, 'id' | 'items'>>) => void;
  removeDataset: (datasetId: string) => void;
  addDatasetItems: (datasetId: string, items: EvaluationDatasetItem[]) => void;
  updateDatasetItem: (
    datasetId: string,
    itemId: string,
    update: Partial<Omit<EvaluationDatasetItem, 'id'>>,
  ) => void;
  removeDatasetItem: (datasetId: string, itemId: string) => void;
  resetDatasets: () => void;
  importDataset: (dataset: EvaluationDataset) => void;
  exportDataset: (datasetId: string) => EvaluationDataset | null;
  startRun: (options: StartRunOptions) => EvaluationRun | null;
  updateRunStatus: (runId: string, status: EvaluationRunStatus, error?: string | null) => void;
  updateRunItem: (runId: string, itemId: string, update: UpdateRunItemOptions) => void;
  attachBaseline: (runId: string, baselineRunId: string | null) => void;
  submitReview: (runId: string, itemId: string, payload: SubmitReviewPayload) => void;
  discardRun: (runId: string) => void;
  clearRuns: () => void;
  appendRunNote: (runId: string, note: string) => void;
  hydrateRuns: (runs: EvaluationRun[]) => void;
}

interface PersistentState {
  datasets: EvaluationDataset[];
  runs: EvaluationRun[];
  activeDatasetId: string | null;
  activeRunId: string | null;
}

const STORAGE_KEY = 'llm-studio-evaluation-state-v1';
const PREFERENCES_KEY = 'llm-studio-evaluation-preferences-v1';

const defaultPreferences: EvaluationPreferences = {
  autoPollIntervalMs: 15000,
  autoAssignBaseline: true,
  defaultReviewer: null,
  qualityGate: 0.7,
  compactMode: false,
};

const createId = (prefix: string) => {
  const uniquePart = typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(16).slice(2, 10)}`;
  return `${prefix}-${uniquePart}`;
};

const createDefaultDataset = (): EvaluationDataset => {
  const now = Date.now();
  const items: EvaluationDatasetItem[] = [
    {
      id: createId('eval-item'),
      domain: 'Математика',
      question: 'Решите уравнение: 2x + 5 = 17',
      referenceAnswer: 'x = 6',
      difficulty: 'easy',
      tags: ['linear'],
    },
    {
      id: createId('eval-item'),
      domain: 'Физика',
      question: 'Сформулируйте второй закон Ньютона и приведите пример.',
      referenceAnswer: 'F = ma. Например, сила 10 Н разгоняет тело массой 2 кг с ускорением 5 м/с².',
      difficulty: 'medium',
      tags: ['theory'],
    },
    {
      id: createId('eval-item'),
      domain: 'История',
      question: 'Назовите три ключевых последствий распада СССР.',
      referenceAnswer: '1) Образование независимых государств; 2) Экономические реформы и кризис; 3) Изменение геополитического баланса.',
      difficulty: 'hard',
      tags: ['politics'],
    },
    {
      id: createId('eval-item'),
      domain: 'Биология',
      question: 'Что такое митоз и каковы его фазы?',
      referenceAnswer: 'Митоз — деление соматической клетки, включает профазу, метафазу, анафазу и телофазу.',
      difficulty: 'medium',
      tags: ['cell'],
    },
    {
      id: createId('eval-item'),
      domain: 'Программирование',
      question: 'Объясните разницу между стеком и кучей в управлении памятью.',
      referenceAnswer: 'Стек — память для статически распределённых кадров функций, куча — динамическое распределение объектов во время выполнения.',
      difficulty: 'medium',
      tags: ['memory'],
    },
    {
      id: createId('eval-item'),
      domain: 'Экономика',
      question: 'Что такое инфляция и какие факторы на неё влияют?',
      referenceAnswer: 'Инфляция — рост общего уровня цен. Влияние оказывают денежная масса, выпуск, ожидания, валютный курс и издержки.',
      difficulty: 'medium',
      tags: ['macro'],
    },
  ];

  return {
    id: createId('eval-dataset'),
    name: 'Базовый эталон',
    description: 'Стартовый набор вопросов по ключевым предметным областям для регрессионного тестирования модели.',
    createdAt: now,
    updatedAt: now,
    items,
  } satisfies EvaluationDataset;
};

const computeAggregateScore = (item: EvaluationRunItem): number | null => {
  const sources: Array<number | null | undefined> = [
    item.humanScore,
    item.remoteScore,
    item.autoScore,
  ];
  const valid = sources.find((value) => typeof value === 'number' && Number.isFinite(value));
  return typeof valid === 'number' ? valid : null;
};

const roundMetric = (value: number): number => Math.round(value * 1000) / 1000;

const computePercentile = (values: number[], percentile: number): number | null => {
  if (!values.length) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const clamped = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * percentile)));
  return sorted[clamped];
};

const computeMetricsForRun = (run: EvaluationRun): EvaluationMetrics => {
  const { items } = run;
  const total = items.length;
  if (total === 0) {
    return {
      total,
      answered: 0,
      scored: 0,
      needsReview: 0,
      coverage: 0,
      overallScore: null,
      autoScoreAverage: null,
      humanApprovalRate: null,
      averageLatencySeconds: null,
      latencyP95Seconds: null,
      averageTokensPerResponse: null,
      tokensPerSecond: null,
      domainSummaries: [],
    } satisfies EvaluationMetrics;
  }

  let answered = 0;
  let scored = 0;
  let needsReview = 0;
  let autoScoreSum = 0;
  let autoScoreCount = 0;
  let aggregateSum = 0;
  let aggregateCount = 0;
  let humanApproved = 0;
  let humanTotal = 0;
  const responseDurations: number[] = [];
  const tokensPerResponse: number[] = [];
  let throughputTokens = 0;
  let throughputSeconds = 0;

  type DomainAccumulator = {
    summary: DomainMetricSummary;
    aggregateSum: number;
    aggregateCount: number;
    autoScoreSum: number;
    autoScoreCount: number;
    humanApproved: number;
    humanTotal: number;
  };

  const domainMap = new Map<string, DomainAccumulator>();

  items.forEach((item) => {
    const domainKey = item.domain || 'Без категории';
    const aggregate = computeAggregateScore(item);

    if (item.status !== 'pending') {
      answered += 1;
    }
    if (item.status === 'needs_review') {
      needsReview += 1;
    }
    if (item.status === 'scored' || item.status === 'reviewed') {
      scored += 1;
    }

    if (typeof item.autoScore === 'number' && Number.isFinite(item.autoScore)) {
      autoScoreSum += item.autoScore;
      autoScoreCount += 1;
    }

    if (typeof aggregate === 'number' && Number.isFinite(aggregate)) {
      aggregateSum += aggregate;
      aggregateCount += 1;
    }

    const startAt = typeof item.answerStartedAt === 'number' ? item.answerStartedAt : null;
    const completedAt = typeof item.answerCompletedAt === 'number' ? item.answerCompletedAt : null;
    const tokensUsed = typeof item.tokens === 'number' && Number.isFinite(item.tokens) && item.tokens >= 0
      ? item.tokens
      : null;

    if (startAt !== null && completedAt !== null && completedAt >= startAt) {
      const seconds = (completedAt - startAt) / 1000;
      responseDurations.push(seconds);
      if (tokensUsed !== null) {
        throughputTokens += tokensUsed;
        throughputSeconds += seconds;
      }
    }

    if (tokensUsed !== null) {
      tokensPerResponse.push(tokensUsed);
    }

    const accumulator = domainMap.get(domainKey) ?? {
      summary: {
        domain: domainKey,
        total: 0,
        answered: 0,
        scored: 0,
        needsReview: 0,
        averageScore: null,
        autoScoreAverage: null,
        humanApprovalRate: null,
      } satisfies DomainMetricSummary,
      aggregateSum: 0,
      aggregateCount: 0,
      autoScoreSum: 0,
      autoScoreCount: 0,
      humanApproved: 0,
      humanTotal: 0,
    } satisfies DomainAccumulator;

    accumulator.summary.total += 1;
    if (item.status !== 'pending') {
      accumulator.summary.answered += 1;
    }
    if (item.status === 'needs_review') {
      accumulator.summary.needsReview += 1;
    }
    if (item.status === 'scored' || item.status === 'reviewed') {
      accumulator.summary.scored += 1;
    }

    if (typeof aggregate === 'number' && Number.isFinite(aggregate)) {
      accumulator.aggregateSum += aggregate;
      accumulator.aggregateCount += 1;
    }

    if (typeof item.autoScore === 'number' && Number.isFinite(item.autoScore)) {
      accumulator.autoScoreSum += item.autoScore;
      accumulator.autoScoreCount += 1;
    }

    if (item.reviewStatus === 'approved') {
      humanApproved += 1;
      humanTotal += 1;
      accumulator.humanApproved += 1;
      accumulator.humanTotal += 1;
    } else if (item.reviewStatus === 'rejected') {
      humanTotal += 1;
      accumulator.humanTotal += 1;
    }

    domainMap.set(domainKey, accumulator);
  });

  const domainSummaries: DomainMetricSummary[] = Array.from(domainMap.values()).map((entry) => {
    const {
      summary,
      aggregateSum: domainAggregateSum,
      aggregateCount: domainAggregateCount,
      autoScoreSum: domainAutoSum,
      autoScoreCount: domainAutoCount,
      humanApproved: domainHumanApproved,
      humanTotal: domainHumanTotal,
    } = entry;

    return {
      ...summary,
      averageScore:
        domainAggregateCount > 0
          ? roundMetric(domainAggregateSum / domainAggregateCount)
          : null,
      autoScoreAverage:
        domainAutoCount > 0
          ? roundMetric(domainAutoSum / domainAutoCount)
          : null,
      humanApprovalRate:
        domainHumanTotal > 0
          ? roundMetric(domainHumanApproved / domainHumanTotal)
          : null,
    } satisfies DomainMetricSummary;
  });

  const averageLatencySeconds = responseDurations.length > 0
    ? roundMetric(responseDurations.reduce((acc, value) => acc + value, 0) / responseDurations.length)
    : null;
  const latencyP95SecondsRaw = computePercentile(responseDurations, 0.95);
  const latencyP95Seconds = latencyP95SecondsRaw === null ? null : roundMetric(latencyP95SecondsRaw);
  const totalTokens = tokensPerResponse.reduce((acc, value) => acc + value, 0);
  const averageTokensPerResponse = tokensPerResponse.length > 0
    ? roundMetric(totalTokens / tokensPerResponse.length)
    : null;
  const tokensPerSecond = throughputSeconds > 0 && throughputTokens > 0
    ? roundMetric(throughputTokens / throughputSeconds)
    : null;

  const coverage = roundMetric(answered / total);
  const overallScore = aggregateCount > 0 ? roundMetric(aggregateSum / aggregateCount) : null;
  const autoScoreAverage = autoScoreCount > 0 ? roundMetric(autoScoreSum / autoScoreCount) : null;
  const qualityGateThreshold = typeof run.config.qualityGateThreshold === 'number'
    ? run.config.qualityGateThreshold
    : null;
  const qualityGatePassed = qualityGateThreshold !== null && overallScore !== null
    ? overallScore >= qualityGateThreshold
    : null;

  return {
    total,
    answered,
    scored,
    needsReview,
    coverage,
    overallScore,
    autoScoreAverage,
    humanApprovalRate: humanTotal > 0 ? roundMetric(humanApproved / humanTotal) : null,
    averageLatencySeconds,
    latencyP95Seconds,
    averageTokensPerResponse,
    tokensPerSecond,
    qualityGateThreshold,
    qualityGatePassed,
    domainSummaries,
  } satisfies EvaluationMetrics;
};

const recomputeMetrics = (runs: EvaluationRun[]): EvaluationRun[] => {
  if (runs.length === 0) {
    return runs;
  }
  const withBaseMetrics = runs.map((run) => ({
    ...run,
    metrics: computeMetricsForRun(run),
  }));

  return withBaseMetrics.map((run) => {
    if (!run.baselineRunId) {
      return run;
    }
    const baseline = withBaseMetrics.find((candidate) => candidate.id === run.baselineRunId);
    if (!baseline) {
      return run;
    }
    const baselineScore = baseline.metrics.overallScore ?? null;
    const currentScore = run.metrics.overallScore ?? null;
    const scoreDelta =
      baselineScore !== null && currentScore !== null
        ? roundMetric(currentScore - baselineScore)
        : null;

    return {
      ...run,
      metrics: {
        ...run.metrics,
        baselineScore,
        scoreDelta,
      },
    } satisfies EvaluationRun;
  });
};

const sanitizeDataset = (dataset: EvaluationDataset): EvaluationDataset => {
  return {
    ...dataset,
    items: dataset.items.map((item) => ({
      ...item,
      tags: Array.isArray(item.tags) ? item.tags.slice(0, 12) : [],
    })),
  } satisfies EvaluationDataset;
};

const loadPersistentState = (): PersistentState => {
  if (typeof window === 'undefined') {
    const dataset = createDefaultDataset();
    return {
      datasets: [dataset],
      runs: [],
      activeDatasetId: dataset.id,
      activeRunId: null,
    } satisfies PersistentState;
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      const dataset = createDefaultDataset();
      return {
        datasets: [dataset],
        runs: [],
        activeDatasetId: dataset.id,
        activeRunId: null,
      } satisfies PersistentState;
    }
    const parsed = JSON.parse(raw) as PersistentState;
    if (!parsed || !Array.isArray(parsed.datasets)) {
      throw new Error('Invalid evaluation state structure');
    }
    const datasets = parsed.datasets.map(sanitizeDataset);
    return {
      datasets: datasets.length > 0 ? datasets : [createDefaultDataset()],
      runs: Array.isArray(parsed.runs) ? recomputeMetrics(parsed.runs) : [],
      activeDatasetId: parsed.activeDatasetId ?? (datasets[0]?.id ?? null),
      activeRunId: parsed.activeRunId ?? null,
    } satisfies PersistentState;
  } catch (error) {
    console.warn('Не удалось загрузить состояние тестирования моделей', error);
    const dataset = createDefaultDataset();
    return {
      datasets: [dataset],
      runs: [],
      activeDatasetId: dataset.id,
      activeRunId: null,
    } satisfies PersistentState;
  }
};

const loadPreferences = (): EvaluationPreferences => {
  if (typeof window === 'undefined') {
    return defaultPreferences;
  }
  try {
    const raw = window.localStorage.getItem(PREFERENCES_KEY);
    if (!raw) {
      return defaultPreferences;
    }
    const parsed = JSON.parse(raw) as Partial<EvaluationPreferences>;
    return {
      ...defaultPreferences,
      ...parsed,
    } satisfies EvaluationPreferences;
  } catch (error) {
    console.warn('Не удалось загрузить настройки тестирования', error);
    return defaultPreferences;
  }
};

const EvaluationContext = createContext<EvaluationContextValue | undefined>(undefined);

export function EvaluationProvider({ children }: { children: ReactNode }) {
  const initialState = useMemo(loadPersistentState, []);
  const [datasets, setDatasets] = useState<EvaluationDataset[]>(initialState.datasets);
  const [runs, setRuns] = useState<EvaluationRun[]>(initialState.runs);
  const [activeDatasetId, setActiveDatasetIdState] = useState<string | null>(initialState.activeDatasetId);
  const [activeRunId, setActiveRunIdState] = useState<string | null>(initialState.activeRunId);
  const [preferences, setPreferences] = useState<EvaluationPreferences>(loadPreferences);
  const { addEvaluationSession } = useHistory();
  const previousRunStatuses = useRef<Record<string, EvaluationRunStatus>>({});

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const payload: PersistentState = {
      datasets,
      runs,
      activeDatasetId,
      activeRunId,
    };
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn('Не удалось сохранить состояние тестирования моделей', error);
    }
  }, [datasets, runs, activeDatasetId, activeRunId]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences));
    } catch (error) {
      console.warn('Не удалось сохранить настройки тестирования', error);
    }
  }, [preferences]);

  useEffect(() => {
    const statuses: Record<string, EvaluationRunStatus> = previousRunStatuses.current;
    runs.forEach((run) => {
      const previous = statuses[run.id];
      statuses[run.id] = run.status;

      const statusChanged = previous !== run.status;
      const shouldRecord = run.status === 'completed'
        || run.status === 'waiting_review'
        || run.status === 'failed';

      if (!statusChanged || !shouldRecord) {
        return;
      }

      const timestamp = run.status === 'completed'
        ? run.completedAt ?? run.updatedAt ?? Date.now()
        : run.updatedAt ?? Date.now();

      addEvaluationSession({
        id: run.id,
        timestamp,
        datasetName: run.config.datasetName,
        modelVariant: run.config.modelVariant,
        total: run.metrics.total,
        scored: run.metrics.scored,
        overallScore: run.metrics.overallScore,
        coverage: run.metrics.coverage,
        status: run.status,
        baselineScore: run.metrics.baselineScore ?? null,
        scoreDelta: run.metrics.scoreDelta ?? null,
        error: run.error ?? null,
      });
    });
  }, [runs, addEvaluationSession]);

  const activeDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === activeDatasetId) ?? null,
    [datasets, activeDatasetId],
  );

  const activeRun = useMemo(
    () => runs.find((run) => run.id === activeRunId) ?? null,
    [runs, activeRunId],
  );

  const reviewTasks = useMemo<EvaluationReviewTask[]>(() => {
    return runs.flatMap((run) => {
      return run.items
        .filter((item) => item.status === 'needs_review')
        .map((item) => ({
          runId: run.id,
          itemId: item.id,
          domain: item.domain,
          question: item.question,
          modelAnswer: item.modelAnswer ?? null,
          referenceAnswer: item.referenceAnswer,
          submittedAt: item.answerCompletedAt ?? run.updatedAt,
          reviewer: item.reviewer ?? null,
          status: 'pending',
        }) satisfies EvaluationReviewTask);
    });
  }, [runs]);

  const setActiveDatasetId = useCallback((datasetId: string | null) => {
    setActiveDatasetIdState(datasetId);
  }, []);

  const setActiveRunId = useCallback((runId: string | null) => {
    setActiveRunIdState(runId);
  }, []);

  const updatePreferences = useCallback((update: Partial<EvaluationPreferences>) => {
    setPreferences((prev) => ({
      ...prev,
      ...update,
    }));
  }, []);

  const createDataset = useCallback((name: string, description?: string) => {
    const dataset: EvaluationDataset = {
      id: createId('eval-dataset'),
      name,
      description,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      items: [],
    };
    setDatasets((prev) => [...prev, dataset]);
    setActiveDatasetIdState(dataset.id);
    return dataset;
  }, []);

  const updateDataset = useCallback((datasetId: string, update: Partial<Omit<EvaluationDataset, 'id' | 'items'>>) => {
    setDatasets((prev) => prev.map((dataset) => {
      if (dataset.id !== datasetId) {
        return dataset;
      }
      return {
        ...dataset,
        ...update,
        updatedAt: Date.now(),
      } satisfies EvaluationDataset;
    }));
  }, []);

  const removeDataset = useCallback((datasetId: string) => {
    setDatasets((prev) => prev.filter((dataset) => dataset.id !== datasetId));
    setRuns((prevRuns) => {
      const remaining = prevRuns.filter((run) => run.config.datasetId !== datasetId);
      return recomputeMetrics(remaining);
    });
    setActiveDatasetIdState((prevId) => (prevId === datasetId ? null : prevId));
  }, []);

  const addDatasetItems = useCallback((datasetId: string, items: EvaluationDatasetItem[]) => {
    if (!items.length) {
      return;
    }
    setDatasets((prev) => prev.map((dataset) => {
      if (dataset.id !== datasetId) {
        return dataset;
      }
      const existingIds = new Set(dataset.items.map((item) => item.id));
      const sanitized = items.filter((item) => {
        if (existingIds.has(item.id)) {
          return false;
        }
        return Boolean(item.question && item.referenceAnswer && item.domain);
      });
      if (!sanitized.length) {
        return dataset;
      }
      return {
        ...dataset,
        items: [...dataset.items, ...sanitized],
        updatedAt: Date.now(),
      } satisfies EvaluationDataset;
    }));
  }, []);

  const updateDatasetItem = useCallback((datasetId: string, itemId: string, update: Partial<Omit<EvaluationDatasetItem, 'id'>>) => {
    setDatasets((prev) => prev.map((dataset) => {
      if (dataset.id !== datasetId) {
        return dataset;
      }
      return {
        ...dataset,
        items: dataset.items.map((item) => {
          if (item.id !== itemId) {
            return item;
          }
          return {
            ...item,
            ...update,
          } satisfies EvaluationDatasetItem;
        }),
        updatedAt: Date.now(),
      } satisfies EvaluationDataset;
    }));
  }, []);

  const removeDatasetItem = useCallback((datasetId: string, itemId: string) => {
    setDatasets((prev) => prev.map((dataset) => {
      if (dataset.id !== datasetId) {
        return dataset;
      }
      return {
        ...dataset,
        items: dataset.items.filter((item) => item.id !== itemId),
        updatedAt: Date.now(),
      } satisfies EvaluationDataset;
    }));
  }, []);

  const resetDatasets = useCallback(() => {
    const dataset = createDefaultDataset();
    setDatasets([dataset]);
    setRuns([]);
    setActiveDatasetIdState(dataset.id);
    setActiveRunIdState(null);
  }, []);

  const importDataset = useCallback((dataset: EvaluationDataset) => {
    const sanitized = sanitizeDataset({
      ...dataset,
      id: dataset.id || createId('eval-dataset'),
      createdAt: dataset.createdAt ?? Date.now(),
      updatedAt: Date.now(),
    });
    setDatasets((prev) => [...prev, sanitized]);
    setActiveDatasetIdState(sanitized.id);
  }, []);

  const exportDataset = useCallback((datasetId: string) => {
    return datasets.find((dataset) => dataset.id === datasetId) ?? null;
  }, [datasets]);

  const hydrateRuns = useCallback((externalRuns: EvaluationRun[]) => {
    setRuns(recomputeMetrics(externalRuns));
  }, []);

  const startRun = useCallback((options: StartRunOptions): EvaluationRun | null => {
    const {
      datasetId,
      name,
      config,
      sampleSize,
      domainFilter,
      difficultyFilter,
      tagFilter,
      manualSelection,
      baselineRunId,
    } = options;

    const dataset = datasets.find((entry) => entry.id === datasetId);
    if (!dataset) {
      console.warn('Dataset not found for evaluation run');
      return null;
    }

    let candidates = dataset.items.slice();
    if (manualSelection && manualSelection.length > 0) {
      const selection = new Set(manualSelection);
      candidates = candidates.filter((item) => selection.has(item.id));
    }
    if (domainFilter && domainFilter.length > 0) {
      const domains = new Set(domainFilter);
      candidates = candidates.filter((item) => domains.has(item.domain));
    }
    if (difficultyFilter && difficultyFilter.length > 0) {
      const difficulties = new Set(difficultyFilter);
      candidates = candidates.filter((item) => difficulties.has(item.difficulty));
    }
    if (tagFilter && tagFilter.length > 0) {
      const tags = new Set(tagFilter);
      candidates = candidates.filter((item) => item.tags.some((tag) => tags.has(tag)));
    }

    if (config.shuffle) {
      candidates = [...candidates].sort(() => Math.random() - 0.5);
    }

    const limited = typeof sampleSize === 'number' && sampleSize > 0
      ? candidates.slice(0, Math.min(sampleSize, candidates.length))
      : candidates;

    if (!limited.length) {
      console.warn('Нет доступных примеров для запуска тестирования');
      return null;
    }

    const now = Date.now();
    const runItems: EvaluationRunItem[] = limited.map((item) => ({
      id: createId('eval-run-item'),
      datasetItemId: item.id,
      domain: item.domain,
      subdomain: item.subdomain,
      question: item.question,
      referenceAnswer: item.referenceAnswer,
      scoringMode: config.scoringMode,
      difficulty: item.difficulty,
      tags: item.tags?.slice() ?? [],
      status: 'pending',
    }));

    const baseRun: EvaluationRun = {
      id: createId('eval-run'),
      name: name || `${dataset.name} — ${new Date(now).toLocaleString()}`,
      createdAt: now,
      updatedAt: now,
      status: 'queued',
      config: {
        ...config,
        datasetId: dataset.id,
        datasetName: dataset.name,
        sampleSize: limited.length,
      },
      items: runItems,
      metrics: {
        total: runItems.length,
        answered: 0,
        scored: 0,
        needsReview: 0,
        coverage: 0,
        overallScore: null,
        autoScoreAverage: null,
        humanApprovalRate: null,
        averageLatencySeconds: null,
        latencyP95Seconds: null,
        averageTokensPerResponse: null,
        tokensPerSecond: null,
        qualityGateThreshold: config.qualityGateThreshold ?? null,
        qualityGatePassed: null,
        domainSummaries: [],
      },
      baselineRunId: baselineRunId ?? null,
    };

    const runWithMetrics: EvaluationRun = {
      ...baseRun,
      metrics: computeMetricsForRun(baseRun),
    };

    setRuns((prev) => recomputeMetrics([...prev, runWithMetrics]));
    setActiveRunIdState(runWithMetrics.id);
    return runWithMetrics;
  }, [datasets]);

  const updateRunStatus = useCallback((runId: string, status: EvaluationRunStatus, error?: string | null) => {
    setRuns((prev) => recomputeMetrics(prev.map((run) => {
      if (run.id !== runId) {
        return run;
      }
      const now = Date.now();
      return {
        ...run,
        status,
        error: error ?? undefined,
        startedAt: run.startedAt ?? (status === 'running' ? now : run.startedAt),
        completedAt: status === 'completed' ? now : run.completedAt,
        updatedAt: now,
      } satisfies EvaluationRun;
    })));
  }, []);

  const updateRunItem = useCallback((runId: string, itemId: string, update: UpdateRunItemOptions) => {
    setRuns((prev) => recomputeMetrics(prev.map((run) => {
      if (run.id !== runId) {
        return run;
      }
      const items = run.items.map((item) => {
        if (item.id !== itemId) {
          return item;
        }
        const next: EvaluationRunItem = {
          ...item,
          ...update,
          autoScore: update.autoScore ?? item.autoScore,
          remoteScore: update.remoteScore ?? item.remoteScore,
          humanScore: update.humanScore ?? item.humanScore,
          status: update.status ?? item.status,
          answerCompletedAt: update.answerCompletedAt ?? item.answerCompletedAt,
          answerStartedAt: update.answerStartedAt ?? item.answerStartedAt,
        } satisfies EvaluationRunItem;
        return next;
      });
      return {
        ...run,
        items,
        updatedAt: Date.now(),
      } satisfies EvaluationRun;
    })));
  }, []);

  const attachBaseline = useCallback((runId: string, baselineRunId: string | null) => {
    setRuns((prev) => recomputeMetrics(prev.map((run) => {
      if (run.id !== runId) {
        return run;
      }
      return {
        ...run,
        baselineRunId,
        updatedAt: Date.now(),
      } satisfies EvaluationRun;
    })));
  }, []);

  const submitReview = useCallback((runId: string, itemId: string, payload: SubmitReviewPayload) => {
    setRuns((prev) => recomputeMetrics(prev.map((run) => {
      if (run.id !== runId) {
        return run;
      }
      const items = run.items.map((item) => {
        if (item.id !== itemId) {
          return item;
      }
      const aggregateScore = typeof payload.scoreOverride === 'number'
        ? payload.scoreOverride
        : computeAggregateScore(item);
        const reviewedItem: EvaluationRunItem = {
          ...item,
          humanScore: typeof payload.scoreOverride === 'number' ? payload.scoreOverride : item.humanScore,
          reviewStatus: payload.status,
          reviewer: payload.reviewer !== undefined ? payload.reviewer : item.reviewer,
          reviewNotes: payload.notes ?? item.reviewNotes,
          status: 'reviewed',
          autoScore: item.autoScore,
          remoteScore: item.remoteScore,
        } satisfies EvaluationRunItem;
        if (aggregateScore !== null) {
          reviewedItem.autoScore = reviewedItem.autoScore ?? aggregateScore;
        }
        return reviewedItem;
      });
      return {
        ...run,
        items,
        status: items.every((entry) => entry.status === 'reviewed' || entry.status === 'scored')
          ? 'completed'
          : run.status,
        updatedAt: Date.now(),
      } satisfies EvaluationRun;
    })));
  }, []);

  const discardRun = useCallback((runId: string) => {
    setRuns((prev) => recomputeMetrics(prev.filter((run) => run.id !== runId)));
    setActiveRunIdState((prev) => (prev === runId ? null : prev));
  }, []);

  const clearRuns = useCallback(() => {
    setRuns([]);
    setActiveRunIdState(null);
  }, []);

  const appendRunNote = useCallback((runId: string, note: string) => {
    setRuns((prev) => prev.map((run) => {
      if (run.id !== runId) {
        return run;
      }
      return {
        ...run,
        notes: [run.notes, note].filter(Boolean).join('\n\n'),
        updatedAt: Date.now(),
      } satisfies EvaluationRun;
    }));
  }, []);

  const value = useMemo<EvaluationContextValue>(() => ({
    datasets,
    activeDatasetId,
    activeDataset,
    runs,
    activeRunId,
    activeRun,
    preferences,
    reviewTasks,
    setActiveDatasetId,
    setActiveRunId,
    updatePreferences,
    createDataset,
    updateDataset,
    removeDataset,
    addDatasetItems,
    updateDatasetItem,
    removeDatasetItem,
    resetDatasets,
    importDataset,
    exportDataset,
    startRun,
    updateRunStatus,
    updateRunItem,
    attachBaseline,
    submitReview,
    discardRun,
    clearRuns,
    appendRunNote,
    hydrateRuns,
  }), [
    datasets,
    activeDatasetId,
    activeDataset,
    runs,
    activeRunId,
    activeRun,
    preferences,
    reviewTasks,
    setActiveDatasetId,
    setActiveRunId,
    updatePreferences,
    createDataset,
    updateDataset,
    removeDataset,
    addDatasetItems,
    updateDatasetItem,
    removeDatasetItem,
    resetDatasets,
    importDataset,
    exportDataset,
    startRun,
    updateRunStatus,
    updateRunItem,
    attachBaseline,
    submitReview,
    discardRun,
    clearRuns,
    appendRunNote,
    hydrateRuns,
  ]);

  return <EvaluationContext.Provider value={value}>{children}</EvaluationContext.Provider>;
}

export function useEvaluation() {
  const context = useContext(EvaluationContext);
  if (context === undefined) {
    throw new Error('useEvaluation must be used within an EvaluationProvider');
  }
  return context;
}
