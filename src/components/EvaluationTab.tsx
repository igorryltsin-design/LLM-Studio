import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Play,
  Square,
  RefreshCw,
  Download,
  Target,
  Layers,
  ShieldCheck,
  TrendingUp,
  ClipboardList,
  Filter,
  Plus,
  Check,
  X,
  AlertTriangle,
  Activity,
  Globe,
  Minimize2,
} from 'lucide-react';
import { useEvaluation } from '../contexts/EvaluationContext';
import { useSettings } from '../contexts/SettingsContext';
import { useStatus } from '../contexts/StatusContext';
import { useTraining } from '../contexts/TrainingContext';
import { loadAutoTrainingConfig, persistAutoTrainingConfig, DEFAULT_AUTO_TRAINING_CONFIG, AutoTrainingConfig } from '../utils/autoTrainingConfig';
import { runProblemPipeline, ProblemPipelinePayload } from '../services/aggregator';
import type {
  DomainMetricSummary,
  EvaluationDataset,
  EvaluationDatasetItem,
  EvaluationDifficulty,
  EvaluationRun,
  EvaluationRunItem,
  EvaluationScoringMode,
} from '../types/evaluation';
import type {
  DashboardDomainInsight,
  DashboardTagSummary,
  DashboardDifficultySummary,
  DashboardHotspot,
  DashboardRecommendation,
} from '../types/evaluationDashboard';
import { scoreAnswerLocally } from '../services/evaluation';
import { callBaseChat } from '../services/baseChat';
import { callRemoteChat, type RemoteChatMessage } from '../services/remoteChat';
import type { DatasetItem } from '../types/training';

const createPipelineDatasetItemId = () => {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return `pipeline-${crypto.randomUUID()}`;
    }
  } catch (error) {
    console.warn('Не удалось сгенерировать UUID через crypto.randomUUID', error);
  }
  const randomPart = Math.random().toString(16).slice(2, 10);
  return `pipeline-${Date.now().toString(36)}-${randomPart}`;
};

interface PlannerState {
  datasetId: string | null;
  modelVariant: 'base' | 'fine_tuned' | 'remote';
  scoringMode: EvaluationScoringMode;
  sampleSize: number | null;
  domains: string[];
  difficulties: EvaluationDifficulty[];
  tags: string[];
  requireHumanReview: boolean;
  baselineRunId: string | null;
}

interface DraftItemForm {
  question: string;
  referenceAnswer: string;
  domain: string;
  difficulty: EvaluationDifficulty;
  tags: string;
}

const INITIAL_FORM: DraftItemForm = {
  question: '',
  referenceAnswer: '',
  domain: '',
  difficulty: 'medium',
  tags: '',
};

const DEFAULT_PLANNER: PlannerState = {
  datasetId: null,
  modelVariant: 'base',
  scoringMode: 'auto',
  sampleSize: null,
  domains: [],
  difficulties: [],
  tags: [],
  requireHumanReview: false,
  baselineRunId: null,
};

const QUALITY_CLASSES: Record<'good' | 'warn' | 'bad', string> = {
  good: 'text-emerald-400',
  warn: 'text-amber-400',
  bad: 'text-rose-400',
};

const DOMAIN_COLORS = [
  '#34d399',
  '#60a5fa',
  '#facc15',
  '#f472b6',
  '#a78bfa',
  '#38bdf8',
  '#fb7185',
  '#2dd4bf',
  '#e879f9',
  '#f97316',
];

const QualityDashboardLazy = lazy(() => import('./evaluation/QualityDashboard'));

const DIFFICULTY_LABELS: Record<EvaluationDifficulty | 'unknown', string> = {
  easy: 'Лёгкая',
  medium: 'Средняя',
  hard: 'Сложная',
  unknown: 'Не указана',
};

const formatPercent = (value: number | null | undefined, fractionDigits = 1) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(fractionDigits)}%`;
};

const formatScore = (value: number | null | undefined, fractionDigits = 3) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(fractionDigits);
};

const formatSeconds = (value: number | null | undefined) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  if (value >= 60) {
    const minutes = Math.floor(value / 60);
    const seconds = value - minutes * 60;
    const secondsText = seconds >= 10 ? seconds.toFixed(0) : seconds.toFixed(1);
    return `${minutes}м ${secondsText}с`;
  }
  const fractionDigits = value >= 10 ? 1 : 2;
  return `${value.toFixed(fractionDigits)} с`;
};

const formatTokensValue = (value: number | null | undefined) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  if (value >= 100) {
    return Math.round(value).toString();
  }
  return value.toFixed(1);
};

const renderTokensLine = (value: number | null | undefined, suffix: string) => {
  const formatted = formatTokensValue(value);
  return formatted === '—' ? formatted : `${formatted} ${suffix}`;
};

const resolveAggregateScore = (item: EvaluationRunItem): number | null => {
  if (typeof item.humanScore === 'number') {
    return item.humanScore;
  }
  if (typeof item.remoteScore === 'number') {
    return item.remoteScore;
  }
  if (typeof item.autoScore === 'number') {
    return item.autoScore;
  }
  return null;
};

const buildQuestionPreview = (question: string, maxLength = 96) => {
  if (question.length <= maxLength) {
    return question;
  }
  return `${question.slice(0, maxLength - 1)}…`;
};

const resolveQualityTone = (score: number | null, threshold: number) => {
  if (score === null) {
    return 'warn' as const;
  }
  if (score >= threshold) {
    return 'good' as const;
  }
  if (score >= threshold * 0.85) {
    return 'warn' as const;
  }
  return 'bad' as const;
};

const buildUserMessage = (question: string) => {
  return `Ответь на вопрос максимально точно и лаконично.\n\nВопрос: ${question}`;
};

const aggregateKeywords = (items: EvaluationDatasetItem[], domains: string[]) => {
  if (!items.length) {
    return [] as string[];
  }
  const filtered = domains.length > 0
    ? items.filter(item => domains.includes(item.domain))
    : items;
  const tokens = filtered.flatMap(item => item.tags ?? []);
  return Array.from(new Set(tokens.map(tag => tag.toLowerCase()))).slice(0, 12);
};

interface LlmScorePayload {
  score: number | null;
  verdict?: string;
  reason?: string;
}

const clampScore = (value: number) => {
  if (!Number.isFinite(value)) {
    return null;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
};

const extractJson = (raw: string) => {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('LLM не вернула корректный JSON');
  }
  const snippet = raw.slice(start, end + 1);
  return JSON.parse(snippet) as Record<string, unknown>;
};

const scoreAnswerWithLlm = async (
  question: string,
  referenceAnswer: string,
  modelAnswer: string,
  qualityThreshold: number,
  remoteConfig: {
    apiUrl: string;
    apiKey: string;
    maxTokens: number;
    temperature: number;
    topP: number;
    modelId: string;
    signal: AbortSignal;
  },
): Promise<LlmScorePayload> => {
  const userPrompt = [
    'Ты выступаешь как строгий проверяющий ответов модели.',
    'Тебе дан вопрос, эталонный ответ и ответ модели.',
    'Поставь оценку от 0 до 1 (1 — точно соответствует эталону).',
    'Верни JSON вида {"score":0.85,"verdict":"короткий вывод","reason":"обоснование"}.',
    'Всегда возвращай только JSON, без дополнительного текста.',
    '',
    `Вопрос: ${question}`,
    `Эталонный ответ: ${referenceAnswer}`,
    `Ответ модели: ${modelAnswer}`,
    `Порог прохода: ${qualityThreshold.toFixed(2)}`,
  ].join('\n');

  const result = await callRemoteChat([
    {
      role: 'system' as const,
      content: 'Ты оцениваешь ответы модели и всегда отвечаешь строгим JSON-объектом.'
        + ' Поля: score (0..1), verdict (строка), reason (строка).',
    },
    { role: 'user' as const, content: userPrompt },
  ], {
    apiUrl: remoteConfig.apiUrl,
    apiKey: remoteConfig.apiKey,
    maxTokens: Math.min(remoteConfig.maxTokens, 256),
    temperature: Math.min(0.2, remoteConfig.temperature),
    topP: remoteConfig.topP,
    model: remoteConfig.modelId,
    signal: remoteConfig.signal,
  });

  const payload = extractJson(result.content);
  const scoreValue = clampScore(Number(payload.score));
  return {
    score: scoreValue,
    verdict: typeof payload.verdict === 'string' ? payload.verdict : undefined,
    reason: typeof payload.reason === 'string' ? payload.reason : undefined,
  } satisfies LlmScorePayload;
};

const EvaluationTab = () => {
  const {
    datasets,
    activeDatasetId,
    activeRun,
    runs,
    reviewTasks,
    preferences,
    setActiveDatasetId,
    setActiveRunId,
    updatePreferences,
    startRun,
    updateRunItem,
    updateRunStatus,
    submitReview,
    addDatasetItems,
    resetDatasets,
    exportDataset,
    importDataset,
    attachBaseline,
    clearRuns,
    appendRunNote,
  } = useEvaluation();
  const { settings } = useSettings();
  const { setActivity, updateActivity, clearActivity } = useStatus();

  const remoteConfigured = Boolean(settings.remoteApiUrl?.trim());
  const isCompact = preferences.compactMode ?? false;

  const [planner, setPlanner] = useState<PlannerState>(() => ({
    ...DEFAULT_PLANNER,
    datasetId: activeDatasetId ?? (datasets[0]?.id ?? null),
    baselineRunId: null,
  }));
  const [qualityGate, setQualityGate] = useState<number>(preferences.qualityGate ?? 0.7);
  const [draftItem, setDraftItem] = useState<DraftItemForm>(INITIAL_FORM);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [importError, setImportError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const runsRef = useRef<EvaluationRun[]>(runs);
  const { config: trainingConfig, addDatasetItems: addTrainingDatasetItems, dataset: trainingDataset } = useTraining();
  const [autoTrainingConfig, setAutoTrainingConfig] = useState<AutoTrainingConfig>(() => loadAutoTrainingConfig());
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [pipelineSummary, setPipelineSummary] = useState<string | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [showPipelineOptions, setShowPipelineOptions] = useState(false);
  const updateAutoTrainingConfig = useCallback((update: Partial<AutoTrainingConfig>) => {
    setAutoTrainingConfig(prev => {
      const next = {
        ...prev,
        ...update,
      };
      persistAutoTrainingConfig(next);
      return next;
    });
  }, []);

  useEffect(() => {
    runsRef.current = runs;
  }, [runs]);

  useEffect(() => {
    if (planner.datasetId !== activeDatasetId && planner.datasetId) {
      setActiveDatasetId(planner.datasetId);
    }
  }, [planner.datasetId, activeDatasetId, setActiveDatasetId]);

  useEffect(() => {
    if (!activeRun) {
      return;
    }
    const baselineId = activeRun.baselineRunId;
    if (baselineId && planner.baselineRunId !== baselineId) {
      setPlanner(prev => ({ ...prev, baselineRunId: baselineId }));
    }
  }, [activeRun, planner.baselineRunId]);

  useEffect(() => {
    if (preferences.qualityGate !== undefined && preferences.qualityGate !== null) {
      setQualityGate(preferences.qualityGate);
    }
  }, [preferences.qualityGate]);

  useEffect(() => {
    if (planner.scoringMode === 'human' && !planner.requireHumanReview) {
      setPlanner(prev => ({ ...prev, requireHumanReview: true }));
    }
  }, [planner.scoringMode, planner.requireHumanReview]);

  const selectedDataset = useMemo<EvaluationDataset | null>(() => {
    if (!planner.datasetId) {
      return null;
    }
    return datasets.find(dataset => dataset.id === planner.datasetId) ?? null;
  }, [datasets, planner.datasetId]);

  const datasetDomains = useMemo(() => {
    if (!selectedDataset) {
      return [] as string[];
    }
    const domains = Array.from(new Set(selectedDataset.items.map(item => item.domain))).sort();
    return domains;
  }, [selectedDataset]);

  const datasetDifficulties = useMemo(() => {
    if (!selectedDataset) {
      return [] as EvaluationDifficulty[];
    }
    const difficulties = Array.from(new Set(selectedDataset.items.map(item => item.difficulty)));
    return difficulties as EvaluationDifficulty[];
  }, [selectedDataset]);

  const availableTags = useMemo(() => {
    if (!selectedDataset) {
      return [] as string[];
    }
    return aggregateKeywords(selectedDataset.items, planner.domains);
  }, [selectedDataset, planner.domains]);

  const filteredItems = useMemo(() => {
    if (!selectedDataset) {
      return [] as EvaluationDatasetItem[];
    }
    return selectedDataset.items.filter((item) => {
      if (planner.domains.length > 0 && !planner.domains.includes(item.domain)) {
        return false;
      }
      if (planner.difficulties.length > 0 && !planner.difficulties.includes(item.difficulty)) {
        return false;
      }
      if (planner.tags.length > 0 && !planner.tags.every(tag => item.tags.includes(tag))) {
        return false;
      }
      return true;
    });
  }, [selectedDataset, planner.domains, planner.difficulties, planner.tags]);

  const domainSummaries = useMemo(() => {
    if (!selectedDataset) {
      return [] as Array<{ domain: string; total: number; difficultyMap: Record<string, number> }>;
    }
    const map = new Map<string, { domain: string; total: number; difficultyMap: Record<string, number> }>();
    selectedDataset.items.forEach((item) => {
      const entry = map.get(item.domain) ?? {
        domain: item.domain,
        total: 0,
        difficultyMap: {},
      };
      entry.total += 1;
      entry.difficultyMap[item.difficulty] = (entry.difficultyMap[item.difficulty] ?? 0) + 1;
      map.set(item.domain, entry);
    });
    return Array.from(map.values()).sort((a, b) => b.total - a.total);
  }, [selectedDataset]);

  const recentRuns = useMemo(() => {
    return runs
      .filter(run => run.config.datasetId === planner.datasetId)
      .sort((a, b) => b.createdAt - a.createdAt);
  }, [runs, planner.datasetId]);

  const appendLog = useCallback((message: string) => {
    setLogs((prev) => {
      const next = [...prev, `${new Date().toLocaleTimeString()} — ${message}`];
      return next.slice(-120);
    });
  }, []);

  const resetPlanner = useCallback(() => {
    setPlanner(prev => ({
      ...DEFAULT_PLANNER,
      datasetId: prev.datasetId,
      baselineRunId: prev.baselineRunId,
    }));
  }, []);

  const handlePlannerChange = useCallback(<Key extends keyof PlannerState>(key: Key, value: PlannerState[Key]) => {
    setPlanner(prev => ({
      ...prev,
      [key]: value,
    }));
  }, []);

  const handleToggleArrayValue = useCallback((key: 'domains' | 'difficulties' | 'tags', value: string) => {
    setPlanner(prev => {
      const current = prev[key] as string[];
      const exists = current.includes(value);
      const nextArray = exists ? current.filter(item => item !== value) : [...current, value];
      if (key === 'difficulties') {
        return {
          ...prev,
          [key]: nextArray as EvaluationDifficulty[],
        } as PlannerState;
      }
      return {
        ...prev,
        [key]: nextArray,
      } as PlannerState;
    });
  }, []);

  const handleAddItem = useCallback(() => {
    if (!planner.datasetId || !draftItem.question.trim() || !draftItem.referenceAnswer.trim() || !draftItem.domain.trim()) {
      return;
    }
    const newItem: EvaluationDatasetItem = {
      id: `${planner.datasetId}-${Date.now()}-${Math.random().toString(16).slice(2, 7)}`,
      domain: draftItem.domain.trim(),
      question: draftItem.question.trim(),
      referenceAnswer: draftItem.referenceAnswer.trim(),
      difficulty: draftItem.difficulty,
      tags: draftItem.tags.split(',').map(tag => tag.trim()).filter(Boolean),
    };
    addDatasetItems(planner.datasetId, [newItem]);
    setDraftItem(INITIAL_FORM);
    appendLog(`Добавлен новый пример в датасет «${draftItem.domain.trim()}».`);
  }, [planner.datasetId, draftItem, addDatasetItems, appendLog]);

  const handleImportDataset = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        if (!parsed || typeof parsed !== 'object') {
          throw new Error('Некорректный формат файла');
        }
        importDataset(parsed as EvaluationDataset);
        setImportError(null);
        appendLog(`Импортирован датасет из файла «${file.name}».`);
      } catch (error) {
        setImportError(error instanceof Error ? error.message : 'Не удалось импортировать датасет');
      }
    };
    reader.readAsText(file);
  }, [importDataset, appendLog]);

  const handleExportDataset = useCallback(() => {
    if (!planner.datasetId) {
      return;
    }
    const dataset = exportDataset(planner.datasetId);
    if (!dataset) {
      return;
    }
    const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${dataset.name.replace(/[^a-zA-Z0-9-_]+/g, '_') || 'dataset'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [planner.datasetId, exportDataset]);

  const updateProgressActivity = useCallback((message: string, progress: number, status: 'running' | 'success' | 'error' = 'running') => {
    updateActivity('evaluation', {
      message,
      progress: Math.round(progress),
      status,
    });
  }, [updateActivity]);

  const finishActivity = useCallback((status: 'success' | 'error' | 'warning', message: string) => {
    updateActivity('evaluation', {
      message,
      status,
      progress: status === 'success' || status === 'warning' ? 100 : undefined,
    });
    setTimeout(() => {
      clearActivity('evaluation');
    }, 1200);
  }, [updateActivity, clearActivity]);

  const resolveModelMessages = useCallback((item: EvaluationRunItem): RemoteChatMessage[] => {
    const main = buildUserMessage(item.question);
    return [
      { role: 'system', content: 'Ты — ассистент, отвечающий на проверочный вопрос максимально точно.' },
      { role: 'user', content: main },
    ];
  }, []);

  const runLocalScoring = useCallback((item: EvaluationRunItem, answer: string) => {
    const keywords = aggregateKeywords([{
      ...item,
      tags: item.tags ?? [],
      difficulty: item.difficulty ?? 'medium',
      referenceAnswer: item.referenceAnswer,
      question: item.question,
      domain: item.domain,
      id: item.datasetItemId,
    }], []);
    return scoreAnswerLocally(item.referenceAnswer, answer, {
      keywords,
      weightExact: 0.25,
      weightF1: 0.5,
      weightJaccard: 0.15,
      weightKeyword: 0.1,
    });
  }, []);

  const evaluateRun = useCallback(async (run: EvaluationRun) => {
    setIsRunning(true);
    appendLog(`Запуск тестирования (${run.items.length} примеров)...`);
    setActivity('evaluation', {
      message: 'Запуск тестирования',
      progress: 0,
      status: 'running',
    });

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const { modelVariant, scoringMode, requireHumanReview: requireReviewConfig, qualityGateThreshold } = run.config;
    const maxTokens = settings.maxTokens;
    const temperature = scoringMode === 'auto' ? Math.min(settings.temperature, 0.4) : settings.temperature;
  const threshold = typeof qualityGateThreshold === 'number' ? qualityGateThreshold : qualityGate;
  const useAutoScoring = scoringMode === 'auto';
    const useLlmScoring = scoringMode === 'llm';
    const forceManualReview = scoringMode === 'human';
    let processed = 0;

    try {
      for (const item of run.items) {
        if (abortController.signal.aborted) {
          appendLog('Тестирование остановлено пользователем.');
          updateRunStatus(run.id, 'failed', 'Отменено пользователем');
          finishActivity('error', 'Тестирование остановлено');
          break;
        }

        updateRunItem(run.id, item.id, {
          status: 'answering',
          answerStartedAt: Date.now(),
        });

        appendLog(`Генерация ответа для вопроса: ${item.question.slice(0, 64)}...`);
        let modelAnswer = '';
        let tokensUsed: number | null = null;
        try {
          if (modelVariant === 'remote') {
            const remoteResult = await callRemoteChat(resolveModelMessages(item), {
              apiUrl: settings.remoteApiUrl,
              apiKey: settings.remoteApiKey,
              maxTokens,
              temperature,
              topP: settings.topP,
              model: settings.remoteModelId,
              signal: abortController.signal,
            });
            modelAnswer = remoteResult.content;
            tokensUsed = remoteResult.tokens ?? null;
          } else {
            let modelPath = settings.baseModelPath;
            let adapterPath: string | undefined;
            if (modelVariant === 'fine_tuned') {
              const fineTunedPath = settings.fineTunedModelPath.trim();
              const fineTunedMethod = settings.fineTunedMethod || 'lora';
              const baseForAdapter = (settings.fineTunedBaseModelPath || settings.baseModelPath).trim();

              if (!fineTunedPath) {
                throw new Error('Не указан путь до дообученной модели');
              }

              if (fineTunedMethod === 'full') {
                modelPath = fineTunedPath;
              } else {
                modelPath = baseForAdapter;
                adapterPath = fineTunedPath;
              }
            }

            const baseResult = await callBaseChat(resolveModelMessages(item), {
              serverUrl: settings.baseModelServerUrl,
              modelPath,
              adapterPath,
              maxTokens,
              temperature,
              topP: settings.topP,
              quantization: settings.quantization,
              device: settings.deviceType,
              signal: abortController.signal,
            });
            modelAnswer = baseResult.content;
            tokensUsed = baseResult.tokens ?? null;
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Неизвестная ошибка генерации';
          appendLog(`⚠️ Ошибка генерации ответа: ${message}`);
          updateRunItem(run.id, item.id, {
            modelAnswer: `Ошибка генерации: ${message}`,
            status: 'needs_review',
            reviewStatus: 'pending',
            answerCompletedAt: Date.now(),
          });
          processed += 1;
          updateProgressActivity('Ожидание проверки проблемных ответов', (processed / run.items.length) * 100);
          // eslint-disable-next-line no-continue
          continue;
        }

        let autoScore: number | null = null;
        let reviewNotes: string | undefined;

        if (useAutoScoring) {
          const localScore = runLocalScoring(item, modelAnswer);
          autoScore = localScore.score;
          reviewNotes = localScore.reasons.join('\n');
        } else if (useLlmScoring) {
          if (!remoteConfigured) {
            appendLog('Удалённая модель не настроена, ответ передан на ручную проверку.');
          } else {
            try {
              const llmResult = await scoreAnswerWithLlm(
                item.question,
                item.referenceAnswer,
                modelAnswer,
                threshold,
                {
                  apiUrl: settings.remoteApiUrl,
                  apiKey: settings.remoteApiKey ?? '',
                  maxTokens: settings.maxTokens,
                  temperature: settings.temperature,
                  topP: settings.topP,
                  modelId: settings.remoteModelId,
                  signal: abortController.signal,
                },
              );
              autoScore = llmResult.score;
              reviewNotes = [llmResult.verdict, llmResult.reason]
                .filter(Boolean)
                .join('\n');
              appendLog(`LLM-оценка ответа: ${autoScore !== null ? autoScore.toFixed(3) : 'нет оценки'}`);
            } catch (error) {
              const message = error instanceof Error ? error.message : 'Ошибка удалённой оценки';
              appendLog(`⚠️ ${message}`);
            }
          }
        }

        const shouldReview = forceManualReview
          || requireReviewConfig
          || (autoScore !== null ? autoScore < Math.max(threshold, 0.6) : true);
        const notesToStore = shouldReview ? reviewNotes : useLlmScoring ? reviewNotes : undefined;

        updateRunItem(run.id, item.id, {
          modelAnswer,
          autoScore,
          status: shouldReview ? 'needs_review' : 'scored',
          reviewStatus: shouldReview ? 'pending' : 'approved',
          answerCompletedAt: Date.now(),
          tokens: tokensUsed,
          reviewNotes: notesToStore,
        });
        processed += 1;
        updateProgressActivity(
          useAutoScoring
            ? 'Расчёт локальных метрик'
            : useLlmScoring
              ? 'Удалённая проверка ответов'
              : 'Подготовка результатов',
          (processed / run.items.length) * 100,
        );
      }

      const latest = runsRef.current.find(candidate => candidate.id === run.id);
      if (!latest) {
        appendLog('Не удалось обновить состояние прогона.');
        finishActivity('error', 'Ошибка состояния теста');
        setIsRunning(false);
        return;
      }

      const pendingReviews = latest.items.filter(item => item.status === 'needs_review').length;
      if (pendingReviews > 0) {
        updateRunStatus(run.id, 'waiting_review');
        finishActivity('warning', 'Требуется ручная проверка');
        appendLog(`Ожидает проверки: ${pendingReviews} ответов.`);
      } else {
        updateRunStatus(run.id, 'completed');
        const gateResult = latest.metrics.qualityGatePassed ?? null;
        const gateThresholdUsed = latest.metrics.qualityGateThreshold ?? threshold;
        const scoreText = formatScore(latest.metrics.overallScore, 3);
        const gateThresholdText = formatScore(gateThresholdUsed, 3);

        if (gateResult === false) {
          finishActivity('warning', 'Скор ниже критерия качества');
          appendLog(`Итоговый скор ${scoreText} ниже порога ${gateThresholdText}.`);
          appendRunNote(run.id, `Скор ${scoreText} ниже порога ${gateThresholdText}. Проверьте проблемные ответы.`);
        } else {
          finishActivity('success', 'Тестирование завершено');
          if (gateResult === true) {
            appendLog(`Итоговый скор ${scoreText} ≥ порога ${gateThresholdText}.`);
          } else {
            appendLog('Тестирование завершено без необходимости ручной проверки.');
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Неизвестная ошибка тестирования';
      appendLog(`Ошибка тестирования: ${message}`);
      updateRunStatus(run.id, 'failed', message);
      finishActivity('error', 'Тестирование завершилось с ошибкой');
    } finally {
      abortControllerRef.current = null;
      setIsRunning(false);
    }
  }, [
    appendLog,
    qualityGate,
    resolveModelMessages,
    runLocalScoring,
    setActivity,
    settings.baseModelPath,
    settings.baseModelServerUrl,
    settings.deviceType,
    settings.fineTunedModelPath,
    settings.maxTokens,
    settings.remoteApiKey,
    settings.remoteApiUrl,
    settings.remoteModelId,
    settings.quantization,
    settings.temperature,
    settings.topP,
    remoteConfigured,
    updateProgressActivity,
    updateRunItem,
    updateRunStatus,
    finishActivity,
    appendRunNote,
  ]);

  const handleStartRun = useCallback(async () => {
    if (!planner.datasetId || !selectedDataset) {
      return;
    }
    let effectiveScoringMode: EvaluationScoringMode = planner.scoringMode;
    if (planner.scoringMode === 'llm' && !remoteConfigured) {
      appendLog('Удалённая модель не настроена. Переключаемся на ручную проверку.');
      effectiveScoringMode = 'human';
      setPlanner(prev => ({ ...prev, scoringMode: 'human', requireHumanReview: true }));
    }

    const requireHumanReview = effectiveScoringMode === 'human' ? true : planner.requireHumanReview;
    const config = {
      datasetId: planner.datasetId,
      datasetName: selectedDataset.name,
      modelVariant: planner.modelVariant,
      scoringMode: effectiveScoringMode,
      sampleSize: planner.sampleSize ?? undefined,
      shuffle: true,
      qualityGateThreshold: qualityGate,
      requireHumanReview,
    };
    const run = startRun({
      datasetId: planner.datasetId,
      name: `Тест ${new Date().toLocaleString()}`,
      config,
      sampleSize: planner.sampleSize,
      domainFilter: planner.domains,
      difficultyFilter: planner.difficulties,
      tagFilter: planner.tags,
      baselineRunId: planner.baselineRunId,
    });
    if (!run) {
      appendLog('Не удалось создать прогон тестирования. Проверьте настройки.');
      return;
    }
    setActiveRunId(run.id);
    updateRunStatus(run.id, 'running');
    void evaluateRun(run);
  }, [
    planner,
    selectedDataset,
    startRun,
    appendLog,
    setActiveRunId,
    updateRunStatus,
    evaluateRun,
    remoteConfigured,
  ]);

  const handleCancelRun = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const handleQualityGateChange = useCallback((value: number) => {
    setQualityGate(value);
    updatePreferences({ qualityGate: value });
  }, [updatePreferences]);

  useEffect(() => {
    if (!preferences.autoAssignBaseline || planner.baselineRunId) {
      return;
    }
    const candidates = runs
      .filter(run => run.id !== activeRun?.id && run.config.datasetId === planner.datasetId)
      .sort((a, b) => (b.completedAt ?? 0) - (a.completedAt ?? 0));
    if (candidates.length > 0) {
      setPlanner(prev => ({ ...prev, baselineRunId: candidates[0].id }));
    }
  }, [preferences.autoAssignBaseline, runs, planner.datasetId, planner.baselineRunId, activeRun?.id]);

  const runQualityGateThreshold = activeRun?.metrics.qualityGateThreshold ?? qualityGate;
  const gateStatus = activeRun?.metrics.qualityGatePassed ?? null;
  const gateStatusLabel = gateStatus === null
    ? 'Нет данных'
    : gateStatus
      ? 'Пройден'
      : 'Не пройден';
  const gateStatusClass = gateStatus === null
    ? 'text-slate-400'
    : gateStatus
      ? 'text-emerald-300'
      : 'text-rose-300';
  const qualityTone = resolveQualityTone(activeRun?.metrics.overallScore ?? null, runQualityGateThreshold);
  const baselineRun = useMemo(() => runs.find(run => run.id === planner.baselineRunId) ?? null, [runs, planner.baselineRunId]);

  const dashboardData = useMemo(() => {
    if (!activeRun) {
      return {
        domains: [] as DashboardDomainInsight[],
        tags: [] as DashboardTagSummary[],
        difficulties: [] as DashboardDifficultySummary[],
        hotspots: [] as DashboardHotspot[],
        recommendations: [] as DashboardRecommendation[],
      };
    }

    const round = (value: number) => Math.round(value * 1000) / 1000;
    const effectiveThreshold = typeof runQualityGateThreshold === 'number' && !Number.isNaN(runQualityGateThreshold)
      ? runQualityGateThreshold
      : 0.7;

    const baselineDomainMap = baselineRun
      ? new Map(baselineRun.metrics.domainSummaries.map(summary => [summary.domain, summary]))
      : new Map<string, DomainMetricSummary>();

    const totalDomainVolume = activeRun.metrics.domainSummaries.reduce((acc, summary) => acc + summary.total, 0);

    const domains: DashboardDomainInsight[] = activeRun.metrics.domainSummaries
      .map((summary, index) => {
        const coverageShare = summary.total > 0 ? summary.scored / summary.total : 0;
        const baselineSummary = baselineDomainMap.get(summary.domain);
        const baselineScore = baselineSummary?.averageScore ?? null;
        const delta = summary.averageScore != null && baselineScore != null
          ? round(summary.averageScore - baselineScore)
          : null;
        const tone = resolveQualityTone(summary.averageScore ?? null, effectiveThreshold);
        const toneClass = tone === 'good'
          ? 'bg-emerald-400'
          : tone === 'warn'
            ? 'bg-amber-400'
            : 'bg-rose-400';
        const share = totalDomainVolume > 0 ? summary.total / totalDomainVolume : 0;
        const color = DOMAIN_COLORS[index % DOMAIN_COLORS.length];
        return {
          ...summary,
          coverageShare,
          delta,
          toneClass,
          share,
          color,
        } satisfies DashboardDomainInsight;
      })
      .sort((a, b) => {
        const scoreA = a.averageScore ?? -1;
        const scoreB = b.averageScore ?? -1;
        return scoreA - scoreB;
      });

    type TagAccumulator = {
      tag: string;
      total: number;
      scored: number;
      needsReview: number;
      scoreSum: number;
      scoreCount: number;
    };

    const tagMap = new Map<string, TagAccumulator>();
    const aggregateByDifficulty = (run: EvaluationRun | null) => {
      type DifficultyAccumulator = {
        total: number;
        scored: number;
        needsReview: number;
        scoreSum: number;
        scoreCount: number;
      };
      const map = new Map<EvaluationDifficulty | 'unknown', DifficultyAccumulator>();
      if (!run) {
        return map;
      }
      run.items.forEach((item) => {
        const difficulty = item.difficulty ?? 'unknown';
        const accumulator = map.get(difficulty) ?? {
          total: 0,
          scored: 0,
          needsReview: 0,
          scoreSum: 0,
          scoreCount: 0,
        };
        accumulator.total += 1;
        if (item.status === 'scored' || item.status === 'reviewed') {
          accumulator.scored += 1;
        }
        if (item.status === 'needs_review') {
          accumulator.needsReview += 1;
        }
        const aggregateScore = resolveAggregateScore(item);
        if (typeof aggregateScore === 'number') {
          accumulator.scoreSum += aggregateScore;
          accumulator.scoreCount += 1;
        }
        map.set(difficulty, accumulator);
      });
      return map;
    };

    activeRun.items.forEach((item) => {
      const aggregateScore = resolveAggregateScore(item);
      const isScored = item.status === 'scored' || item.status === 'reviewed';
      const tags = item.tags ?? [];
      tags.forEach((tagRaw) => {
        const tag = tagRaw.trim();
        if (!tag) {
          return;
        }
        const accumulator = tagMap.get(tag) ?? {
          tag,
          total: 0,
          scored: 0,
          needsReview: 0,
          scoreSum: 0,
          scoreCount: 0,
        } satisfies TagAccumulator;
        accumulator.total += 1;
        if (isScored) {
          accumulator.scored += 1;
        }
        if (item.status === 'needs_review') {
          accumulator.needsReview += 1;
        }
        if (typeof aggregateScore === 'number') {
          accumulator.scoreSum += aggregateScore;
          accumulator.scoreCount += 1;
        }
        tagMap.set(tag, accumulator);
      });
    });

    const tags: DashboardTagSummary[] = Array.from(tagMap.values())
      .map(({ tag, total, scored, needsReview, scoreSum, scoreCount }) => {
        const coverageShare = total > 0 ? scored / total : 0;
        const averageScore = scoreCount > 0 ? round(scoreSum / scoreCount) : null;
        const tone = resolveQualityTone(averageScore, effectiveThreshold);
        const toneClass = tone === 'good'
          ? 'text-emerald-300'
          : tone === 'warn'
            ? 'text-amber-300'
            : 'text-rose-300';
        return {
          tag,
          total,
          scored,
          needsReview,
          coverageShare,
          averageScore,
          toneClass,
        } satisfies DashboardTagSummary;
      })
      .sort((a, b) => {
        const scoreA = a.averageScore ?? -1;
        const scoreB = b.averageScore ?? -1;
        if (scoreA === scoreB) {
          return b.total - a.total;
        }
        return scoreA - scoreB;
      });

    const currentDifficulty = aggregateByDifficulty(activeRun);
    const baselineDifficulty = aggregateByDifficulty(baselineRun);
    const difficultyKeys = Array.from(new Set([
      ...currentDifficulty.keys(),
      ...baselineDifficulty.keys(),
    ]));

    const difficulties: DashboardDifficultySummary[] = difficultyKeys
      .map((difficulty) => {
        const current = currentDifficulty.get(difficulty) ?? {
          total: 0,
          scored: 0,
          needsReview: 0,
          scoreSum: 0,
          scoreCount: 0,
        };
        const baseline = baselineDifficulty.get(difficulty) ?? {
          total: 0,
          scored: 0,
          needsReview: 0,
          scoreSum: 0,
          scoreCount: 0,
        };
        const averageScore = current.scoreCount > 0 ? round(current.scoreSum / current.scoreCount) : null;
        const baselineScore = baseline.scoreCount > 0 ? round(baseline.scoreSum / baseline.scoreCount) : null;
        const delta = averageScore !== null && baselineScore !== null
          ? round(averageScore - baselineScore)
          : null;
        const tone = resolveQualityTone(averageScore, effectiveThreshold);
        const toneClass = tone === 'good'
          ? 'text-emerald-300'
          : tone === 'warn'
            ? 'text-amber-300'
            : 'text-rose-300';
        return {
          difficulty,
          total: current.total,
          scored: current.scored,
          needsReview: current.needsReview,
          averageScore,
          baselineScore,
          delta,
          toneClass,
        } satisfies DashboardDifficultySummary;
      })
      .sort((a, b) => {
        const order = ['hard', 'medium', 'easy', 'unknown'];
        const indexA = order.indexOf(a.difficulty);
        const indexB = order.indexOf(b.difficulty);
        return (indexA === -1 ? order.length : indexA) - (indexB === -1 ? order.length : indexB);
      });

    const HOTSPOT_LIMIT = 8;
    const hotspotCandidates = activeRun.items
      .map((item) => ({
        item,
        aggregateScore: resolveAggregateScore(item),
      }))
      .filter(({ item, aggregateScore }) => {
        if (item.status === 'needs_review') {
          return true;
        }
        if (aggregateScore === null) {
          return false;
        }
        return aggregateScore < effectiveThreshold * 0.95;
      })
      .sort((a, b) => {
        const scoreA = a.aggregateScore ?? 2;
        const scoreB = b.aggregateScore ?? 2;
        return scoreA - scoreB;
      });

    const hotspots: DashboardHotspot[] = hotspotCandidates
      .slice(0, HOTSPOT_LIMIT)
      .map(({ item, aggregateScore }) => {
        const tone = resolveQualityTone(aggregateScore, effectiveThreshold);
        const toneClass = tone === 'good'
          ? 'text-emerald-300'
          : tone === 'warn'
            ? 'text-amber-300'
            : 'text-rose-300';
        return {
          id: item.id,
          question: buildQuestionPreview(item.question),
          domain: item.domain,
          tags: item.tags ?? [],
          difficulty: item.difficulty ?? 'unknown',
          score: aggregateScore,
          status: item.status,
          needsReview: item.status === 'needs_review',
          reviewNotes: item.reviewNotes ?? null,
          toneClass,
        } satisfies DashboardHotspot;
      });

    const recommendationMap = new Map<string, DashboardRecommendation>();
    const registerRecommendation = (key: string, label: string) => {
      const entry = recommendationMap.get(key) ?? { key, label, count: 0 };
      entry.count += 1;
      recommendationMap.set(key, entry);
    };

    hotspots.forEach((hotspot) => {
      registerRecommendation(
        `domain:${hotspot.domain}`,
        `Добавить QA по домену «${hotspot.domain}»`,
      );
      hotspot.tags.forEach((tag) => {
        registerRecommendation(
          `tag:${tag}`,
          `Расширить примеры с тегом «${tag}»`,
        );
      });
      registerRecommendation(
        `difficulty:${hotspot.difficulty}`,
        `Укрепить задания уровня «${DIFFICULTY_LABELS[hotspot.difficulty]}»`,
      );
    });

    const recommendations = Array.from(recommendationMap.values())
      .sort((a, b) => b.count - a.count)
      .slice(0, 6);

    return {
      domains,
      tags,
      difficulties,
      hotspots,
      recommendations,
    };
  }, [activeRun, baselineRun, runQualityGateThreshold]);

  const { domains, tags, difficulties, hotspots, recommendations } = dashboardData;

  const handleSubmitReview = useCallback((taskId: { runId: string; itemId: string }, approve: boolean) => {
    submitReview(taskId.runId, taskId.itemId, {
      status: approve ? 'approved' : 'rejected',
      reviewer: preferences.defaultReviewer ?? 'analyst',
      scoreOverride: undefined,
    });
    appendLog(`Ответ ${approve ? 'одобрен' : 'отклонён'} вручную.`);
  }, [submitReview, preferences.defaultReviewer, appendLog]);

  const downloadActiveRun = useCallback(() => {
    if (!activeRun) {
      return;
    }
    const payload = {
      id: activeRun.id,
      name: activeRun.name,
      createdAt: activeRun.createdAt,
      metrics: activeRun.metrics,
      items: activeRun.items,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${activeRun.name.replace(/[^a-zA-Z0-9-_]+/g, '_') || 'evaluation'}-results.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [activeRun]);

  const handleSendToAggregator = useCallback(async () => {
    if (!activeRun) {
      appendLog('Нет активного прогона для отправки в Agregator.');
      return;
    }

    const aggregatorUrl = autoTrainingConfig.aggregatorUrl.trim();
    if (!aggregatorUrl) {
      appendLog('URL сервиса Agregator не задан. Откройте раздел автообучения и укажите адрес.');
      setPipelineError('Не задан URL Agregator');
      return;
    }

    const baseModelPath = settings.baseModelPath.trim();
    const baseModelServerUrl = settings.baseModelServerUrl.trim();
    if (autoTrainingConfig.autoLaunchFineTune && (!baseModelPath || !baseModelServerUrl)) {
      appendLog('Для запуска дообучения укажите путь к базовой модели и URL сервера модели во вкладке «Настройки».');
      setPipelineError('Не заданы параметры базовой модели или сервера');
      return;
    }

    setIsPipelineRunning(true);
    setPipelineSummary(null);
    setPipelineError(null);
    appendLog('Передаём проблемные вопросы в Agregator…');

    const evaluationItems = activeRun.items.map(item => ({
      question: item.question,
      referenceAnswer: item.referenceAnswer,
      modelAnswer: item.modelAnswer ?? '',
      autoScore: item.autoScore ?? null,
      humanScore: item.humanScore ?? null,
      remoteScore: item.remoteScore ?? null,
      status: item.status,
      reviewStatus: item.reviewStatus,
      domain: item.domain,
      difficulty: item.difficulty ?? null,
      tags: item.tags ?? [],
      datasetItemId: item.datasetItemId,
    }));

    const evaluationPayload = {
      id: activeRun.id,
      name: activeRun.name,
      created_at: activeRun.createdAt,
      metrics: activeRun.metrics,
      quality_gate: activeRun.metrics.qualityGateThreshold ?? runQualityGateThreshold ?? null,
      items: evaluationItems,
    };

    const scoreThreshold = autoTrainingConfig.scoreThreshold ?? runQualityGateThreshold ?? null;
    const generationOptions = {
      pairs_per_snippet: Math.max(1, autoTrainingConfig.pairsPerSnippet),
      include_reference_pair: autoTrainingConfig.includeReferencePair,
      min_paragraph_chars: Math.max(40, autoTrainingConfig.minParagraphChars),
      max_paragraph_chars: Math.max(autoTrainingConfig.minParagraphChars, autoTrainingConfig.maxParagraphChars),
      max_segments: Math.max(1, autoTrainingConfig.maxSegments),
    };

    const searchOptions = {
      top_k: Math.max(1, autoTrainingConfig.topK),
      deep_search: autoTrainingConfig.deepSearch,
      max_candidates: Math.max(5, autoTrainingConfig.maxCandidates),
      max_snippets: Math.max(1, autoTrainingConfig.maxSnippets),
      chunk_chars: Math.max(500, autoTrainingConfig.chunkChars),
      max_chunks: Math.max(1, autoTrainingConfig.maxChunks),
      max_segments: Math.max(1, autoTrainingConfig.maxSegments),
    };

    const pipelinePayload: ProblemPipelinePayload = {
      evaluation: evaluationPayload,
      score_threshold: scoreThreshold,
      target_pairs: Math.max(1, autoTrainingConfig.targetPairs),
      generation: generationOptions,
      search: searchOptions,
      dry_run: autoTrainingConfig.pipelineDryRun,
      include_dataset: autoTrainingConfig.includeDatasetPreview,
    };

    if (autoTrainingConfig.autoLaunchFineTune && baseModelPath && baseModelServerUrl) {
      const targetModules = trainingConfig.targetModules
        ? trainingConfig.targetModules
            .split(',')
            .map(entry => entry.trim())
            .filter(Boolean)
        : undefined;

      const fineTuneConfig: Record<string, unknown> = {
        server_url: baseModelServerUrl,
        base_model_path: baseModelPath,
        output_dir: trainingConfig.outputDir.trim() || undefined,
        include_previous_dataset: autoTrainingConfig.includePreviousDataset,
        previous_fine_tune_path: autoTrainingConfig.previousFineTunePath.trim() || undefined,
        deduplicate: autoTrainingConfig.deduplicate,
        min_examples: Math.max(1, autoTrainingConfig.minExamples),
        timeout: Math.max(60, autoTrainingConfig.fineTuneTimeout),
        config: {
          method: trainingConfig.method,
          quantization: trainingConfig.quantization,
          lora_rank: trainingConfig.loraRank,
          lora_alpha: trainingConfig.loraAlpha,
          learning_rate: trainingConfig.learningRate,
          batch_size: trainingConfig.batchSize,
          epochs: trainingConfig.epochs,
          max_length: trainingConfig.maxLength,
          warmup_steps: trainingConfig.warmupSteps,
          target_modules: targetModules && targetModules.length > 0 ? targetModules : undefined,
          initial_adapter_path: trainingConfig.initialAdapterPath?.trim() || undefined,
        },
      };
      pipelinePayload.fine_tune = fineTuneConfig;
    }

    const headers = autoTrainingConfig.authToken.trim()
      ? {
          [autoTrainingConfig.authHeader.trim() || 'Authorization']: autoTrainingConfig.authToken.trim(),
        }
      : undefined;

    try {
      const response = await runProblemPipeline(
        pipelinePayload,
        {
          baseUrl: aggregatorUrl,
          headers,
          timeout: Math.max(60000, autoTrainingConfig.fineTuneTimeout * 1000),
          credentials: autoTrainingConfig.authToken.trim() ? 'omit' : 'include',
        },
      );

      const datasetSize = response.dataset_size ?? 0;
      const logsCount = response.logs?.length ?? 0;
      appendLog(`Agregator: сформировано ${datasetSize} QA-пар (логов: ${logsCount}).`);
      if (response.fine_tune_job?.id) {
        appendLog(`Agregator: запущено дообучение (задача ${response.fine_tune_job.id}).`);
      }

      const rawDataset = Array.isArray(response.dataset)
        ? response.dataset
        : Array.isArray(response.dataset_preview_full)
          ? response.dataset_preview_full
          : [];

      if (autoTrainingConfig.pipelineDryRun) {
        if (rawDataset.length > 0) {
          appendLog('Agregator: dry-run — датасет не импортирован во вкладку «Датасеты», доступен только в ответе.');
        }
      } else if (rawDataset.length > 0) {
        const existingPairs = new Set<string>();
        for (const item of trainingDataset) {
          const signature = `${item.input.trim()}\u0001${item.output.trim()}`;
          existingPairs.add(signature);
        }

        const importedItems = rawDataset.reduce<DatasetItem[]>((acc, entry) => {
          if (!entry || typeof entry !== 'object') {
            return acc;
          }
          const record = entry as Record<string, unknown>;
          const input = typeof record.input === 'string' ? record.input.trim() : '';
          const output = typeof record.output === 'string' ? record.output.trim() : '';
          if (!input || !output) {
            return acc;
          }
          const signature = `${input}\u0001${output}`;
          if (existingPairs.has(signature)) {
            return acc;
          }
          existingPairs.add(signature);
          const sourceValue = record.source;
          const source = typeof sourceValue === 'string' && sourceValue.trim() ? sourceValue.trim() : undefined;
          acc.push({
            id: createPipelineDatasetItemId(),
            input,
            output,
            source,
          });
          return acc;
        }, []);

        if (importedItems.length > 0) {
          addTrainingDatasetItems(importedItems);
          appendLog(`Agregator: ${importedItems.length} новых примеров добавлены во вкладку «Датасеты».`);
        }
      } else if (datasetSize > 0 && !autoTrainingConfig.includeDatasetPreview) {
        appendLog('Agregator: датасет не загружен в интерфейс — включите опцию «Вернуть полный датасет», чтобы увидеть его на вкладке «Датасеты».');
      }

      setPipelineSummary(`QA-пар: ${datasetSize}${response.fine_tune_job?.id ? ` • задача ${response.fine_tune_job.id}` : ''}`);
      setPipelineError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error ?? 'Неизвестная ошибка');
      appendLog(`Agregator: ошибка — ${message}`);
      setPipelineError(message);
      setPipelineSummary(null);
    } finally {
      setIsPipelineRunning(false);
    }
  }, [
    activeRun,
    appendLog,
    addTrainingDatasetItems,
    autoTrainingConfig,
    runQualityGateThreshold,
    settings.baseModelPath,
    settings.baseModelServerUrl,
    trainingConfig,
    trainingDataset,
  ]);

  if (isCompact) {
    return (
      <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <Layers className="h-5 w-5 text-sky-400" />
            <div className="flex flex-col text-xs text-slate-500 dark:text-slate-300">
              <span className="text-sm font-semibold text-slate-700 dark:text-slate-100">{selectedDataset ? selectedDataset.name : 'Датасет не выбран'}</span>
              {activeRun && <span>Прогон: {activeRun.name}</span>}
            </div>
          </div>
          <button
            type="button"
            onClick={() => updatePreferences({ compactMode: false })}
            className="inline-flex items-center gap-1 rounded-lg border border-emerald-500/60 bg-emerald-500/15 px-2 py-1 text-xs text-emerald-200 hover:border-emerald-400 hover:text-emerald-100"
          >
            <Minimize2 className="h-3 w-3" /> Обычный вид
          </button>
        </div>

        {activeRun ? (
          <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-none">
            <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
              <div className="rounded-xl border border-slate-200 bg-white/90 p-3 dark:border-slate-800 dark:bg-slate-900/70">
                <p className="text-[11px] text-slate-500">Итоговый скор</p>
                <p className={`mt-1 text-lg font-semibold ${QUALITY_CLASSES[resolveQualityTone(activeRun.metrics.overallScore ?? null, qualityGate)]}`}>
                  {formatScore(activeRun.metrics.overallScore, 3)}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white/90 p-3 dark:border-slate-800 dark:bg-slate-900/70">
                <p className="text-[11px] text-slate-500">Покрытие</p>
                <p className="mt-1 text-lg font-semibold text-sky-300">
                  {formatPercent(activeRun.metrics.coverage)}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white/90 p-3 dark:border-slate-800 dark:bg-slate-900/70">
                <p className="text-[11px] text-slate-500">Ручные оценки</p>
                <p className="mt-1 text-lg font-semibold text-amber-300">
                  {formatPercent(activeRun.metrics.humanApprovalRate)}
                </p>
              </div>
            </div>
          </section>
        ) : (
          <section className="rounded-2xl border border-dashed border-slate-300 bg-white/80 p-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/40">
            Запустите прогон, чтобы увидеть метрики.
          </section>
        )}

        <Suspense
          fallback={(
            <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-none">
              <div className="h-40 animate-pulse rounded-2xl bg-slate-200/60 dark:bg-slate-700/40" />
            </section>
          )}
        >
          <QualityDashboardLazy
            activeRun={activeRun}
            domains={domains}
            tags={tags}
            difficulties={difficulties}
            hotspots={hotspots}
            recommendations={recommendations}
            compact
            formatScore={formatScore}
            formatPercent={formatPercent}
            difficultyLabels={DIFFICULTY_LABELS}
          />
        </Suspense>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="h-5 w-5 text-sky-400" />
              <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Эталонный датасет</h2>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => updatePreferences({ compactMode: true })}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:border-emerald-400 hover:text-emerald-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:text-emerald-200"
              >
                <Minimize2 className="h-3 w-3" /> Компактно
              </button>
              <button
                type="button"
                onClick={handleExportDataset}
                className="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-500 dark:text-slate-300 hover:border-slate-500 hover:text-white"
              >
                <Download className="mr-1 inline h-3 w-3" />Экспорт
              </button>
              <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-500 dark:text-slate-300 hover:border-slate-500 hover:text-white">
                <input type="file" accept="application/json" className="hidden" onChange={handleImportDataset} />
                <RefreshCw className="h-3 w-3" />
                Импорт
              </label>
              <button
                type="button"
                onClick={resetDatasets}
                className="rounded-lg border border-rose-400 bg-rose-100 px-2 py-1 text-xs text-rose-600 hover:border-rose-500 hover:text-rose-700 dark:border-rose-500/50 dark:bg-rose-500/10 dark:text-rose-300"
              >
                Сброс
              </button>
            </div>
          </header>
          <div className="mt-3 space-y-3 text-xs text-slate-500 dark:text-slate-300">
            <div>
              <label className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Датасет</label>
              <select
                value={planner.datasetId ?? ''}
                onChange={(event) => handlePlannerChange('datasetId', event.target.value || null)}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-sky-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              >
                {datasets.map(dataset => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} ({dataset.items.length})
                  </option>
                ))}
              </select>
              {importError && <p className="mt-1 text-[11px] text-rose-300">{importError}</p>}
            </div>
            <div className="flex flex-wrap gap-3 text-[11px] text-slate-500 dark:text-slate-400">
              <span className="rounded-full border border-slate-300 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-800">Всего: {selectedDataset?.items.length ?? 0}</span>
              <span className="rounded-full border border-slate-300 bg-white px-2 py-1 flex items-center gap-1 dark:border-slate-700 dark:bg-slate-800">
                <Filter className="h-3 w-3" />
                Фильтр: {filteredItems.length}
              </span>
              <span className="rounded-full border border-slate-300 bg-white px-2 py-1 flex items-center gap-1 dark:border-slate-700 dark:bg-slate-800">
                <ClipboardList className="h-3 w-3" />
                Домены: {datasetDomains.length}
              </span>
            </div>
            <div className="space-y-2">
              <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Домены</p>
              <div className="flex flex-wrap gap-2">
                {datasetDomains.map(domain => {
                  const isActive = planner.domains.includes(domain);
                  return (
                    <button
                      key={domain}
                      type="button"
                      onClick={() => handleToggleArrayValue('domains', domain)}
                      className={`rounded-full border px-3 py-1 text-[11px] transition ${
                        isActive
                          ? 'border-sky-400 bg-sky-100 text-sky-700 dark:border-sky-500 dark:bg-sky-500/20 dark:text-sky-100'
                          : 'border-slate-300 bg-white text-slate-600 hover:border-slate-400 hover:text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:text-slate-100'
                      }`}
                    >
                      {domain}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Сложность</p>
              <div className="flex flex-wrap gap-2">
                {datasetDifficulties.map(difficulty => {
                  const isActive = planner.difficulties.includes(difficulty);
                  return (
                    <button
                      key={difficulty}
                      type="button"
                      onClick={() => handleToggleArrayValue('difficulties', difficulty)}
                      className={`rounded-full border px-3 py-1 text-[11px] transition ${
                        isActive
                          ? 'border-amber-400 bg-amber-100 text-amber-700 dark:border-amber-500 dark:bg-amber-500/20 dark:text-amber-100'
                          : 'border-slate-300 bg-white text-slate-600 hover:border-slate-400 hover:text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:text-slate-100'
                      }`}
                    >
                      {difficulty}
                    </button>
                  );
                })}
              </div>
            </div>
            {availableTags.length > 0 && (
              <div className="space-y-2">
                <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Теги</p>
                <div className="flex flex-wrap gap-2">
                  {availableTags.map(tag => {
                    const isActive = planner.tags.includes(tag);
                    return (
                      <button
                        key={tag}
                        type="button"
                        onClick={() => handleToggleArrayValue('tags', tag)}
                        className={`rounded-full border px-3 py-1 text-[11px] transition ${
                          isActive
                            ? 'border-emerald-400 bg-emerald-100 text-emerald-700 dark:border-emerald-500 dark:bg-emerald-500/20 dark:text-emerald-100'
                            : 'border-slate-300 bg-white text-slate-600 hover:border-slate-400 hover:text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:text-slate-100'
                        }`}
                      >
                        {tag}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
          <footer className="mt-4 border-t border-slate-200 pt-3 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
            <p>Распределение: {domainSummaries.map(summary => `${summary.domain} (${summary.total})`).join(', ')}</p>
          </footer>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none xl:col-span-2">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-emerald-400" />
              <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Планировщик прогона</h2>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <ShieldCheck className="h-3.5 w-3.5" /> Критерий качества: {formatScore(qualityGate, 2)}
              <input
                type="range"
                min={0.4}
                max={0.95}
                step={0.01}
                value={qualityGate}
                onChange={(event) => handleQualityGateChange(Number(event.target.value))}
                className="ml-2 w-32"
              />
            </div>
          </header>
          <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <label className="text-[11px] uppercase tracking-wide text-slate-500">Вариант модели</label>
              <select
                value={planner.modelVariant}
                onChange={(event) => handlePlannerChange('modelVariant', event.target.value as PlannerState['modelVariant'])}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              >
                <option value="base">Базовая модель</option>
                <option value="fine_tuned" disabled={!settings.fineTunedModelPath}>
                  Дообученная модель
                </option>
                <option value="remote">Удалённый API</option>
              </select>
            </div>
            <div>
              <label className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Режим скоринга</label>
              <select
                value={planner.scoringMode}
                onChange={(event) => handlePlannerChange('scoringMode', event.target.value as EvaluationScoringMode)}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              >
                <option value="auto">Только локальные метрики</option>
                <option value="llm" disabled={!remoteConfigured}>
                  Оценка удалённой моделью
                </option>
                <option value="human">Только ручная проверка</option>
              </select>
              {!remoteConfigured && planner.scoringMode === 'llm' && (
                <p className="mt-1 text-[11px] text-rose-400">
                  Укажите настройки удалённой модели в разделе «Настройки», чтобы использовать автоматическую проверку.
                </p>
              )}
            </div>
            <div>
              <label className="text-[11px] uppercase tracking-wide text-slate-500">Размер выборки</label>
              <input
                type="number"
                min={1}
                max={filteredItems.length}
                value={planner.sampleSize ?? ''}
                onChange={(event) => handlePlannerChange('sampleSize', event.target.value ? Number(event.target.value) : null)}
                placeholder={`до ${filteredItems.length}`}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </div>
            <div className="flex items-center gap-2">
              <input
                id="require-human-review"
                type="checkbox"
                checked={planner.requireHumanReview}
                onChange={(event) => handlePlannerChange('requireHumanReview', event.target.checked)}
                disabled={planner.scoringMode === 'human'}
                className="h-4 w-4 rounded border border-slate-300 bg-white text-emerald-500 focus:ring-emerald-500 disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900"
              />
              <label htmlFor="require-human-review" className="text-xs text-slate-500 dark:text-slate-300">
                Требовать ручную проверку независимо от авто-скора
              </label>
            </div>
            <div className="md:col-span-2">
              <label className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Базовый прогон для сравнения</label>
              <select
                value={planner.baselineRunId ?? ''}
                onChange={(event) => {
                  const value = event.target.value || null;
                  handlePlannerChange('baselineRunId', value);
                  if (activeRun) {
                    attachBaseline(activeRun.id, value);
                  }
                }}
                className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              >
                <option value="">Без сравнения</option>
                {recentRuns.filter(run => run.id !== activeRun?.id).map(run => (
                  <option key={run.id} value={run.id}>
                    {run.name} — {formatScore(run.metrics.overallScore, 2)}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <footer className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleStartRun}
              disabled={isRunning || filteredItems.length === 0}
              className="inline-flex items-center gap-2 rounded-xl border border-emerald-500 bg-emerald-500/15 px-4 py-2 text-sm font-medium text-emerald-200 hover:border-emerald-400 hover:text-emerald-100 disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-slate-800 disabled:text-slate-500"
            >
              <Play className="h-4 w-4" />Запустить тестирование
            </button>
            <button
              type="button"
              onClick={handleCancelRun}
              disabled={!isRunning}
              className="inline-flex items-center gap-2 rounded-xl border border-rose-500 bg-rose-500/15 px-4 py-2 text-sm font-medium text-rose-200 hover:border-rose-400 hover:text-rose-100 disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-slate-800 disabled:text-slate-500"
            >
              <Square className="h-4 w-4" />Остановить
            </button>
            <button
              type="button"
              onClick={resetPlanner}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-600 hover:border-slate-400 hover:text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
            >
              <RefreshCw className="h-4 w-4" />Сбросить фильтры
            </button>
            <span className="ml-auto text-xs text-slate-400">
              Выбрано {filteredItems.length} примеров
            </span>
          </footer>
        </section>
      </div>

      <div className="grid flex-1 grid-cols-1 gap-4 xl:grid-cols-2">
        <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-none">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-emerald-400" />
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Итоговые метрики</h3>
            </div>
            {activeRun && (
              <button
                type="button"
                onClick={downloadActiveRun}
                className="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-500 dark:text-slate-300 hover:border-slate-500 hover:text-slate-100"
              >
                <Download className="mr-1 inline h-3 w-3" />JSON
              </button>
            )}
          </header>
          {activeRun ? (
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
                <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3">
                  <p className="text-[11px] text-slate-500">Итоговый скор</p>
                  <p className={`mt-1 text-lg font-semibold ${QUALITY_CLASSES[qualityTone]}`}>
                    {formatScore(activeRun.metrics.overallScore, 3)}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    Порог: {formatScore(runQualityGateThreshold, 2)}
                  </p>
                  <p className={`text-[11px] font-medium ${gateStatusClass}`}>
                    Критерий: {gateStatusLabel}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3">
                  <p className="text-[11px] text-slate-500">Покрытие</p>
                  <p className="mt-1 text-lg font-semibold text-sky-300">
                    {formatPercent(activeRun.metrics.coverage)}
                  </p>
                  <p className="text-[11px] text-slate-500">Оценено: {activeRun.metrics.scored}/{activeRun.metrics.total}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3">
                  <p className="text-[11px] text-slate-500">Авто-скор</p>
                  <p className="mt-1 text-lg font-semibold text-emerald-300">
                    {formatScore(activeRun.metrics.autoScoreAverage, 3)}
                  </p>
                  <p className="text-[11px] text-slate-500">Локальные метрики</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3">
                  <p className="text-[11px] text-slate-500">Ручные оценки</p>
                  <p className="mt-1 text-lg font-semibold text-amber-300">
                    {formatPercent(activeRun.metrics.humanApprovalRate)}
                  </p>
                  <p className="text-[11px] text-slate-500">Отклонения: {activeRun.metrics.needsReview}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3">
                  <p className="text-[11px] text-slate-500">Время ответа</p>
                  <p className="mt-1 text-lg font-semibold text-sky-300">
                    {formatSeconds(activeRun.metrics.averageLatencySeconds)}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    P95: {formatSeconds(activeRun.metrics.latencyP95Seconds)}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3">
                  <p className="text-[11px] text-slate-500">Токены</p>
                  <p className="mt-1 text-lg font-semibold text-violet-300">
                    {renderTokensLine(activeRun.metrics.averageTokensPerResponse, 'ток./ответ')}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    Скорость: {renderTokensLine(activeRun.metrics.tokensPerSecond, 'ток./с')}
                  </p>
                </div>
              </div>
              {baselineRun && (
                <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3 text-xs text-slate-500 dark:text-slate-300">
                  <div className="flex items-center gap-2 text-[11px] text-slate-500">
                    <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
                    Сравнение с базовым прогоном
                  </div>
                  <p className="mt-2 text-sm text-slate-600 dark:text-slate-200">
                    Δ общий скор: {formatScore(activeRun.metrics.scoreDelta, 3)} (база: {formatScore(activeRun.metrics.baselineScore, 3)})
                  </p>
                  <p className="text-[11px] text-slate-500">{baselineRun.name}</p>
                </div>
              )}
              <div className="rounded-xl border border-slate-200 bg-white/90 dark:bg-slate-900/70 p-3 text-xs text-slate-500 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-400">
                <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Доменные показатели</p>
                <div className="mt-2 space-y-2 max-h-60 overflow-y-auto pr-1">
                  {activeRun.metrics.domainSummaries.map((domain: DomainMetricSummary) => {
                    const domainDelta = baselineRun?.metrics.domainSummaries.find(item => item.domain === domain.domain);
                    const delta = domainDelta?.averageScore != null && domain.averageScore != null
                      ? domain.averageScore - domainDelta.averageScore
                      : null;
                    const tone = delta == null ? 'text-slate-300' : delta >= 0 ? 'text-emerald-300' : 'text-rose-300';
                    return (
                      <div key={domain.domain} className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-600 dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-200">
                        <div className="flex items-center justify-between">
                          <span>{domain.domain}</span>
                          <span className="text-sm font-semibold text-emerald-500 dark:text-emerald-200">{formatScore(domain.averageScore, 3)}</span>
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                          <span>Оценено: {domain.scored}/{domain.total}</span>
                          <span>Авто: {formatScore(domain.autoScoreAverage, 3)}</span>
                          <span className={tone}>{delta != null ? `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}` : '—'}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="mt-6 rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/40">
              Запустите прогон тестирования, чтобы увидеть метрики.
            </div>
          )}
        </section>

        <Suspense
          fallback={(
            <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-none xl:col-span-2">
              <div className="h-40 animate-pulse rounded-2xl bg-slate-200/60 dark:bg-slate-700/40" />
            </section>
          )}
        >
          <QualityDashboardLazy
            activeRun={activeRun}
            domains={domains}
            tags={tags}
            difficulties={difficulties}
            hotspots={hotspots}
            recommendations={recommendations}
            compact={false}
            formatScore={formatScore}
            formatPercent={formatPercent}
            difficultyLabels={DIFFICULTY_LABELS}
          />
        </Suspense>
        <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none xl:col-span-2">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-sky-400" />
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Очередь проверки</h3>
            </div>
            <span className="rounded-full border border-slate-300 bg-white px-2 py-1 text-[11px] text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
              В очереди: {reviewTasks.length}
            </span>
          </header>
          <div className="mt-3 space-y-3 overflow-y-auto pr-1 text-xs text-slate-500 dark:text-slate-300" style={{ maxHeight: '320px' }}>
            {reviewTasks.length === 0 && (
              <div className="rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-slate-500 dark:border-slate-700 dark:bg-slate-900/50">
                Нет ответов, требующих ручной проверки.
              </div>
            )}
            {reviewTasks.map(task => (
              <article key={`${task.runId}-${task.itemId}`} className="rounded-xl border border-slate-200 bg-white p-3 text-slate-600 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-200">
                <header className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                  <span>{task.domain}</span>
                  <span>{new Date(task.submittedAt).toLocaleTimeString()}</span>
                </header>
                <p className="mt-2 text-sm font-medium text-slate-700 dark:text-slate-100">{task.question}</p>
                <p className="mt-2 whitespace-pre-wrap text-xs text-slate-500 dark:text-slate-400">
                  <span className="text-slate-500 dark:text-slate-400">Ответ модели:</span> {task.modelAnswer ?? '—'}
                </p>
                <p className="mt-2 whitespace-pre-wrap text-xs text-emerald-600 dark:text-emerald-300">
                  <span className="text-emerald-600 dark:text-emerald-400">Эталон:</span> {task.referenceAnswer}
                </p>
                <footer className="mt-3 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleSubmitReview({ runId: task.runId, itemId: task.itemId }, true)}
                    className="inline-flex items-center gap-1 rounded-lg border border-emerald-500 bg-emerald-500/15 px-3 py-1 text-xs text-emerald-600 hover:border-emerald-400 hover:text-emerald-500 dark:text-emerald-200 dark:hover:text-emerald-100"
                  >
                    <Check className="h-3 w-3" />Одобрить
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSubmitReview({ runId: task.runId, itemId: task.itemId }, false)}
                    className="inline-flex items-center gap-1 rounded-lg border border-rose-500 bg-rose-500/15 px-3 py-1 text-xs text-rose-600 hover:border-rose-400 hover:text-rose-500 dark:text-rose-200 dark:hover:text-rose-100"
                  >
                    <X className="h-3 w-3" />Отклонить
                  </button>
                </footer>
              </article>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ClipboardList className="h-5 w-5 text-amber-400" />
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Последние прогоны</h3>
            </div>
            <button
              type="button"
              onClick={clearRuns}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs text-slate-600 hover:border-slate-400 hover:text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
            >
              Очистить историю
            </button>
          </header>
          <div className="mt-3 space-y-2 overflow-y-auto pr-1 text-xs text-slate-500 dark:text-slate-300" style={{ maxHeight: '320px' }}>
            {recentRuns.length === 0 && (
              <div className="rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-slate-500 dark:border-slate-700 dark:bg-slate-900/50">
                История тестов появится после первого запуска.
              </div>
            )}
            {recentRuns.map(run => (
              <div
                key={run.id}
                className={`cursor-pointer rounded-xl border px-3 py-2 transition ${
                  run.id === activeRun?.id
                    ? 'border-emerald-500 bg-emerald-500/10'
                    : 'border-slate-200 bg-white hover:border-slate-400 dark:border-slate-800 dark:bg-slate-950/30 dark:hover:border-slate-600'
                }`}
                onClick={() => setActiveRunId(run.id)}
                role="button"
                tabIndex={0}
              >
                <div className="flex items-center justify-between text-slate-700 dark:text-slate-200">
                  <span className="text-sm font-medium">{run.name}</span>
                  <span className="text-xs text-slate-500 dark:text-slate-400">{new Date(run.updatedAt).toLocaleString()}</span>
                </div>
                <div className="mt-1 flex items-center gap-3 text-[11px] text-slate-500 dark:text-slate-400">
                  <span className="flex items-center gap-1">
                    <ShieldCheck className="h-3 w-3 text-emerald-400" />
                    {formatScore(run.metrics.overallScore, 2)}
                  </span>
                  <span className="flex items-center gap-1">
                    <Activity className="h-3 w-3 text-sky-400" />
                    {run.status}
                  </span>
                  <span className="flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3 text-amber-400" />
                    {run.metrics.needsReview}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none">
        <header className="flex flex-wrap items-center justify-between gap-3 text-sm font-semibold text-slate-700 dark:text-slate-200">
          <div className="flex items-center gap-2">
            <Target className="h-5 w-5 text-emerald-400" />
            <span>Agregator: генерация датасета по проблемным вопросам</span>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-500 dark:text-slate-400">
            <span>URL: {autoTrainingConfig.aggregatorUrl || 'не задан'}</span>
            <span>Цель: {autoTrainingConfig.targetPairs}</span>
            <span>Порог: {autoTrainingConfig.scoreThreshold ?? runQualityGateThreshold ?? '—'}</span>
          </div>
        </header>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={handleSendToAggregator}
            disabled={!activeRun || isPipelineRunning}
            className="inline-flex items-center gap-2 rounded-xl border border-emerald-500 bg-emerald-500/15 px-4 py-2 text-sm font-medium text-emerald-200 hover:border-emerald-400 hover:text-emerald-100 disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-slate-800 disabled:text-slate-500"
          >
            {isPipelineRunning ? <Activity className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {isPipelineRunning ? 'Отправка…' : 'Сформировать датасет'}
          </button>
          <button
            type="button"
            onClick={() => setShowPipelineOptions(prev => !prev)}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs text-slate-600 hover:border-slate-400 hover:text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
          >
            {showPipelineOptions ? <Minimize2 className="h-3.5 w-3.5" /> : <Filter className="h-3.5 w-3.5" />}
            {showPipelineOptions ? 'Скрыть настройки' : 'Показать настройки'}
          </button>
          {pipelineSummary && <span className="text-xs text-emerald-400">{pipelineSummary}</span>}
          {pipelineError && <span className="text-xs text-rose-400">{pipelineError}</span>}
        </div>
        {showPipelineOptions && (
          <div className="mt-4 grid gap-4 text-xs text-slate-600 dark:text-slate-300 md:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span>URL Agregator</span>
              <input
                type="text"
                value={autoTrainingConfig.aggregatorUrl}
                onChange={(event) => updateAutoTrainingConfig({ aggregatorUrl: event.target.value })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Количество пар в датасете</span>
              <input
                type="number"
                min={1}
                value={autoTrainingConfig.targetPairs}
                onChange={(event) => updateAutoTrainingConfig({ targetPairs: Math.max(1, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.targetPairs) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Порог проблемного скора</span>
              <input
                type="number"
                step="0.05"
                value={autoTrainingConfig.scoreThreshold ?? ''}
                onChange={(event) => {
                  const raw = event.target.value;
                  if (raw === '') {
                    updateAutoTrainingConfig({ scoreThreshold: null });
                    return;
                  }
                  const parsed = Number(raw);
                  updateAutoTrainingConfig({ scoreThreshold: Number.isFinite(parsed) ? parsed : null });
                }}
                placeholder={`${runQualityGateThreshold ?? 0.7}`}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Пар на сниппет</span>
              <input
                type="number"
                min={1}
                value={autoTrainingConfig.pairsPerSnippet}
                onChange={(event) => updateAutoTrainingConfig({ pairsPerSnippet: Math.max(1, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.pairsPerSnippet) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Мин. длина абзаца</span>
              <input
                type="number"
                min={40}
                value={autoTrainingConfig.minParagraphChars}
                onChange={(event) => updateAutoTrainingConfig({ minParagraphChars: Math.max(40, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.minParagraphChars) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Макс. длина абзаца</span>
              <input
                type="number"
                min={autoTrainingConfig.minParagraphChars}
                value={autoTrainingConfig.maxParagraphChars}
                onChange={(event) => updateAutoTrainingConfig({ maxParagraphChars: Math.max(autoTrainingConfig.minParagraphChars, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.maxParagraphChars) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Top k поиска</span>
              <input
                type="number"
                min={1}
                value={autoTrainingConfig.topK}
                onChange={(event) => updateAutoTrainingConfig({ topK: Math.max(1, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.topK) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Макс. кандидатов</span>
              <input
                type="number"
                min={5}
                value={autoTrainingConfig.maxCandidates}
                onChange={(event) => updateAutoTrainingConfig({ maxCandidates: Math.max(5, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.maxCandidates) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Макс. сниппетов</span>
              <input
                type="number"
                min={1}
                value={autoTrainingConfig.maxSnippets}
                onChange={(event) => updateAutoTrainingConfig({ maxSnippets: Math.max(1, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.maxSnippets) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Размер чанка (символов)</span>
              <input
                type="number"
                min={500}
                value={autoTrainingConfig.chunkChars}
                onChange={(event) => updateAutoTrainingConfig({ chunkChars: Math.max(500, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.chunkChars) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Макс. чанков</span>
              <input
                type="number"
                min={1}
                value={autoTrainingConfig.maxChunks}
                onChange={(event) => updateAutoTrainingConfig({ maxChunks: Math.max(1, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.maxChunks) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>Таймаут дообучения (сек)</span>
              <input
                type="number"
                min={60}
                value={autoTrainingConfig.fineTuneTimeout}
                onChange={(event) => updateAutoTrainingConfig({ fineTuneTimeout: Math.max(60, Number(event.target.value) || DEFAULT_AUTO_TRAINING_CONFIG.fineTuneTimeout) })}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <div className="flex flex-col gap-2 md:col-span-2">
              <span className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Дополнительно</span>
              <div className="flex flex-wrap gap-3">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoTrainingConfig.includeReferencePair}
                    onChange={(event) => updateAutoTrainingConfig({ includeReferencePair: event.target.checked })}
                    className="h-4 w-4 rounded border border-slate-300 text-emerald-500 focus:ring-emerald-500 dark:border-slate-700 dark:bg-slate-900"
                  />
                  <span>Добавлять эталонные пары</span>
                </label>
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoTrainingConfig.deepSearch}
                    onChange={(event) => updateAutoTrainingConfig({ deepSearch: event.target.checked })}
                    className="h-4 w-4 rounded border border-slate-300 text-emerald-500 focus:ring-emerald-500 dark:border-slate-700 dark:bg-slate-900"
                  />
                  <span>Глубокий поиск</span>
                </label>
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoTrainingConfig.autoLaunchFineTune}
                    onChange={(event) => updateAutoTrainingConfig({ autoLaunchFineTune: event.target.checked })}
                    className="h-4 w-4 rounded border border-slate-300 text-emerald-500 focus:ring-emerald-500 dark:border-slate-700 dark:bg-slate-900"
                  />
                  <span>Запускать дообучение</span>
                </label>
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoTrainingConfig.pipelineDryRun}
                    onChange={(event) => updateAutoTrainingConfig({ pipelineDryRun: event.target.checked })}
                    className="h-4 w-4 rounded border border-slate-300 text-emerald-500 focus:ring-emerald-500 dark:border-slate-700 dark:bg-slate-900"
                  />
                  <span>Только собрать датасет</span>
                </label>
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoTrainingConfig.includeDatasetPreview}
                    onChange={(event) => updateAutoTrainingConfig({ includeDatasetPreview: event.target.checked })}
                    className="h-4 w-4 rounded border border-slate-300 text-emerald-500 focus:ring-emerald-500 dark:border-slate-700 dark:bg-slate-900"
                  />
                  <span>Вернуть полный датасет</span>
                </label>
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Globe className="h-5 w-5 text-slate-300" />
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Журнал</h3>
          </div>
          <button
            type="button"
            onClick={() => setLogs([])}
            className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:border-slate-400 hover:text-slate-800 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
          >
            Очистить
          </button>
        </header>
        <div className="mt-3 max-h-40 overflow-y-auto rounded-xl border border-slate-200 bg-white p-3 text-[11px] text-slate-600 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-400">
          {logs.length === 0 && <p className="text-slate-500 dark:text-slate-500">Журнал событий появится во время тестирования.</p>}
          {logs.map(entry => (
            <div key={entry}>{entry}</div>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none">
        <header className="flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
          <Plus className="h-5 w-5 text-emerald-400" />Добавить новый пример
        </header>
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="md:col-span-2">
            <label className="text-[11px] uppercase tracking-wide text-slate-500">Вопрос</label>
            <textarea
              value={draftItem.question}
              onChange={(event) => setDraftItem(prev => ({ ...prev, question: event.target.value }))}
              rows={2}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <div className="md:col-span-2">
            <label className="text-[11px] uppercase tracking-wide text-slate-500">Эталонный ответ</label>
            <textarea
              value={draftItem.referenceAnswer}
              onChange={(event) => setDraftItem(prev => ({ ...prev, referenceAnswer: event.target.value }))}
              rows={3}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-[11px] uppercase tracking-wide text-slate-500">Домен</label>
            <input
              value={draftItem.domain}
              onChange={(event) => setDraftItem(prev => ({ ...prev, domain: event.target.value }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="text-[11px] uppercase tracking-wide text-slate-500">Сложность</label>
            <select
              value={draftItem.difficulty}
              onChange={(event) => setDraftItem(prev => ({ ...prev, difficulty: event.target.value as EvaluationDifficulty }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
            >
              <option value="easy">easy</option>
              <option value="medium">medium</option>
              <option value="hard">hard</option>
            </select>
          </div>
          <div className="md:col-span-2">
            <label className="text-[11px] uppercase tracking-wide text-slate-500">Теги (через запятую)</label>
            <input
              value={draftItem.tags}
              onChange={(event) => setDraftItem(prev => ({ ...prev, tags: event.target.value }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-emerald-500 focus:outline-none"
            />
          </div>
        </div>
        <footer className="mt-4 flex items-center gap-2">
          <button
            type="button"
            onClick={handleAddItem}
            className="inline-flex items-center gap-2 rounded-xl border border-emerald-500 bg-emerald-500/15 px-4 py-2 text-sm font-medium text-emerald-200 hover:border-emerald-400 hover:text-emerald-100"
          >
            <Plus className="h-4 w-4" />Добавить пример
          </button>
          <span className="text-xs text-slate-500">Используйте новые примеры для расширения покрытия предметных областей.</span>
        </footer>
      </section>
    </div>
  );
};

export default EvaluationTab;
