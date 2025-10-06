import { useMemo, useState, type ComponentType } from 'react';
import { X, Search, MessageSquare, Brain, FileText, Target } from 'lucide-react';
import { useHistory } from '../contexts/HistoryContext';
import type { EvaluationRunStatus } from '../types/evaluation';

type HistoryFilter = 'all' | 'chat' | 'training' | 'processing' | 'evaluation';

type HistoryStatus = 'completed' | 'failed' | 'running' | 'cancelled';

type HistoryEntryType = 'chat' | 'training' | 'processing' | 'evaluation';

interface HistorySidebarProps {
  onClose: () => void;
  lastViewedAt: number;
}

interface HistoryEntry {
  id: string;
  type: HistoryEntryType;
  timestamp: number;
  title: string;
  subtitle?: string;
  description?: string;
  status?: HistoryStatus;
  metrics?: Record<string, string | number> | undefined;
  searchText: string;
}

interface SparklineSummary {
  completed: number;
  failed: number;
  running: number;
}

interface SparklineProps {
  data: number[];
}

interface SparklineCardProps {
  label: string;
  data: number[];
  summary: SparklineSummary;
}

const statusToValue = (status: HistoryStatus | undefined) => {
  if (!status) {
    return 0;
  }
  if (status === 'completed') {
    return 3;
  }
  if (status === 'running') {
    return 2;
  }
  if (status === 'cancelled') {
    return 1;
  }
  return 1;
};

const Sparkline = ({ data }: SparklineProps) => {
  if (data.length === 0) {
    return (
      <div className="flex h-10 items-center justify-center text-[11px] text-slate-500">
        Нет данных
      </div>
    );
  }

  const maxValue = Math.max(...data, 1);

  return (
    <div className="flex h-10 items-end gap-[2px]" aria-hidden="true">
      {data.map((value, index) => {
        const normalizedHeight = (value / maxValue) * 100;
        const height = Math.max(6, normalizedHeight);
        const color = value >= 3
          ? 'bg-emerald-400/80'
          : value >= 2
            ? 'bg-amber-400/80'
            : 'bg-rose-500/80';
        return (
          <span
            key={`${value}-${index}`}
            className={`block flex-1 rounded-sm ${color}`}
            style={{ height: `${height}%` }}
          />
        );
      })}
    </div>
  );
};

const SparklineCard = ({ label, data, summary }: SparklineCardProps) => {
  const total = summary.completed + summary.failed + summary.running;

  return (
    <div className="rounded-xl border border-slate-200 bg-white/80 px-4 py-3 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-none">
      <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
        <span>{label}</span>
        <span>{total} записей</span>
      </div>
      <div className="mt-3">
        <Sparkline data={data} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 text-center text-[11px] text-slate-500 dark:text-slate-400">
        <div className="flex flex-col items-center gap-1">
          <span className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Успех</span>
          <span className="text-xs font-semibold text-emerald-500 dark:text-emerald-300">{summary.completed}</span>
        </div>
        <div className="flex flex-col items-center gap-1">
          <span className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">В работе</span>
          <span className="text-xs font-semibold text-amber-500 dark:text-amber-300">{summary.running}</span>
        </div>
        <div className="flex flex-col items-center gap-1">
          <span className="text-[10px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Сбой/Отмена</span>
          <span className="text-xs font-semibold text-rose-500 dark:text-rose-300">{summary.failed}</span>
        </div>
      </div>
    </div>
  );
};

const formatTimestamp = (timestamp: number) => {
  const date = new Date(timestamp);
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const truncate = (value: string, max = 160) => {
  if (value.length <= max) {
    return value;
  }
  return `${value.slice(0, max).trim()}...`;
};

const mapEvaluationStatusToHistory = (status: EvaluationRunStatus): HistoryStatus => {
  if (status === 'completed') {
    return 'completed';
  }
  if (status === 'failed') {
    return 'failed';
  }
  return 'running';
};

const describeEvaluationStatus = (status: EvaluationRunStatus, error?: string | null) => {
  if (status === 'waiting_review') {
    return 'Требуется ручная проверка результатов';
  }
  if (status === 'failed') {
    if (error && error.trim().length > 0) {
      return truncate(`Ошибка тестирования: ${error.trim()}`, 200);
    }
    return 'Тестирование завершилось с ошибкой';
  }
  return 'Результаты тестирования модели';
};

const formatCompletionRatio = (scored: number, total: number) => {
  if (total === 0) {
    return '0/0';
  }
  return `${scored}/${total}`;
};

const HistorySidebar = ({ onClose, lastViewedAt }: HistorySidebarProps) => {
  const {
    chatHistory,
    trainingHistory,
    processingHistory,
    evaluationHistory,
    exportHistory,
    clearHistory,
  } = useHistory();
  const [activeFilter, setActiveFilter] = useState<HistoryFilter>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const searchNormalized = searchTerm.trim().toLowerCase();

  const combinedHistory = useMemo<HistoryEntry[]>(() => {
    const chatEntries = chatHistory.map((message) => {
      const baseTitle = message.role === 'assistant' ? 'Ответ ассистента' : 'Сообщение пользователя';
      const modelLabel = message.model ? `Модель: ${message.model}` : undefined;

      return {
        id: message.id,
        type: 'chat' as HistoryEntryType,
        timestamp: message.timestamp,
        title: baseTitle,
        subtitle: modelLabel,
        description: truncate(message.content, 220),
        status: undefined,
        metrics: undefined,
        searchText: [baseTitle, message.content, message.model ?? '', message.role].join(' ').toLowerCase(),
      } satisfies HistoryEntry;
    });

    const trainingEntries = trainingHistory.map((session) => {
      return {
        id: session.id,
        type: 'training' as HistoryEntryType,
        timestamp: session.timestamp,
        title: session.modelName,
        subtitle: `Датасет: ${session.datasetSize} примеров`,
        description: 'История обучения модели',
        status: session.status,
        metrics: session.metrics,
        searchText: [
          session.modelName,
          session.status,
          session.datasetSize,
          Object.entries(session.metrics ?? {})
            .map(([key, value]) => `${key} ${value}`)
            .join(' '),
        ].join(' ').toLowerCase(),
      } satisfies HistoryEntry;
    });

    const processingEntries = processingHistory.map((job) => {
      return {
        id: job.id,
        type: 'processing' as HistoryEntryType,
        timestamp: job.timestamp,
        title: job.fileName,
        subtitle: `QA-пары: ${job.qaGenerated}`,
        description: 'Обработка датасета',
        status: job.status,
        metrics: undefined,
        searchText: [job.fileName, job.status, job.qaGenerated].join(' ').toLowerCase(),
      } satisfies HistoryEntry;
    });

    const evaluationEntries = evaluationHistory.map((session) => {
      const formattedScore = session.overallScore != null ? session.overallScore.toFixed(3) : '—';
      const formattedCoverage = session.coverage != null ? `${Math.round(session.coverage * 100)}%` : '—';
      const completion = formatCompletionRatio(session.scored, session.total);
      const status = mapEvaluationStatusToHistory(session.status);
      const description = describeEvaluationStatus(session.status, session.error);
      return {
        id: session.id,
        type: 'evaluation' as HistoryEntryType,
        timestamp: session.timestamp,
        title: session.datasetName,
        subtitle: `Модель: ${session.modelVariant}`,
        description,
        status,
        metrics: {
          Score: formattedScore,
          Coverage: formattedCoverage,
          'Оценено': completion,
        },
        searchText: [
          session.datasetName,
          session.modelVariant,
          session.overallScore ?? '',
          session.coverage ?? '',
          session.status,
          session.error ?? '',
        ].join(' ').toLowerCase(),
      } satisfies HistoryEntry;
    });

    return [...chatEntries, ...trainingEntries, ...processingEntries, ...evaluationEntries].sort(
      (a, b) => b.timestamp - a.timestamp,
    );
  }, [chatHistory, trainingHistory, processingHistory, evaluationHistory]);

  const filteredHistory = useMemo(() => {
    return combinedHistory.filter((entry) => {
      if (activeFilter !== 'all' && entry.type !== activeFilter) {
        return false;
      }
      if (!searchNormalized) {
        return true;
      }
      return entry.searchText.includes(searchNormalized);
    });
  }, [combinedHistory, activeFilter, searchNormalized]);

  const trainingSparklineData = useMemo(
    () => trainingHistory.slice(-24).map((session) => statusToValue(session.status)),
    [trainingHistory],
  );

  const processingSparklineData = useMemo(
    () => processingHistory.slice(-24).map((job) => statusToValue(job.status)),
    [processingHistory],
  );

  const evaluationSparklineData = useMemo(
    () => evaluationHistory.slice(-24).map((session) => {
      if (session.status === 'failed') {
        return 1;
      }
      if (session.status === 'completed') {
        if (session.overallScore === null || session.overallScore === undefined) {
          return 2;
        }
        if (session.overallScore >= 0.75) {
          return 3;
        }
        if (session.overallScore >= 0.5) {
          return 2;
        }
        return 1;
      }
      return 2;
    }),
    [evaluationHistory],
  );

  const trainingSummary = useMemo<SparklineSummary>(() => {
    let completed = 0;
    let failed = 0;
    let running = 0;

    trainingHistory.forEach((session) => {
      if (session.status === 'completed') {
        completed += 1;
      } else if (session.status === 'failed') {
        failed += 1;
      } else {
        running += 1;
      }
    });

    return { completed, failed, running };
  }, [trainingHistory]);

  const processingSummary = useMemo<SparklineSummary>(() => {
    let completed = 0;
    let failed = 0;
    let running = 0;

    processingHistory.forEach((job) => {
      if (job.status === 'completed') {
        completed += 1;
      } else if (job.status === 'failed' || job.status === 'cancelled') {
        failed += 1;
      } else {
        running += 1;
      }
    });

    return { completed, failed, running };
  }, [processingHistory]);

  const evaluationSummary = useMemo<SparklineSummary>(() => {
    let completed = 0;
    let failed = 0;
    let running = 0;

    evaluationHistory.forEach((session) => {
      if (session.status === 'completed') {
        completed += 1;
      } else if (session.status === 'failed') {
        failed += 1;
      } else {
        running += 1;
      }
    });

    return { completed, failed, running };
  }, [evaluationHistory]);

  const filterTabs: { id: HistoryFilter; label: string }[] = [
    { id: 'all', label: 'Все' },
    { id: 'chat', label: 'Чат' },
    { id: 'training', label: 'Обучение' },
    { id: 'evaluation', label: 'Тесты' },
    { id: 'processing', label: 'Датасеты' },
  ];

  const iconByType: Record<HistoryEntryType, ComponentType<{ className?: string }>> = {
    chat: MessageSquare,
    training: Brain,
    processing: FileText,
    evaluation: Target,
  };

  return (
    <div className="flex h-full flex-col border-l border-slate-900 bg-slate-950 text-slate-100">
      <div className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-900 dark:bg-slate-950/95">
        <div className="flex items-center justify-between px-5 py-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">История</h2>
            <p className="text-xs text-slate-500 dark:text-slate-500">Поиск по чатам, обучению и обработке датасетов</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white/80 text-slate-600 shadow-sm shadow-slate-900/5 transition-colors hover:border-blue-500 hover:text-blue-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:shadow-none dark:hover:text-white"
            title="Закрыть историю"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-3 px-5 pb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400 dark:text-slate-500" />
            <input
              type="search"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Поиск по истории"
              className="w-full rounded-lg border border-slate-200 bg-white py-2 pl-10 pr-3 text-sm text-slate-700 placeholder:text-slate-400 shadow-sm shadow-slate-900/5 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:placeholder:text-slate-500 dark:shadow-none"
            />
          </div>
          <div className="flex gap-2 overflow-x-auto text-xs">
            {filterTabs.map((tab) => {
              const isActive = activeFilter === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveFilter(tab.id)}
                  className={`rounded-full px-3 py-1 font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-500 text-white shadow shadow-blue-900/40'
                      : 'border border-slate-800 bg-slate-900 text-slate-300 hover:border-blue-500 hover:text-white'
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <SparklineCard label="Обучение" data={trainingSparklineData} summary={trainingSummary} />
          <SparklineCard label="Обработка" data={processingSparklineData} summary={processingSummary} />
          <SparklineCard label="Тестирование" data={evaluationSparklineData} summary={evaluationSummary} />
        </div>

        <div className="space-y-3">
          {filteredHistory.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white/70 px-4 py-8 text-center text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900/40">
              Ничего не найдено. Попробуйте изменить фильтр или запрос.
            </div>
          ) : (
            filteredHistory.map((entry) => {
              const Icon = iconByType[entry.type];
              const isNew = entry.timestamp > lastViewedAt;

              return (
                <article
                  key={`${entry.type}-${entry.id}`}
                  className="rounded-xl border border-slate-200 bg-white/90 px-4 py-3 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-slate-900/20"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-3">
                      <span className="mt-1 rounded-lg border border-slate-200 bg-white/70 p-2 text-slate-500 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400 dark:shadow-none">
                        <Icon className="h-4 w-4" />
                      </span>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">{entry.title}</h3>
                          {entry.status && (
                            <span
                              className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                                entry.status === 'completed'
                                  ? 'bg-emerald-500/10 text-emerald-300'
                                  : entry.status === 'running'
                                    ? 'bg-amber-500/10 text-amber-300'
                                    : 'bg-rose-500/10 text-rose-300'
                              }`}
                            >
                              {entry.status === 'completed' && 'Готово'}
                              {entry.status === 'running' && 'В работе'}
                              {entry.status === 'failed' && 'Ошибка'}
                              {entry.status === 'cancelled' && 'Отменено'}
                            </span>
                          )}
                          {isNew && (
                            <span className="rounded-full bg-blue-500/10 px-2 py-0.5 text-[11px] font-semibold text-blue-300">
                              Новое
                            </span>
                          )}
                        </div>
                        {entry.subtitle && (
                          <p className="text-[13px] text-slate-500 dark:text-slate-400">{entry.subtitle}</p>
                        )}
                      </div>
                    </div>
                    <span className="shrink-0 text-[11px] text-slate-500">
                      {formatTimestamp(entry.timestamp)}
                    </span>
                  </div>
                  {entry.description && (
                    <p className="mt-3 whitespace-pre-wrap text-[13px] leading-relaxed text-slate-600 dark:text-slate-300">
                      {entry.type === 'chat' ? entry.description : entry.description}
                    </p>
                  )}
                  {entry.metrics && Object.keys(entry.metrics).length > 0 && (
                    <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                  {Object.entries(entry.metrics).map(([key, value]) => (
                    <div
                      key={key}
                      className="rounded-lg border border-slate-200 bg-white/80 px-3 py-2 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-950/40"
                    >
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-slate-500 dark:text-slate-400">{key}</span>
                        <span className="font-semibold text-slate-700 dark:text-slate-100">
                          {typeof value === 'number' ? value : value}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </article>
              );
            })
          )}
        </div>
      </div>

      <div className="border-t border-slate-200 bg-white/90 px-5 py-4 text-xs text-slate-500 dark:border-slate-900 dark:bg-slate-950/95 dark:text-slate-400">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => {
              const blob = new Blob([exportHistory('chat', 'json')], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `chat-history-${Date.now()}.json`;
              link.click();
              URL.revokeObjectURL(url);
            }}
            className="rounded-lg border border-slate-200 bg-white/80 px-3 py-1.5 text-slate-600 shadow-sm shadow-slate-900/5 transition-colors hover:border-blue-500 hover:text-blue-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:shadow-none dark:hover:text-white"
          >
            Экспорт чатов
          </button>
          <button
            type="button"
            onClick={() => {
              const blob = new Blob([exportHistory('evaluation', 'json')], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `evaluation-history-${Date.now()}.json`;
              link.click();
              URL.revokeObjectURL(url);
            }}
            className="rounded-lg border border-slate-200 bg-white/80 px-3 py-1.5 text-slate-600 shadow-sm shadow-slate-900/5 transition-colors hover:border-blue-500 hover:text-blue-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:shadow-none dark:hover:text-white"
          >
            Экспорт тестов
          </button>
          <button
            type="button"
            onClick={() => {
              if (window.confirm('Очистить все записи истории?')) {
                clearHistory('all');
              }
            }}
            className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-1.5 text-rose-500 shadow-sm shadow-rose-200/40 transition-colors hover:border-rose-400 hover:text-rose-600 dark:border-rose-500/40 dark:bg-rose-500/10 dark:text-rose-300 dark:shadow-none"
          >
            Очистить всё
          </button>
        </div>
      </div>
    </div>
  );
};

export default HistorySidebar;
