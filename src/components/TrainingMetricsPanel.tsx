import { useMemo } from 'react';
import { XCircle } from 'lucide-react';
import type { FineTuneJob } from '../services/fineTune';
import type { TrainingMetricPoint } from '../types/training';

interface TrainingMetricsPanelProps {
  history: TrainingMetricPoint[];
  job: FineTuneJob | null;
  onClose?: () => void;
}

interface MetricSeries {
  key: string;
  label: string;
  color: string;
  lastValue: number | null;
  formattedLastValue: string;
  formattedDelta: string | null;
  trend: 'up' | 'down' | 'flat';
  minFormatted: string;
  maxFormatted: string;
  points: { x: number; y: number }[];
  epochs: number[];
  lastEpoch: number | null;
}

type MetricFormatter = (value: number) => string;

interface MetricMeta {
  label: string;
  color: string;
  formatter?: MetricFormatter;
  emphasis?: boolean;
  showDelta?: boolean;
}

const METRIC_META: Record<string, MetricMeta> = {
  loss: {
    label: 'Train Loss',
    color: '#2563eb',
    formatter: (value: number) => value.toFixed(3),
    emphasis: true,
  },
  train_loss: {
    label: 'Rolling Loss',
    color: '#1d4ed8',
    formatter: (value: number) => value.toFixed(3),
  },
  perplexity: {
    label: 'Perplexity',
    color: '#dc2626',
    formatter: (value: number) => (value > 999 ? value.toExponential(2) : value.toFixed(2)),
    emphasis: true,
  },
  learning_rate: {
    label: 'Learning Rate',
    color: '#16a34a',
    formatter: (value: number) => value.toExponential(2),
  },
  grad_norm: {
    label: 'Grad Norm',
    color: '#f97316',
    formatter: (value: number) => value.toFixed(3),
  },
  epoch: {
    label: 'Epoch',
    color: '#0ea5e9',
    formatter: (value: number) => value.toFixed(2),
    showDelta: false,
  },
  train_runtime: {
    label: 'Runtime (s)',
    color: '#2563eb',
    formatter: (value: number) => `${value.toFixed(1)} s`,
    showDelta: false,
  },
  train_samples_per_second: {
    label: 'Samples / s',
    color: '#10b981',
    formatter: (value: number) => value.toFixed(2),
    emphasis: true,
  },
  train_steps_per_second: {
    label: 'Steps / s',
    color: '#14b8a6',
    formatter: (value: number) => value.toFixed(2),
  },
  train_tokens_per_second: {
    label: 'Tokens / s',
    color: '#8b5cf6',
    formatter: (value: number) => (value > 9999 ? value.toExponential(2) : value.toFixed(0)),
  },
  eval_loss: {
    label: 'Eval Loss',
    color: '#a855f7',
    formatter: (value: number) => value.toFixed(3),
    emphasis: true,
  },
  eval_runtime: {
    label: 'Eval Runtime (s)',
    color: '#6366f1',
    formatter: (value: number) => `${value.toFixed(1)} s`,
    showDelta: false,
  },
  eval_samples_per_second: {
    label: 'Eval Samples / s',
    color: '#f59e0b',
    formatter: (value: number) => value.toFixed(2),
  },
};

const METRIC_ORDER = [
  'loss',
  'train_loss',
  'perplexity',
  'eval_loss',
  'learning_rate',
  'grad_norm',
  'train_samples_per_second',
  'train_steps_per_second',
  'train_tokens_per_second',
  'train_runtime',
  'epoch',
  'eval_runtime',
  'eval_samples_per_second',
];

const CHART_WIDTH = 260;
const CHART_HEIGHT = 120;
const PAD_X = 16;
const PAD_Y = 16;
const EPSILON = 1e-6;

const formatDefault = (value: number) => {
  if (!Number.isFinite(value)) {
    return '—';
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(2);
  }
  return value.toPrecision(3);
};

const getFormatter = (key: string): MetricFormatter => METRIC_META[key]?.formatter ?? formatDefault;

const formatDelta = (key: string, delta: number): string => {
  const formatter = getFormatter(key);
  const magnitude = formatter(Math.abs(delta));
  return `${delta >= 0 ? '+' : '-'}${magnitude}`;
};

function buildSeries(history: TrainingMetricPoint[], key: string): MetricSeries {
  const meta = METRIC_META[key];
  const label = meta?.label ?? key;
  const color = meta?.color ?? '#7c3aed';
  const formatter = getFormatter(key);

  const epochBuckets = new Map<number, number>();
  let inferredEpoch = 1;

  history.forEach((point) => {
    const rawValue = point.metrics[key];
    if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
      return;
    }

    const rawEpoch = typeof point.metrics.epoch === 'number' && Number.isFinite(point.metrics.epoch)
      ? point.metrics.epoch
      : null;
    const epochIndex = rawEpoch !== null
      ? Math.max(1, Math.floor(rawEpoch + EPSILON))
      : inferredEpoch;

    epochBuckets.set(epochIndex, rawValue);

    if (rawEpoch !== null) {
      inferredEpoch = Math.max(inferredEpoch, epochIndex);
    } else if (!meta?.showDelta) {
      // keep epoch steady for cumulative metrics when no epoch info
      inferredEpoch += 0;
    } else {
      inferredEpoch += 1;
    }
  });

  if (epochBuckets.size === 0) {
    return {
      key,
      label,
      color,
      lastValue: null,
      formattedLastValue: '—',
      formattedDelta: null,
      trend: 'flat',
      minFormatted: '—',
      maxFormatted: '—',
      epochs: [],
      lastEpoch: null,
      points: [],
    };
  }

  const samples = Array.from(epochBuckets.entries())
    .map(([epoch, value]) => ({ epoch, value }))
    .sort((a, b) => a.epoch - b.epoch);

  const values = samples.map((sample) => sample.value);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const range = rawMax - rawMin;
  const safeRange = range === 0 ? Math.max(Math.abs(rawMax) * 0.1, 1e-4) : range;
  const chartMin = rawMin - safeRange * 0.05;
  const chartMax = rawMax + safeRange * 0.05;
  const minEpoch = samples[0].epoch;
  const maxEpoch = samples[samples.length - 1].epoch;
  const epochRange = Math.max(maxEpoch - minEpoch, 1);
  const points = samples.map((sample) => {
    const progress = (sample.epoch - minEpoch) / (epochRange || 1);
    const clamped = (sample.value - chartMin) / (chartMax - chartMin || 1);
    const x = PAD_X + progress * (CHART_WIDTH - PAD_X * 2);
    const y = CHART_HEIGHT - PAD_Y - clamped * (CHART_HEIGHT - PAD_Y * 2);
    return { x, y };
  });

  const lastValue = samples[samples.length - 1].value;
  const lastEpoch = samples[samples.length - 1].epoch;
  const previousValue = samples.length > 1 ? samples[samples.length - 2].value : null;
  const deltaRaw = previousValue === null ? null : lastValue - previousValue;
  const showDelta = meta?.showDelta ?? true;
  const hasDelta = showDelta && deltaRaw !== null && Math.abs(deltaRaw) > EPSILON;
  const trend: 'up' | 'down' | 'flat' = hasDelta
    ? deltaRaw! > 0
      ? 'up'
      : 'down'
    : 'flat';

  return {
    key,
    label,
    color,
    lastValue,
    formattedLastValue: formatter(lastValue),
    formattedDelta: hasDelta ? formatDelta(key, deltaRaw!) : null,
    trend,
    minFormatted: formatter(rawMin),
    maxFormatted: formatter(rawMax),
    epochs: samples.map((sample) => sample.epoch),
    lastEpoch,
    points,
  };
}

function LineChart({ series }: { series: MetricSeries }) {
  if (series.points.length === 0) {
    return (
      <div className="h-28 flex items-center justify-center text-sm text-slate-500 bg-slate-950/40 border border-dashed border-slate-800 rounded-lg">
        Нет данных для визуализации
      </div>
    );
  }

  const pathPoints = series.points.map((point) => `${point.x},${point.y}`).join(' ');

  return (
    <svg width="100%" height={CHART_HEIGHT} viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} className="rounded-md bg-slate-900">
      <defs>
        <linearGradient id={`gradient-${series.key}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={series.color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={series.color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        fill="none"
        stroke={series.color}
        strokeWidth={2.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        points={pathPoints}
      />
      <polygon
        fill={`url(#gradient-${series.key})`}
        points={`${PAD_X},${CHART_HEIGHT - PAD_Y} ${pathPoints} ${CHART_WIDTH - PAD_X},${CHART_HEIGHT - PAD_Y}`}
        opacity={0.18}
      />
      {series.points.length > 0 && (
        <circle
          cx={series.points[series.points.length - 1].x}
          cy={series.points[series.points.length - 1].y}
          r={4}
          fill="#ffffff"
          stroke={series.color}
          strokeWidth={2}
        />
      )}
    </svg>
  );
}

export default function TrainingMetricsPanel({ history, job, onClose }: TrainingMetricsPanelProps) {
  const seriesList = useMemo(() => {
    if (!history.length) {
      return [] as MetricSeries[];
    }
    const keys = new Set<string>();
    history.forEach((point) => {
      Object.keys(point.metrics).forEach((key) => {
        if (typeof point.metrics[key] === 'number' && Number.isFinite(point.metrics[key] as number)) {
          keys.add(key);
        }
      });
    });

    const sortedKeys = Array.from(keys).sort((a, b) => {
      const indexA = METRIC_ORDER.indexOf(a);
      const indexB = METRIC_ORDER.indexOf(b);
      if (indexA === -1 && indexB === -1) {
        return a.localeCompare(b);
      }
      if (indexA === -1) {
        return 1;
      }
      if (indexB === -1) {
        return -1;
      }
      return indexA - indexB;
    });

    return sortedKeys.map((key) => buildSeries(history, key));
  }, [history]);

  const summaryMetrics = useMemo(
    () => seriesList.filter((series) => METRIC_META[series.key]?.emphasis && series.lastValue !== null).slice(0, 4),
    [seriesList],
  );

  const lastUpdate = history.length > 0 ? new Date(history[history.length - 1].timestamp).toLocaleTimeString() : null;
  const totalSamples = history.length;
  const hasMetrics = seriesList.length > 0;

  return (
    <div className="w-96 bg-slate-900 border-l border-slate-800 p-4 flex flex-col gap-4 overflow-hidden">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-lg font-semibold">Метрики обучения</h3>
          <div className="text-xs text-slate-400 mt-1">
            {totalSamples > 0
              ? `Обновлено ${totalSamples} раз${totalSamples === 1 ? '' : totalSamples < 5 ? 'а' : ''}`
              : 'Метрики появятся после первых шагов обучения'}
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="p-1 text-slate-500 hover:text-slate-300"
          aria-label="Закрыть панель метрик"
          title="Закрыть панель метрик"
        >
          <XCircle className="w-5 h-5" />
        </button>
      </div>

      {job && (
        <div className="text-xs text-slate-400 space-y-1">
          <div><span className="text-slate-200">Статус:</span> {job.status}</div>
          <div><span className="text-slate-200">Прогресс:</span> {Math.round(job.progress)}%</div>
          {lastUpdate && <div><span className="text-slate-200">Последнее обновление:</span> {lastUpdate}</div>}
        </div>
      )}

      {summaryMetrics.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium uppercase tracking-wide text-slate-400">Быстрый обзор</div>
          <div className="grid grid-cols-2 gap-2">
                {summaryMetrics.map((series) => (
                  <div
                    key={`${series.key}-summary`}
                    className="rounded-md border border-slate-800 bg-slate-900/60 px-2 py-2"
                    title={`Последняя эпоха: ${series.lastEpoch ?? '—'}`}
                  >
                    <div className="text-[11px] uppercase tracking-wide text-slate-400">{series.label}</div>
                    <div className="text-sm font-semibold text-slate-50" style={{ color: series.color }}>
                      {series.formattedLastValue}
                    </div>
                    {series.formattedDelta && (
                  <div className={`text-[11px] ${series.trend === 'down' ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {series.formattedDelta}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto pr-1">
        {hasMetrics ? (
          <div className="space-y-3 pb-2">
            {seriesList.map((series) => {
              const trendClass = series.trend === 'up'
                ? 'text-emerald-400'
                : series.trend === 'down'
                  ? 'text-rose-400'
                  : 'text-slate-500';
              return (
                <div
                  key={series.key}
                  className="border border-slate-800 rounded-lg p-3 shadow-md shadow-slate-950/20 bg-slate-900"
                  title={`Эпохи: ${series.epochs.join(', ')}`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-400">{series.label}</div>
                      <div className="text-lg font-semibold text-slate-50" style={{ color: series.color }}>
                        {series.formattedLastValue}
                      </div>
                      {series.lastEpoch !== null && (
                        <div className="text-[11px] text-slate-500">Эпоха {series.lastEpoch}</div>
                      )}
                      {series.formattedDelta && (
                        <div className={`text-xs ${trendClass}`}>
                          Δ {series.formattedDelta} за эпоху
                        </div>
                      )}
                    </div>
                    <div className="text-right text-[11px] text-slate-400 leading-tight">
                      <div>мин {series.minFormatted}</div>
                      <div>макс {series.maxFormatted}</div>
                    </div>
                  </div>
                  <div className="mt-3">
                    <LineChart series={series} />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="h-48 flex flex-col items-center justify-center text-sm text-slate-400 bg-slate-950/40 border border-dashed border-slate-800 rounded-lg">
            <div>Ждем первые метрики от тренировки…</div>
            <div className="text-xs text-slate-500 mt-1">Панель обновляется автоматически</div>
          </div>
        )}
      </div>
    </div>
  );
}
