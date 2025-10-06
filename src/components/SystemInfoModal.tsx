import { useEffect, useMemo } from 'react';
import { X } from 'lucide-react';
import { useSettings } from '../contexts/SettingsContext';
import { useHistory } from '../contexts/HistoryContext';
import type { SystemStats } from '../services/systemMonitor';
import { APP_VERSION } from '../version';

interface SystemInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  systemStats: SystemStats;
}

const formatPercent = (value: number | null) => {
  if (value === null || Number.isNaN(value)) {
    return '—';
  }
  return `${Math.round(value)}%`;
};

const formatMemory = (used: number | null, total: number | null) => {
  if (used === null || total === null || Number.isNaN(used) || Number.isNaN(total)) {
    return '—';
  }
  return `${used.toFixed(1)} / ${total.toFixed(1)} GB`;
};

const formatMemoryWithPercent = (used: number | null, total: number | null, percent: number | null) => {
  const base = formatMemory(used, total);
  if (base === '—') {
    return base;
  }
  const percentLabel = formatPercent(percent);
  return percentLabel === '—' ? base : `${base} (${percentLabel})`;
};

const formatLatency = (latencyMs: number | null) => {
  if (latencyMs === null || Number.isNaN(latencyMs)) {
    return '—';
  }
  return `${Math.round(latencyMs)} мс`;
};

const formatTimestamp = (value: number | null) => {
  if (value === null || Number.isNaN(value)) {
    return '—';
  }

  const normalized = value < 1e12 ? value * 1000 : value;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime()) || date.getFullYear() < 2000) {
    return '—';
  }

  const time = date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  const day = date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });

  return `${time} (${day})`;
};

const SystemInfoModal = ({ isOpen, onClose, systemStats }: SystemInfoModalProps) => {
  const { settings } = useSettings();
  const { trainingHistory } = useHistory();

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const clientPlatform = useMemo(() => {
    if (typeof navigator === 'undefined') {
      return 'Web Browser';
    }
    const nav = navigator as Navigator & { userAgentData?: { platform?: string } };
    return nav.userAgentData?.platform || nav.platform || 'Web Browser';
  }, []);

  const trainingStats = useMemo(() => {
    let completed = 0;
    let failed = 0;
    let running = 0;

    trainingHistory.forEach(session => {
      if (session.status === 'completed') {
        completed += 1;
      } else if (session.status === 'failed') {
        failed += 1;
      } else if (session.status === 'running') {
        running += 1;
      }
    });

    return {
      completed,
      failed,
      running,
      total: trainingHistory.length,
    };
  }, [trainingHistory]);

  if (!isOpen) {
    return null;
  }

  const gpuBackendLabel: Record<SystemStats['gpuBackend'], string> = {
    cpu: 'CPU',
    cuda: 'CUDA',
    mps: 'MPS',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-modal="true"
        className="relative z-10 w-full max-w-5xl overflow-hidden rounded-3xl border border-slate-200 bg-white/95 shadow-2xl shadow-slate-900/40 backdrop-blur dark:border-slate-800 dark:bg-slate-900"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-200 px-6 py-4 dark:border-slate-800">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400 dark:text-slate-500">Сводка системы</p>
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Текущая конфигурация</h2>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Данные обновлены {formatTimestamp(systemStats.timestamp)}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 text-slate-500 transition-colors hover:border-blue-500 hover:text-blue-500 dark:border-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
            aria-label="Закрыть системную информацию"
          >
            <X className="h-5 w-5" />
          </button>
        </header>

        {systemStats.message && (
          <div className="border-b border-amber-200 bg-amber-50/70 px-6 py-3 text-sm text-amber-700 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
            {systemStats.message}
          </div>
        )}

        <div className="grid gap-6 px-6 py-6 lg:grid-cols-2">
          <section className="space-y-3 rounded-2xl border border-slate-200/70 bg-slate-50/80 p-5 text-sm text-slate-600 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-300">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Среда</h3>
            <dl className="space-y-2">
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Платформа</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{clientPlatform}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Версия интерфейса</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{APP_VERSION}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Тема</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {settings.theme === 'light' ? 'Светлая' : 'Тёмная'}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Автосохранение</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{settings.autoSave ? 'Да' : 'Нет'}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Сервер</dt>
                <dd className="max-w-[260px] text-right font-semibold text-slate-900 dark:text-slate-100 break-words">
                  {settings.baseModelServerUrl || 'Не указан'}
                </dd>
              </div>
            </dl>
          </section>

          <section className="space-y-3 rounded-2xl border border-slate-200/70 bg-slate-50/80 p-5 text-sm text-slate-600 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-300">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Производительность</h3>
            <dl className="space-y-2">
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">CPU (система)</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {formatPercent(systemStats.cpuPercent ?? null)}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">CPU (приложение)</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {formatPercent(systemStats.cpuProcessPercent ?? null)}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">RAM</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {formatMemoryWithPercent(systemStats.ramUsedGb ?? null, systemStats.ramTotalGb ?? null, systemStats.ramPercent ?? null)}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Папка моделей</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {systemStats.modelsDirGb !== null && systemStats.modelsDirGb !== undefined
                    ? `${systemStats.modelsDirGb.toFixed(2)} GB`
                    : '—'}
                </dd>
              </div>
            </dl>
          </section>

          <section className="space-y-3 rounded-2xl border border-slate-200/70 bg-slate-50/80 p-5 text-sm text-slate-600 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-300">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Графика и задержки</h3>
            <dl className="space-y-2">
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">GPU Backend</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {gpuBackendLabel[systemStats.gpuBackend]}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">GPU</dt>
                <dd className="max-w-[260px] text-right font-semibold text-slate-900 dark:text-slate-100 break-words">
                  {systemStats.gpuName ? systemStats.gpuName : systemStats.gpuBackend === 'cpu' ? '—' : 'Не определено'}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">GPU память</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {formatMemory(systemStats.gpuMemoryUsedGb ?? null, systemStats.gpuMemoryTotalGb ?? null)}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Задержка</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{formatLatency(systemStats.latencyMs)}</dd>
              </div>
            </dl>
          </section>

          <section className="space-y-3 rounded-2xl border border-slate-200/70 bg-slate-50/80 p-5 text-sm text-slate-600 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-300">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Обучение моделей</h3>
            <dl className="space-y-2">
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Успешно</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{trainingStats.completed}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">С ошибкой</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{trainingStats.failed}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">В процессе</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{trainingStats.running}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Всего</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{trainingStats.total}</dd>
              </div>
            </dl>
          </section>

          <section className="space-y-3 rounded-2xl border border-slate-200/70 bg-slate-50/80 p-5 text-sm text-slate-600 shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-300 lg:col-span-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Текущие настройки</h3>
            <dl className="grid gap-3 sm:grid-cols-2">
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Устройство</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {settings.deviceType.toUpperCase()}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Квантизация</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">
                  {settings.quantization === 'none' ? 'Отключена' : settings.quantization}
                </dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Макс. токены</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{settings.maxTokens}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-slate-500 dark:text-slate-400">Макс. история</dt>
                <dd className="text-right font-semibold text-slate-900 dark:text-slate-100">{settings.maxHistoryItems}</dd>
              </div>
              <div className="flex justify-between gap-4 sm:col-span-2">
                <dt className="text-slate-500 dark:text-slate-400">Адаптер</dt>
                <dd className="max-w-[420px] text-right font-semibold text-slate-900 dark:text-slate-100 break-words">
                  {settings.fineTunedModelPath || 'Не выбран'}
                </dd>
              </div>
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
};

export default SystemInfoModal;
