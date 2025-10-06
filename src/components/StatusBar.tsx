import { useEffect, useState } from 'react';
import { Wifi, WifiOff, Activity, Clock, HardDrive, Timer, Info, HelpCircle } from 'lucide-react';
import { useSettings } from '../contexts/SettingsContext';
import { useStatus } from '../contexts/StatusContext';
import { openModelsDirectory } from '../services/systemMonitor';
import type { SystemStats } from '../services/systemMonitor';

interface StatusBarProps {
  systemStats: SystemStats;
  serverOnline: boolean;
  onOpenSystemInfo?: () => void;
  onOpenHelp?: () => void;
}

const StatusBar = ({ systemStats, serverOnline, onOpenSystemInfo, onOpenHelp }: StatusBarProps) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [isOnline, setIsOnline] = useState(serverOnline);
  const [isOpeningModelsDir, setIsOpeningModelsDir] = useState(false);
  const { settings } = useSettings();
  const { currentActivity } = useStatus();

  useEffect(() => {
    const timer = window.setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => {
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    setIsOnline(serverOnline);
  }, [serverOnline]);

  const getUsageClass = (percent: number | null) => {
    if (percent === null || Number.isNaN(percent)) {
      return 'text-slate-500';
    }
    if (percent < 60) {
      return 'text-emerald-400';
    }
    if (percent < 85) {
      return 'text-amber-400';
    }
    return 'text-rose-400';
  };

  const activityColor = currentActivity?.status === 'error'
    ? 'text-rose-400'
    : currentActivity?.status === 'success'
      ? 'text-emerald-400'
      : currentActivity?.status === 'warning'
        ? 'text-amber-400'
        : 'text-sky-400';
  const activityBarClass = currentActivity?.status === 'error'
    ? 'bg-rose-400'
    : currentActivity?.status === 'success'
      ? 'bg-emerald-400'
      : currentActivity?.status === 'warning'
        ? 'bg-amber-400'
        : 'bg-sky-400';
  const statusLabel = currentActivity?.message ?? 'Готов';
  const progressLabel =
    typeof currentActivity?.progress === 'number'
      ? `${Math.round(Math.max(0, Math.min(currentActivity.progress, 100)))}%`
      : null;
  const progressValue =
    typeof currentActivity?.progress === 'number'
      ? Math.max(0, Math.min(currentActivity.progress, 100))
      : null;

  const memoryLabel = systemStats.modelsDirGb !== null
    ? `${systemStats.modelsDirGb.toFixed(2)} GB`
    : 'Нет данных';

  const latencyMs = systemStats.latencyMs;
  const latencyLabel = latencyMs !== null ? `${Math.round(latencyMs)} мс` : '—';
  const latencyState = latencyMs === null
    ? 'idle'
    : latencyMs < 150
      ? 'good'
      : latencyMs < 350
        ? 'warn'
        : 'bad';
  const latencyDotClass =
    latencyState === 'good'
      ? 'bg-emerald-400'
      : latencyState === 'warn'
        ? 'bg-amber-400'
        : latencyState === 'bad'
          ? 'bg-rose-400'
          : 'bg-slate-600';
  const latencyTextClass =
    latencyState === 'good'
      ? 'text-emerald-600 dark:text-emerald-300'
      : latencyState === 'warn'
        ? 'text-amber-600 dark:text-amber-300'
        : latencyState === 'bad'
          ? 'text-rose-600 dark:text-rose-300'
          : 'text-slate-600 dark:text-slate-400';

  const chipBase = 'flex items-center gap-1 rounded-full border px-2 py-1 text-[11px] shadow-sm transition-colors text-slate-600 dark:text-slate-200';
  const chipLight = 'border-slate-200 bg-white/80 shadow-slate-900/5';
  const chipDark = 'dark:border-slate-700 dark:bg-slate-900/70 dark:shadow-none';
  const iconButton = 'flex h-7 w-7 items-center justify-center rounded-full border border-slate-200 bg-white/80 text-slate-500 transition-colors hover:border-blue-500 hover:text-blue-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:text-white';

  const formattedTime = currentTime.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });

  const handleOpenModelsDirectory = async () => {
    if (isOpeningModelsDir) {
      return;
    }
    if (!isOnline) {
      alert('Сервер локальной модели недоступен. Запустите его, чтобы открыть каталог моделей.');
      return;
    }
    if (!settings.baseModelServerUrl) {
      alert('Укажите URL сервера базовой модели на вкладке «Настройки».');
      return;
    }

    setIsOpeningModelsDir(true);
    try {
      await openModelsDirectory(settings.baseModelServerUrl);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Не удалось открыть каталог моделей';
      alert(message);
    } finally {
      setIsOpeningModelsDir(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <div className={`flex items-center gap-2 rounded-xl border px-3 py-1 shadow-sm ${chipLight} ${chipDark} min-w-[280px]`}>
        <Activity className={`h-3 w-3 ${activityColor}`} />
        <div className="flex items-center gap-2">
          <span className="text-slate-600 dark:text-slate-300">{statusLabel}</span>
          {progressLabel && <span className="text-slate-500 dark:text-slate-500">{progressLabel}</span>}
        </div>
        {progressValue !== null && (
          <div className="ml-2 h-1.5 w-80 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
            <div className={`h-full ${activityBarClass} transition-[width] duration-300 ease-out`} style={{ width: `${progressValue}%` }} />
          </div>
        )}
      </div>

      <div className={`${chipBase} ${chipLight} ${chipDark}`}>
        {isOnline ? (
          <Wifi className="h-3 w-3 text-emerald-400" />
        ) : (
          <WifiOff className="h-3 w-3 text-rose-400" />
        )}
        <span>{isOnline ? 'Сервер онлайн' : 'Сервер офлайн'}</span>
      </div>

      <div className={`${chipBase} ${chipLight} ${chipDark}`}>
        <span className={`flex h-2.5 w-2.5 items-center justify-center rounded-full ${latencyDotClass}`} />
        <Timer className={`h-3 w-3 ${latencyTextClass}`} />
        <span className="text-slate-500 dark:text-slate-400">Задержка</span>
        <span className={`min-w-[3.5rem] tabular-nums font-mono text-right ${latencyTextClass}`}>
          {latencyLabel}
        </span>
      </div>

      <button
        type="button"
        onClick={handleOpenModelsDirectory}
        className={`${chipBase} ${chipLight} ${chipDark} ${
          isOpeningModelsDir
            ? 'cursor-wait opacity-70'
            : isOnline
              ? 'cursor-pointer hover:border-blue-500 hover:text-blue-500'
              : 'cursor-pointer opacity-75'
        }`}
        title={isOnline ? 'Открыть каталог моделей' : 'Сервер недоступен — запустите его, чтобы открыть каталог моделей'}
        aria-label="Папка моделей"
      >
        <HardDrive className={`h-3 w-3 ${getUsageClass(systemStats.ramPercent)}`} />
        <span>Папка моделей: {memoryLabel}</span>
      </button>

      {systemStats.cpuProcessPercent !== null && (
        <div className={`${chipBase} ${chipLight} ${chipDark}`}>
          <Activity className="h-3 w-3 text-slate-500" />
          <span>Процессор: {Math.round(systemStats.cpuProcessPercent)}%</span>
        </div>
      )}

      <div className={`${chipBase} ${chipLight} ${chipDark}`}>
        <Clock className="h-3 w-3 text-slate-500" />
        <span className="min-w-[4.5rem] tabular-nums font-mono text-right text-slate-600 dark:text-slate-200">
          {formattedTime}
        </span>
      </div>

      {onOpenSystemInfo && (
        <button
          type="button"
          onClick={onOpenSystemInfo}
          className={iconButton}
          title="Информация о системе"
          aria-label="Информация о системе"
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      )}

      {onOpenHelp && (
        <button
          type="button"
          onClick={onOpenHelp}
          className={iconButton}
          title="Справка"
          aria-label="Справка"
        >
          <HelpCircle className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
};

export default StatusBar;
