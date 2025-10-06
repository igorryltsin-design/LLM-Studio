import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  Save,
  RotateCcw,
  Folder,
  Key,
  Cpu,
  Palette,
  Sun,
  Moon,
  Info,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useSettings } from '../contexts/SettingsContext';
import type { Settings } from '../contexts/SettingsContext';
import { listAvailableFineTunes, type FineTunedModelInfo } from '../services/fineTune';
import type { SystemStats } from '../services/systemMonitor';

interface SettingsPanelProps {
  systemStats?: SystemStats;
}

interface SectionCardProps {
  icon: LucideIcon;
  title: string;
  subtitle?: string;
  actions?: JSX.Element | null;
  children: ReactNode;
}

const SectionCard = ({ icon: Icon, title, subtitle, actions = null, children }: SectionCardProps) => (
  <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-900/80 dark:shadow-slate-950/30">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-start gap-3">
        <span className="mt-1 flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-200">
          <Icon className="h-5 w-5" />
        </span>
        <div>
          <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">{title}</h3>
          {subtitle && <p className="text-sm text-slate-500 dark:text-slate-400">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
    <div className="mt-5 space-y-5 text-sm text-slate-700 dark:text-slate-200">{children}</div>
  </section>
);

const SectionHint = ({ tone = 'info', children }: { tone?: 'info' | 'warning'; children: ReactNode }) => {
  const toneClass = tone === 'warning'
    ? 'bg-amber-50 border-amber-300 text-amber-700 dark:bg-amber-500/10 dark:border-amber-400/40 dark:text-amber-100'
    : 'bg-sky-50 border-sky-300 text-sky-700 dark:bg-sky-500/10 dark:border-sky-400/40 dark:text-sky-100';
  return (
    <p className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-xs shadow-sm ${toneClass}`}>
      <Info className="mt-0.5 h-3.5 w-3.5 flex-none" />
      <span>{children}</span>
    </p>
  );
};

const SettingsPanel = ({ systemStats }: SettingsPanelProps) => {
  const { settings, updateSettings, resetSettings } = useSettings();
  const [fineTuneOptions, setFineTuneOptions] = useState<FineTunedModelInfo[]>([]);
  const [fineTuneError, setFineTuneError] = useState<string | null>(null);
  const [loadingFineTunes, setLoadingFineTunes] = useState(false);
  const [showAdvancedGeneration, setShowAdvancedGeneration] = useState(false);
  const [showRemoteSettings, setShowRemoteSettings] = useState(
    Boolean(settings.remoteApiUrl || settings.remoteApiKey || settings.remoteModelId),
  );
  const [showFineTuneDetails, setShowFineTuneDetails] = useState(Boolean(settings.fineTunedModelPath));

  const baseModelPathError = settings.baseModelPath.trim() === '';
  const serverUrlValue = settings.baseModelServerUrl.trim();
  const serverUrlError = Boolean(serverUrlValue) && !/^https?:\/\//i.test(serverUrlValue);

  useEffect(() => {
    let cancelled = false;
    const fetchFineTunes = async () => {
      if (!settings.baseModelServerUrl) {
        setFineTuneOptions([]);
        return;
      }
      setLoadingFineTunes(true);
      setFineTuneError(null);
      try {
        const items = await listAvailableFineTunes(settings.baseModelServerUrl);
        if (!cancelled) {
          setFineTuneOptions(items);
        }
      } catch (error) {
        if (!cancelled) {
          setFineTuneOptions([]);
          setFineTuneError(
            error instanceof Error
              ? error.message
              : 'Не удалось получить список дообученных моделей',
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingFineTunes(false);
        }
      }
    };

    void fetchFineTunes();

    return () => {
      cancelled = true;
    };
  }, [settings.baseModelServerUrl]);

  const selectedFineTune = useMemo(
    () => fineTuneOptions.find(option => option.path === settings.fineTunedModelPath) ?? null,
    [fineTuneOptions, settings.fineTunedModelPath],
  );

  useEffect(() => {
    if (!selectedFineTune) {
      return;
    }

    const updates: Partial<typeof settings> = {};

    if (
      selectedFineTune.base_model_path
      && selectedFineTune.base_model_path !== settings.fineTunedBaseModelPath
    ) {
      updates.fineTunedBaseModelPath = selectedFineTune.base_model_path;
    }

    const normalizedMethod = selectedFineTune.method === 'full' ? 'full' : 'lora';
    if (normalizedMethod !== settings.fineTunedMethod) {
      updates.fineTunedMethod = normalizedMethod;
    }

    if (Object.keys(updates).length > 0) {
      updateSettings(updates);
    }
  }, [selectedFineTune, settings.fineTunedBaseModelPath, settings.fineTunedMethod, updateSettings]);

  const handleFineTuneSelect = (path: string) => {
    if (!path) {
      updateSettings({ fineTunedModelPath: '', fineTunedBaseModelPath: '', fineTunedMethod: '' });
      return;
    }
    const matched = fineTuneOptions.find(option => option.path === path) ?? null;
    const nextMethod = matched
      ? matched.method === 'full'
        ? 'full'
        : 'lora'
      : settings.fineTunedMethod || 'lora';

    updateSettings({
      fineTunedModelPath: path,
      fineTunedBaseModelPath: matched?.base_model_path ?? (settings.fineTunedBaseModelPath || settings.baseModelPath),
      fineTunedMethod: nextMethod,
    });
  };

  const detectedBackend = systemStats?.gpuBackend ?? 'cpu';
  const quantizationSupported = detectedBackend === 'cuda';

  const availableDevices = useMemo(() => ([
    { value: 'cpu', label: 'CPU', disabled: false },
    { value: 'cuda', label: 'CUDA (GPU)', disabled: detectedBackend !== 'cuda' },
    { value: 'mps', label: 'MPS (Apple Silicon)', disabled: detectedBackend !== 'mps' },
  ]), [detectedBackend]);

  const quantizationOptions = useMemo(() => ([
    { value: 'none', label: 'Без квантизации', disabled: false },
    { value: '8bit', label: '8-bit', disabled: !quantizationSupported },
    { value: '4bit', label: '4-bit', disabled: !quantizationSupported },
  ]), [quantizationSupported]);

  const themeOptions = [
    {
      value: 'light' as const,
      label: 'Светлая',
      description: 'Мягкий светлый интерфейс для дневной работы',
      icon: Sun,
    },
    {
      value: 'dark' as const,
      label: 'Тёмная',
      description: 'Контрастная тёмная палитра для вечерних сессий',
      icon: Moon,
    },
  ];

  useEffect(() => {
    if (!quantizationSupported && settings.quantization !== 'none') {
      updateSettings({ quantization: 'none' });
    }
  }, [quantizationSupported, settings.quantization, updateSettings]);

  useEffect(() => {
    setShowFineTuneDetails(Boolean(settings.fineTunedModelPath));
  }, [settings.fineTunedModelPath]);

  useEffect(() => {
    if (settings.remoteApiUrl || settings.remoteApiKey || settings.remoteModelId) {
      setShowRemoteSettings(true);
    }
  }, [settings.remoteApiKey, settings.remoteApiUrl, settings.remoteModelId]);

  useEffect(() => {
    const deviceAvailable = availableDevices.some(option => option.value === settings.deviceType && !option.disabled);
    if (!deviceAvailable && detectedBackend) {
      updateSettings({ deviceType: detectedBackend });
    }
  }, [availableDevices, detectedBackend, settings.deviceType, updateSettings]);

  const handleSave = () => {
    alert('Настройки сохранены');
  };

  const handleReset = () => {
    if (confirm('Вы уверены, что хотите сбросить все настройки?')) {
      resetSettings();
    }
  };

  return (
    <div className="h-full overflow-y-auto bg-slate-950/40 p-6">
      <div className="mx-auto max-w-4xl space-y-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">Настройки</h2>
            <p className="text-sm text-slate-400">Соберите рабочее окружение под вашу инфраструктуру</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleReset}
              className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-slate-200 transition-colors hover:border-slate-500 hover:bg-slate-900"
              type="button"
            >
              <RotateCcw className="h-4 w-4" />
              Сбросить
            </button>
            <button
              onClick={handleSave}
              className="flex items-center gap-2 rounded-lg bg-blue-500 px-4 py-2 text-white shadow-lg shadow-blue-900/40 transition-colors hover:bg-blue-600"
              type="button"
            >
              <Save className="h-4 w-4" />
              Сохранить
            </button>
          </div>
        </div>

        <SectionCard
          icon={Folder}
          title="Рабочие модели"
          subtitle="Укажите базовую модель, сервер и подключение адаптеров"
        >
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Каталог базовой модели</label>
                <input
                  type="text"
                  value={settings.baseModelPath}
                  onChange={(e) => updateSettings({ baseModelPath: e.target.value })}
                  placeholder="models/gemma"
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                {baseModelPathError && (
                  <div className="mt-2">
                    <SectionHint tone="warning">Укажите путь к директории модели. Без него локальный сервер не сможет загрузить веса.</SectionHint>
                  </div>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Адрес локального сервера</label>
                <input
                  type="text"
                  value={settings.baseModelServerUrl}
                  onChange={(e) => updateSettings({ baseModelServerUrl: e.target.value })}
                  placeholder="http://127.0.0.1:8001"
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                {serverUrlError && (
                  <div className="mt-2">
                    <SectionHint tone="warning">URL должен начинаться с http:// или https://. Проверьте настройки прокси и порт.</SectionHint>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Выберите дообученную модель</label>
                <select
                  value={selectedFineTune ? selectedFineTune.path : ''}
                  onChange={(e) => handleFineTuneSelect(e.target.value)}
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                >
                  <option value="">Без адаптера</option>
                  {fineTuneOptions.map(option => (
                    <option key={option.path} value={option.path}>
                      {option.name || option.path}
                    </option>
                  ))}
                </select>
                <div className="mt-2 space-y-2">
                  {loadingFineTunes && <SectionHint>Загружаю список дообученных моделей…</SectionHint>}
                  {fineTuneError && <SectionHint tone="warning">{fineTuneError}</SectionHint>}
                </div>
                <button
                  type="button"
                  onClick={() => setShowFineTuneDetails((prev) => !prev)}
                  className="mt-3 text-xs font-semibold text-blue-400 hover:text-blue-300"
                >
                  {showFineTuneDetails ? 'Скрыть параметры адаптера' : 'Расширенные параметры адаптера'}
                </button>
              </div>

              {showFineTuneDetails && (
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                  <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Путь адаптера</label>
                  <input
                    type="text"
                    value={settings.fineTunedModelPath}
                    onChange={(e) => handleFineTuneSelect(e.target.value)}
                    placeholder="/path/to/finetune"
                    className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                  <div className="mt-3 space-y-1 text-xs text-slate-400">
                    {settings.fineTunedBaseModelPath && (
                      <p>Базовая модель: <span className="font-mono break-all text-slate-200">{settings.fineTunedBaseModelPath}</span></p>
                    )}
                    {settings.fineTunedMethod && (
                      <p>Режим: {settings.fineTunedMethod === 'full' ? 'полное дообучение' : 'LoRA адаптер'}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          icon={Cpu}
          title="Устройство и генерация"
          subtitle="Настройте вычислительный бэкенд и параметры вывода"
          actions={(
            <button
              type="button"
              onClick={() => setShowAdvancedGeneration(prev => !prev)}
              className="text-xs font-semibold text-blue-400 hover:text-blue-300"
            >
              {showAdvancedGeneration ? 'Скрыть расширенные параметры' : 'Показать расширенные параметры'}
            </button>
          )}
        >
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Устройство</label>
                <select
                  value={settings.deviceType}
                  onChange={(e) => {
                    const nextDevice = e.target.value as Settings['deviceType'];
                    updateSettings({ deviceType: nextDevice });
                  }}
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                >
                  {availableDevices.map(option => (
                    <option key={option.value} value={option.value} disabled={option.disabled}>
                      {option.label}
                    </option>
                  ))}
                </select>
                {detectedBackend !== settings.deviceType && (
                  <div className="mt-2">
                    <SectionHint tone="warning">Обнаружено устройство {detectedBackend.toUpperCase()}. Переключитесь, чтобы использовать его возможности.</SectionHint>
                  </div>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Квантизация</label>
                <select
                  value={quantizationSupported ? settings.quantization : 'none'}
                  onChange={(e) => {
                    const nextQuantization = e.target.value as Settings['quantization'];
                    updateSettings({ quantization: nextQuantization });
                  }}
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                >
                  {quantizationOptions.map(option => (
                    <option key={option.value} value={option.value} disabled={option.disabled}>
                      {option.label}
                    </option>
                  ))}
                </select>
                {!quantizationSupported && (
                  <div className="mt-2">
                    <SectionHint>Квантизация 4/8-bit активируется автоматически, когда обнаружен CUDA-бэкенд.</SectionHint>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Максимум токенов</label>
                <div className="mt-2 flex items-center gap-3">
                  <input
                    type="range"
                    min="128"
                    max="4096"
                    step="128"
                    value={settings.maxTokens}
                    onChange={(e) => updateSettings({ maxTokens: parseInt(e.target.value, 10) })}
                    className="flex-1"
                  />
                  <span className="w-14 text-right text-xs font-semibold text-slate-200">{settings.maxTokens}</span>
                </div>
              </div>

              {showAdvancedGeneration && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Temperature</label>
                    <div className="mt-2 flex items-center gap-3">
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={settings.temperature}
                        onChange={(e) => updateSettings({ temperature: parseFloat(e.target.value) })}
                        className="flex-1"
                      />
                      <span className="w-12 text-right text-xs font-semibold text-slate-200">{settings.temperature.toFixed(1)}</span>
                    </div>
                    <p className="mt-2 text-xs text-slate-400">Для чувствительных сценариев держите значение 0–0.3, чтобы избежать шумных ответов.</p>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Top P</label>
                    <div className="mt-2 flex items-center gap-3">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={settings.topP}
                        onChange={(e) => updateSettings({ topP: parseFloat(e.target.value) })}
                        className="flex-1"
                      />
                      <span className="w-12 text-right text-xs font-semibold text-slate-200">{settings.topP.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          icon={Key}
          title="Удалённый API"
          subtitle="Подключите сторонний провайдер при необходимости"
          actions={(
            <button
              type="button"
              onClick={() => setShowRemoteSettings(prev => !prev)}
              className="text-xs font-semibold text-blue-400 hover:text-blue-300"
            >
              {showRemoteSettings ? 'Скрыть' : 'Настроить'}
            </button>
          )}
        >
          {showRemoteSettings ? (
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">URL API</label>
                <input
                  type="text"
                  value={settings.remoteApiUrl}
                  onChange={(e) => updateSettings({ remoteApiUrl: e.target.value })}
                  placeholder="http://127.0.0.1:1234/v1"
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Модель</label>
                <input
                  type="text"
                  value={settings.remoteModelId}
                  onChange={(e) => updateSettings({ remoteModelId: e.target.value })}
                  placeholder="google/gemma-3n-e4b"
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
              </div>
              <div className="lg:col-span-3">
                <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">API ключ</label>
                <input
                  type="password"
                  value={settings.remoteApiKey}
                  onChange={(e) => updateSettings({ remoteApiKey: e.target.value })}
                  placeholder="sk-..."
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                <p className="mt-2 text-xs text-slate-400">Ключ хранится локально в браузере и используется только для запросов к вашему провайдеру.</p>
              </div>
            </div>
          ) : (
            <SectionHint>Добавьте URL и ключ, если хотите подключить внешнее API вместо локального сервера.</SectionHint>
          )}
        </SectionCard>

        <SectionCard icon={Palette} title="Интерфейс" subtitle="Выберите тему и управление историей">
          <div className="space-y-5">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Тема</label>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                {themeOptions.map((option) => {
                  const ThemeIcon = option.icon;
                  const isActive = settings.theme === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => updateSettings({ theme: option.value })}
                      className={`flex items-start gap-3 rounded-xl border px-4 py-3 text-left transition-all ${
                        isActive
                          ? 'border-blue-500 bg-blue-500/10 shadow-md shadow-blue-900/40'
                          : 'border-slate-700 hover:border-blue-400 hover:bg-slate-900/40'
                      }`}
                      aria-pressed={isActive}
                    >
                      <span
                        className={`mt-1 flex h-10 w-10 items-center justify-center rounded-full ${
                          isActive ? 'bg-blue-500 text-white' : 'bg-slate-800 text-slate-300'
                        }`}
                      >
                        <ThemeIcon className="h-5 w-5" />
                      </span>
                      <span className="flex-1">
                        <span className="block font-semibold text-slate-100">{option.label}</span>
                        <span className="text-sm text-slate-400">{option.description}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="autoSave"
                checked={settings.autoSave}
                onChange={(e) => updateSettings({ autoSave: e.target.checked })}
                className="h-4 w-4 rounded border-slate-700 bg-slate-950 text-blue-500 focus:ring-blue-500"
              />
              <label htmlFor="autoSave" className="text-sm font-medium text-slate-200">
                Автоматически сохранять изменения
              </label>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-slate-400">Максимум записей в истории</label>
              <div className="mt-2 flex items-center gap-3">
                <input
                  type="range"
                  min="50"
                  max="1000"
                  step="50"
                  value={settings.maxHistoryItems}
                  onChange={(e) => updateSettings({ maxHistoryItems: parseInt(e.target.value, 10) })}
                  className="flex-1"
                />
                <span className="w-14 text-right text-xs font-semibold text-slate-200">{settings.maxHistoryItems}</span>
              </div>
            </div>
          </div>
        </SectionCard>

      </div>
    </div>
  );
};

export default SettingsPanel;
