import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  MessageSquare,
  Brain,
  FileText,
  Settings,
  History,
  Menu,
  ChevronLeft,
  ChevronRight,
  Lightbulb,
  Target,
} from 'lucide-react';
import ChatTab from './components/ChatTab';
import TrainingTab from './components/TrainingTab';
import FileProcessingTab from './components/FileProcessingTab';
import SettingsPanel from './components/SettingsPanel';
import StatusBar from './components/StatusBar';
import HistorySidebar from './components/HistorySidebar';
import EvaluationTab from './components/EvaluationTab';
import { SettingsProvider, useSettings } from './contexts/SettingsContext';
import { HistoryProvider, useHistory } from './contexts/HistoryContext';
import { StatusProvider } from './contexts/StatusContext';
import { FileProcessingProvider } from './contexts/FileProcessingContext';
import { EvaluationProvider } from './contexts/EvaluationContext';
import { TrainingProvider } from './contexts/TrainingContext';
import { fetchSystemStats, SystemStats } from './services/systemMonitor';
import { APP_VERSION } from './version';
import SystemInfoModal from './components/SystemInfoModal';
import HelpModal from './components/HelpModal';

type TabType = 'chat' | 'training' | 'evaluation' | 'files' | 'settings';

function AppContent() {
  const [activeTab, setActiveTab] = useState<TabType>('chat');
  const [showHistory, setShowHistory] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const { settings, updateSettings } = useSettings();
  const { chatHistory, processingHistory, trainingHistory, evaluationHistory } = useHistory();
  const [lastHistoryViewedAt, setLastHistoryViewedAt] = useState(() => {
    if (typeof window === 'undefined') {
      return Date.now();
    }
    const stored = window.localStorage.getItem('llm-studio-history-last-opened');
    const parsed = stored ? Number.parseInt(stored, 10) : NaN;
    return Number.isFinite(parsed) ? parsed : Date.now();
  });
  const handleCloseHistory = useCallback(() => {
    setShowHistory(false);
    setLastHistoryViewedAt(Date.now());
  }, [setShowHistory, setLastHistoryViewedAt]);

  const [systemStats, setSystemStats] = useState<SystemStats>({
    status: 'error',
    cpuPercent: null,
    cpuProcessPercent: null,
    ramPercent: null,
    ramUsedGb: null,
    ramTotalGb: null,
    modelsDirGb: null,
    gpuPercent: null,
    gpuMemoryUsedGb: null,
    gpuMemoryTotalGb: null,
    gpuName: null,
    gpuBackend: 'cpu',
    message: undefined,
    timestamp: null,
    latencyMs: null,
  });
  const [isSystemInfoOpen, setIsSystemInfoOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const tabs = [
    { id: 'chat' as TabType, icon: MessageSquare, label: 'Чат' },
    { id: 'training' as TabType, icon: Brain, label: 'Обучение' },
    { id: 'evaluation' as TabType, icon: Target, label: 'Тестирование' },
    { id: 'files' as TabType, icon: FileText, label: 'Датасеты' },
    { id: 'settings' as TabType, icon: Settings, label: 'Настройки' },
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatTab />;
      case 'training':
        return <TrainingTab systemStats={systemStats} />;
      case 'evaluation':
        return <EvaluationTab />;
      case 'files':
        return <FileProcessingTab />;
      case 'settings':
        return <SettingsPanel systemStats={systemStats} />;
      default:
        return <ChatTab />;
    }
  };

  useEffect(() => {
    let isMounted = true;
    let intervalId: number | null = null;

    const loadStats = async () => {
      const stats = await fetchSystemStats(settings.baseModelServerUrl);
      if (isMounted) {
        setSystemStats(stats);
      }
    };

    void loadStats();
    intervalId = window.setInterval(() => {
      void loadStats();
    }, 5000);

    return () => {
      isMounted = false;
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [settings.baseModelServerUrl]);

  useEffect(() => {
    if (!showHistory) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        handleCloseHistory();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showHistory, handleCloseHistory]);

  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }
    const isDark = settings.theme === 'dark';
    const root = document.documentElement;
    const { body } = document;

    root.classList.toggle('dark', isDark);
    body.classList.remove('theme-dark', 'theme-light');
    body.classList.add(isDark ? 'theme-dark' : 'theme-light');
    body.setAttribute('data-theme', settings.theme);
  }, [settings.theme]);

  const getUsageClass = (percent: number | null) => {
    if (percent === null || Number.isNaN(percent)) {
      return 'text-slate-600';
    }
    if (percent < 60) {
      return 'text-emerald-400';
    }
    if (percent < 85) {
      return 'text-amber-400';
    }
    return 'text-rose-400';
  };

  const formatPercent = (percent: number | null) => {
    if (percent === null || Number.isNaN(percent)) {
      return '—';
    }
    return `${Math.round(percent)}%`;
  };

  const formatMemory = (used: number | null, total: number | null) => {
    if (used === null || total === null) {
      return '';
    }
    return `${used.toFixed(1)} / ${total.toFixed(1)} GB`;
  };

  const historyCounts = useMemo(
    () => ({
      chat: chatHistory.length,
      training: trainingHistory.length,
      files: processingHistory.length,
      evaluation: evaluationHistory.length,
      total: chatHistory.length + trainingHistory.length + processingHistory.length + evaluationHistory.length,
    }),
    [chatHistory, trainingHistory, processingHistory, evaluationHistory],
  );

  const newHistoryCounts = useMemo(() => {
    const countNew = (items: { timestamp: number }[]) =>
      items.filter(item => item.timestamp > lastHistoryViewedAt).length;

    const chat = countNew(chatHistory);
    const training = countNew(trainingHistory);
    const files = countNew(processingHistory);
    const evaluation = countNew(evaluationHistory);

    return {
      chat,
      training,
      files,
      evaluation,
      total: chat + training + files + evaluation,
    };
  }, [chatHistory, trainingHistory, processingHistory, evaluationHistory, lastHistoryViewedAt]);

  const historySummaryItems = useMemo(
    () => [
      {
        id: 'chat',
        label: 'Чат',
        compactLabel: 'Чат',
        total: historyCounts.chat,
        fresh: newHistoryCounts.chat,
      },
      {
        id: 'training',
        label: 'Обучение',
        compactLabel: 'Обуч.',
        total: historyCounts.training,
        fresh: newHistoryCounts.training,
      },
      {
        id: 'evaluation',
        label: 'Тесты',
        compactLabel: 'Тесты',
        total: historyCounts.evaluation,
        fresh: newHistoryCounts.evaluation,
      },
      {
        id: 'files',
        label: 'Датасеты',
        compactLabel: 'Датасеты',
        total: historyCounts.files,
        fresh: newHistoryCounts.files,
      },
    ],
    [historyCounts, newHistoryCounts],
  );

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem('llm-studio-history-last-opened', String(lastHistoryViewedAt));
  }, [lastHistoryViewedAt]);

  const sidebarWidthClass = isSidebarCollapsed ? 'w-64 lg:w-20' : 'w-64';
  const shouldShowLabels = !isSidebarCollapsed || isSidebarOpen;
  const navButtonClasses = (isActive: boolean) =>
    `w-full flex items-center ${shouldShowLabels ? 'gap-3 px-4 justify-start' : 'justify-center px-2'} py-3 rounded-lg transition-all duration-200 ${
      isActive
        ? 'bg-blue-500 text-white shadow-lg shadow-blue-900/40'
        : 'text-slate-300 hover:bg-slate-900 hover:text-white'
    }`;

  const SidebarToggleIcon = isSidebarCollapsed ? ChevronRight : ChevronLeft;
  const handleToggleTheme = () => updateSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' });
  const handleOpenHistory = () => {
    setShowHistory(true);
    setIsSidebarOpen(false);
  };

  const sidebarClassName = `${sidebarWidthClass} bg-slate-950 text-slate-100 flex flex-col border-r border-slate-900 transition-all duration-300 fixed inset-y-0 z-30 transform ${
    isSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
  } lg:static lg:z-0`;

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100">
      {/* Sidebar */}
      <div className={sidebarClassName}>
        <div className="flex items-center justify-between gap-2 border-b border-slate-900 px-4 py-4">
          <div className={`flex items-center ${shouldShowLabels ? 'gap-3' : 'justify-center w-full'}`}>
            <Brain className="h-6 w-6 text-blue-400" />
            {shouldShowLabels && (
              <div>
                <h1 className="text-xl font-bold leading-tight">Obuchator</h1>
                <p className="text-[10px] leading-tight text-slate-500">Made by Ryltsin I.A.</p>
                <p className="text-xs text-slate-500">Версия {APP_VERSION}</p>
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => setIsSidebarCollapsed(prev => !prev)}
            className="hidden lg:flex h-9 w-9 items-center justify-center rounded-lg border border-slate-800 bg-slate-900 text-slate-300 hover:border-blue-500 hover:text-white"
            title={isSidebarCollapsed ? 'Раскрыть меню' : 'Свернуть меню'}
          >
            <SidebarToggleIcon className="h-4 w-4" />
          </button>
        </div>

        <nav className={`flex-1 ${shouldShowLabels ? 'px-4 py-4' : 'px-2 py-4'}`}>
          <ul className="space-y-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <li key={tab.id}>
                  <button
                    onClick={() => {
                      setActiveTab(tab.id);
                      setIsSidebarOpen(false);
                    }}
                    className={navButtonClasses(isActive)}
                  >
                    <Icon className="h-5 w-5" />
                    {shouldShowLabels && tab.label}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <div
          className={`border-t border-slate-900 ${shouldShowLabels ? 'px-4 py-3' : 'px-2 py-3'}`}
        >
          <button
            type="button"
            onClick={handleOpenHistory}
            className={`w-full rounded-xl border border-slate-200 bg-white/80 text-slate-800 shadow-sm shadow-slate-900/5 transition-colors hover:border-blue-500 hover:bg-white dark:border-slate-800 dark:bg-slate-900/80 dark:text-slate-100 dark:shadow-none ${
              shouldShowLabels ? 'px-4 py-3' : 'px-2 py-2'
            }`}
          >
            <div className={`flex items-center ${shouldShowLabels ? 'justify-between gap-3' : 'flex-col gap-3 text-center'}`}>
              <div className={`flex items-center ${shouldShowLabels ? 'gap-3' : 'flex-col gap-1 text-[11px] w-full'}`}>
                <History className="h-4 w-4 shrink-0 text-blue-500 dark:text-blue-400" />
                {shouldShowLabels ? (
                  <span className="text-sm font-semibold leading-snug text-slate-700 dark:text-slate-100">История</span>
                ) : (
                  <span className="w-full max-w-[60px] break-words text-center text-[11px] font-semibold leading-tight text-slate-700 dark:text-slate-100">
                    История
                  </span>
                )}
              </div>
              <div className={`flex items-center ${shouldShowLabels ? 'gap-2' : 'gap-1 text-[11px]'}`}>
                <span className="rounded-full bg-blue-500/15 px-2 py-0.5 text-[11px] font-semibold text-blue-600 dark:bg-blue-500/20 dark:text-blue-100">
                  {historyCounts.total}
                </span>
                {newHistoryCounts.total > 0 && (
                  <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-semibold text-emerald-300">
                    +{newHistoryCounts.total}
                  </span>
                )}
              </div>
            </div>
            {shouldShowLabels ? (
              <div className="mt-3 space-y-1 text-[10px] text-slate-500 dark:text-slate-400">
                {historySummaryItems.map(item => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between rounded-lg border border-slate-200 bg-white/70 px-2.5 py-1 shadow-sm shadow-slate-900/10 dark:border-slate-800/60 dark:bg-slate-950/40 dark:text-slate-400 dark:shadow-none"
                  >
                    <span className="font-medium text-slate-600 dark:text-slate-300">{item.label}</span>
                    <div className="flex items-center gap-1 text-slate-700 dark:text-slate-100">
                      <span className="text-[11px] font-semibold">{item.total}</span>
                      {item.fresh > 0 && (
                        <span className="text-[11px] font-semibold text-emerald-500 dark:text-emerald-300">+{item.fresh}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="mt-2 flex w-full justify-center">
                <span className="min-w-[48px] rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-center text-[11px] font-semibold text-slate-700 dark:border-slate-800 dark:bg-slate-900/70 dark:text-slate-100">
                  {historyCounts.total}
                </span>
              </div>
            )}
          </button>
        </div>

        {/* Resource Monitor */}
        <div className={`border-t border-slate-900 ${shouldShowLabels ? 'px-4 py-4' : 'px-2 py-4 text-center'}`}>
          {shouldShowLabels && <div className="mb-2 text-xs text-slate-500">Ресурсы</div>}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>CPU</span>
              <span className={getUsageClass(systemStats.cpuPercent)}>
                {formatPercent(systemStats.cpuPercent)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span>RAM</span>
              <span className={getUsageClass(systemStats.ramPercent)}>
                {formatPercent(systemStats.ramPercent)}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span>GPU</span>
              <span className={getUsageClass(systemStats.gpuPercent)}>
                {formatPercent(systemStats.gpuPercent)}
              </span>
            </div>
          </div>
          {systemStats.ramUsedGb !== null && systemStats.ramTotalGb !== null && (
            <div className="mt-3 text-[11px] text-slate-500">
              RAM: {formatMemory(systemStats.ramUsedGb, systemStats.ramTotalGb)}
            </div>
          )}
          {systemStats.message && (
            <div className="mt-3 text-[11px] text-amber-400">
              {systemStats.message}
            </div>
          )}
        </div>
      </div>

      {isSidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-slate-950/40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col bg-slate-900">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white/90 px-4 py-3 shadow-sm shadow-slate-900/5 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80 dark:shadow-none">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsSidebarOpen(prev => !prev)}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 shadow-sm shadow-slate-900/5 transition-colors hover:border-blue-500 hover:text-blue-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:shadow-none dark:hover:text-white lg:hidden"
              title="Меню"
            >
              <Menu className="h-5 w-5" />
            </button>
          </div>

          <div className="flex flex-1 items-center justify-end gap-3">
            <StatusBar
              systemStats={systemStats}
              serverOnline={systemStats.status === 'ok'}
              onOpenSystemInfo={() => setIsSystemInfoOpen(true)}
              onOpenHelp={() => setIsHelpOpen(true)}
            />
            <button
              type="button"
              className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white/80 text-slate-700 shadow-sm shadow-slate-900/5 transition-colors hover:border-blue-500 hover:text-blue-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:shadow-none"
              onClick={handleToggleTheme}
              title={settings.theme === 'dark' ? 'Переключить на светлую тему' : 'Переключить на тёмную тему'}
            >
              <Lightbulb className="h-4 w-4" />
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-hidden">
          {renderTabContent()}
        </main>
      </div>

      {/* History Sidebar */}
      <SystemInfoModal
        isOpen={isSystemInfoOpen}
        onClose={() => setIsSystemInfoOpen(false)}
        systemStats={systemStats}
      />
      <HelpModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
      />

      {showHistory && (
        <>
          <div
            className="fixed inset-0 bg-slate-950/40 backdrop-blur-sm"
            onClick={handleCloseHistory}
            aria-hidden="true"
          />
          <div className="fixed inset-y-0 right-0 z-40 w-full max-w-3xl shadow-2xl">
            <HistorySidebar onClose={handleCloseHistory} lastViewedAt={lastHistoryViewedAt} />
          </div>
        </>
      )}
    </div>
  );
}

function App() {
  return (
    <SettingsProvider>
      <HistoryProvider>
        <StatusProvider>
          <TrainingProvider>
            <EvaluationProvider>
              <FileProcessingProvider>
                <AppContent />
              </FileProcessingProvider>
            </EvaluationProvider>
          </TrainingProvider>
        </StatusProvider>
      </HistoryProvider>
    </SettingsProvider>
  );
}

export default App;
