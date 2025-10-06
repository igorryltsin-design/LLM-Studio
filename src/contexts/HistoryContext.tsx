import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { EvaluationRunStatus } from '../types/evaluation';
import { useSettings } from './SettingsContext';

export interface ChatMessage {
  id: string;
  timestamp: number;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  tokens?: number;
  source?: 'base' | 'finetuned' | 'remote';
}

type TrainingMetrics = Record<string, number>;

interface TrainingSession {
  id: string;
  timestamp: number;
  modelName: string;
  datasetSize: number;
  status: 'completed' | 'failed' | 'running';
  metrics?: TrainingMetrics;
}

interface ProcessingJob {
  id: string;
  timestamp: number;
  fileName: string;
  qaGenerated: number;
  status: 'completed' | 'failed' | 'running' | 'cancelled';
}

export interface EvaluationSession {
  id: string;
  timestamp: number;
  datasetName: string;
  modelVariant: string;
  total: number;
  scored: number;
  overallScore: number | null;
  coverage: number;
  status: EvaluationRunStatus;
  baselineScore?: number | null;
  scoreDelta?: number | null;
  error?: string | null;
}

type PersistedEvaluationSession = Omit<EvaluationSession, 'status' | 'error'> & {
  status?: EvaluationRunStatus;
  error?: string | null;
};

const normalizeEvaluationSession = (session: PersistedEvaluationSession): EvaluationSession | null => {
  if (!session || typeof session !== 'object') {
    return null;
  }

  if (typeof session.id !== 'string' || typeof session.timestamp !== 'number') {
    return null;
  }

  const status: EvaluationRunStatus = session.status ?? 'completed';

  return {
    ...session,
    status,
    error: session.error ?? null,
  } satisfies EvaluationSession;
};

interface HistoryContextType {
  chatHistory: ChatMessage[];
  trainingHistory: TrainingSession[];
  processingHistory: ProcessingJob[];
  evaluationHistory: EvaluationSession[];
  addChatMessage: (message: ChatMessage) => void;
  addTrainingSession: (session: TrainingSession) => void;
  addProcessingJob: (job: ProcessingJob) => void;
  addEvaluationSession: (session: EvaluationSession) => void;
  clearHistory: (type: 'chat' | 'training' | 'processing' | 'evaluation' | 'all') => void;
  exportHistory: (type: 'chat' | 'training' | 'processing' | 'evaluation', format: 'json' | 'markdown') => string;
}

const HistoryContext = createContext<HistoryContextType | undefined>(undefined);
export function HistoryProvider({ children }: { children: ReactNode }) {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<TrainingSession[]>([]);
  const [processingHistory, setProcessingHistory] = useState<ProcessingJob[]>([]);
  const [evaluationHistory, setEvaluationHistory] = useState<EvaluationSession[]>([]);
  const {
    settings: { maxHistoryItems },
  } = useSettings();

  useEffect(() => {
    const savedChatHistory = localStorage.getItem('llm-studio-chat-history');
    const savedTrainingHistory = localStorage.getItem('llm-studio-training-history');
    const savedProcessingHistory = localStorage.getItem('llm-studio-processing-history');
    const savedEvaluationHistory = localStorage.getItem('llm-studio-evaluation-history');

    if (savedChatHistory) {
      try {
        const parsed = JSON.parse(savedChatHistory) as ChatMessage[];
        setChatHistory(parsed.slice(-maxHistoryItems));
      } catch (error) {
        console.error('Failed to parse chat history:', error);
      }
    }

    if (savedTrainingHistory) {
      try {
        const parsed = JSON.parse(savedTrainingHistory) as TrainingSession[];
        setTrainingHistory(parsed.slice(-maxHistoryItems));
      } catch (error) {
        console.error('Failed to parse training history:', error);
      }
    }

    if (savedProcessingHistory) {
      try {
        const parsed = JSON.parse(savedProcessingHistory) as ProcessingJob[];
        setProcessingHistory(parsed.slice(-maxHistoryItems));
      } catch (error) {
        console.error('Failed to parse processing history:', error);
      }
    }

    if (savedEvaluationHistory) {
      try {
        const parsed = JSON.parse(savedEvaluationHistory) as PersistedEvaluationSession[];
        const sanitized = parsed
          .map(normalizeEvaluationSession)
          .filter((entry): entry is EvaluationSession => entry !== null);
        setEvaluationHistory(sanitized.slice(-maxHistoryItems));
      } catch (error) {
        console.error('Failed to parse evaluation history:', error);
      }
    }
  }, [maxHistoryItems]);

  const addChatMessage = (message: ChatMessage) => {
    setChatHistory((prev) => {
      const updated = [...prev, message].slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-chat-history', JSON.stringify(updated));
      return updated;
    });
  };

  const addTrainingSession = (session: TrainingSession) => {
    setTrainingHistory((prev) => {
      const updated = [...prev, session].slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-training-history', JSON.stringify(updated));
      return updated;
    });
  };

  const addProcessingJob = (job: ProcessingJob) => {
    setProcessingHistory((prev) => {
      const updated = [...prev, job].slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-processing-history', JSON.stringify(updated));
      return updated;
    });
  };

  const addEvaluationSession = (session: EvaluationSession) => {
    setEvaluationHistory((prev) => {
      const filtered = prev.filter((entry) => entry.id !== session.id);
      const updated = [...filtered, session]
        .sort((a, b) => a.timestamp - b.timestamp)
        .slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-evaluation-history', JSON.stringify(updated));
      return updated;
    });
  };

  useEffect(() => {
    setChatHistory((prev) => {
      if (prev.length <= maxHistoryItems) {
        return prev;
      }
      const trimmed = prev.slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-chat-history', JSON.stringify(trimmed));
      return trimmed;
    });

    setTrainingHistory((prev) => {
      if (prev.length <= maxHistoryItems) {
        return prev;
      }
      const trimmed = prev.slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-training-history', JSON.stringify(trimmed));
      return trimmed;
    });

    setProcessingHistory((prev) => {
      if (prev.length <= maxHistoryItems) {
        return prev;
      }
      const trimmed = prev.slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-processing-history', JSON.stringify(trimmed));
      return trimmed;
    });

    setEvaluationHistory((prev) => {
      if (prev.length <= maxHistoryItems) {
        return prev;
      }
      const trimmed = prev.slice(-maxHistoryItems);
      localStorage.setItem('llm-studio-evaluation-history', JSON.stringify(trimmed));
      return trimmed;
    });
  }, [maxHistoryItems]);

  const clearHistory = (type: 'chat' | 'training' | 'processing' | 'evaluation' | 'all') => {
    if (type === 'chat' || type === 'all') {
      setChatHistory([]);
      localStorage.removeItem('llm-studio-chat-history');
    }
    if (type === 'training' || type === 'all') {
      setTrainingHistory([]);
      localStorage.removeItem('llm-studio-training-history');
    }
    if (type === 'processing' || type === 'all') {
      setProcessingHistory([]);
      localStorage.removeItem('llm-studio-processing-history');
    }
    if (type === 'evaluation' || type === 'all') {
      setEvaluationHistory([]);
      localStorage.removeItem('llm-studio-evaluation-history');
    }
  };

  const exportHistory = (
    type: 'chat' | 'training' | 'processing' | 'evaluation',
    format: 'json' | 'markdown',
  ) => {
    if (type === 'chat') {
      const data = chatHistory;
      if (format === 'json') {
        return JSON.stringify(data, null, 2);
      }

      const header = '# Chat History\n\n';
      const body = data
        .map((item) => {
          const timestamp = new Date(item.timestamp).toLocaleString();
          return `## ${timestamp}\n**${item.role}**: ${item.content}\n\n`;
        })
        .join('');
      return `${header}${body}`;
    }

    if (type === 'training') {
      const data = trainingHistory;
      if (format === 'json') {
        return JSON.stringify(data, null, 2);
      }

      const header = '# Training History\n\n';
      const body = data
        .map((item) => {
          const timestamp = new Date(item.timestamp).toLocaleString();
          const metricsEntries = item.metrics
            ? Object.entries(item.metrics)
                .map(([key, value]) => `  - ${key}: ${value}`)
                .join('\n')
            : '  - Метрики отсутствуют';

          return [
            `## ${timestamp}`,
            `- Модель: ${item.modelName}`,
            `- Датасет: ${item.datasetSize} примеров`,
            `- Статус: ${item.status}`,
            '- Метрики:',
            metricsEntries,
            '',
          ].join('\n');
        })
        .join('\n');

      return `${header}${body}`;
    }

    if (type === 'processing') {
      const data = processingHistory;
      if (format === 'json') {
        return JSON.stringify(data, null, 2);
      }

      const header = '# Processing History\n\n';
      const body = data
        .map((item) => {
          const timestamp = new Date(item.timestamp).toLocaleString();
          return [
            `## ${timestamp}`,
            `- Файл: ${item.fileName}`,
            `- Сгенерированные QA-пары: ${item.qaGenerated}`,
            `- Статус: ${item.status}`,
            '',
          ].join('\n');
        })
        .join('\n');

      return `${header}${body}`;
    }

    const evaluations = evaluationHistory;
    if (format === 'json') {
      return JSON.stringify(evaluations, null, 2);
    }

    const header = '# Evaluation History\n\n';
    const statusLabels: Record<EvaluationRunStatus, string> = {
      draft: 'Черновик',
      queued: 'В очереди',
      running: 'Выполняется',
      waiting_review: 'Ожидает проверки',
      completed: 'Завершён',
      failed: 'Ошибка',
    };

    const body = evaluations
      .map((item) => {
        const timestamp = new Date(item.timestamp).toLocaleString();
        const statusLabel = statusLabels[item.status] ?? item.status;
        const lines = [
          `## ${timestamp}`,
          `- Датасет: ${item.datasetName}`,
          `- Модель: ${item.modelVariant}`,
          `- Примеров: ${item.total}`,
          `- Оценено: ${item.scored}`,
          `- Итоговый скор: ${item.overallScore ?? '—'}`,
          `- Покрытие: ${(item.coverage * 100).toFixed(1)}%`,
          `- Статус: ${statusLabel}`,
        ];
        if (typeof item.baselineScore === 'number') {
          lines.push(`- Базовый скор: ${item.baselineScore}`);
        }
        if (typeof item.scoreDelta === 'number') {
          const delta = item.scoreDelta >= 0 ? `+${item.scoreDelta}` : String(item.scoreDelta);
          lines.push(`- Δ к базе: ${delta}`);
        }
        if (item.error) {
          lines.push(`- Ошибка: ${item.error}`);
        }
        lines.push('');
        return lines.join('\n');
      })
      .join('\n');

    return `${header}${body}`;
  };

  return (
    <HistoryContext.Provider value={{
      chatHistory,
      trainingHistory,
      processingHistory,
      evaluationHistory,
      addChatMessage,
      addTrainingSession,
      addProcessingJob,
      addEvaluationSession,
      clearHistory,
      exportHistory,
    }}>
      {children}
    </HistoryContext.Provider>
  );
}

export function useHistory() {
  const context = useContext(HistoryContext);
  if (context === undefined) {
    throw new Error('useHistory must be used within a HistoryProvider');
  }
  return context;
}
