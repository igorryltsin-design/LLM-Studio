import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { Send, Download, Trash2, Bot, User, Loader } from 'lucide-react';
import { useSettings } from '../contexts/SettingsContext';
import { useHistory } from '../contexts/HistoryContext';
import { useStatus } from '../contexts/StatusContext';
import type { ChatMessage } from '../contexts/HistoryContext';
import { callRemoteChat, RemoteChatError, RemoteChatMessage } from '../services/remoteChat';
import { callBaseChat, BaseChatError } from '../services/baseChat';

interface ModelResponse {
  model: string;
  content: string;
  tokens?: number;
  source?: 'base' | 'finetuned' | 'remote';
  error?: string;
}

const DEFAULT_SELECTED_MODELS = {
  base: true,
  finetuned: false,
  remote: false,
};

const MODEL_SELECTION_STORAGE_KEY = 'llm-studio-chat-selected-models';

type ModelKey = 'base' | 'finetuned' | 'remote';

type ModelSelectionMap = Record<ModelKey, boolean>;

const SOURCE_LABELS: Record<ModelKey, string> = {
  base: 'Базовая модель',
  finetuned: 'Дообученная',
  remote: 'Удалённая API',
};

const SOURCE_BADGE_CLASSES: Record<ModelKey, string> = {
  base: 'bg-slate-800 text-slate-200',
  finetuned: 'bg-emerald-900 text-emerald-200',
  remote: 'bg-sky-900 text-sky-200',
};

const SOURCE_ORDER: ModelKey[] = ['base', 'finetuned', 'remote'];

const getGridColumnsClass = (count: number) => {
  if (count >= 3) {
    return 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3';
  }
  if (count === 2) {
    return 'grid-cols-1 md:grid-cols-2';
  }
  return 'grid-cols-1';
};

interface SanitizeOptions {
  collapseUser?: boolean;
  collapseAssistant?: boolean;
}

const sanitizeConversation = (
  items: RemoteChatMessage[],
  options: SanitizeOptions = {},
): RemoteChatMessage[] => {
  const result: RemoteChatMessage[] = [];

  items.forEach((item) => {
    const content = typeof item.content === 'string' ? item.content.trim() : item.content;
    if (!content) {
      return;
    }

    const normalized: RemoteChatMessage = { ...item, content };

    const last = result[result.length - 1];
    const shouldCollapse =
      last &&
      last.role === normalized.role &&
      ((normalized.role === 'user' && options.collapseUser) ||
        (normalized.role === 'assistant' && options.collapseAssistant));

    if (shouldCollapse) {
      result[result.length - 1] = normalized;
    } else {
      result.push(normalized);
    }
  });

  if (result[0]?.role === 'assistant' && result.length > 1) {
    result.shift();
  }

  return result;
};

interface ConversationGroup {
  user?: ChatMessage;
  assistants: ChatMessage[];
}

const ChatTab = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModels, setSelectedModels] = useState<ModelSelectionMap>(() => {
    if (typeof window === 'undefined') {
      return { ...DEFAULT_SELECTED_MODELS };
    }

    try {
      const stored = localStorage.getItem(MODEL_SELECTION_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        return {
          ...DEFAULT_SELECTED_MODELS,
          ...parsed,
        };
      }
    } catch (error) {
      console.warn('Не удалось восстановить выбранные модели чата из localStorage:', error);
    }

    return { ...DEFAULT_SELECTED_MODELS };
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { settings } = useSettings();
  const { chatHistory, addChatMessage, exportHistory, clearHistory } = useHistory();
  const { setActivity, updateActivity, clearActivity } = useStatus();

  const conversationGroups = useMemo(() => {
    const groups: ConversationGroup[] = [];
    let currentGroup: ConversationGroup | null = null;

    messages.forEach((message) => {
      if (message.role === 'user') {
        currentGroup = { user: message, assistants: [] };
        groups.push(currentGroup);
        return;
      }

      if (!currentGroup) {
        currentGroup = { assistants: [message] };
        groups.push(currentGroup);
        return;
      }

      currentGroup.assistants.push(message);
    });

    return groups;
  }, [messages]);

  useEffect(() => {
    setMessages(chatHistory);
  }, [chatHistory]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      localStorage.setItem(MODEL_SELECTION_STORAGE_KEY, JSON.stringify(selectedModels));
    } catch (error) {
      console.warn('Не удалось сохранить выбранные модели чата:', error);
    }
  }, [selectedModels]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    addChatMessage(userMessage);
    setInput('');
    setIsLoading(true);

    const modelPromises: Promise<ModelResponse>[] = [];

    if (selectedModels.base) {
      const baseHistory = buildModelHistory(messages, userMessage, 'base');

      const basePromise = (async (): Promise<ModelResponse> => {
        try {
          const result = await callBaseChat(baseHistory, {
            serverUrl: settings.baseModelServerUrl,
            modelPath: settings.baseModelPath,
            maxTokens: settings.maxTokens,
            temperature: settings.temperature,
            topP: settings.topP,
            quantization: settings.quantization,
            device: settings.deviceType,
          });

          return {
            model: result.model ?? 'base',
            content: result.content,
            tokens: result.tokens,
            source: 'base',
          };
        } catch (error) {
          const message = error instanceof BaseChatError
            ? error.message
            : 'Неизвестная ошибка локального сервера';

          console.error('Base chat error:', error);

          return {
            model: 'base',
            content: `[Базовая модель] Ошибка: ${message}`,
            source: 'base',
            error: message,
          };
        }
      })();

      modelPromises.push(basePromise);
    }
    if (selectedModels.finetuned) {
      const fineTunedPath = settings.fineTunedModelPath.trim();
      const fineTunedMethod = settings.fineTunedMethod || 'lora';
      const baseForAdapter = (settings.fineTunedBaseModelPath || settings.baseModelPath).trim();

      if (!fineTunedPath) {
        modelPromises.push(Promise.resolve({
          model: 'finetuned',
          content: '[Дообученная модель] Путь до модели не указан в настройках',
          source: 'finetuned',
          error: 'Путь до модели не указан',
        }));
      } else if (fineTunedMethod === 'lora' && !baseForAdapter) {
        modelPromises.push(Promise.resolve({
          model: 'finetuned',
          content: '[Дообученная модель] Не указан путь к базовой модели для адаптера',
          source: 'finetuned',
          error: 'Не указан путь к базовой модели',
        }));
      } else {
        const finetuneHistory = buildModelHistory(messages, userMessage, 'finetuned');
        const adapterLabel = fineTunedPath.split(/[\\/]/).filter(Boolean).pop() ?? 'finetuned';

        const finetunePromise = (async (): Promise<ModelResponse> => {
          try {
            const result = await callBaseChat(
              finetuneHistory,
              fineTunedMethod === 'full'
                ? {
                    serverUrl: settings.baseModelServerUrl,
                    modelPath: fineTunedPath,
                    maxTokens: settings.maxTokens,
                    temperature: settings.temperature,
                    topP: settings.topP,
                    quantization: settings.quantization,
                    device: settings.deviceType,
                  }
                : {
                    serverUrl: settings.baseModelServerUrl,
                    modelPath: baseForAdapter,
                    adapterPath: fineTunedPath,
                    maxTokens: settings.maxTokens,
                    temperature: settings.temperature,
                    topP: settings.topP,
                    quantization: settings.quantization,
                    device: settings.deviceType,
                  },
            );

            return {
              model: result.model ?? adapterLabel,
              content: result.content,
              tokens: result.tokens,
              source: 'finetuned',
            };
          } catch (error) {
            const message = error instanceof BaseChatError
              ? error.message
              : 'Неизвестная ошибка при обращении к дообученной модели';

            console.error('Fine-tuned chat error:', error);

            return {
              model: 'finetuned',
              content: `[Дообученная модель] Ошибка: ${message}`,
              source: 'finetuned',
              error: message,
            };
          }
        })();

        modelPromises.push(finetunePromise);
      }
    }
    if (selectedModels.remote) {
      const remoteHistory = buildModelHistory(messages, userMessage, 'remote');

      const remotePromise = (async (): Promise<ModelResponse> => {
        try {
          const result = await callRemoteChat(remoteHistory, {
            apiUrl: settings.remoteApiUrl,
            apiKey: settings.remoteApiKey,
            maxTokens: settings.maxTokens,
            temperature: settings.temperature,
            topP: settings.topP,
            model: settings.remoteModelId,
          });

          return {
            model: result.model ?? 'remote',
            content: result.content,
            tokens: result.tokens,
            source: 'remote',
          };
        } catch (error) {
          const message = error instanceof RemoteChatError
            ? error.message
            : 'Неизвестная ошибка удалённого API';

          console.error('Remote chat error:', error);

          return {
            model: 'remote',
            content: `[Удалённая модель] Ошибка: ${message}`,
            source: 'remote',
            error: message,
          };
        }
      })();

      modelPromises.push(remotePromise);
    }

    if (modelPromises.length === 0) {
      setIsLoading(false);
      return;
    }

    setActivity('chat', { message: 'Формирует ответ' });

    let activityStatus: 'success' | 'error' | null = null;

    try {
      const responses = await Promise.all(modelPromises);
      
      responses.forEach(response => {
        const assistantMessage: ChatMessage = {
          id: Date.now().toString() + Math.random(),
          role: 'assistant',
          content: response.content,
          model: response.model,
          timestamp: Date.now(),
          tokens: response.tokens,
          source: response.source,
        };

        setMessages(prev => [...prev, assistantMessage]);
        addChatMessage(assistantMessage);
      });

      const hadErrors = responses.some(response => response.error);
      if (hadErrors) {
        updateActivity('chat', { status: 'error', message: 'Ответ получен с ошибками' });
        activityStatus = 'error';
      } else {
        updateActivity('chat', { status: 'success', message: 'Ответ готов' });
        activityStatus = 'success';
      }
    } catch (error) {
      console.error('Error getting model responses:', error);
      updateActivity('chat', { status: 'error', message: 'Ошибка генерации ответа' });
      activityStatus = 'error';
    } finally {
      setIsLoading(false);
      if (activityStatus) {
        const timeout = activityStatus === 'error' ? 5000 : 2000;
        window.setTimeout(() => {
          clearActivity('chat');
        }, timeout);
      } else {
        clearActivity('chat');
      }
    }
  };

  const buildModelHistory = (
    history: ChatMessage[],
    pendingMessage: ChatMessage,
    source: 'base' | 'finetuned' | 'remote',
  ): RemoteChatMessage[] => {
    const relevantMessages = [...history, pendingMessage]
      .filter(message => (
        message.role === 'user'
        || message.source === source
        || typeof message.source === 'undefined'
      ))
      .map(({ role, content }) => ({ role, content })) as RemoteChatMessage[];

    const sanitized = (() => {
      if (source === 'remote') {
        return sanitizeConversation(relevantMessages, { collapseUser: true, collapseAssistant: true });
      }
      if (source === 'base') {
        return sanitizeConversation(relevantMessages, { collapseAssistant: true });
      }
      return sanitizeConversation(relevantMessages, { collapseUser: true, collapseAssistant: true });
    })();

    if (source === 'base' && sanitized.length > 0) {
      const lastIndex = sanitized.length - 1;
      const lastMessage = sanitized[lastIndex];

      if (lastMessage.role === 'user') {
        const recentUserMessages = sanitized
          .slice(Math.max(0, lastIndex - 6), lastIndex)
          .filter(msg => msg.role === 'user');

        if (recentUserMessages.length > 0) {
          const contextSummary = recentUserMessages
            .map(msg => `- ${msg.content}`)
            .join('\n');

          sanitized[lastIndex] = {
            ...lastMessage,
            content: `${lastMessage.content}\n\nКонтекст последних реплик пользователя:\n${contextSummary}`,
          };
        }
      }
    }

    return sanitized;
  };

  const handleExport = () => {
    const jsonData = exportHistory('chat', 'json');
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-history-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleClear = () => {
    clearHistory('chat');
    setMessages([]);
  };

  return (
    <div className="h-full flex flex-col bg-slate-900">
      {/* Chat Header */}
      <div className="border-b border-slate-800 p-4 bg-slate-900">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Чат с моделями</h2>
          <div className="flex gap-2">
            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              <Download className="w-4 h-4" />
              Экспорт
            </button>
            <button
              onClick={handleClear}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-rose-600 text-white rounded-lg hover:bg-rose-700 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Очистить
            </button>
          </div>
        </div>

        {/* Model Selection */}
        <div className="flex gap-4 text-slate-300">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={selectedModels.base}
              onChange={(e) => setSelectedModels(prev => ({ ...prev, base: e.target.checked }))}
              className="rounded border-slate-300 bg-white accent-blue-500 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950"
            />
            <span className="text-sm">Базовая модель</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={selectedModels.finetuned}
              onChange={(e) => setSelectedModels(prev => ({ ...prev, finetuned: e.target.checked }))}
              className="rounded border-slate-300 bg-white accent-blue-500 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950"
            />
            <span className="text-sm">Дообученная</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={selectedModels.remote}
              onChange={(e) => setSelectedModels(prev => ({ ...prev, remote: e.target.checked }))}
              className="rounded border-slate-300 bg-white accent-blue-500 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950"
            />
            <span className="text-sm">Удалённая API</span>
          </label>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {conversationGroups.length === 0 && !isLoading && (
          <div className="pointer-events-none mx-auto mt-16 max-w-xl text-center text-slate-400">
            <div className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-600">
              Начните диалог
            </div>
            <h3 className="mt-4 text-2xl font-semibold text-slate-100">Задайте вопрос, чтобы сравнить модели</h3>
            <p className="mt-3 text-sm text-slate-400">
              Выберите базовую, дообученную или удалённую модель сверху, затем сформулируйте задачу. Ответы появятся рядом для удобного сравнения.
            </p>
            <div className="mt-6 flex flex-wrap justify-center gap-2 text-xs text-slate-500">
              <span className="rounded-full border border-slate-700/60 bg-slate-900/40 px-3 py-1">Например: Что такое ПВО?</span>
              <span className="rounded-full border border-slate-700/60 bg-slate-900/40 px-3 py-1">Что такое научная новизна</span>
  
            </div>
          </div>
        )}

        {conversationGroups.map((group, index) => {
          const groupKey = group.user?.id ?? group.assistants[0]?.id ?? `group-${index}`;
          const isLastGroup = index === conversationGroups.length - 1;

          const assistantBySource: Partial<Record<ModelKey, ChatMessage>> = {};
          const additionalAssistants: ChatMessage[] = [];

          group.assistants.forEach((assistant) => {
            if (assistant.source && SOURCE_ORDER.includes(assistant.source)) {
              if (!assistantBySource[assistant.source]) {
                assistantBySource[assistant.source] = assistant;
              } else {
                additionalAssistants.push(assistant);
              }
            } else {
              additionalAssistants.push(assistant);
            }
          });

          type AssistantCardDescriptor = {
            key: string;
            message: ChatMessage | null;
            source: ModelKey | null;
            state: 'ready' | 'pending';
          };

          const assistantCards: AssistantCardDescriptor[] = [
            ...SOURCE_ORDER.flatMap((source) => {
              const message = assistantBySource[source];
              return message
                ? [{
                    key: message.id,
                    message,
                    source,
                    state: 'ready' as const,
                  }]
                : [];
            }),
            ...additionalAssistants.map((message) => ({
              key: message.id,
              message,
              source: null,
              state: 'ready' as const,
            })),
          ];

          const activeSelectedSources = SOURCE_ORDER.filter((source) => selectedModels[source]);
          const pendingSources = isLastGroup && isLoading
            ? (activeSelectedSources.length > 0
              ? activeSelectedSources.filter((source) => !assistantBySource[source])
              : [])
            : [];

          pendingSources.forEach((source) => {
            assistantCards.push({
              key: `pending-${source}-${groupKey}`,
              message: null,
              source,
              state: 'pending',
            });
          });

          if (isLastGroup && isLoading && pendingSources.length === 0 && assistantCards.length === 0) {
            assistantCards.push({
              key: `pending-generic-${groupKey}`,
              message: null,
              source: null,
              state: 'pending',
            });
          }

          const gridClass = getGridColumnsClass(assistantCards.length || 1);

          return (
            <div key={groupKey} className="space-y-4">
              {group.user && (
                <div className="flex justify-end">
                  <div className="flex items-end gap-3">
                    <div className="max-w-2xl rounded-2xl bg-blue-500 px-4 py-3 text-white shadow-lg shadow-blue-900/30">
                      <div className="whitespace-pre-wrap text-sm leading-relaxed">{group.user.content}</div>
                      {group.user.timestamp && (
                        <div className="mt-2 text-right text-[11px] text-blue-100/80">
                          {new Date(group.user.timestamp).toLocaleTimeString()}
                        </div>
                      )}
                    </div>
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-700 text-white">
                      <User className="h-5 w-5" />
                    </div>
                  </div>
                </div>
              )}

              {assistantCards.length > 0 && (
                <div className={`grid ${gridClass} gap-4`}>
                  {assistantCards.map((card) => {
                    if (card.state === 'pending') {
                      const pendingLabel = card.source ? SOURCE_LABELS[card.source] : 'Модель';
                      return (
                        <div
                          key={card.key}
                          className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white/85 p-4 text-slate-600 shadow-sm shadow-slate-900/10 dark:border-slate-800 dark:bg-slate-950/60 dark:text-slate-300"
                        >
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-500/60 text-white">
                            <Loader className="h-5 w-5 animate-spin" />
                          </div>
                          <div>
                            <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">{pendingLabel}</div>
                            <div className="mt-1 text-sm text-slate-300">Генерирую ответ...</div>
                          </div>
                        </div>
                      );
                    }

                    const source = card.message?.source ?? null;
                    const badgeLabel = source ? SOURCE_LABELS[source] : 'Ассистент';
                    const badgeClass = source
                      ? SOURCE_BADGE_CLASSES[source]
                      : 'border border-slate-700 bg-slate-900/60 text-slate-200';
                    const modelName = card.message?.model;
                    const timestamp = card.message?.timestamp
                      ? new Date(card.message.timestamp).toLocaleTimeString()
                      : null;

                    return (
                      <div
                        key={card.key}
                        className="rounded-2xl border border-slate-200 bg-white p-4 shadow-lg shadow-slate-900/10 transition-colors dark:border-slate-800 dark:bg-slate-950/70 dark:shadow-slate-950/25"
                      >
                        <div className="flex items-start gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-500 text-white">
                            <Bot className="h-5 w-5" />
                          </div>
                          <div className="flex-1">
                            {(badgeLabel || modelName) && (
                              <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                                {badgeLabel && (
                                  <span className={`px-2 py-0.5 rounded-full font-semibold uppercase tracking-wide ${badgeClass}`}>
                                    {badgeLabel}
                                  </span>
                                )}
                                {modelName && <span className="text-slate-400">{modelName}</span>}
                              </div>
                            )}
                            <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700 dark:text-slate-100">
                              {card.message?.content}
                            </div>
                            {(card.message?.tokens || timestamp) && (
                              <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-slate-500 dark:text-slate-500">
                                {card.message?.tokens && <span>{card.message.tokens} токенов</span>}
                                {timestamp && <span>{timestamp}</span>}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {isLastGroup && <div ref={messagesEndRef} />}
            </div>
          );
        })}
      </div>

      {/* Input Form */}
      <div className="border-t border-slate-800 p-4 bg-slate-900">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Введите ваше сообщение..."
            className="flex-1 rounded-lg border border-slate-300 bg-white px-4 py-2 text-slate-900 placeholder-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:placeholder-slate-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatTab;
