export class RemoteChatError extends Error {
  public readonly details?: unknown;

  constructor(message: string, details?: unknown) {
    super(message);
    this.name = 'RemoteChatError';
    this.details = details;
  }
}

export interface RemoteChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface RemoteChatConfig {
  apiUrl: string;
  apiKey?: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  model?: string;
  signal?: AbortSignal;
}

export interface RemoteChatResult {
  content: string;
  model: string;
  tokens?: number;
  raw?: unknown;
}

interface RemoteUsage {
  total_tokens?: number;
}

interface RemoteChoice {
  message?: {
    content?: string;
  };
  text?: string;
}

interface RemotePayload {
  choices?: RemoteChoice[];
  content?: string;
  model?: string;
  tokens?: number;
  usage?: RemoteUsage;
  error?: unknown;
  message?: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const DEFAULT_REMOTE_MODEL = 'google/gemma-3n-e4b';

const needsChatCompletionsSuffix = (url: string) => {
  const sanitized = url.replace(/\/$/, '');
  return !/\/(chat\/)?completions$/i.test(sanitized);
};

const buildEndpoint = (apiUrl: string) => {
  if (!apiUrl) {
    throw new RemoteChatError('Не указан URL удалённого API');
  }

  const trimmed = apiUrl.trim();
  if (needsChatCompletionsSuffix(trimmed)) {
    return `${trimmed.replace(/\/$/, '')}/chat/completions`;
  }
  return trimmed;
};

const parseErrorMessage = (payload: unknown) => {
  if (!payload) return 'Неизвестная ошибка удалённого API';
  if (typeof payload === 'string') return payload;
  if (isRecord(payload) && 'error' in payload) {
    return parseErrorMessage(payload.error);
  }
  if (isRecord(payload) && typeof payload.message === 'string') {
    return payload.message;
  }
  try {
    return JSON.stringify(payload);
  } catch (error) {
    console.warn('Не удалось сериализовать ответ ошибки удалённого API:', error);
    return 'Неизвестная ошибка удалённого API';
  }
};

export async function callRemoteChat(
  messages: RemoteChatMessage[],
  config: RemoteChatConfig,
): Promise<RemoteChatResult> {
  const endpoint = buildEndpoint(config.apiUrl);
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (config.apiKey?.trim()) {
    headers.Authorization = `Bearer ${config.apiKey.trim()}`;
  }

  const body = {
    model: config.model ?? DEFAULT_REMOTE_MODEL,
    messages,
    max_tokens: config.maxTokens,
    temperature: config.temperature,
    top_p: config.topP,
  };

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: config.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new RemoteChatError('Запрос отменён пользователем', error);
    }
    throw new RemoteChatError('Не удалось обратиться к удалённому API', error);
  }

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch (error) {
    throw new RemoteChatError('Удалённый API вернул неверный JSON', error);
  }

  if (!response.ok) {
    throw new RemoteChatError(parseErrorMessage(payload), payload);
  }

  if (isRecord(payload)) {
    const data = payload as RemotePayload;

    if (Array.isArray(data.choices) && data.choices.length > 0) {
      const firstChoice = data.choices[0];
      const content = firstChoice?.message?.content ?? firstChoice?.text;
      if (typeof content !== 'string' || !content.trim()) {
        throw new RemoteChatError('Удалённый API не вернул текст ответа', payload);
      }

      return {
        content: content.trim(),
        model: data.model ?? config.model ?? 'remote',
        tokens: data.usage?.total_tokens,
        raw: payload,
      };
    }

    if (typeof data.content === 'string') {
      return {
        content: data.content.trim(),
        model: data.model ?? config.model ?? 'remote',
        tokens: data.tokens,
        raw: payload,
      };
    }
  }

  throw new RemoteChatError('Не удалось распознать формат ответа удалённого API', payload);
}
