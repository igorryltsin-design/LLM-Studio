import { RemoteChatMessage } from './remoteChat';

export class BaseChatError extends Error {
  public readonly details?: unknown;

  constructor(message: string, details?: unknown) {
    super(message);
    this.name = 'BaseChatError';
    this.details = details;
  }
}

interface BaseChatConfig {
  serverUrl: string;
  modelPath: string;
  adapterPath?: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  quantization?: 'none' | '4bit' | '8bit';
  device?: 'auto' | 'cpu' | 'cuda' | 'mps';
  signal?: AbortSignal;
}

interface BaseChatResponse {
  content: string;
  model?: string;
  tokens?: number;
}

interface BaseChatUsage {
  total_tokens?: number;
}

interface BaseChatChoice {
  message?: {
    content?: string;
  };
  text?: string;
}

interface BaseChatPayload {
  choices?: BaseChatChoice[];
  content?: string;
  model?: string;
  tokens?: number;
  usage?: BaseChatUsage;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const buildEndpoint = (serverUrl: string) => {
  if (!serverUrl) {
    throw new BaseChatError('Не указан URL сервера базовой модели');
  }

  const trimmed = serverUrl.trim().replace(/\/$/, '');

  if (/\/chat\/completions$/i.test(trimmed)) {
    return trimmed;
  }

  if (/\/v1$/i.test(trimmed)) {
    return `${trimmed}/chat/completions`;
  }

  return `${trimmed}/v1/chat/completions`;
};

const parseErrorMessage = (payload: unknown): string => {
  if (!payload) return 'Неизвестная ошибка локального сервера';
  if (typeof payload === 'string') return payload;
  if (Array.isArray(payload) && payload.length > 0) {
    return parseErrorMessage(payload[0]);
  }
  if (isRecord(payload)) {
    if ('detail' in payload) {
      return parseErrorMessage(payload.detail);
    }
    if ('error' in payload) {
      return parseErrorMessage(payload.error);
    }
    const message = payload.message;
    if (typeof message === 'string') {
      return message;
    }
  }
  try {
    return JSON.stringify(payload);
  } catch (error) {
    console.warn('Не удалось сериализовать ответ ошибки локального сервера:', error);
    return 'Неизвестная ошибка локального сервера';
  }
};

export async function callBaseChat(
  messages: RemoteChatMessage[],
  config: BaseChatConfig,
): Promise<BaseChatResponse> {
  const endpoint = buildEndpoint(config.serverUrl);

  if (!config.modelPath) {
    throw new BaseChatError('Не указан путь к базовой модели');
  }

  const body = {
    model_path: config.modelPath,
    adapter_path: config.adapterPath,
    messages,
    max_tokens: config.maxTokens,
    temperature: config.temperature,
    top_p: config.topP,
    quantization: config.quantization ?? 'none',
    device: config.device ?? undefined,
  };

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: config.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new BaseChatError('Запрос отменён пользователем', error);
    }
    throw new BaseChatError('Не удалось обратиться к локальному серверу базовой модели', error);
  }

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch (error) {
    throw new BaseChatError('Локальный сервер вернул некорректный JSON', error);
  }

  if (!response.ok) {
    throw new BaseChatError(parseErrorMessage(payload), payload);
  }

  if (isRecord(payload)) {
    const data = payload as BaseChatPayload;

    if (Array.isArray(data.choices) && data.choices.length > 0) {
      const firstChoice = data.choices[0];
      const content = firstChoice?.message?.content ?? firstChoice?.text;
      if (typeof content !== 'string' || !content.trim()) {
        throw new BaseChatError('Локальный сервер не вернул текст ответа', payload);
      }

      return {
        content: content.trim(),
        model: data.model ?? 'base',
        tokens: data.usage?.total_tokens,
      };
    }

    if (typeof data.content === 'string') {
      return {
        content: data.content.trim(),
        model: data.model ?? 'base',
        tokens: data.tokens,
      };
    }
  }

  throw new BaseChatError('Не удалось распознать формат ответа локального сервера', payload);
}
