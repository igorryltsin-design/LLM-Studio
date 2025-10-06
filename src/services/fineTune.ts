export type FineTuneMethod = 'lora' | 'qlora' | 'full';
export type FineTuneQuantization = 'none' | '4bit' | '8bit';
export type FineTuneJobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'paused'
  | 'pausing';

export interface FineTuneDatasetItem {
  input: string;
  output: string;
  source?: string;
}

export interface FineTuneConfigPayload {
  method: FineTuneMethod;
  quantization: FineTuneQuantization;
  lora_rank: number;
  lora_alpha: number;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  max_length: number;
  warmup_steps: number;
  target_modules?: string[];
  initial_adapter_path?: string | null;
}

export interface FineTuneRequestPayload {
  base_model_path: string;
  output_dir?: string;
  dataset: FineTuneDatasetItem[];
  config: FineTuneConfigPayload;
}

export interface FineTuneEvent {
  timestamp: number;
  level: string;
  message: string;
}

export interface FineTuneJob {
  id: string;
  status: FineTuneJobStatus;
  progress: number;
  message: string;
  metrics: Record<string, number>;
  error?: string;
  createdAt: number;
  updatedAt: number;
  startedAt?: number;
  finishedAt?: number;
  datasetSize: number;
  outputDir: string;
  baseModelPath: string;
  config: FineTuneConfigPayload;
  events?: FineTuneEvent[];
  resumeCheckpoint?: string | null;
}

export class FineTuneError extends Error {
  public readonly details?: unknown;

  constructor(message: string, details?: unknown) {
    super(message);
    this.name = 'FineTuneError';
    this.details = details;
  }
}

const parseErrorMessage = (payload: unknown): string => {
  if (!payload) return 'Неизвестная ошибка сервера дообучения';
  if (typeof payload === 'string') return payload;
  if (Array.isArray(payload) && payload.length > 0) {
    return parseErrorMessage(payload[0]);
  }
  if (typeof payload === 'object') {
    const record = payload as Record<string, unknown>;
    if ('detail' in record) {
      return parseErrorMessage(record.detail);
    }
    if (typeof record.message === 'string') {
      return record.message;
    }
    if (typeof record.error === 'string') {
      return record.error;
    }
  }
  try {
    return JSON.stringify(payload);
  } catch (error) {
    console.warn('Не удалось сериализовать ошибку fine-tune:', error);
    return 'Неизвестная ошибка сервера дообучения';
  }
};

const normalizeBaseUrl = (rawUrl: string): string => {
  if (!rawUrl || typeof rawUrl !== 'string') {
    throw new FineTuneError('URL сервера дообучения не задан');
  }

  const trimmed = rawUrl.trim();
  if (!trimmed) {
    throw new FineTuneError('URL сервера дообучения не задан');
  }

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch (error) {
    throw new FineTuneError(`Некорректный URL сервера: ${String(error)}`);
  }

  const sanitizedPath = url.pathname
    .replace(/\/+$/, '')
    .replace(/\/chat\/completions$/i, '')
    .replace(/\/system\/status$/i, '')
    .replace(/\/v1$/i, '');

  const segments = sanitizedPath.split('/').filter(Boolean);
  url.pathname = segments.length ? `/${segments.join('/')}` : '';
  url.search = '';
  url.hash = '';

  const normalized = url.toString().replace(/\/$/, '');
  return normalized || `${url.protocol}//${url.host}`;
};

const buildEndpoint = (serverUrl: string, path: string): string => {
  const base = normalizeBaseUrl(serverUrl);
  const suffix = path.startsWith('/') ? path.slice(1) : path;
  return `${base}/${suffix}`;
};

const fetchJson = async <T>(input: RequestInfo, init?: RequestInit): Promise<T> => {
  let response: Response;
  try {
    response = await fetch(input, init);
  } catch (error) {
    throw new FineTuneError('Не удалось обратиться к серверу дообучения', error);
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    throw new FineTuneError('Сервер дообучения вернул некорректный JSON', error);
  }

  if (!response.ok) {
    throw new FineTuneError(parseErrorMessage(payload), payload);
  }

  return payload as T;
};

export interface FineTunedModelInfo {
  id: string;
  name: string;
  path: string;
  base_model_path?: string | null;
  method?: string | null;
  dataset_size?: number | null;
  created_at?: number | null;
  finished_at?: number | null;
}

export interface AgregatorRequestPayload {
  base_url: string;
  endpoint?: string;
  params?: Record<string, unknown>;
  headers?: Record<string, string>;
  timeout?: number;
}

export interface AutoFineTuneRequestPayload {
  aggregator: AgregatorRequestPayload;
  base_model_path: string;
  output_dir?: string;
  config: FineTuneConfigPayload;
  include_previous_dataset?: boolean;
  previous_fine_tune_path?: string;
  deduplicate?: boolean;
  min_examples?: number;
}

export const createFineTuneJob = async (
  serverUrl: string,
  payload: FineTuneRequestPayload,
): Promise<FineTuneJob> => {
  const endpoint = buildEndpoint(serverUrl, '/v1/fine-tunes');
  return fetchJson<FineTuneJob>(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
};

export const listFineTuneJobs = async (serverUrl: string): Promise<FineTuneJob[]> => {
  const endpoint = buildEndpoint(serverUrl, '/v1/fine-tunes');
  return fetchJson<FineTuneJob[]>(endpoint, {
    method: 'GET',
    cache: 'no-store',
  });
};

export const getFineTuneJob = async (
  serverUrl: string,
  jobId: string,
): Promise<FineTuneJob> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/fine-tunes/${jobId}`);
  return fetchJson<FineTuneJob>(endpoint, {
    method: 'GET',
    cache: 'no-store',
  });
};

export const cancelFineTuneJob = async (
  serverUrl: string,
  jobId: string,
): Promise<FineTuneJob> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/fine-tunes/${jobId}/cancel`);
  return fetchJson<FineTuneJob>(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
};

export const pauseFineTuneJob = async (
  serverUrl: string,
  jobId: string,
): Promise<FineTuneJob> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/fine-tunes/${jobId}/pause`);
  return fetchJson<FineTuneJob>(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
};

export const resumeFineTuneJob = async (
  serverUrl: string,
  jobId: string,
): Promise<FineTuneJob> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/fine-tunes/${jobId}/resume`);
  return fetchJson<FineTuneJob>(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
};

export const listAvailableFineTunes = async (
  serverUrl: string,
): Promise<FineTunedModelInfo[]> => {
  const endpoint = buildEndpoint(serverUrl, '/v1/fine-tunes/available');
  return fetchJson<FineTunedModelInfo[]>(endpoint, {
    method: 'GET',
    cache: 'no-store',
  });
};

export const autoFineTuneFromAgregator = async (
  serverUrl: string,
  payload: AutoFineTuneRequestPayload,
): Promise<FineTuneJob> => {
  const endpoint = buildEndpoint(serverUrl, '/v1/fine-tunes/from-agregator');
  return fetchJson<FineTuneJob>(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
};
