interface SystemStatusDevice {
  id?: number;
  name?: string;
  memory_total_gb?: number | null;
  memory_used_gb?: number | null;
  memory_percent?: number | null;
  utilization_percent?: number | null;
}

interface SystemStatusPayload {
  status?: string;
  message?: string;
  timestamp?: number;
  cpu?: {
    percent?: number;
    process_percent?: number | null;
  };
  memory?: {
    percent?: number;
    used_gb?: number;
    total_gb?: number;
    models_dir_gb?: number;
  };
  gpu?: {
    available?: boolean;
    backend?: string;
    devices?: SystemStatusDevice[];
  };
}

export interface SystemStats {
  status: 'ok' | 'error';
  cpuPercent: number | null;
  cpuProcessPercent: number | null;
  ramPercent: number | null;
  ramUsedGb: number | null;
  ramTotalGb: number | null;
  modelsDirGb: number | null;
  gpuPercent: number | null;
  gpuMemoryUsedGb: number | null;
  gpuMemoryTotalGb: number | null;
  gpuName: string | null;
  gpuBackend: 'cpu' | 'cuda' | 'mps';
  message?: string;
  timestamp: number | null;
  latencyMs: number | null;
}

const DEFAULT_STATS: SystemStats = {
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
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const buildSystemEndpoint = (serverUrl: string, suffix: string): string => {
  if (!serverUrl || typeof serverUrl !== 'string') {
    throw new Error('Не указан URL локального сервера');
  }

  const trimmed = serverUrl.trim();
  if (!trimmed) {
    throw new Error('Не указан URL локального сервера');
  }

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch (error) {
    throw new Error(`Некорректный URL сервера: ${String(error)}`);
  }

  const normalizedPath = url.pathname
    .replace(/\/+$/, '')
    .replace(/\/chat\/completions$/i, '')
    .replace(/\/v1$/i, '');

  const endpointPath = `${normalizedPath}/${suffix}`;
  const sanitizedSegments = endpointPath.split('/').filter(Boolean);
  url.pathname = `/${sanitizedSegments.join('/')}`;
  url.search = '';
  url.hash = '';

  return url.toString();
};

const buildSystemStatusEndpoint = (serverUrl: string): string =>
  buildSystemEndpoint(serverUrl, 'system/status');

const formatNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && !Number.isNaN(value)) {
    return value;
  }
  return null;
};

const normalizeEpochTimestamp = (value: unknown): number | null => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null;
  }

  const milliseconds = value < 1e12 ? value * 1000 : value;
  return Number.isFinite(milliseconds) && milliseconds > 0 ? milliseconds : null;
};

const parseSystemPayload = (payload: SystemStatusPayload, latencyMs: number | null): SystemStats => {
  const timestamp = normalizeEpochTimestamp(payload.timestamp) ?? Date.now();

  const cpuPercent = formatNumber(payload.cpu?.percent);
  const cpuProcessPercent = formatNumber(payload.cpu?.process_percent);
  const ramPercent = formatNumber(payload.memory?.percent);
  const ramUsedGb = formatNumber(payload.memory?.used_gb);
  const ramTotalGb = formatNumber(payload.memory?.total_gb);
  const modelsDirGb = formatNumber(payload.memory?.models_dir_gb);

  const gpuDevice = Array.isArray(payload.gpu?.devices) ? payload.gpu!.devices![0] : undefined;
  const gpuPercent = formatNumber(gpuDevice?.utilization_percent ?? gpuDevice?.memory_percent);
  const gpuMemoryUsedGb = formatNumber(gpuDevice?.memory_used_gb);
  const gpuMemoryTotalGb = formatNumber(gpuDevice?.memory_total_gb);
  const gpuName = typeof gpuDevice?.name === 'string' ? gpuDevice?.name : null;
  const rawBackend = typeof payload.gpu?.backend === 'string' ? payload.gpu?.backend.toLowerCase() : '';
  const gpuBackend: 'cpu' | 'cuda' | 'mps' = rawBackend === 'cuda' ? 'cuda' : rawBackend === 'mps' ? 'mps' : 'cpu';

  return {
    status: 'ok',
    cpuPercent,
    cpuProcessPercent,
    ramPercent,
    ramUsedGb,
    ramTotalGb,
    modelsDirGb,
    gpuPercent,
    gpuMemoryUsedGb,
    gpuMemoryTotalGb,
    gpuName,
    gpuBackend,
    timestamp,
    latencyMs,
  };
};

export async function fetchSystemStats(serverUrl: string): Promise<SystemStats> {
  let endpoint: string;
  try {
    endpoint = buildSystemStatusEndpoint(serverUrl);
  } catch (error) {
    return {
      ...DEFAULT_STATS,
      message: error instanceof Error ? error.message : String(error),
      latencyMs: null,
    };
  }

  let response: Response;
  const getNow = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  let latencyMs: number | null = null;
  try {
    const requestStartedAt = getNow();
    response = await fetch(endpoint, {
      method: 'GET',
      cache: 'no-store',
    });
    latencyMs = Math.max(0, getNow() - requestStartedAt);
  } catch (error) {
    return {
      ...DEFAULT_STATS,
      message: error instanceof Error ? error.message : 'Не удалось получить системные метрики',
      latencyMs,
    };
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (parseError) {
    console.warn('Не удалось преобразовать ответ сервера системного статуса в JSON', parseError);
    return {
      ...DEFAULT_STATS,
      message: 'Некорректный формат ответа сервера системного статуса',
      latencyMs,
    };
  }

  if (!isRecord(payload)) {
    return {
      ...DEFAULT_STATS,
      message: 'Некорректный ответ сервера системного статуса',
      latencyMs,
    };
  }

  const statusField = typeof payload.status === 'string' ? payload.status : '';
  if (!response.ok || (statusField && statusField.toLowerCase() !== 'ok')) {
    const message = typeof payload.message === 'string' ? payload.message : 'Сервер вернул ошибку';
    return {
      ...DEFAULT_STATS,
      message,
      latencyMs,
    };
  }

  return parseSystemPayload(payload as SystemStatusPayload, latencyMs);
}

const parseErrorMessage = (payload: unknown, fallback: string): string => {
  if (!payload) {
    return fallback;
  }

  if (typeof payload === 'string') {
    return payload;
  }

  if (isRecord(payload)) {
    const detail = payload.detail;
    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }
    const message = payload.message;
    if (typeof message === 'string' && message.trim()) {
      return message;
    }
    const error = payload.error;
    if (typeof error === 'string' && error.trim()) {
      return error;
    }
  }

  try {
    return JSON.stringify(payload);
  } catch (stringifyError) {
    console.warn('Не удалось преобразовать сообщение об ошибке в строку', stringifyError);
    return fallback;
  }
};

export async function openModelsDirectory(serverUrl: string): Promise<void> {
  const endpoint = buildSystemEndpoint(serverUrl, 'system/open-models-directory');

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      cache: 'no-store',
    });
  } catch (error) {
    throw new Error('Не удалось обратиться к локальному серверу базовой модели');
  }

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch (error) {
    if (response.ok) {
      return;
    }
  }

  if (!response.ok) {
    const message = parseErrorMessage(payload, 'Не удалось открыть каталог моделей');
    throw new Error(message);
  }

  if (isRecord(payload)) {
    const status = payload.status;
    if (typeof status === 'string' && status.toLowerCase() !== 'ok') {
      const message = parseErrorMessage(payload, 'Не удалось открыть каталог моделей');
      throw new Error(message);
    }
  }
}
