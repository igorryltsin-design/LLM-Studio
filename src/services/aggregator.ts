export interface ProblemPipelineOptions {
  baseUrl: string;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  timeout?: number;
  credentials?: RequestCredentials;
}

export interface ProblemPipelinePayload {
  evaluation: Record<string, unknown>;
  score_threshold?: number | null;
  target_pairs?: number;
  generation?: Record<string, unknown>;
  search?: Record<string, unknown>;
  fine_tune?: Record<string, unknown>;
  dry_run?: boolean;
  include_dataset?: boolean;
}

export interface ProblemPipelineResponse {
  ok: boolean;
  dataset_size?: number;
  logs?: string[];
  meta?: Record<string, unknown>;
  preview?: Array<Record<string, unknown>>;
  dataset?: Array<Record<string, unknown>>;
  dataset_preview_full?: Array<Record<string, unknown>>;
  fine_tune_job?: Record<string, unknown>;
  error?: string;
}

export const runProblemPipeline = async (
  payload: ProblemPipelinePayload,
  options: ProblemPipelineOptions,
): Promise<ProblemPipelineResponse> => {
  const base = options.baseUrl.replace(/\/$/, '');
  const endpoint = `${base}/api/training/problem-pipeline`;
  const controller = new AbortController();
  const timeout = options.timeout ?? 180000;
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
      body: JSON.stringify(payload),
      signal: options.signal ?? controller.signal,
      credentials: options.credentials ?? 'same-origin',
    });

    const raw = await response.text();
    let data: ProblemPipelineResponse;
    try {
      data = raw ? (JSON.parse(raw) as ProblemPipelineResponse) : { ok: response.ok };
    } catch (error) {
      throw new Error(raw || `Agregator pipeline failed with status ${response.status}`);
    }

    if (!response.ok || data.ok === false) {
      const message = data?.error || raw || `Agregator pipeline failed with status ${response.status}`;
      throw new Error(message);
    }
    return data;
  } finally {
    clearTimeout(timeoutId);
  }
};
