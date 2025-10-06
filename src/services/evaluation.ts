import type {
  EvaluationDataset,
  EvaluationRunStatus,
} from '../types/evaluation';

export class EvaluationServiceError extends Error {
  public readonly details?: unknown;

  constructor(message: string, details?: unknown) {
    super(message);
    this.name = 'EvaluationServiceError';
    this.details = details;
  }
}

export interface RemoteEvaluationDatasetSummary {
  id: string;
  name: string;
  description?: string;
  language?: string;
  size: number;
  updated_at?: number;
  metadata?: Record<string, unknown>;
}

export interface RemoteEvaluationRunSummary {
  id: string;
  status: EvaluationRunStatus;
  progress: number;
  dataset_id: string;
  dataset_name: string;
  created_at: number;
  updated_at: number;
  completed_at?: number | null;
  metrics?: {
    overall_score?: number | null;
    coverage?: number | null;
    per_domain?: Record<string, number>;
    approvals?: Record<string, number>;
  };
  error?: string | null;
}

export interface RemoteEvaluationRunRequest {
  dataset_id: string;
  model_variant: 'base' | 'fine_tuned' | 'remote';
  sample_size?: number;
  scoring_mode?: 'auto' | 'remote' | 'human' | 'hybrid';
  webhook_url?: string;
  require_human_review?: boolean;
  metadata?: Record<string, unknown>;
}

export interface LocalScoreBreakdown {
  exactMatch: number;
  tokenPrecision: number;
  tokenRecall: number;
  tokenF1: number;
  jaccard: number;
  lengthRatio: number;
  keywordRecall: number;
  weightedScore: number;
}

export interface LocalScoreResult {
  score: number;
  breakdown: LocalScoreBreakdown;
  reasons: string[];
}

export interface LocalScoringOptions {
  keywords?: string[];
  weightExact?: number;
  weightF1?: number;
  weightJaccard?: number;
  weightKeyword?: number;
}

interface FetchJsonOptions extends RequestInit {
  expected?: number[];
}

const DEFAULT_KEYWORDS: string[] = [];

const sanitizeText = (value: string) => value
  .toLowerCase()
  .replace(/\s+/g, ' ')
  .trim();

const tokenize = (value: string) => sanitizeText(value)
  .split(/[^\p{L}\p{N}]+/u)
  .filter(Boolean);

const uniqueTokens = (tokens: string[]) => Array.from(new Set(tokens));

const toSet = (tokens: string[]) => {
  return tokens.reduce<Record<string, number>>((acc, token) => {
    acc[token] = (acc[token] ?? 0) + 1;
    return acc;
  }, {});
};

const computeF1 = (precision: number, recall: number) => {
  if (precision === 0 && recall === 0) {
    return 0;
  }
  return (2 * precision * recall) / (precision + recall);
};

const jaccardSimilarity = (a: string[], b: string[]) => {
  if (!a.length || !b.length) {
    return 0;
  }
  const setA = new Set(a);
  const setB = new Set(b);
  const intersection = Array.from(setA).filter(token => setB.has(token));
  const unionSize = setA.size + setB.size - intersection.length;
  return unionSize === 0 ? 0 : intersection.length / unionSize;
};

const calculatePrecisionRecall = (reference: Record<string, number>, candidate: Record<string, number>) => {
  let correct = 0;
  let totalCandidate = 0;
  let totalReference = 0;

  Object.entries(candidate).forEach(([token, count]) => {
    totalCandidate += count;
    if (reference[token]) {
      correct += Math.min(count, reference[token]);
    }
  });

  Object.values(reference).forEach((count) => {
    totalReference += count;
  });

  const precision = totalCandidate === 0 ? 0 : correct / totalCandidate;
  const recall = totalReference === 0 ? 0 : correct / totalReference;
  return { precision, recall };
};

export const scoreAnswerLocally = (
  referenceRaw: string,
  candidateRaw: string,
  options: LocalScoringOptions = {},
): LocalScoreResult => {
  const reference = sanitizeText(referenceRaw);
  const candidate = sanitizeText(candidateRaw);

  if (!reference || !candidate) {
    return {
      score: 0,
      breakdown: {
        exactMatch: Number(reference === candidate && reference.length > 0),
        tokenPrecision: 0,
        tokenRecall: 0,
        tokenF1: 0,
        jaccard: 0,
        lengthRatio: reference ? candidate.length / reference.length : 0,
        keywordRecall: 0,
        weightedScore: 0,
      },
      reasons: [reference ? 'Ответ модели пустой' : 'Эталон пустой'],
    } satisfies LocalScoreResult;
  }

  const referenceTokens = tokenize(referenceRaw);
  const candidateTokens = tokenize(candidateRaw);
  const referenceBag = toSet(referenceTokens);
  const candidateBag = toSet(candidateTokens);
  const { precision, recall } = calculatePrecisionRecall(referenceBag, candidateBag);
  const f1 = computeF1(precision, recall);
  const jaccard = jaccardSimilarity(referenceTokens, candidateTokens);
  const exactMatch = Number(reference === candidate);
  const lengthRatio = reference.length > 0 ? Math.min(candidate.length / reference.length, 2) : 0;

  const keywords = options.keywords?.length ? options.keywords : DEFAULT_KEYWORDS;
  let keywordRecall = 0;
  if (keywords.length > 0) {
    const normalizedCandidate = new Set(uniqueTokens(candidateTokens));
    const normalizedReference = new Set(uniqueTokens(referenceTokens));
    const meaningful = keywords.filter(keyword => normalizedReference.has(keyword));
    if (meaningful.length > 0) {
      const matched = meaningful.filter(keyword => normalizedCandidate.has(keyword));
      keywordRecall = matched.length / meaningful.length;
    }
  }

  const weightExact = options.weightExact ?? 0.2;
  const weightF1 = options.weightF1 ?? 0.5;
  const weightJaccard = options.weightJaccard ?? 0.2;
  const weightKeyword = options.weightKeyword ?? 0.1;

  const weightedScore = Math.max(0, Math.min(1,
    exactMatch * weightExact
      + f1 * weightF1
      + jaccard * weightJaccard
      + keywordRecall * weightKeyword,
  ));

  const reasons: string[] = [];
  if (exactMatch === 1) {
    reasons.push('Ответ совпадает с эталоном');
  } else {
    if (f1 > 0.75) {
      reasons.push('Высокое пересечение ключевых слов');
    }
    if (keywordRecall < 0.5 && keywords.length > 0) {
      reasons.push('Покрытие ключевых терминов ниже 50%');
    }
    if (jaccard < 0.3) {
      reasons.push('Слабое совпадение лексики с эталоном');
    }
  }

  return {
    score: weightedScore,
    breakdown: {
      exactMatch,
      tokenPrecision: precision,
      tokenRecall: recall,
      tokenF1: f1,
      jaccard,
      lengthRatio,
      keywordRecall,
      weightedScore,
    },
    reasons,
  } satisfies LocalScoreResult;
};

const normalizeBaseUrl = (serverUrl: string) => {
  if (!serverUrl) {
    throw new EvaluationServiceError('Не указан URL сервера тестирования');
  }
  const trimmed = serverUrl.trim();
  if (!trimmed) {
    throw new EvaluationServiceError('Не указан URL сервера тестирования');
  }
  try {
    const url = new URL(trimmed);
    url.search = '';
    url.hash = '';
    url.pathname = url.pathname.replace(/\/+$/, '');
    return url.toString();
  } catch (error) {
    throw new EvaluationServiceError('Некорректный URL сервера тестирования', error);
  }
};

const buildEndpoint = (serverUrl: string, path: string) => {
  const base = normalizeBaseUrl(serverUrl);
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${base}${suffix}`;
};

const fetchJson = async <T>(url: string, options: FetchJsonOptions = {}): Promise<T> => {
  let response: Response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new EvaluationServiceError('Не удалось обратиться к серверу тестирования', error);
  }

  const { expected = [200, 201] } = options;
  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    throw new EvaluationServiceError('Сервер тестирования вернул некорректный JSON', error);
  }

  if (!expected.includes(response.status)) {
    throw new EvaluationServiceError('Ошибка сервера тестирования', payload);
  }

  return payload as T;
};

export const listEvaluationDatasets = async (
  serverUrl: string,
): Promise<RemoteEvaluationDatasetSummary[]> => {
  const endpoint = buildEndpoint(serverUrl, '/v1/evaluations/datasets');
  return fetchJson<RemoteEvaluationDatasetSummary[]>(endpoint, {
    method: 'GET',
    cache: 'no-store',
  });
};

export const fetchEvaluationDataset = async (
  serverUrl: string,
  datasetId: string,
): Promise<EvaluationDataset> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/evaluations/datasets/${datasetId}`);
  return fetchJson<EvaluationDataset>(endpoint, {
    method: 'GET',
    cache: 'no-store',
  });
};

export const createEvaluationRun = async (
  serverUrl: string,
  payload: RemoteEvaluationRunRequest,
): Promise<RemoteEvaluationRunSummary> => {
  const endpoint = buildEndpoint(serverUrl, '/v1/evaluations/runs');
  return fetchJson<RemoteEvaluationRunSummary>(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
};

export const getEvaluationRun = async (
  serverUrl: string,
  runId: string,
): Promise<RemoteEvaluationRunSummary> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/evaluations/runs/${runId}`);
  return fetchJson<RemoteEvaluationRunSummary>(endpoint, {
    method: 'GET',
    cache: 'no-store',
  });
};

export const cancelEvaluationRun = async (
  serverUrl: string,
  runId: string,
): Promise<void> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/evaluations/runs/${runId}`);
  await fetchJson(endpoint, {
    method: 'DELETE',
    expected: [200, 204],
  });
};

export const submitEvaluationReview = async (
  serverUrl: string,
  runId: string,
  itemId: string,
  payload: {
    status: 'approved' | 'rejected';
    score?: number;
    reviewer?: string;
    notes?: string;
  },
): Promise<void> => {
  const endpoint = buildEndpoint(serverUrl, `/v1/evaluations/runs/${runId}/items/${itemId}/review`);
  await fetchJson(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    expected: [200, 204],
  });
};

export const aggregateLocalScores = (results: LocalScoreResult[]) => {
  if (!results.length) {
    return {
      averageScore: 0,
      averageF1: 0,
      exactMatchRate: 0,
      jaccard: 0,
      keywordRecall: 0,
    };
  }

  const totals = results.reduce(
    (acc, result) => {
      acc.averageScore += result.score;
      acc.averageF1 += result.breakdown.tokenF1;
      acc.exactMatchRate += result.breakdown.exactMatch;
      acc.jaccard += result.breakdown.jaccard;
      acc.keywordRecall += result.breakdown.keywordRecall;
      return acc;
    },
    {
      averageScore: 0,
      averageF1: 0,
      exactMatchRate: 0,
      jaccard: 0,
      keywordRecall: 0,
    },
  );

  return {
    averageScore: totals.averageScore / results.length,
    averageF1: totals.averageF1 / results.length,
    exactMatchRate: totals.exactMatchRate / results.length,
    jaccard: totals.jaccard / results.length,
    keywordRecall: totals.keywordRecall / results.length,
  };
};

export const compareWithBaseline = (current: number | null, baseline: number | null) => {
  if (current === null || baseline === null) {
    return null;
  }
  return current - baseline;
};
