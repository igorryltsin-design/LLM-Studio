export interface AutoTrainingConfig {
  aggregatorUrl: string;
  endpoint: string;
  collection: string;
  tags: string;
  status: string;
  query: string;
  limit: number;
  authHeader: string;
  authToken: string;
  includePreviousDataset: boolean;
  previousFineTunePath: string;
  deduplicate: boolean;
  minExamples: number;
  scoreThreshold: number | null;
  targetPairs: number;
  pairsPerSnippet: number;
  includeReferencePair: boolean;
  minParagraphChars: number;
  maxParagraphChars: number;
  maxSegments: number;
  topK: number;
  deepSearch: boolean;
  maxCandidates: number;
  maxSnippets: number;
  chunkChars: number;
  maxChunks: number;
  autoLaunchFineTune: boolean;
  pipelineDryRun: boolean;
  includeDatasetPreview: boolean;
  fineTuneTimeout: number;
}

export const AUTO_TRAINING_STORAGE_KEY = 'llm-studio-auto-training';

export const DEFAULT_AUTO_TRAINING_CONFIG: AutoTrainingConfig = {
  aggregatorUrl: 'http://127.0.0.1:5050',
  endpoint: '/export/csv',
  collection: 'LLM Feedback',
  tags: 'ready-for-training',
  status: 'ready',
  query: '',
  limit: 500,
  authHeader: 'Authorization',
  authToken: '',
  includePreviousDataset: true,
  previousFineTunePath: '',
  deduplicate: true,
  minExamples: 1,
  scoreThreshold: null,
  targetPairs: 30,
  pairsPerSnippet: 1,
  includeReferencePair: true,
  minParagraphChars: 140,
  maxParagraphChars: 680,
  maxSegments: 4,
  topK: 5,
  deepSearch: true,
  maxCandidates: 20,
  maxSnippets: 3,
  chunkChars: 5000,
  maxChunks: 25,
  autoLaunchFineTune: true,
  pipelineDryRun: false,
  includeDatasetPreview: false,
  fineTuneTimeout: 900,
};

const parseNumber = (value: unknown, fallback: number): number => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

const parseNullableNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const parseBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    if (value.toLowerCase() === 'true') {
      return true;
    }
    if (value.toLowerCase() === 'false') {
      return false;
    }
  }
  return fallback;
};

export const loadAutoTrainingConfig = (): AutoTrainingConfig => {
  if (typeof window === 'undefined') {
    return DEFAULT_AUTO_TRAINING_CONFIG;
  }
  try {
    const raw = window.localStorage.getItem(AUTO_TRAINING_STORAGE_KEY);
    if (!raw) {
      return DEFAULT_AUTO_TRAINING_CONFIG;
    }
    const parsed = JSON.parse(raw) as Partial<AutoTrainingConfig>;

    return {
      ...DEFAULT_AUTO_TRAINING_CONFIG,
      ...parsed,
      limit: parseNumber(parsed?.limit, DEFAULT_AUTO_TRAINING_CONFIG.limit),
      minExamples: parseNumber(parsed?.minExamples, DEFAULT_AUTO_TRAINING_CONFIG.minExamples),
      scoreThreshold: parseNullableNumber(parsed?.scoreThreshold),
      targetPairs: parseNumber(parsed?.targetPairs, DEFAULT_AUTO_TRAINING_CONFIG.targetPairs),
      pairsPerSnippet: parseNumber(parsed?.pairsPerSnippet, DEFAULT_AUTO_TRAINING_CONFIG.pairsPerSnippet),
      minParagraphChars: parseNumber(parsed?.minParagraphChars, DEFAULT_AUTO_TRAINING_CONFIG.minParagraphChars),
      maxParagraphChars: parseNumber(parsed?.maxParagraphChars, DEFAULT_AUTO_TRAINING_CONFIG.maxParagraphChars),
      maxSegments: parseNumber(parsed?.maxSegments, DEFAULT_AUTO_TRAINING_CONFIG.maxSegments),
      topK: parseNumber(parsed?.topK, DEFAULT_AUTO_TRAINING_CONFIG.topK),
      deepSearch: parseBoolean(parsed?.deepSearch, DEFAULT_AUTO_TRAINING_CONFIG.deepSearch),
      maxCandidates: parseNumber(parsed?.maxCandidates, DEFAULT_AUTO_TRAINING_CONFIG.maxCandidates),
      maxSnippets: parseNumber(parsed?.maxSnippets, DEFAULT_AUTO_TRAINING_CONFIG.maxSnippets),
      chunkChars: parseNumber(parsed?.chunkChars, DEFAULT_AUTO_TRAINING_CONFIG.chunkChars),
      maxChunks: parseNumber(parsed?.maxChunks, DEFAULT_AUTO_TRAINING_CONFIG.maxChunks),
      autoLaunchFineTune: parseBoolean(parsed?.autoLaunchFineTune, DEFAULT_AUTO_TRAINING_CONFIG.autoLaunchFineTune),
      pipelineDryRun: parseBoolean(parsed?.pipelineDryRun, DEFAULT_AUTO_TRAINING_CONFIG.pipelineDryRun),
      includeDatasetPreview: parseBoolean(parsed?.includeDatasetPreview, DEFAULT_AUTO_TRAINING_CONFIG.includeDatasetPreview),
      fineTuneTimeout: parseNumber(parsed?.fineTuneTimeout, DEFAULT_AUTO_TRAINING_CONFIG.fineTuneTimeout),
    };
  } catch (error) {
    console.warn('Failed to load auto training config from storage', error);
    return DEFAULT_AUTO_TRAINING_CONFIG;
  }
};

export const persistAutoTrainingConfig = (config: AutoTrainingConfig): void => {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    window.localStorage.setItem(AUTO_TRAINING_STORAGE_KEY, JSON.stringify(config));
  } catch (error) {
    console.warn('Failed to persist auto training config', error);
  }
};
