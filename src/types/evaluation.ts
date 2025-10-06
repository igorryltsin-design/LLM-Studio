export type EvaluationDifficulty = 'easy' | 'medium' | 'hard';

export type EvaluationScoringMode = 'auto' | 'human' | 'llm';

export type EvaluationItemStatus =
  | 'pending'
  | 'answering'
  | 'awaiting_score'
  | 'scored'
  | 'needs_review'
  | 'reviewed';

export type EvaluationReviewStatus = 'pending' | 'approved' | 'rejected';

export type EvaluationRunStatus =
  | 'draft'
  | 'queued'
  | 'running'
  | 'waiting_review'
  | 'completed'
  | 'failed';

export interface EvaluationDatasetItem {
  id: string;
  domain: string;
  subdomain?: string;
  question: string;
  referenceAnswer: string;
  difficulty: EvaluationDifficulty;
  tags: string[];
  metadata?: Record<string, string>;
}

export interface EvaluationDataset {
  id: string;
  name: string;
  description?: string;
  createdAt: number;
  updatedAt: number;
  items: EvaluationDatasetItem[];
}

export interface EvaluationRunConfig {
  datasetId: string;
  datasetName: string;
  modelVariant: 'base' | 'fine_tuned' | 'remote';
  scoringMode: EvaluationScoringMode;
  sampleSize?: number | null;
  shuffle: boolean;
  qualityGateThreshold?: number | null;
  requireHumanReview: boolean;
}

export interface EvaluationRunItem {
  id: string;
  datasetItemId: string;
  domain: string;
  subdomain?: string;
  question: string;
  referenceAnswer: string;
  modelAnswer?: string;
  answerStartedAt?: number;
  answerCompletedAt?: number;
  scoringMode: EvaluationScoringMode;
  status: EvaluationItemStatus;
  autoScore?: number | null;
  remoteScore?: number | null;
  humanScore?: number | null;
  reviewStatus?: EvaluationReviewStatus;
  reviewer?: string | null;
  reviewNotes?: string | null;
  tokens?: number | null;
  difficulty?: EvaluationDifficulty;
  tags?: string[];
}

export interface DomainMetricSummary {
  domain: string;
  total: number;
  answered: number;
  scored: number;
  needsReview: number;
  averageScore: number | null;
  autoScoreAverage: number | null;
  humanApprovalRate: number | null;
}

export interface EvaluationMetrics {
  total: number;
  answered: number;
  scored: number;
  needsReview: number;
  coverage: number;
  overallScore: number | null;
  autoScoreAverage: number | null;
  humanApprovalRate: number | null;
  averageLatencySeconds: number | null;
  latencyP95Seconds: number | null;
  averageTokensPerResponse: number | null;
  tokensPerSecond: number | null;
  qualityGateThreshold?: number | null;
  qualityGatePassed?: boolean | null;
  baselineScore?: number | null;
  scoreDelta?: number | null;
  domainSummaries: DomainMetricSummary[];
}

export interface EvaluationRun {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  startedAt?: number;
  completedAt?: number;
  status: EvaluationRunStatus;
  config: EvaluationRunConfig;
  items: EvaluationRunItem[];
  metrics: EvaluationMetrics;
  error?: string;
  notes?: string;
  baselineRunId?: string | null;
}

export interface EvaluationReviewTask {
  runId: string;
  itemId: string;
  domain: string;
  question: string;
  modelAnswer: string | null;
  referenceAnswer: string;
  submittedAt: number;
  reviewer?: string | null;
  status: EvaluationReviewStatus;
}
