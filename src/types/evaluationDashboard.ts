import type {
  DomainMetricSummary,
  EvaluationDifficulty,
  EvaluationRunItem,
} from './evaluation';

export interface DashboardDomainInsight extends DomainMetricSummary {
  coverageShare: number;
  toneClass: string;
  delta: number | null;
  share: number;
  color: string;
}

export interface DashboardTagSummary {
  tag: string;
  total: number;
  scored: number;
  needsReview: number;
  coverageShare: number;
  averageScore: number | null;
  toneClass: string;
}

export interface DashboardDifficultySummary {
  difficulty: EvaluationDifficulty | 'unknown';
  total: number;
  scored: number;
  needsReview: number;
  averageScore: number | null;
  baselineScore: number | null;
  delta: number | null;
  toneClass: string;
}

export interface DashboardHotspot {
  id: string;
  question: string;
  domain: string;
  tags: string[];
  difficulty: EvaluationDifficulty | 'unknown';
  score: number | null;
  status: EvaluationRunItem['status'];
  needsReview: boolean;
  reviewNotes: string | null;
  toneClass: string;
}

export interface DashboardRecommendation {
  key: string;
  label: string;
  count: number;
}
