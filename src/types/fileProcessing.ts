export type GenerationSource = 'remote' | 'base';
export type ValidationMode = 'auto' | 'manual';

export interface ParagraphConfig {
  id: string;
  index: number;
  text: string;
  desiredPairs: number;
  tokenCount: number;
}

export interface DocumentFile {
  id: string;
  name: string;
  type: string;
  size: number;
  content: string;
  processed: boolean;
  timestamp: number;
  paragraphs: ParagraphConfig[];
}

export interface QAPair {
  id: string;
  fileId: string;
  paragraphId: string;
  question: string;
  answer: string;
  source: string;
  quality: number;
  approved: boolean;
  rejected: boolean;
  issues?: string[];
  generationSource: GenerationSource;
}

export interface GenerationLog {
  id: string;
  timestamp: number;
  level: 'info' | 'warn' | 'error';
  message: string;
  context?: string;
}
