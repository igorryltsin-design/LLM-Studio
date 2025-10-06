export interface DatasetItem {
  id: string;
  input: string;
  output: string;
  source?: string;
}

export type TrainingMethod = 'lora' | 'qlora' | 'full';
export type TrainingQuantization = '4bit' | '8bit' | 'none';

export interface TrainingConfig {
  method: TrainingMethod;
  quantization: TrainingQuantization;
  loraRank: number;
  loraAlpha: number;
  learningRate: number;
  batchSize: number;
  epochs: number;
  maxLength: number;
  warmupSteps: number;
  outputDir: string;
  targetModules: string;
  initialAdapterPath: string;
}

export interface TrainingMetricPoint {
  timestamp: number;
  metrics: Record<string, number>;
}
