import { createContext, useContext, useMemo, useState, useEffect, useRef, ReactNode } from 'react';
import {
  DocumentFile,
  GenerationLog,
  GenerationSource,
  QAPair,
  ValidationMode,
} from '../types/fileProcessing';

const STORAGE_KEY = 'llm-studio-file-processing-state';

interface PersistentState {
  files: DocumentFile[];
  qaPairs: QAPair[];
  logs: GenerationLog[];
  isProcessing: boolean;
  processingProgress: number;
  generationSource: GenerationSource;
  validationMode: ValidationMode;
  defaultPairsPerParagraph: number;
}

const defaultState: PersistentState = {
  files: [],
  qaPairs: [],
  logs: [],
  isProcessing: false,
  processingProgress: 0,
  generationSource: 'remote',
  validationMode: 'auto',
  defaultPairsPerParagraph: 2,
};

const loadState = (): PersistentState => {
  if (typeof window === 'undefined') {
    return defaultState;
  }

  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) {
      return defaultState;
    }
    const parsed = JSON.parse(saved) as Partial<PersistentState>;
    return {
      ...defaultState,
      ...parsed,
    };
  } catch (error) {
    console.warn('Не удалось загрузить состояние обработки файлов', error);
    return defaultState;
  }
};

interface FileProcessingContextValue extends PersistentState {
  setFiles: React.Dispatch<React.SetStateAction<DocumentFile[]>>;
  setQaPairs: React.Dispatch<React.SetStateAction<QAPair[]>>;
  setLogs: React.Dispatch<React.SetStateAction<GenerationLog[]>>;
  setIsProcessing: React.Dispatch<React.SetStateAction<boolean>>;
  setProcessingProgress: React.Dispatch<React.SetStateAction<number>>;
  setGenerationSource: React.Dispatch<React.SetStateAction<GenerationSource>>;
  setValidationMode: React.Dispatch<React.SetStateAction<ValidationMode>>;
  setDefaultPairsPerParagraph: React.Dispatch<React.SetStateAction<number>>;
  resetState: () => void;
}

const FileProcessingContext = createContext<FileProcessingContextValue | undefined>(undefined);

export function FileProcessingProvider({ children }: { children: ReactNode }) {
  const initial = loadState();
  const [files, setFiles] = useState<DocumentFile[]>(initial.files);
  const [qaPairs, setQaPairs] = useState<QAPair[]>(initial.qaPairs);
  const [logs, setLogs] = useState<GenerationLog[]>(initial.logs);
  const [isProcessing, setIsProcessing] = useState<boolean>(initial.isProcessing);
  const [processingProgress, setProcessingProgress] = useState<number>(initial.processingProgress);
  const [generationSource, setGenerationSource] = useState<GenerationSource>(initial.generationSource);
  const [validationMode, setValidationMode] = useState<ValidationMode>(initial.validationMode);
  const [defaultPairsPerParagraph, setDefaultPairsPerParagraph] = useState<number>(
    initial.defaultPairsPerParagraph,
  );
  const shouldResetProcessingRef = useRef(initial.isProcessing);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const payload: PersistentState = {
      files,
      qaPairs,
      logs,
      isProcessing,
      processingProgress,
      generationSource,
      validationMode,
      defaultPairsPerParagraph,
    };

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn('Не удалось сохранить состояние обработки файлов', error);
    }
  }, [
    files,
    qaPairs,
    logs,
    isProcessing,
    processingProgress,
    generationSource,
    validationMode,
    defaultPairsPerParagraph,
  ]);

  useEffect(() => {
    if (shouldResetProcessingRef.current) {
      shouldResetProcessingRef.current = false;
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  }, []);

  const resetState = () => {
    setFiles([]);
    setQaPairs([]);
    setLogs([]);
    setIsProcessing(false);
    setProcessingProgress(0);
    setGenerationSource('remote');
    setValidationMode('auto');
    setDefaultPairsPerParagraph(2);
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  };

  const value = useMemo<FileProcessingContextValue>(
    () => ({
      files,
      qaPairs,
      logs,
      isProcessing,
      processingProgress,
      generationSource,
      validationMode,
      defaultPairsPerParagraph,
      setFiles,
      setQaPairs,
      setLogs,
      setIsProcessing,
      setProcessingProgress,
      setGenerationSource,
      setValidationMode,
      setDefaultPairsPerParagraph,
      resetState,
    }),
    [
      files,
      qaPairs,
      logs,
      isProcessing,
      processingProgress,
      generationSource,
      validationMode,
      defaultPairsPerParagraph,
    ],
  );

  return <FileProcessingContext.Provider value={value}>{children}</FileProcessingContext.Provider>;
}

export function useFileProcessing() {
  const context = useContext(FileProcessingContext);
  if (!context) {
    throw new Error('useFileProcessing must be used within a FileProcessingProvider');
  }
  return context;
}
