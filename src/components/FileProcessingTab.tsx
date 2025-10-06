import { ChangeEvent, useCallback, useMemo, useRef, useState } from 'react';
import {
  Upload,
  FileText,
  Download,
  Trash2,
  Play,
  CheckCircle,
  XCircle,
  Eye,
  Plus,
  Filter,
  Settings,
  AlertTriangle,
  Info,
  Square,
} from 'lucide-react';
import { useHistory } from '../contexts/HistoryContext';
import { useStatus } from '../contexts/StatusContext';
import { useSettings } from '../contexts/SettingsContext';
import { callRemoteChat, RemoteChatError, RemoteChatMessage } from '../services/remoteChat';
import { callBaseChat, BaseChatError } from '../services/baseChat';
import { extractTextFromFile } from '../utils/textExtraction';
import {
  DocumentFile,
  GenerationLog,
  GenerationSource,
  ParagraphConfig,
  QAPair,
  ValidationMode,
} from '../types/fileProcessing';
import { useFileProcessing } from '../contexts/FileProcessingContext';
import { useTraining } from '../contexts/TrainingContext';
import type { DatasetItem } from '../types/training';
import { useLazyList } from '../hooks/useLazyList';

interface ProcessingStats {
  totalFiles: number;
  processedFiles: number;
  generatedPairs: number;
  approvedPairs: number;
  rejectedPairs: number;
  avgQuality: number;
}

interface RawGeneratedPair {
  question: string;
  answer: string;
}

const SPECIAL_SYMBOL_REGEX = /[^-a-zA-Zа-яА-ЯёЁ0-9\s.,;:!?'"()]/g;
const REPEATED_SEQUENCE_REGEX = /(.)\1{3,}/;
const SOURCE_REFERENCE_PHRASES = [
  'согласно тексту',
  'согласно статье',
  'согласно абзацу',
  'в тексте',
  'в абзаце',
  'в статье',
  'по тексту',
  'по статье',
  'в документе',
  'в данном тексте',
  'в данном абзаце',
  'приведённом тексте',
  'представленном тексте',
];

const createId = (prefix: string) => `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;

const countWords = (text: string) =>
  text
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;

const estimatePairsForParagraph = (wordCount: number) => {
  if (wordCount >= 180) return 4;
  if (wordCount >= 120) return 3;
  if (wordCount >= 60) return 2;
  return wordCount >= 30 ? 1 : 0;
};

const splitIntoParagraphs = (content: string): ParagraphConfig[] => {
  const sanitized = content.replace(/\r\n/g, '\n').trim();

  if (!sanitized) {
    return [];
  }

  const rawParagraphs = sanitized
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter((paragraph) => paragraph.length > 0);

  if (rawParagraphs.length === 0) {
    return [];
  }

  return rawParagraphs.map((paragraph, index) => {
    const tokenCount = countWords(paragraph);
    return {
      id: createId('paragraph'),
      index,
      text: paragraph,
      desiredPairs: estimatePairsForParagraph(tokenCount),
      tokenCount,
    };
  });
};

const normalizeField = (value: string) =>
  value
    .replace(/^\s*[-*\d.()]+\s*/g, '')
    .replace(/^ответ[:\-\s]+/i, '')
    .replace(/^вопрос[:\-\s]+/i, '')
    .replace(/^(согласно|по|в|во)\s+(данному\s+)?(тексту|абзацу|документу|статье)[^,.:!?]*[,.:!?\s]*/i, '')
    .trim();

const extractJsonPayload = (raw: string): unknown => {
  if (!raw) {
    throw new Error('Модель не вернула данных');
  }

  const fencedMatch = raw.match(/```json([\s\S]*?)```/i);
  const candidate = fencedMatch ? fencedMatch[1] : raw;

  const startIndex = candidate.indexOf('{');
  const endIndex = candidate.lastIndexOf('}');

  if (startIndex === -1 || endIndex === -1 || endIndex <= startIndex) {
    throw new Error('Не удалось найти JSON в ответе модели');
  }

  const jsonString = candidate.slice(startIndex, endIndex + 1);

  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.warn('Не удалось распарсить JSON модели. Исходный ответ:', raw, error);
    throw new Error('Ответ модели содержит некорректный JSON');
  }
};

const pickPairsFromPayload = (payload: unknown): RawGeneratedPair[] => {
  if (!payload || typeof payload !== 'object') {
    return [];
  }

  if (Array.isArray(payload)) {
    return payload
      .map((item) => {
        if (!item || typeof item !== 'object') return null;
        const question = 'question' in item && typeof item.question === 'string' ? item.question : null;
        const answer = 'answer' in item && typeof item.answer === 'string' ? item.answer : null;
        if (!question || !answer) return null;
        return { question: normalizeField(question), answer: normalizeField(answer) };
      })
      .filter((item): item is RawGeneratedPair => Boolean(item));
  }

  const record = payload as Record<string, unknown>;
  if (Array.isArray(record.pairs)) {
    return pickPairsFromPayload(record.pairs);
  }

  if (record.question && record.answer && typeof record.question === 'string' && typeof record.answer === 'string') {
    return [
      {
        question: normalizeField(String(record.question)),
        answer: normalizeField(String(record.answer)),
      },
    ];
  }

  return [];
};

const evaluatePairQuality = (question: string, answer: string, paragraph: string) => {
  const issues: string[] = [];
  let score = 1;

  const questionWordCount = countWords(question);
  const answerWordCount = countWords(answer);

  if (questionWordCount < 5) {
    issues.push('Вопрос слишком короткий');
    score -= 0.25;
  }

  if (!question.includes('?')) {
    issues.push('Вопрос не содержит вопросительного знака');
    score -= 0.1;
  }

  if (answerWordCount < 8) {
    issues.push('Ответ слишком короткий');
    score -= 0.25;
  }

  const questionLower = question.toLowerCase();
  const answerLower = answer.toLowerCase();

  if (SOURCE_REFERENCE_PHRASES.some((phrase) => questionLower.includes(phrase))) {
    issues.push('Вопрос ссылается на текст вместо самостоятельной формулировки');
    score -= 0.3;
  }

  if (SOURCE_REFERENCE_PHRASES.some((phrase) => answerLower.includes(phrase))) {
    issues.push('Ответ содержит ссылку на текст');
    score -= 0.2;
  }

  const combined = `${question} ${answer}`;
  const specialSymbols = combined.match(SPECIAL_SYMBOL_REGEX);
  if (specialSymbols && specialSymbols.length / combined.length > 0.08) {
    issues.push('Слишком много специальных символов');
    score -= 0.2;
  }

  if (REPEATED_SEQUENCE_REGEX.test(combined)) {
    issues.push('Повторяющиеся символы в вопросе или ответе');
    score -= 0.15;
  }

  const paragraphNormalized = paragraph.toLowerCase();
  const salientWords = answer
    .toLowerCase()
    .match(/[a-zа-яё0-9]{4,}/g);

  if (salientWords && salientWords.length > 0) {
    const uniqueWords = Array.from(new Set(salientWords));
    const matched = uniqueWords.filter((word) => paragraphNormalized.includes(word)).length;
    const coverage = matched / uniqueWords.length;
    if (coverage < 0.4) {
      issues.push('Ответ плохо опирается на исходный текст');
      score -= 0.25;
    }
  }

  if (questionWordCount > 0 && answerWordCount > 0) {
    const overlap = new Set(
      question
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean)
        .filter((token) => answer.toLowerCase().includes(token)),
    ).size;

    if (overlap / Math.max(questionWordCount, 1) > 0.7) {
      issues.push('Вопрос повторяет ответ без переформулировки');
      score -= 0.15;
    }
  }

  score = Math.max(0, Math.min(1, Number(score.toFixed(2))));

  return { score, issues };
};

const FileProcessingTab = () => {
  const {
    files,
    setFiles,
    qaPairs,
    setQaPairs,
    isProcessing,
    setIsProcessing,
    processingProgress,
    setProcessingProgress,
    generationSource,
    setGenerationSource,
    validationMode,
    setValidationMode,
    defaultPairsPerParagraph,
    setDefaultPairsPerParagraph,
    logs,
    setLogs,
  } = useFileProcessing();
  const { dataset: trainingDataset, addDatasetItems } = useTraining();

  const [showQualityPanel, setShowQualityPanel] = useState(false);
  const [filter, setFilter] = useState<'all' | 'approved' | 'rejected' | 'pending'>('all');
  const [expandedFileId, setExpandedFileId] = useState<string | null>(null);
  const [cancellationRequested, setCancellationRequested] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const cancelRequestedRef = useRef(false);
  const { addProcessingJob } = useHistory();
  const { setActivity, updateActivity, clearActivity } = useStatus();
  const { settings } = useSettings();

  const appendLog = useCallback((level: GenerationLog['level'], message: string, context?: string) => {
    const entry: GenerationLog = {
      id: createId('log'),
      timestamp: Date.now(),
      level,
      message,
      context,
    };
    setLogs((prev) => [...prev, entry].slice(-300));
  }, [setLogs]);

  const clampDesiredPairs = (value: number) => Math.max(0, Math.min(5, Math.floor(value)));

  const handleParagraphPairsChange = (fileId: string, paragraphId: string, value: number) => {
    setFiles(prev => prev.map(file => {
      if (file.id !== fileId) return file;
      return {
        ...file,
        paragraphs: file.paragraphs.map(paragraph =>
          paragraph.id === paragraphId
            ? { ...paragraph, desiredPairs: clampDesiredPairs(value) }
            : paragraph,
        ),
      };
    }));
  };

  const applyDefaultPairsToFile = (fileId: string, value: number) => {
    setFiles(prev => prev.map(file => {
      if (file.id !== fileId) return file;
      const desired = clampDesiredPairs(value);
      return {
        ...file,
        paragraphs: file.paragraphs.map(paragraph => ({ ...paragraph, desiredPairs: desired })),
      };
    }));
  };

  const applyDefaultPairsToAllFiles = () => {
    const desired = clampDesiredPairs(defaultPairsPerParagraph);
    setFiles(prev =>
      prev.map(file => ({
        ...file,
        paragraphs: file.paragraphs.map(paragraph => ({ ...paragraph, desiredPairs: desired })),
      })),
    );
    appendLog('info', `Применено ${desired} QA на абзац для всех файлов`, 'Глобальные настройки');
  };

  const clearLogs = () => setLogs([]);

  const totalPlannedPairs = useMemo(
    () =>
      files.reduce(
        (sum, file) =>
          sum + file.paragraphs.reduce((acc, paragraph) => acc + (paragraph.desiredPairs ?? 0), 0),
        0,
      ),
    [files],
  );

  const stats: ProcessingStats = {
    totalFiles: files.length,
    processedFiles: files.filter(f => f.processed).length,
    generatedPairs: qaPairs.length,
    approvedPairs: qaPairs.filter(q => q.approved).length,
    rejectedPairs: qaPairs.filter(q => q.rejected).length,
    avgQuality: qaPairs.length > 0 ? qaPairs.reduce((sum, q) => sum + q.quality, 0) / qaPairs.length : 0,
  };

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = Array.from(event.target.files || []);
    if (uploadedFiles.length === 0) {
      return;
    }

    const preparedFiles: DocumentFile[] = [];

    for (const file of uploadedFiles) {
      try {
        const rawContent = await extractTextFromFile(file);
        const paragraphs = splitIntoParagraphs(rawContent);
        preparedFiles.push({
          id: createId('file'),
          name: file.name,
          type: file.type,
          size: file.size,
          content: rawContent,
          processed: false,
          timestamp: Date.now(),
          paragraphs,
        });

        appendLog(
          'info',
          `Файл "${file.name}" загружен (${paragraphs.length} абзацев обнаружено)`,
          file.name,
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : 'неизвестная ошибка';
        appendLog('error', `Не удалось прочитать файл "${file.name}": ${message}`, file.name);
      }
    }

    if (preparedFiles.length > 0) {
      setFiles(prev => [...prev, ...preparedFiles]);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const requestPairsFromModel = async (
    paragraph: ParagraphConfig,
    desiredPairs: number,
    signal?: AbortSignal,
  ): Promise<RawGeneratedPair[]> => {
    const baseMessage = `Тебе дан абзац текста на русском языке. Сформируй ${desiredPairs} пар "вопрос-ответ" для обучения модели. Делай вопросы полезными для дообучения LLM, а не ради формальности:
- Выявляй конкретные факты, связи, выводы, аргументы, перечисления, условия, последствия, сравнения и численные данные из абзаца.
- Формулируй вопросы как самостоятельные — не используй выражения вроде "согласно тексту", "в абзаце", "по статье" и т.п.
- Избегай общих или определяющих вопросов вроде "Что такое...", "О чём этот текст?" если абзац не даёт точного ответа.
- Не задавай вопросы, на которые нельзя ответить строго по абзацу, но делай так, чтобы ответ выглядел универсальным знанием, пригодным вне контекста абзаца.
Ответы формулируй кратко, но содержательно и опираясь только на текст. Не выдумывай факты, не ссылайся на внешний контекст. Возвращай только JSON.`;

    const messages: RemoteChatMessage[] = [
      {
        role: 'system',
        content:
          'Ты — помощник по подготовке данных для обучения. Отвечай на русском языке, формируй содержательные вопросы и ответы и соблюдай формат JSON без дополнительных комментариев.',
      },
      {
        role: 'user',
        content: `${baseMessage}\n\nАбзац:\n"""\n${paragraph.text}\n"""\n\nФормат ответа:\n{\n  "pairs": [\n    { "question": "...", "answer": "..." }\n  ]\n}\n\nКоличество элементов в массиве должно быть ровно ${desiredPairs}. Не добавляй других полей.`,
      },
    ];

    if (generationSource === 'remote') {
      if (!settings.remoteApiUrl?.trim()) {
        throw new Error('Не указан URL удалённого API');
      }
      const result = await callRemoteChat(messages, {
        apiUrl: settings.remoteApiUrl,
        apiKey: settings.remoteApiKey,
        model: settings.remoteModelId,
        maxTokens: Math.min(settings.maxTokens, 1024),
        temperature: settings.temperature,
        topP: settings.topP,
        signal,
      });
      const payload = extractJsonPayload(result.content);
      return pickPairsFromPayload(payload);
    }

    if (!settings.baseModelServerUrl?.trim() || !settings.baseModelPath?.trim()) {
      throw new Error('Не заданы параметры базовой модели');
    }
    const baseResult = await callBaseChat(messages, {
      serverUrl: settings.baseModelServerUrl,
      modelPath: settings.baseModelPath,
      maxTokens: Math.min(settings.maxTokens, 1024),
      temperature: settings.temperature,
      topP: settings.topP,
      quantization: settings.quantization,
      device: settings.deviceType,
      signal,
    });
    const payload = extractJsonPayload(baseResult.content);
    return pickPairsFromPayload(payload);
  };

  const generateQAPairs = async (
    fileId: string,
    options?: { resetLogs?: boolean },
  ): Promise<'success' | 'error' | 'cancelled'> => {
    const { resetLogs = true } = options ?? {};
    const file = files.find(f => f.id === fileId);
    if (!file) {
      return 'error';
    }

    if (!file.paragraphs || file.paragraphs.length === 0) {
      appendLog('error', `В файле "${file.name}" не найдено абзацев`, file.name);
      updateActivity('dataset', { status: 'error', message: 'Не удалось выделить абзацы', progress: 0 });
      return 'error';
    }

    const tasks = file.paragraphs.filter(paragraph => paragraph.desiredPairs > 0);
    if (tasks.length === 0) {
      appendLog('warn', `Для файла "${file.name}" не задано ни одного абзаца с генерацией QA`, file.name);
      updateActivity('dataset', { status: 'error', message: 'Нет параграфов для генерации', progress: 0 });
      return 'error';
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    cancelRequestedRef.current = false;
    setCancellationRequested(false);

    const totalPairsPlanned = tasks.reduce((sum, paragraph) => sum + paragraph.desiredPairs, 0);
    let generatedPairsCount = 0;
    let successfulPairs = 0;
    let rejectedPairs = 0;
    let activityStatus: 'success' | 'error' = 'success';
    let cancelledByUser = false;
    let clearedExistingPairs = false;
    let lastProgress = 0;
    const newPairs: QAPair[] = [];

    const pushPairs = (pairs: QAPair[]) => {
      if (pairs.length === 0) {
        return;
      }
      newPairs.push(...pairs);
      setQaPairs(prev => {
        const base = clearedExistingPairs ? prev : prev.filter(qa => qa.fileId !== file.id);
        clearedExistingPairs = true;
        return [...base, ...pairs];
      });
    };

    setIsProcessing(true);
    setProcessingProgress(0);
    setActivity('dataset', { message: 'Формирует датасет', progress: 0 });
    if (resetLogs) {
      setLogs([]);
    }
    appendLog(
      'info',
      `Запуск генерации QA для файла "${file.name}" (${tasks.length} абзацев, план: ${totalPairsPlanned} QA, источник: ${generationSource === 'remote' ? 'удалённая модель' : 'базовая модель'})`,
      file.name,
    );

    let completion: { status: 'success' | 'error' | 'cancelled'; timeout: number } = {
      status: 'error',
      timeout: 4000,
    };

    try {
      const processParagraph = async (paragraph: ParagraphConfig, index: number) => {
        if (cancelRequestedRef.current) {
          cancelledByUser = true;
          return;
        }

        appendLog(
          'info',
          `Абзац ${index + 1}/${tasks.length}: планируется ${paragraph.desiredPairs} QA (~${paragraph.tokenCount} слов)`,
          file.name,
        );

        try {
          const rawPairs = await requestPairsFromModel(
            paragraph,
            paragraph.desiredPairs,
            abortController.signal,
          );

          if (cancelRequestedRef.current) {
            cancelledByUser = true;
            return;
          }

          if (rawPairs.length === 0) {
            appendLog('warn', `Модель не сформировала пары для абзаца ${index + 1}`, file.name);
            return;
          }

          const limit = Math.min(paragraph.desiredPairs, rawPairs.length);
          const baseIndex = generatedPairsCount;

          if (limit < paragraph.desiredPairs) {
            appendLog(
              'warn',
              `Получено только ${limit} из ${paragraph.desiredPairs} пар для абзаца ${index + 1}`,
              file.name,
            );
          }

          const preparedPairs: QAPair[] = [];

          rawPairs.slice(0, limit).forEach((pair, pairIndex) => {
            const pairNumber = baseIndex + pairIndex + 1;
            const question = normalizeField(pair.question);
            const answer = normalizeField(pair.answer);
            const { score, issues } = evaluatePairQuality(question, answer, paragraph.text);

            const autoApproved = validationMode === 'auto' && issues.length === 0;
            const autoRejected = validationMode === 'auto' && issues.length > 0;

            if (autoApproved) {
              successfulPairs += 1;
              appendLog('info', `QA пара ${pairNumber} автоматически одобрена`, file.name);
            } else if (autoRejected) {
              rejectedPairs += 1;
              appendLog('warn', `QA пара ${pairNumber} отклонена: ${issues.join('; ')}`, file.name);
            } else {
              appendLog('info', `QA пара ${pairNumber} создана и ждёт проверки`, file.name);
            }

            preparedPairs.push({
              id: createId('qa'),
              fileId: file.id,
              paragraphId: paragraph.id,
              question,
              answer,
              source: `${file.name} • абзац ${paragraph.index + 1}`,
              quality: score,
              approved: autoApproved,
              rejected: autoRejected,
              issues: issues.length > 0 ? issues : undefined,
              generationSource,
            });
          });

          generatedPairsCount += limit;
          pushPairs(preparedPairs);

          const progress = tasks.length > 0 ? Math.min(100, ((index + 1) / tasks.length) * 100) : 100;
          lastProgress = progress;
          setProcessingProgress(progress);
          updateActivity('dataset', { progress, message: `Абзац ${index + 1} из ${tasks.length}` });
        } catch (error) {
          if (cancelRequestedRef.current) {
            cancelledByUser = true;
            appendLog('warn', `Генерация остановлена пользователем на абзаце ${index + 1}`, file.name);
            return;
          }

          const isAbortError =
            (error instanceof RemoteChatError || error instanceof BaseChatError) &&
            error.message === 'Запрос отменён пользователем';

          if (isAbortError) {
            cancelledByUser = true;
            appendLog('warn', `Генерация остановлена пользователем на абзаце ${index + 1}`, file.name);
            return;
          }

          activityStatus = 'error';
          const errorMessage =
            error instanceof RemoteChatError || error instanceof BaseChatError
              ? error.message
              : error instanceof Error
                ? error.message
                : 'Неизвестная ошибка генерации';
          appendLog('error', `Ошибка при обработке абзаца ${index + 1}: ${errorMessage}`, file.name);
        }
      };

      for (let index = 0; index < tasks.length; index += 1) {
        if (cancelRequestedRef.current) {
          cancelledByUser = true;
          break;
        }
        await processParagraph(tasks[index], index);
        if (cancelRequestedRef.current) {
          cancelledByUser = true;
          break;
        }
      }

      if (cancelledByUser) {
        const summaryMessage =
          newPairs.length > 0
            ? `Генерация остановлена пользователем: получено ${newPairs.length} QA из запланированных ${totalPairsPlanned}`
            : 'Генерация остановлена пользователем до получения новых QA';
        appendLog('warn', summaryMessage, file.name);
        updateActivity('dataset', {
          status: 'error',
          message: 'Генерация остановлена пользователем',
          progress: lastProgress,
        });

        if (newPairs.length > 0) {
          addProcessingJob({
            id: Date.now().toString(),
            timestamp: Date.now(),
            fileName: file.name,
            qaGenerated: newPairs.length,
            status: 'cancelled',
          });
        }

        completion = { status: 'cancelled', timeout: 4000 };
        return 'cancelled';
      }

      if (newPairs.length === 0) {
        activityStatus = 'error';
        updateActivity('dataset', {
          status: 'error',
          message: 'Не удалось сгенерировать QA пары',
          progress: 0,
        });
        completion = { status: 'error', timeout: 5000 };
        return 'error';
      }

      setFiles(prev => prev.map(f => (f.id === fileId ? { ...f, processed: true } : f)));

      addProcessingJob({
        id: Date.now().toString(),
        timestamp: Date.now(),
        fileName: file.name,
        qaGenerated: newPairs.length,
        status: activityStatus === 'success' ? 'completed' : 'failed',
      });

      if (activityStatus === 'success') {
        appendLog(
          'info',
          `Генерация завершена: создано ${newPairs.length} QA пар из запланированных ${totalPairsPlanned} (автоодобрено: ${successfulPairs}, отклонено: ${rejectedPairs})`,
          file.name,
        );
        updateActivity('dataset', { status: 'success', message: 'Датасет сформирован', progress: 100 });
        completion = { status: 'success', timeout: 2500 };
        return 'success';
      }

      updateActivity('dataset', { status: 'error', message: 'Генерация завершилась с ошибками', progress: 100 });
      completion = { status: 'error', timeout: 5000 };
      return 'error';
    } finally {
      abortControllerRef.current = null;
      cancelRequestedRef.current = false;
      setCancellationRequested(false);
      setIsProcessing(false);
      setProcessingProgress(0);
      window.setTimeout(() => clearActivity('dataset'), completion.timeout);
    }
  };

  const handleProcessAllFiles = async () => {
    if (isProcessing) {
      appendLog('warn', 'Генерация уже выполняется, дождитесь завершения текущей задачи', 'Глобальные настройки');
      return;
    }

    const pendingFiles = files.filter(
      file => !file.processed && file.paragraphs.some(paragraph => paragraph.desiredPairs > 0),
    );

    if (pendingFiles.length === 0) {
      appendLog('warn', 'Нет файлов, готовых к пакетной обработке', 'Глобальные настройки');
      return;
    }

    appendLog('info', `Пакетная обработка запущена (${pendingFiles.length} файлов)`, 'Глобальные настройки');

    let cancelled = false;

    for (let index = 0; index < pendingFiles.length; index += 1) {
      const result = await generateQAPairs(pendingFiles[index].id, { resetLogs: index === 0 });
      if (result === 'cancelled') {
        cancelled = true;
        break;
      }
    }

    if (cancelled) {
      appendLog('warn', 'Пакетная обработка остановлена пользователем', 'Глобальные настройки');
      return;
    }

    appendLog('info', 'Пакетная обработка завершена', 'Глобальные настройки');
  };

  const handleCancelProcessing = () => {
    if (!isProcessing || cancelRequestedRef.current) {
      return;
    }
    cancelRequestedRef.current = true;
    setCancellationRequested(true);
    appendLog('warn', 'Остановка генерации запрошена пользователем', 'Глобальные настройки');
    updateActivity('dataset', { message: 'Останавливаем обработку...', status: 'running' });
    abortControllerRef.current?.abort();
  };

  const handleApproveQA = (id: string) => {
    setQaPairs(prev => prev.map(qa => 
      qa.id === id ? { ...qa, approved: true, rejected: false } : qa
    ));
  };

  const handleRejectQA = (id: string) => {
    setQaPairs(prev => prev.map(qa => 
      qa.id === id ? { ...qa, approved: false, rejected: true } : qa
    ));
  };

  const handleDeleteFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
    setQaPairs(prev => prev.filter(qa => qa.fileId !== id));
    appendLog('info', `Файл удалён вместе с QA парами`, id);
  };

  const handleExportDataset = () => {
    const approvedPairs = qaPairs.filter(qa => qa.approved);
    const dataset = approvedPairs.map(qa => ({
      input: qa.question,
      output: qa.answer,
      source: qa.source,
      quality: qa.quality,
    }));

    const jsonData = JSON.stringify(dataset, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `qa-dataset-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleAddToTraining = () => {
    const approvedPairs = qaPairs.filter(qa => qa.approved);
    if (approvedPairs.length === 0) {
      appendLog('warn', 'Нет одобренных QA пар для добавления в обучение', 'Глобальные настройки');
      return;
    }

    const existingIds = new Set(trainingDataset.map(item => item.id));
    const itemsToAdd: DatasetItem[] = approvedPairs
      .map(qa => ({
        id: qa.id,
        input: qa.question.trim(),
        output: qa.answer.trim(),
        source: qa.source,
      }))
      .filter(item => !existingIds.has(item.id));

    if (itemsToAdd.length === 0) {
      appendLog('info', 'Все одобренные QA пары уже находятся в датасете обучения', 'Глобальные настройки');
      if (!isProcessing) {
        setActivity('dataset', { message: 'Одобренные пары уже в обучении', status: 'warning' });
        window.setTimeout(() => clearActivity('dataset'), 2500);
      }
      return;
    }

    addDatasetItems(itemsToAdd);
    appendLog(
      'info',
      `Добавлено ${itemsToAdd.length} QA пар в датасет обучения (из ${approvedPairs.length} одобренных)`,
      'Глобальные настройки',
    );

    if (!isProcessing) {
      setActivity('dataset', {
        message: `В обучение добавлено ${itemsToAdd.length} QA пар`,
        status: 'success',
      });
      window.setTimeout(() => clearActivity('dataset'), 2500);
    }
  };

  const filteredQAPairs = qaPairs.filter(qa => {
    switch (filter) {
      case 'approved': return qa.approved;
      case 'rejected': return qa.rejected;
      case 'pending': return !qa.approved && !qa.rejected;
      default: return true;
    }
  });

  const {
    containerRef: qaListRef,
    handleScroll: handleQaListScroll,
    visibleItems: visibleQAPairs,
    hasMore: hasMoreQAPairs,
    loadMore: loadMoreQAPairs,
  } = useLazyList(filteredQAPairs, {
    initialBatchSize: 20,
    batchSize: 20,
    resetKey: `${filter}-${qaPairs.length}`,
  });

  const getQualityColor = (quality: number) => {
    if (quality > 0.8) return 'text-green-400 bg-green-900/30';
    if (quality > 0.6) return 'text-amber-300 bg-amber-900/30';
    return 'text-rose-400 bg-rose-900/30';
  };

  return (
    <div className="h-full flex">
      {/* Main Content */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Обработка файлов</h2>
            <div className="flex gap-3">
              <button
                onClick={() => setShowQualityPanel(!showQualityPanel)}
                className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                title={showQualityPanel ? 'Скрыть аналитику качества' : 'Показать аналитику качества'}
              >
                <Eye className="w-4 h-4" />
                Аналитика
              </button>
              <button
                onClick={handleExportDataset}
                disabled={stats.approvedPairs === 0}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title="Экспортировать одобренные QA пары"
              >
                <Download className="w-4 h-4" />
                Экспорт
              </button>
              <button
                onClick={handleAddToTraining}
                disabled={stats.approvedPairs === 0}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title="Добавить одобренные пары в тренировочный датасет"
              >
                <Plus className="w-4 h-4" />
                В обучение
              </button>
            </div>
          </div>

          {/* Generation Settings */}
          <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
            <div className="flex flex-wrap gap-6 items-end">
              <div className="flex flex-col gap-1">
                <label className="text-sm font-medium text-slate-200">Источник генерации</label>
                <div className="flex items-center gap-2">
                  <Settings className="w-4 h-4 text-slate-400" />
                  <select
                    value={generationSource}
                    onChange={(e) => setGenerationSource(e.target.value as GenerationSource)}
                    className="border border-slate-700 bg-slate-950 text-slate-100 placeholder-slate-500 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="remote">Удалённая модель (API)</option>
                    <option value="base">Базовая локальная модель</option>
                  </select>
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-sm font-medium text-slate-200">Режим проверки</label>
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-slate-400" />
                  <select
                    value={validationMode}
                    onChange={(e) => setValidationMode(e.target.value as ValidationMode)}
                    className="border border-slate-700 bg-slate-950 text-slate-100 placeholder-slate-500 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="auto">Автоматический (одобрять/отклонять)</option>
                    <option value="manual">Ручной (только флаги качества)</option>
                  </select>
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-sm font-medium text-slate-200">QA на абзац (по умолчанию)</label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={5}
                    value={defaultPairsPerParagraph}
                    onChange={(e) => {
                      const nextValue = Number(e.target.value);
                      setDefaultPairsPerParagraph(clampDesiredPairs(Number.isNaN(nextValue) ? 0 : nextValue));
                    }}
                    className="w-20 border border-slate-700 bg-slate-950 text-slate-100 placeholder-slate-500 rounded-lg px-3 py-2 text-sm"
                  />
                  <button
                    onClick={applyDefaultPairsToAllFiles}
                    className="px-3 py-2 text-sm bg-slate-900 text-slate-200 rounded-lg hover:bg-slate-800 transition-colors"
                    title="Применить выбранное значение ко всем загруженным файлам"
                  >
                    Применить ко всем
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2 text-sm text-slate-300 bg-slate-950/40 border border-slate-800 rounded-lg px-3 py-2">
                <Info className="w-4 h-4 text-blue-500" />
                <div>
                  Запланировано генераций: <span className="font-semibold text-slate-100">{totalPlannedPairs}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
              <div className="text-sm text-slate-300">Файлов</div>
              <div className="text-2xl font-bold text-slate-50">{stats.totalFiles}</div>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
              <div className="text-sm text-slate-300">Обработано</div>
              <div className="text-2xl font-bold text-blue-400">{stats.processedFiles}</div>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
              <div className="text-sm text-slate-300">QA пар</div>
              <div className="text-2xl font-bold text-purple-400">{stats.generatedPairs}</div>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
              <div className="text-sm text-slate-300">Одобрено</div>
              <div className="text-2xl font-bold text-emerald-400">{stats.approvedPairs}</div>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
              <div className="text-sm text-slate-300">Отклонено</div>
              <div className="text-2xl font-bold text-rose-400">{stats.rejectedPairs}</div>
            </div>
            <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
              <div className="text-sm text-slate-300">Качество</div>
              <div className="text-2xl font-bold text-orange-400">{(stats.avgQuality * 100).toFixed(0)}%</div>
            </div>
          </div>

          {/* File Upload */}
          <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-4">
              <h3 className="text-lg font-semibold">Загрузка документов</h3>
              <button
                onClick={handleProcessAllFiles}
                disabled={isProcessing || files.length === 0}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isProcessing || files.length === 0
                    ? 'bg-slate-800 text-slate-400 cursor-not-allowed'
                    : 'bg-emerald-600 text-white hover:bg-emerald-500'
                }`}
                title="Запустить генерацию QA для всех файлов"
              >
                Обработать все
              </button>
            </div>
            
          <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-slate-500 transition-colors">
              <Upload className="w-12 h-12 text-slate-500 mx-auto mb-4" />
              <p className="text-slate-300 mb-4">
                Перетащите файлы сюда или нажмите для выбора
              </p>
              <p className="text-sm text-slate-400 mb-4">
                Поддерживаются: PDF, DOCX, TXT и другие текстовые форматы
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.txt,.md"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                title="Выбрать документы для загрузки"
              >
                Выбрать файлы
              </button>
            </div>
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <h3 className="text-lg font-semibold mb-4">Загруженные файлы</h3>
              
              <div className="space-y-3">
                {files.map((file) => {
                  const plannedPairs = file.paragraphs.reduce((acc, paragraph) => acc + paragraph.desiredPairs, 0);
                  const isExpanded = expandedFileId === file.id;
                  return (
                    <div key={file.id} className="bg-slate-950/40 border border-slate-800 rounded-lg p-4 space-y-3">
                      <div className="flex flex-wrap justify-between gap-3 items-start">
                        <div className="flex items-start gap-3">
                          <FileText className="w-5 h-5 text-blue-400 mt-1" />
                          <div>
                            <div className="font-medium text-slate-50">{file.name}</div>
                            <div className="text-sm text-slate-400">
                              {(file.size / 1024).toFixed(1)} KB • {new Date(file.timestamp).toLocaleString()}
                            </div>
                            <div className="text-xs text-slate-400 mt-1">
                              Абзацев: {file.paragraphs.length} • План: {plannedPairs} QA
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setExpandedFileId(isExpanded ? null : file.id)}
                            className="px-3 py-2 text-sm bg-slate-900 border border-slate-700 text-slate-200 rounded-lg hover:bg-slate-900 transition-colors"
                            title={isExpanded ? 'Скрыть настройки абзацев' : 'Открыть настройки абзацев'}
                          >
                            {isExpanded ? 'Скрыть абзацы' : 'Настроить абзацы'}
                          </button>

                          {file.processed ? (
                            <span className="flex items-center gap-1 text-sm text-emerald-400">
                              <CheckCircle className="w-4 h-4" />
                              Обработан
                            </span>
                          ) : (
                            <button
                              onClick={() => generateQAPairs(file.id)}
                              disabled={isProcessing || plannedPairs === 0}
                              className="flex items-center gap-2 px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                              title="Сгенерировать QA пары для файла"
                            >
                              <Play className="w-4 h-4" />
                              Обработать
                            </button>
                          )}

                          <button
                            onClick={() => handleDeleteFile(file.id)}
                            className="p-2 text-rose-400 hover:bg-rose-900/30 rounded-lg transition-colors"
                            title="Удалить файл"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      {plannedPairs === 0 && (
                        <div className="text-sm text-amber-300 bg-amber-900/30 border border-amber-700 rounded-lg px-3 py-2">
                          Для этого файла не запланировано генераций. Установите количество QA на абзац.
                        </div>
                      )}

                      {isExpanded && (
                        <div className="border-t border-slate-800 pt-3">
                          <div className="flex justify-between items-center mb-3">
                            <span className="text-sm font-semibold text-slate-200">Параметры абзацев</span>
                            <button
                              onClick={() => applyDefaultPairsToFile(file.id, defaultPairsPerParagraph)}
                              className="text-sm px-3 py-1 border border-slate-700 text-slate-200 rounded-lg hover:bg-slate-900 transition-colors"
                              title="Установить одинаковое количество QA для всех абзацев файла"
                            >
                              Установить {defaultPairsPerParagraph} всем
                            </button>
                          </div>
                          <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
                            {file.paragraphs.map((paragraph) => {
                              const preview = paragraph.text.length > 240
                                ? `${paragraph.text.slice(0, 240)}...`
                                : paragraph.text;
                              return (
                                <div key={paragraph.id} className="bg-slate-900 border border-slate-800 rounded-lg p-3">
                                  <div className="flex justify-between items-center mb-2 text-xs text-slate-400">
                                    <span>Абзац {paragraph.index + 1} • {paragraph.tokenCount} слов</span>
                                    <div className="flex items-center gap-2">
                                      <span className="text-slate-400">QA</span>
                                      <input
                                        type="number"
                                        min={0}
                                        max={5}
                                        value={paragraph.desiredPairs}
                                        onChange={(e) => {
                                          const next = Number(e.target.value);
                                          handleParagraphPairsChange(
                                            file.id,
                                            paragraph.id,
                                            Number.isNaN(next) ? 0 : next,
                                          );
                                        }}
                                        className="w-16 border border-slate-700 bg-slate-950 text-slate-100 placeholder-slate-500 rounded px-2 py-1"
                                      />
                                    </div>
                                  </div>
                                  <p className="text-sm text-slate-200 whitespace-pre-wrap max-h-24 overflow-y-auto">
                                    {preview}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Processing Progress */}
              {isProcessing && (
                <div className="mt-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 text-sm mb-2">
                    <span>{cancellationRequested ? 'Останавливаем генерацию...' : 'Генерация QA пар...'}</span>
                    <div className="flex items-center gap-3">
                      <button
                        onClick={handleCancelProcessing}
                        disabled={cancellationRequested}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                          cancellationRequested
                            ? 'bg-slate-800 text-slate-400 cursor-not-allowed'
                            : 'bg-rose-600 text-white hover:bg-rose-500'
                        }`}
                        title="Остановить текущую обработку"
                      >
                        <Square className="w-3.5 h-3.5" />
                        Остановить
                      </button>
                      <span>{Math.round(processingProgress)}%</span>
                    </div>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${processingProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* QA Pairs */}
          {qaPairs.length > 0 && (
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Сгенерированные QA пары</h3>
                
                <div className="flex items-center gap-3">
                  <Filter className="w-4 h-4 text-slate-400" />
                  <select
                    value={filter}
                    onChange={(e) =>
                      setFilter(
                        e.target.value as 'all' | 'approved' | 'rejected' | 'pending',
                      )
                    }
                    className="px-3 py-2 border border-slate-700 bg-slate-950 text-slate-100 placeholder-slate-500 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">Все ({qaPairs.length})</option>
                    <option value="pending">Ожидают ({qaPairs.filter(q => !q.approved && !q.rejected).length})</option>
                    <option value="approved">Одобрено ({stats.approvedPairs})</option>
                    <option value="rejected">Отклонено ({stats.rejectedPairs})</option>
                  </select>
                </div>
              </div>
              
              <div
                ref={qaListRef}
                onScroll={handleQaListScroll}
                className="space-y-4 max-h-96 overflow-y-auto pr-1"
              >
                {visibleQAPairs.map((qa) => (
                  <div key={qa.id} className="border border-slate-800 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getQualityColor(qa.quality)}`}>
                          {(qa.quality * 100).toFixed(0)}%
                        </span>
                        <span className="text-xs text-slate-400">из {qa.source}</span>
                        <span className="text-xs uppercase font-semibold tracking-wide text-slate-500">
                          {qa.generationSource === 'remote' ? 'REMOTE' : 'BASE'}
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {!qa.approved && !qa.rejected && (
                          <>
                            <button
                              onClick={() => handleApproveQA(qa.id)}
                              className="flex items-center gap-1 px-3 py-1 text-sm bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 transition-colors"
                              title="Отметить пару как одобренную"
                            >
                              <CheckCircle className="w-3 h-3" />
                              Одобрить
                            </button>
                            <button
                              onClick={() => handleRejectQA(qa.id)}
                              className="flex items-center gap-1 px-3 py-1 text-sm bg-rose-600 text-white rounded-lg hover:bg-rose-500 transition-colors"
                              title="Отклонить пару"
                            >
                              <XCircle className="w-3 h-3" />
                              Отклонить
                            </button>
                          </>
                        )}
                        
                        {qa.approved && (
                          <span className="flex items-center gap-1 text-sm text-emerald-400">
                            <CheckCircle className="w-4 h-4" />
                            Одобрено
                          </span>
                        )}
                        
                        {qa.rejected && (
                          <span className="flex items-center gap-1 text-sm text-rose-400">
                            <XCircle className="w-4 h-4" />
                            Отклонено
                          </span>
                        )}
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div>
                        <div className="text-sm font-medium text-slate-200 mb-1">Вопрос:</div>
                        <div className="text-sm text-slate-50">{qa.question}</div>
                      </div>
                      
                      <div>
                        <div className="text-sm font-medium text-slate-200 mb-1">Ответ:</div>
                        <div className="text-sm text-slate-50">{qa.answer}</div>
                      </div>
                      
                      {qa.issues && qa.issues.length > 0 && (
                        <div>
                          <div className="text-sm font-medium text-slate-200 mb-1">Предупреждения:</div>
                          <div className="flex flex-wrap gap-1">
                            {qa.issues.map((issue, index) => (
                              <span key={index} className="px-2 py-1 bg-amber-900/30 text-xs text-amber-300 rounded">
                                {issue}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {hasMoreQAPairs && (
                  <div className="flex justify-center py-2">
                    <button
                      type="button"
                      onClick={loadMoreQAPairs}
                      className="rounded-lg border border-slate-700 px-4 py-1.5 text-sm text-slate-200 transition-colors hover:border-blue-500 hover:text-blue-300"
                    >
                      Показать ещё
                    </button>
                  </div>
                )}

                {filteredQAPairs.length === 0 && (
                  <div className="text-center py-8 text-slate-400">
                    {filter === 'all' ? 'QA пары не созданы' : `Нет ${filter === 'approved' ? 'одобренных' : filter === 'rejected' ? 'отклонённых' : 'ожидающих'} пар`}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Logs */}
          <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Журнал обработки</h3>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">Всего записей: {logs.length}</span>
                <button
                  onClick={clearLogs}
                  disabled={logs.length === 0}
                  className="px-3 py-1 text-sm border border-slate-700 bg-slate-950 text-slate-300 rounded-lg hover:bg-slate-900 disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Очистить журнал событий"
                >
                  Очистить
                </button>
              </div>
            </div>
            <div className="max-h-64 overflow-y-auto space-y-2 pr-1 text-sm">
              {logs.length === 0 && (
                <div className="text-slate-400 text-center py-6">Журнал пуст</div>
              )}
              {logs.map((log) => (
                <div
                  key={log.id}
                  className={`border rounded-lg px-3 py-2 ${
                    log.level === 'error'
                      ? 'border-rose-800 bg-rose-950/40 text-rose-300'
                      : log.level === 'warn'
                        ? 'border-amber-700 bg-amber-950/40 text-amber-300'
                        : 'border-slate-800 bg-slate-950/40 text-slate-200'
                  }`}
                >
                  <div className="flex justify-between items-center text-xs mb-1">
                    <span className="font-semibold uppercase tracking-wide">
                      {log.level === 'error' ? 'Ошибка' : log.level === 'warn' ? 'Предупреждение' : 'Инфо'}
                    </span>
                    <span className="text-slate-400">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div>{log.message}</div>
                  {log.context && (
                    <div className="text-xs text-slate-400 mt-1">Контекст: {log.context}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Quality Analysis Panel */}
      {showQualityPanel && (
        <div className="w-80 bg-slate-900 border-l border-slate-800 p-4">
          <h3 className="text-lg font-semibold mb-4">Анализ качества</h3>
          
          <div className="space-y-4">
            <div className="bg-blue-950/30 p-4 rounded-lg border border-blue-900">
              <div className="text-sm text-blue-300 font-medium">Распределение качества</div>
              <div className="mt-2 space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Высокое (80%+)</span>
                  <span className="font-medium">{qaPairs.filter(q => q.quality > 0.8).length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Среднее (60-80%)</span>
                  <span className="font-medium">{qaPairs.filter(q => q.quality > 0.6 && q.quality <= 0.8).length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Низкое (&lt;60%)</span>
                  <span className="font-medium">{qaPairs.filter(q => q.quality <= 0.6).length}</span>
                </div>
              </div>
            </div>
            
            <div className="bg-emerald-950/30 p-4 rounded-lg border border-emerald-900">
              <div className="text-sm text-emerald-300 font-medium">Готовность датасета</div>
              <div className="text-2xl font-bold text-emerald-400">
                {qaPairs.length > 0 ? Math.round((stats.approvedPairs / stats.generatedPairs) * 100) : 0}%
              </div>
              <div className="text-xs text-emerald-300 mt-1">
                {stats.approvedPairs} из {stats.generatedPairs} одобрено
              </div>
            </div>
            
            <div className="bg-amber-950/30 p-4 rounded-lg border border-amber-900">
              <div className="text-sm text-amber-300 font-medium">Рекомендации</div>
              <div className="text-xs text-amber-200 mt-1">
                {stats.approvedPairs < 10 && 'Добавьте больше одобренных пар для качественного обучения'}
                {stats.avgQuality < 0.7 && 'Рассмотрите улучшение качества генерации'}
                {stats.rejectedPairs > stats.approvedPairs && 'Слишком много отклонённых пар - проверьте настройки'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileProcessingTab;
