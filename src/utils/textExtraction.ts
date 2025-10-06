import JSZip from 'jszip';
import { GlobalWorkerOptions, getDocument } from 'pdfjs-dist';
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import type { TextItem } from 'pdfjs-dist/types/src/display/api';

const isTextItem = (value: unknown): value is TextItem =>
  typeof value === 'object' && value !== null && 'str' in value;

let pdfWorkerConfigured = false;

const configurePdfWorker = () => {
  if (pdfWorkerConfigured) {
    return;
  }

  if (typeof window === 'undefined') {
    pdfWorkerConfigured = true;
    return;
  }

  try {
    GlobalWorkerOptions.workerSrc = pdfWorkerSrc;
    pdfWorkerConfigured = true;
  } catch (error) {
    console.warn('Не удалось настроить PDF worker', error);
  }
};

export const extractPdfText = async (file: File): Promise<string> => {
  const data = await file.arrayBuffer();
  configurePdfWorker();

  const pdf = await getDocument({ data }).promise;

  try {
    const pages: string[] = [];
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      const content = await page.getTextContent();
      const text = (content.items as unknown[])
        .map((item: unknown) => (isTextItem(item) && typeof item.str === 'string' ? item.str : ''))
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim();

      if (text) {
        pages.push(text);
      }
    }

    return pages.join('\n\n');
  } finally {
    pdf.destroy();
  }
};

const DOCUMENT_XML_PATHS = [
  'word/document.xml',
  'word/document2.xml',
  'word/document3.xml',
];

const extractTextNodes = (element: Element): string[] => {
  const texts: string[] = [];
  element.childNodes.forEach(node => {
    if (node.nodeType === Node.TEXT_NODE && node.textContent) {
      texts.push(node.textContent);
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      const childElement = node as Element;
      if (childElement.tagName === 'w:tab') {
        texts.push('\t');
      } else if (childElement.tagName === 'w:br') {
        texts.push('\n');
      } else {
        texts.push(...extractTextNodes(childElement));
      }
    }
  });
  return texts;
};

export const extractDocxText = async (file: File): Promise<string> => {
  const zip = await JSZip.loadAsync(await file.arrayBuffer());

  const documentEntry = DOCUMENT_XML_PATHS
    .map(path => zip.file(path))
    .find(entry => entry !== null && typeof entry !== 'undefined');

  if (!documentEntry) {
    throw new Error('В документе DOCX не найден основной текст');
  }

  const xmlContent = await documentEntry.async('string');
  const parser = new DOMParser();
  const xml = parser.parseFromString(xmlContent, 'application/xml');

  const parserError = xml.getElementsByTagName('parsererror');
  if (parserError.length > 0) {
    throw new Error('DOCX содержит некорректный XML');
  }

  const paragraphs = Array.from(xml.getElementsByTagName('w:p'));
  const result: string[] = [];

  paragraphs.forEach(paragraph => {
    const raw = extractTextNodes(paragraph).join('');
    const cleaned = raw
      .replace(/\r/g, '')
      .replace(/\t/g, ' ')
      .replace(/ {2,}/g, ' ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    if (cleaned) {
      result.push(cleaned);
    }
  });

  return result.join('\n\n');
};

const DOC_MIME_TYPES = [
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/msword',
];

const decodeBufferWithEncoding = (data: Uint8Array, encoding: string, fatal = true) => {
  try {
    return new TextDecoder(encoding, { fatal }).decode(data);
  } catch (error) {
    if (fatal) {
      return null;
    }
    console.warn(`Не удалось декодировать текст как ${encoding}`, error);
    return null;
  }
};

const detectEncodingFromBom = (data: Uint8Array): string | null => {
  if (data.length >= 3 && data[0] === 0xef && data[1] === 0xbb && data[2] === 0xbf) {
    return 'utf-8';
  }
  if (data.length >= 2 && data[0] === 0xff && data[1] === 0xfe) {
    return 'utf-16le';
  }
  if (data.length >= 2 && data[0] === 0xfe && data[1] === 0xff) {
    return 'utf-16be';
  }
  return null;
};

const extractPlainText = async (file: File): Promise<string> => {
  const buffer = await file.arrayBuffer();
  const data = new Uint8Array(buffer);

  const bomEncoding = detectEncodingFromBom(data);
  if (bomEncoding) {
    const sliced = bomEncoding === 'utf-8' ? data.subarray(3) : data.subarray(2);
    const decoded = decodeBufferWithEncoding(sliced, bomEncoding, false);
    if (decoded) {
      return decoded.replace(/\r\n?/g, '\n');
    }
  }

  const fallbacks = ['utf-8', 'windows-1251', 'koi8-r', 'utf-16le', 'utf-16be'];
  for (const encoding of fallbacks) {
    const decoded = decodeBufferWithEncoding(data, encoding, encoding !== 'utf-8');
    if (decoded) {
      return decoded.replace(/\r\n?/g, '\n');
    }
  }

  return new TextDecoder().decode(data).replace(/\r\n?/g, '\n');
};

export const extractTextFromFile = async (file: File): Promise<string> => {
  const extension = file.name.split('.').pop()?.toLowerCase();

  if (file.type === 'application/pdf' || extension === 'pdf') {
    return extractPdfText(file);
  }

  if (DOC_MIME_TYPES.includes(file.type) || extension === 'docx') {
    return extractDocxText(file);
  }

  if (extension === 'doc') {
    throw new Error('Формат DOC не поддерживается. Сохраните файл в DOCX и попробуйте снова.');
  }

  if (file.type.startsWith('text/') || !file.type) {
    return extractPlainText(file);
  }

  throw new Error(`Неподдерживаемый формат файла: ${file.type || extension || file.name}`);
};
