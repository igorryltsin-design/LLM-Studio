declare module 'jszip' {
  export type JSZipInput = string | ArrayBuffer | Uint8Array | Blob | Promise<ArrayBuffer> | Promise<Uint8Array>;

  export interface JSZipObject {
    async(type: 'string'): Promise<string>;
    async(type: 'uint8array'): Promise<Uint8Array>;
    async(type: string): Promise<unknown>;
  }

  export default class JSZip {
    static loadAsync(data: JSZipInput, options?: Record<string, unknown>): Promise<JSZip>;
    loadAsync(data: JSZipInput, options?: Record<string, unknown>): Promise<JSZip>;
    file(name: string): JSZipObject | null;
    file(name: string, data: string | ArrayBuffer | Uint8Array, options?: Record<string, unknown>): this;
    folder(name: string): JSZip;
    generateAsync(options?: Record<string, unknown>): Promise<Uint8Array>;
  }
}

declare module 'pdfjs-dist' {
  export interface PDFDocumentLoadingTask {
    promise: Promise<PDFDocumentProxy>;
  }

  export interface PDFDocumentProxy {
    numPages: number;
    getPage(pageNumber: number): Promise<PDFPageProxy>;
    destroy(): void;
  }

  export interface PDFPageProxy {
    getTextContent(): Promise<{ items: unknown[] }>;
  }

  export interface WorkerOptions {
    workerSrc: string;
  }

  export const GlobalWorkerOptions: WorkerOptions;
  export function getDocument(options: { data: ArrayBuffer | Uint8Array | string }): PDFDocumentLoadingTask;
}

declare module 'pdfjs-dist/types/src/display/api' {
  export interface TextItem {
    str?: string;
  }
}

declare module 'pdfjs-dist/build/pdf.worker.min.mjs?url' {
  const workerSrc: string;
  export default workerSrc;
}
