import { useCallback, useEffect, useMemo, useRef, useState, type RefObject, type UIEvent } from 'react';

interface UseLazyListOptions {
  initialBatchSize?: number;
  batchSize?: number;
  threshold?: number;
  resetKey?: unknown;
}

interface UseLazyListResult<T> {
  containerRef: RefObject<HTMLDivElement>;
  handleScroll: (event: UIEvent<HTMLDivElement>) => void;
  visibleItems: T[];
  hasMore: boolean;
  loadMore: () => void;
  visibleCount: number;
}

/**
 * Lightweight lazy renderer that reveals items chunk-by-chunk as the user scrolls.
 */
export function useLazyList<T>(items: T[], options: UseLazyListOptions = {}): UseLazyListResult<T> {
  const {
    initialBatchSize = 30,
    batchSize = initialBatchSize,
    threshold = 0.85,
    resetKey,
  } = options;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [visibleCount, setVisibleCount] = useState(() => Math.min(initialBatchSize, items.length));

  useEffect(() => {
    setVisibleCount((prev) => Math.min(prev, items.length));
  }, [items.length]);

  useEffect(() => {
    setVisibleCount(Math.min(initialBatchSize, items.length));
  }, [items.length, initialBatchSize, resetKey]);

  const loadMore = useCallback(() => {
    setVisibleCount((prev) => {
      if (prev >= items.length) {
        return prev;
      }
      return Math.min(items.length, prev + batchSize);
    });
  }, [batchSize, items.length]);

  const handleScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      const target = event.currentTarget;
      if (!target || target.scrollHeight <= target.clientHeight) {
        return;
      }
      const ratio = (target.scrollTop + target.clientHeight) / target.scrollHeight;
      if (ratio >= threshold) {
        loadMore();
      }
    },
    [loadMore, threshold],
  );

  useEffect(() => {
    const element = containerRef.current;
    if (!element || visibleCount >= items.length) {
      return;
    }

    if (element.scrollHeight <= element.clientHeight * 1.05) {
      if (typeof window === 'undefined') {
        loadMore();
        return;
      }
      const id = window.requestAnimationFrame(() => loadMore());
      return () => window.cancelAnimationFrame(id);
    }

    return undefined;
  }, [items.length, loadMore, visibleCount]);

  const visibleItems = useMemo(() => items.slice(0, visibleCount), [items, visibleCount]);
  const hasMore = visibleCount < items.length;

  return {
    containerRef,
    handleScroll,
    visibleItems,
    hasMore,
    loadMore,
    visibleCount,
  };
}
