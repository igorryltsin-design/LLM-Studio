import { createContext, useCallback, useContext, useMemo, useState, ReactNode } from 'react';

type ActivityKey = 'chat' | 'training' | 'dataset' | 'files' | 'evaluation';

type ActivityStatus = 'running' | 'success' | 'error' | 'warning';

interface ActivityState {
  key: ActivityKey;
  message: string;
  progress?: number;
  status: ActivityStatus;
  startedAt: number;
  updatedAt: number;
}

interface ActivityPayload {
  message: string;
  progress?: number;
  status?: ActivityStatus;
}

interface ActivityUpdate {
  message?: string;
  progress?: number;
  status?: ActivityStatus;
}

interface StatusContextValue {
  activities: Partial<Record<ActivityKey, ActivityState | null>>;
  currentActivity: ActivityState | null;
  setActivity: (key: ActivityKey, payload: ActivityPayload) => void;
  updateActivity: (key: ActivityKey, update: ActivityUpdate) => void;
  clearActivity: (key: ActivityKey) => void;
}

const PRIORITY: ActivityKey[] = ['training', 'evaluation', 'dataset', 'files', 'chat'];

const StatusContext = createContext<StatusContextValue | undefined>(undefined);

export function StatusProvider({ children }: { children: ReactNode }) {
  const [activities, setActivities] = useState<Partial<Record<ActivityKey, ActivityState | null>>>(
    {}
  );

  const setActivity = useCallback((key: ActivityKey, payload: ActivityPayload) => {
    setActivities((prev) => ({
      ...prev,
      [key]: {
        key,
        message: payload.message,
        progress: payload.progress,
        status: payload.status ?? 'running',
        startedAt: Date.now(),
        updatedAt: Date.now(),
      },
    }));
  }, []);

  const updateActivity = useCallback((key: ActivityKey, update: ActivityUpdate) => {
    setActivities((prev) => {
      const current = prev[key];
      if (!current) {
        return prev;
      }
      return {
        ...prev,
        [key]: {
          ...current,
          ...update,
          updatedAt: Date.now(),
        },
      };
    });
  }, []);

  const clearActivity = useCallback((key: ActivityKey) => {
    setActivities((prev) => ({
      ...prev,
      [key]: null,
    }));
  }, []);

  const currentActivity = useMemo<ActivityState | null>(() => {
    for (const key of PRIORITY) {
      const activity = activities[key];
      if (activity && activity.status === 'running') {
        return activity;
      }
    }

    for (const key of PRIORITY) {
      const activity = activities[key];
      if (activity) {
        return activity;
      }
    }

    return null;
  }, [activities]);

  const contextValue = useMemo(
    () => ({ activities, currentActivity, setActivity, updateActivity, clearActivity }),
    [activities, currentActivity, setActivity, updateActivity, clearActivity]
  );

  return <StatusContext.Provider value={contextValue}>{children}</StatusContext.Provider>;
}

export function useStatus() {
  const context = useContext(StatusContext);
  if (context === undefined) {
    throw new Error('useStatus must be used within a StatusProvider');
  }
  return context;
}
