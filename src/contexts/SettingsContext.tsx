import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { fetchSystemStats } from '../services/systemMonitor';

export interface Settings {
  baseModelPath: string;
  baseModelServerUrl: string;
  fineTunedModelPath: string;
  fineTunedBaseModelPath: string;
  fineTunedMethod: 'lora' | 'full' | '';
  remoteApiUrl: string;
  remoteApiKey: string;
  remoteModelId: string;
  quantization: '4bit' | '8bit' | 'none';
  maxTokens: number;
  temperature: number;
  topP: number;
  deviceType: 'cpu' | 'cuda' | 'mps';
  theme: 'light' | 'dark';
  autoSave: boolean;
  maxHistoryItems: number;
}

const DEVICE_DETECTION_KEY = 'llm-studio-device-detected';

const defaultSettings: Settings = {
  baseModelPath: 'models/gemma',
  baseModelServerUrl: 'http://127.0.0.1:8001',
  fineTunedModelPath: '',
  fineTunedBaseModelPath: '',
  fineTunedMethod: '',
  remoteApiUrl: 'http://127.0.0.1:1234/v1',
  remoteApiKey: '',
  remoteModelId: 'google/gemma-3n-e4b',
  quantization: 'none',
  maxTokens: 2048,
  temperature: 0.7,
  topP: 0.9,
  deviceType: 'cpu',
  theme: 'dark',
  autoSave: true,
  maxHistoryItems: 200,
};

interface SettingsContextType {
  settings: Settings;
  updateSettings: (newSettings: Partial<Settings>) => void;
  resetSettings: () => void;
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(defaultSettings);

  useEffect(() => {
    const savedSettings = localStorage.getItem('llm-studio-settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setSettings({ ...defaultSettings, ...parsed });
        return;
      } catch (error) {
        console.error('Failed to parse saved settings:', error);
      }
    }

    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setSettings((prev) => {
        const nextTheme: Settings['theme'] = prefersDark ? 'dark' : 'light';
        if (prev.theme === nextTheme) {
          return prev;
        }
        const next: Settings = { ...prev, theme: nextTheme };
        localStorage.setItem('llm-studio-settings', JSON.stringify(next));
        return next;
      });
    }
  }, []);

  const updateSettings = (newSettings: Partial<Settings>) => {
    const updatedSettings = { ...settings, ...newSettings };
    setSettings(updatedSettings);
    localStorage.setItem('llm-studio-settings', JSON.stringify(updatedSettings));
    if (Object.prototype.hasOwnProperty.call(newSettings, 'deviceType')) {
      localStorage.setItem(DEVICE_DETECTION_KEY, 'manual');
    }
  };

  const resetSettings = () => {
    setSettings(defaultSettings);
    localStorage.removeItem('llm-studio-settings');
    localStorage.removeItem(DEVICE_DETECTION_KEY);
  };

  useEffect(() => {
    const detectionState = localStorage.getItem(DEVICE_DETECTION_KEY);
    if (detectionState) {
      return;
    }

    let cancelled = false;

    const detectDevice = async () => {
      try {
        const stats = await fetchSystemStats(settings.baseModelServerUrl);
        if (cancelled) {
          return;
        }
        if (stats.status !== 'ok') {
          return;
        }
        if (stats.gpuBackend === 'cuda' || stats.gpuBackend === 'mps') {
          setSettings((prev) => {
            if (prev.deviceType !== 'cpu') {
              return prev;
            }
            const nextDevice: Settings['deviceType'] = stats.gpuBackend;
            const nextSettings: Settings = { ...prev, deviceType: nextDevice };
            localStorage.setItem('llm-studio-settings', JSON.stringify(nextSettings));
            localStorage.setItem(DEVICE_DETECTION_KEY, stats.gpuBackend);
            return nextSettings;
          });
        } else {
          localStorage.setItem(DEVICE_DETECTION_KEY, 'cpu');
        }
      } catch (error) {
        console.error('Не удалось определить устройство выполнения:', error);
      }
    };

    void detectDevice();

    return () => {
      cancelled = true;
    };
  }, [settings.baseModelServerUrl]);

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, resetSettings }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}
