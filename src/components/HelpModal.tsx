import { useEffect } from 'react';
import { X, BookOpen } from 'lucide-react';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const keyFeatures = [
  'Запуск локальных языковых моделей и подключение к удалённым провайдерам через единый интерфейс.',
  'Дообучение моделей на своих данных с отслеживанием прогресса и истории запусков.',
  'Подготовка датасетов: загрузка файлов, извлечение знаний и генерация Q&A.',
  'Мониторинг системных ресурсов, задержек и состояния серверов в реальном времени.',
  'Настройка интерфейса, автоматического сохранения и параметров генерации под конкретные сценарии.',
];

const quickStart = [
  'Укажите путь к базовой модели и URL локального сервера на вкладке «Рабочие модели».',
  'Подключите удалённый API, если требуется комбинированная инфраструктура.',
  'Загрузите материалы во вкладке «Датасеты» и запустите генерацию Q&A или fine-tuning.',
  'Переключайтесь между чатами, обучением и датасетами с помощью навигации слева.',
  'Следите за статусом задач и ресурсами через статус-бар и модальное окно системы.',
];

const HelpModal = ({ isOpen, onClose }: HelpModalProps) => {
  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-modal="true"
        className="relative z-10 w-full max-w-3xl overflow-hidden rounded-3xl border border-slate-200 bg-white/95 shadow-2xl shadow-slate-900/40 backdrop-blur dark:border-slate-800 dark:bg-slate-900"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-200 px-6 py-4 dark:border-slate-800">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400 dark:text-slate-500">Справка</p>
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Об LLM Studio</h2>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Краткое описание возможностей и сценариев использования
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 text-slate-500 transition-colors hover:border-blue-500 hover:text-blue-500 dark:border-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
            aria-label="Закрыть справку"
          >
            <X className="h-5 w-5" />
          </button>
        </header>

        <div className="space-y-6 px-6 py-6 text-sm text-slate-600 dark:text-slate-300">
          <section className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 shadow-sm shadow-slate-900/5 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">
                <BookOpen className="h-5 w-5" />
              </span>
              <div>
                <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">Назначение</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  LLM Studio помогает разрабатывать и сопровождать прикладные решения на базе языковых моделей: от подготовки данных до мониторинга и интеграции в продукты.
                </p>
              </div>
            </div>
          </section>

          <section className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Основные функции</h4>
            <ul className="space-y-2">
              {keyFeatures.map(feature => (
                <li key={feature} className="rounded-lg border border-slate-200/60 bg-white/60 px-4 py-2 text-slate-600 shadow-sm shadow-slate-900/5 dark:border-slate-700/70 dark:bg-slate-900/70 dark:text-slate-300">
                  {feature}
                </li>
              ))}
            </ul>
          </section>

          <section className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Как начать</h4>
            <ol className="list-decimal space-y-2 pl-5">
              {quickStart.map(step => (
                <li key={step} className="text-slate-600 dark:text-slate-300">
                  {step}
                </li>
              ))}
            </ol>
          </section>
        </div>
      </div>
    </div>
  );
};

export default HelpModal;
