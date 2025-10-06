import type { EvaluationRun, EvaluationDifficulty } from '../../types/evaluation';
import type {
  DashboardDifficultySummary,
  DashboardDomainInsight,
  DashboardHotspot,
  DashboardRecommendation,
  DashboardTagSummary,
} from '../../types/evaluationDashboard';

interface QualityDashboardProps {
  activeRun: EvaluationRun | null;
  domains: DashboardDomainInsight[];
  tags: DashboardTagSummary[];
  difficulties: DashboardDifficultySummary[];
  hotspots: DashboardHotspot[];
  recommendations: DashboardRecommendation[];
  compact: boolean;
  formatScore: (value: number | null | undefined, fractionDigits?: number) => string;
  formatPercent: (value: number | null | undefined, fractionDigits?: number) => string;
  difficultyLabels: Record<EvaluationDifficulty | 'unknown', string>;
}

const QualityDashboard = ({
  activeRun,
  domains,
  tags,
  difficulties,
  hotspots,
  recommendations,
  compact,
  formatScore,
  formatPercent,
  difficultyLabels,
}: QualityDashboardProps) => {
  if (!activeRun) {
    return (
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/60 dark:shadow-none">
        <div className="mt-6 rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-900/40">
          Запустите прогон, чтобы построить дашборд.
        </div>
      </section>
    );
  }

  const domainSample = compact ? domains.slice(0, 3) : domains;
  const tagSample = compact ? tags.slice(0, 4) : tags;
  const difficultySample = compact ? difficulties.slice(0, 2) : difficulties;
  const hotspotSample = compact ? hotspots.slice(0, 3) : hotspots;
  const recommendationSample = compact ? recommendations.slice(0, 3) : recommendations;

  const contentSpacing = compact ? 'space-y-3' : 'space-y-4';
  const listGap = compact ? 'gap-2' : 'gap-3';
  const cardPadding = compact ? 'p-2.5' : 'p-3';

  const worstDomain = domains
    .slice()
    .sort((a, b) => (a.averageScore ?? 2) - (b.averageScore ?? 2))
    .find(domain => domain.averageScore != null);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm shadow-slate-900/20 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-none xl:col-span-2">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-5 w-5 rounded-full bg-emerald-400/20" />
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Дашборд качества</h3>
        </div>
        <span className="text-[11px] text-slate-500 dark:text-slate-400">
          Горячих тем: {hotspotSample.length}/{hotspots.length} · Рекомендаций: {recommendationSample.length}/{recommendations.length}
        </span>
      </header>

      <div className={`mt-3 ${contentSpacing} text-xs text-slate-500 dark:text-slate-300`}>
        <div className={`flex flex-col items-center ${compact ? 'gap-3 md:flex-row md:items-start' : 'gap-4 md:flex-row md:items-start'}`}>
          <div className={`relative ${compact ? 'h-40 w-40' : 'h-48 w-48'}`}>
            <div
              className="h-full w-full rounded-full shadow-inner shadow-slate-900/10"
              style={{
                background: (() => {
                  if (!domains.length) {
                    return 'conic-gradient(#334155 0 100%)';
                  }
                  let cursor = 0;
                  const segments = domains.map((domain) => {
                    const start = cursor * 100;
                    cursor += domain.share;
                    const end = Math.min(100, cursor * 100);
                    return `${domain.color} ${start}% ${end}%`;
                  });
                  return `conic-gradient(${segments.join(', ')})`;
                })(),
              }}
            />
            <div className="absolute inset-6 flex flex-col items-center justify-center rounded-full bg-white/95 text-center shadow-inner shadow-slate-900/10 dark:bg-slate-950/80">
              <span className="text-[11px] uppercase tracking-wide text-slate-400 dark:text-slate-500">Всего</span>
              <span className="text-lg font-semibold text-slate-800 dark:text-slate-100">{activeRun.metrics.total}</span>
              {worstDomain && (
                <div className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">
                  <p className="font-semibold text-slate-600 dark:text-slate-200">Риск: {worstDomain.domain}</p>
                  <p>{formatScore(worstDomain.averageScore, 3)}</p>
                </div>
              )}
            </div>
          </div>

          <div className={`flex-1 ${contentSpacing}`}>
            {domainSample.map((domain) => {
              const sharePercent = Math.max(1, Math.round(domain.share * 100));
              const coveragePercent = Math.round(domain.coverageShare * 100);
              const deltaText = domain.delta != null
                ? `${domain.delta >= 0 ? '+' : ''}${domain.delta.toFixed(3)}`
                : '—';

              return (
                <div
                  key={domain.domain}
                  className={`rounded-xl border border-slate-200 bg-white/95 ${cardPadding} shadow-sm shadow-slate-900/5 dark:border-slate-800 dark:bg-slate-900/80`}
                >
                  <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                    <span className="flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-100">
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: domain.color }}
                      />
                      {domain.domain}
                    </span>
                    <span className={`text-sm font-semibold ${domain.toneClass.replace('bg-', 'text-')}`}>
                      {formatScore(domain.averageScore, 3)}
                    </span>
                  </div>
                  {compact ? (
                    <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-500 dark:text-slate-400">
                      <span>{sharePercent}% выборки</span>
                      <span>Δ {deltaText}</span>
                    </div>
                  ) : (
                    <div className={`mt-2 flex flex-wrap items-center ${listGap} text-[10px] text-slate-500 dark:text-slate-400`}>
                      <span>Доля: {sharePercent}%</span>
                      <span>Покрытие: {coveragePercent}%</span>
                      <span>Оценено: {domain.scored}/{domain.total}</span>
                      <span>Δ: {deltaText}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className={`grid grid-cols-1 ${listGap} md:grid-cols-2`}>
          <div>
            <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Теги</p>
            <div className={`mt-2 ${compact ? 'space-y-2' : 'space-y-3'} overflow-y-auto pr-1 ${compact ? 'max-h-36' : 'max-h-48'}`}>
              {tagSample.length > 0 ? tagSample.map((tag) => (
                <div
                  key={tag.tag}
                  className={`rounded-lg border border-slate-200 bg-white/95 ${cardPadding} text-slate-600 dark:border-slate-800 dark:bg-slate-900/70 dark:text-slate-200`}
                >
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold">#{tag.tag}</span>
                    <span className={`font-semibold ${tag.toneClass}`}>{formatScore(tag.averageScore, 3)}</span>
                  </div>
                  {!compact ? (
                    <div className={`mt-1 flex flex-wrap items-center ${listGap} text-[10px] text-slate-500 dark:text-slate-400`}>
                      <span>Оценено: {tag.scored}/{tag.total}</span>
                      <span>Покрытие: {formatPercent(tag.coverageShare)}</span>
                      <span>Review: {tag.needsReview}</span>
                    </div>
                  ) : (
                    <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">Покрытие {formatPercent(tag.coverageShare)}</p>
                  )}
                </div>
              )) : (
                <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-center text-[11px] text-slate-400 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-500">
                  Вопросы без тегов.
                </div>
              )}
            </div>
          </div>

          <div>
            <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Сложность</p>
            <div className={`mt-2 ${compact ? 'space-y-2' : 'space-y-3'}`}>
              {difficultySample.length > 0 ? difficultySample.map((item) => {
                const deltaText = item.delta != null
                  ? `${item.delta >= 0 ? '+' : ''}${item.delta.toFixed(3)}`
                  : '—';
                return (
                  <div
                    key={item.difficulty}
                    className={`rounded-lg border border-slate-200 bg-white/95 ${cardPadding} text-slate-600 dark:border-slate-800 dark:bg-slate-900/70 dark:text-slate-200`}
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold">{difficultyLabels[item.difficulty]}</span>
                      <span className={`font-semibold ${item.toneClass}`}>{formatScore(item.averageScore, 3)}</span>
                    </div>
                    {!compact ? (
                      <div className={`mt-1 flex flex-wrap items-center ${listGap} text-[10px] text-slate-500 dark:text-slate-400`}>
                        <span>Оценено: {item.scored}/{item.total}</span>
                        <span>База: {formatScore(item.baselineScore, 3)}</span>
                        <span>Δ: {deltaText}</span>
                      </div>
                    ) : (
                      <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">Δ {deltaText}</p>
                    )}
                  </div>
                );
              }) : (
                <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-center text-[11px] text-slate-400 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-500">
                  Нет данных по сложности.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className={`grid grid-cols-1 ${listGap} md:grid-cols-3`}>
          <div className="md:col-span-2">
            <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Горячие вопросы</p>
            <div className={`mt-2 ${compact ? 'space-y-2 max-h-44' : 'space-y-3 max-h-60'} overflow-y-auto pr-1`}>
              {hotspotSample.length > 0 ? hotspotSample.map((item) => (
                <div
                  key={item.id}
                  className={`rounded-lg border ${item.needsReview ? 'border-rose-400 dark:border-rose-500/60' : 'border-slate-200 dark:border-slate-800'} bg-white/95 ${cardPadding} text-slate-600 dark:bg-slate-900/70 dark:text-slate-200`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-semibold text-slate-700 dark:text-slate-100">{item.question}</span>
                    <span className={`text-sm font-semibold ${item.toneClass}`}>{formatScore(item.score, 3)}</span>
                  </div>
                  {compact ? (
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-slate-500 dark:text-slate-400">
                      <span>{item.domain}</span>
                      <span className="mx-1">·</span>
                      <span>{difficultyLabels[item.difficulty]}</span>
                    </div>
                  ) : (
                    <div className={`mt-1 flex flex-wrap items-center ${listGap} text-[10px] text-slate-500 dark:text-slate-400`}>
                      <span>Домен: {item.domain}</span>
                      <span>Сложность: {difficultyLabels[item.difficulty]}</span>
                      {item.tags.length > 0 && <span>Теги: {item.tags.join(', ')}</span>}
                      <span>Статус: {item.status}</span>
                    </div>
                  )}
                  {!compact && item.reviewNotes && (
                    <p className="mt-1 rounded-lg bg-slate-800/40 p-2 text-[10px] italic text-slate-400 dark:bg-slate-800/60">
                      {item.reviewNotes}
                    </p>
                  )}
                </div>
              )) : (
                <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-center text-[11px] text-slate-400 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-500">
                  Все вопросы проходят критерии качества.
                </div>
              )}
            </div>
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-wide text-slate-500 dark:text-slate-400">Рекомендации к датасету</p>
            <div className={`mt-2 ${compact ? 'space-y-2' : 'space-y-3'}`}>
              {recommendationSample.length > 0 ? recommendationSample.map((item) => (
                <div
                  key={item.key}
                  className={`rounded-lg border border-slate-200 bg-white/95 ${cardPadding} text-slate-600 dark:border-slate-800 dark:bg-slate-900/70 dark:text-slate-200`}
                >
                  <p className="text-sm font-semibold text-slate-700 dark:text-slate-100">{item.label}</p>
                  {!compact && (
                    <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">Повторяющиеся проблемы: {item.count}</p>
                  )}
                </div>
              )) : (
                <div className="rounded-xl border border-dashed border-slate-300 bg-white p-4 text-center text-[11px] text-slate-400 dark:border-slate-700 dark:bg-slate-900/40 dark:text-slate-500">
                  Перекосы не обнаружены.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default QualityDashboard;
