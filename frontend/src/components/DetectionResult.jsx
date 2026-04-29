import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { CheckCircle2, AlertTriangle } from 'lucide-react';

const DetectionResult = ({ result, language = 'zh', variant = 'card' }) => {
  if (!result) return null;

  const isEmbedded = variant === 'embedded';
  const isAIGC = result.prediction === 'AIGC';
  const confidencePercent = Math.round((result.confidence || 0) * 100);
  const aigcProbability = typeof result.probabilities?.aigc === 'number'
    ? result.probabilities.aigc
    : (typeof result.probability === 'number' ? result.probability : 0);

  const copy = {
    zh: {
      title: '最终判定',
      aigc: 'AI生成',
      real: '真实图像',
      confidence: '置信度',
      probability: 'AIGC 概率',
      currentMode: '当前模式',
      currentThreshold: '当前阈值',
      decisionRule: '判定规则',
      boundaryTitle: '判定边界指示',
      realRegion: '真实区间',
      aigcRegion: 'AIGC 区间',
      markerLabel: '当前概率',
      sensitivityTitle: '模式敏感性分析',
      deltaLabel: '距离阈值',
      flipWarning: '当前样本对阈值敏感，不同模式下可能出现不同判定。',
      deltaPositive: '判定为 AI 生成',
      deltaNegative: '判定为真实图像',
      summaryTitle: '判定摘要',
      aigcSummary: '结合语义结构、频域分布与残差证据的综合观察，当前样本整体上更符合生成图像的取证特征。',
      realSummary: '结合语义结构、频域分布与残差证据的综合观察，当前样本整体上更接近真实图像的取证表现。',
      pathNote: '系统以语义结构、频域分布与噪声残差证据形成综合判断。阈值模式仅改变最终分类边界，不改变模型概率或取证证据。',
      ruleText: '当 AIGC 概率 ≥ 当前阈值时判定为 AI 生成，否则判定为真实图像。',
    },
    en: {
      title: 'Final Decision',
      aigc: 'AI Generated',
      real: 'Real Image',
      confidence: 'Confidence',
      probability: 'AIGC Probability',
      currentMode: 'Current Mode',
      currentThreshold: 'Current Threshold',
      decisionRule: 'Decision Rule',
      boundaryTitle: 'Decision Boundary Indicator',
      realRegion: 'REAL region',
      aigcRegion: 'AIGC region',
      markerLabel: 'Current probability',
      sensitivityTitle: 'Mode Sensitivity Analysis',
      deltaLabel: 'Distance to threshold',
      flipWarning: 'This sample is sensitive to threshold changes and may flip labels across modes.',
      deltaPositive: 'Classified as AIGC',
      deltaNegative: 'Classified as REAL',
      summaryTitle: 'Decision Summary',
      aigcSummary: 'Considering semantic structure, spectral characteristics, and residual evidence together, the current sample is overall more consistent with forensic patterns of AI-generated imagery.',
      realSummary: 'Considering semantic structure, spectral characteristics, and residual evidence together, the current sample is overall closer to the forensic profile of a real image.',
      pathNote: 'The system forms an integrated judgment from semantic structure, frequency distribution, and noise residual evidence. Threshold mode only changes the final decision boundary and does not change model probabilities or forensic evidence.',
      ruleText: 'Classify as AIGC when the AIGC probability is greater than or equal to the current threshold; otherwise classify as REAL.',
    },
  }[language];

  const summaryText = isAIGC ? copy.aigcSummary : copy.realSummary;
  const thresholdModeLabel = result.threshold_mode_label || result.threshold_mode_key || '-';
  const thresholdValue = typeof result.threshold_used === 'number'
    ? result.threshold_used
    : (typeof result.threshold === 'number' ? result.threshold : null);
  const deltaToThreshold = typeof result.delta_to_threshold === 'number'
    ? result.delta_to_threshold
    : (thresholdValue !== null ? aigcProbability - thresholdValue : null);
  const markerPercent = Math.max(0, Math.min(100, aigcProbability * 100));
  const thresholdPercent = thresholdValue !== null ? Math.max(0, Math.min(100, thresholdValue * 100)) : 0;
  const sensitivityText = deltaToThreshold !== null
    ? `${deltaToThreshold >= 0 ? '+' : ''}${deltaToThreshold.toFixed(3)} → ${deltaToThreshold >= 0 ? copy.deltaPositive : copy.deltaNegative}`
    : '-';
  const probabilityPercentText = `${(aigcProbability * 100).toFixed(1)}%`;
  const thresholdPercentText = thresholdValue !== null ? `${(thresholdValue * 100).toFixed(1)}%` : '-';
  const decisionRuleText = result.decision_rule_text || (
    language === 'zh'
      ? `AIGC 概率 ${probabilityPercentText} ${isAIGC ? '≥' : '<'} 当前阈值 ${thresholdPercentText}，因此判定为 ${isAIGC ? copy.aigc : copy.real}。`
      : `AIGC probability ${probabilityPercentText} ${isAIGC ? '>=' : '<'} threshold ${thresholdPercentText}, prediction=${result.prediction}.`
  );

  if (isEmbedded) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.08 }}
        className="flex h-full flex-col"
      >
        <p className="section-title mb-4">{copy.title}</p>

        <div className="flex h-full flex-col justify-between gap-6">
          <div className="flex items-start gap-4">
            <div
              className={clsx(
                'flex h-16 w-16 shrink-0 items-center justify-center rounded-[22px] border',
                isAIGC ? 'border-red-200 bg-red-50 text-red-600' : 'border-emerald-200 bg-emerald-50 text-emerald-700'
              )}
            >
              {isAIGC ? <AlertTriangle className="h-8 w-8" /> : <CheckCircle2 className="h-8 w-8" />}
            </div>

            <div className="min-w-0">
              <div className="inline-flex items-center rounded-full border border-line bg-panel px-3 py-1.5 text-xs font-medium text-ink">
                {copy.confidence} {confidencePercent}%
              </div>
              <h2 className="mt-4 text-4xl font-semibold tracking-tight text-ink 2xl:text-5xl">
                {isAIGC ? copy.aigc : copy.real}
              </h2>
            </div>
          </div>

          <div className="rounded-[24px] border border-line bg-panel p-4">
            <div className="section-title mb-2">{copy.summaryTitle}</div>
            <p className="text-sm leading-7 text-muted">{summaryText}</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <div className="rounded-[18px] border border-line bg-white p-3">
                <div className="section-title mb-1">{copy.currentMode}</div>
                <div className="text-sm font-semibold text-ink">{thresholdModeLabel}</div>
              </div>
              <div className="rounded-[18px] border border-line bg-white p-3">
                <div className="section-title mb-1">{copy.currentThreshold}</div>
                <div className="text-sm font-semibold text-ink">{thresholdPercentText}</div>
              </div>
              <div className="rounded-[18px] border border-line bg-white p-3">
                <div className="section-title mb-1">{copy.probability}</div>
                <div className="text-sm font-semibold text-ink">{probabilityPercentText}</div>
              </div>
            </div>
            <div className="mt-4 rounded-[18px] border border-line bg-white p-4">
              <div className="section-title mb-2">{copy.boundaryTitle}</div>
              <p className="mb-3 text-sm leading-7 text-muted">{decisionRuleText}</p>
              <div className="mb-2 flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.12em] text-subtle">
                <span>{copy.realRegion}</span>
                <span>{copy.aigcRegion}</span>
              </div>
              <div className="relative h-4 overflow-hidden rounded-full border border-line bg-slate-100">
                <div className="absolute inset-y-0 left-0 bg-emerald-100" style={{ width: `${thresholdPercent}%` }} />
                <div className="absolute inset-y-0 right-0 bg-red-100" style={{ left: `${thresholdPercent}%` }} />
                <div className="absolute inset-y-[-6px] w-[2px] bg-slate-700" style={{ left: `calc(${thresholdPercent}% - 1px)` }} />
                <div className="absolute top-1/2 h-4 w-4 -translate-y-1/2 rounded-full border-2 border-white bg-slate-900 shadow" style={{ left: `calc(${markerPercent}% - 8px)` }} />
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.1 }}
      className="flex h-full flex-col rounded-[28px] border border-line bg-white p-5 shadow-card lg:p-6"
    >
      <p className="section-title mb-3">{copy.title}</p>
      <div className="rounded-[24px] border border-line bg-panel p-5 lg:p-6">
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.05fr)_minmax(320px,0.95fr)] xl:items-start">
          <div className="flex items-start gap-4">
            <div
              className={clsx(
                'flex h-14 w-14 shrink-0 items-center justify-center rounded-[20px] border',
                isAIGC ? 'border-red-200 bg-red-50 text-red-600' : 'border-emerald-200 bg-emerald-50 text-emerald-700'
              )}
            >
              {isAIGC ? <AlertTriangle className="h-7 w-7" /> : <CheckCircle2 className="h-7 w-7" />}
            </div>
            <div className="min-w-0">
              <h2 className="text-2xl font-semibold tracking-tight text-ink">
                {isAIGC ? copy.aigc : copy.real}
              </h2>
              <p className="mt-2 text-sm leading-7 text-muted">{summaryText}</p>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-[20px] border border-line bg-white p-4">
              <span className="section-title">{copy.confidence}</span>
              <div className="mt-3 text-3xl font-semibold tracking-tight text-ink">{confidencePercent}%</div>
            </div>
            <div className="rounded-[20px] border border-line bg-white p-4">
              <span className="section-title">{copy.probability}</span>
              <div className="mt-3 text-3xl font-semibold tracking-tight text-ink">{probabilityPercentText}</div>
            </div>
            <div className="rounded-[20px] border border-line bg-white p-4">
              <span className="section-title">{copy.currentMode}</span>
              <div className="mt-3 text-lg font-semibold tracking-tight text-ink">{thresholdModeLabel}</div>
            </div>
            <div className="rounded-[20px] border border-line bg-white p-4">
              <span className="section-title">{copy.currentThreshold}</span>
              <div className="mt-3 text-3xl font-semibold tracking-tight text-ink">{thresholdPercentText}</div>
            </div>
            <div className="rounded-[20px] border border-line bg-white p-4 sm:col-span-2">
              <div className="section-title mb-2">{copy.decisionRule}</div>
              <p className="text-sm leading-7 text-muted">{copy.ruleText}</p>
              <p className="mt-2 text-sm font-semibold leading-7 text-ink">{decisionRuleText}</p>
            </div>
            <div className="rounded-[20px] border border-line bg-white p-4 sm:col-span-2">
              <div className="section-title mb-2">{copy.boundaryTitle}</div>
              <div className="mb-3 flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.12em] text-subtle">
                <span>{copy.realRegion}</span>
                <span>{copy.aigcRegion}</span>
              </div>
              <div className="relative h-5 overflow-hidden rounded-full border border-line bg-slate-100">
                <div className="absolute inset-y-0 left-0 bg-emerald-100" style={{ width: `${thresholdPercent}%` }} />
                <div className="absolute inset-y-0 right-0 bg-red-100" style={{ left: `${thresholdPercent}%` }} />
                <div className="absolute inset-y-[-8px] w-[2px] bg-slate-700" style={{ left: `calc(${thresholdPercent}% - 1px)` }} />
                <div className="absolute top-1/2 h-5 w-5 -translate-y-1/2 rounded-full border-2 border-white bg-slate-900 shadow" style={{ left: `calc(${markerPercent}% - 10px)` }} />
              </div>
              <div className="mt-3 flex items-center justify-between text-sm text-muted">
                <span>{copy.markerLabel}: {probabilityPercentText}</span>
                <span>{copy.currentThreshold}: {thresholdPercentText}</span>
              </div>
            </div>
            <div className="rounded-[20px] border border-line bg-white p-4 sm:col-span-2">
              <div className="section-title mb-2">{copy.sensitivityTitle}</div>
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-[16px] border border-line bg-panel p-3">
                  <div className="section-title mb-1">{copy.probability}</div>
                  <div className="text-lg font-semibold text-ink">{probabilityPercentText}</div>
                </div>
                <div className="rounded-[16px] border border-line bg-panel p-3">
                  <div className="section-title mb-1">{copy.currentThreshold}</div>
                  <div className="text-lg font-semibold text-ink">{thresholdPercentText}</div>
                </div>
                <div className="rounded-[16px] border border-line bg-panel p-3">
                  <div className="section-title mb-1">{copy.deltaLabel}</div>
                  <div className="text-lg font-semibold text-ink">{sensitivityText}</div>
                </div>
              </div>
              {result.threshold_sensitive && (
                <div className="mt-3 rounded-[16px] border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
                  {copy.flipWarning}
                </div>
              )}
            </div>
            <div className="rounded-[20px] border border-line bg-white p-4 sm:col-span-2">
              <div className="section-title mb-2">{copy.summaryTitle}</div>
              <p className="text-sm leading-7 text-muted">{copy.pathNote}</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default DetectionResult;
