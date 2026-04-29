import React from 'react';
import { motion } from 'framer-motion';
import { normalizeBranchScores, normalizeDecisionWeights } from '../utils/resultUtils';

const copyMap = {
  zh: {
    eyebrow: '三分支取证分析',
    title: '三分支分析概览',
    overview: '系统通过语义结构、频域分布与噪声残差三类线索进行协同分析。',
    evidence: '证据线索',
    decisionWeight: '最终判定权重',
    noiseDisabled: '辅助证据分支，当前不参与最终判定',
    auxiliary: '辅助证据',
    mainDecision: '同级融合分支',
    semantic: {
      title: '语义结构分支',
      role: '内容级特征分析',
      description: '关注主体结构、语义布局与场景逻辑的一致性，用于判断图像在内容层面是否呈现自然、合理的组织方式。',
    },
    frequency: {
      title: '频域分布分支',
      role: '频谱/伪迹分析',
      description: '关注频谱分布、生成伪影与频率异常特征，用于识别图像在频域层面的非自然模式与取证线索。',
    },
    noise: {
      title: '噪声残差分支',
      role: '残差/噪声证据分析',
      description: '提供残差一致性与噪声统计特征方面证据，用来增强解释性',
    },
  },
  en: {
    eyebrow: 'Three-Branch Forensics',
    title: 'Three-Branch Analysis Overview',
    overview: 'The system performs collaborative analysis across semantic, frequency, and noise residual evidence.',
    evidence: 'Evidence cue',
    decisionWeight: 'Decision weight',
    noiseDisabled: 'Auxiliary evidence branch; not used for the current final decision',
    auxiliary: 'Auxiliary evidence',
    mainDecision: 'Peer fusion branch',
    semantic: {
      title: 'Semantic Branch',
      role: 'Content-level feature analysis',
      description: 'Examines object structure, semantic layout, and scene coherence to assess whether the image remains natural and internally consistent at the content level.',
    },
    frequency: {
      title: 'Frequency Branch',
      role: 'Spectral / forensic abnormality analysis',
      description: 'Examines spectral distribution, generative artifacts, and frequency-domain anomalies to identify non-natural forensic patterns.',
    },
    noise: {
      title: 'Noise Branch',
      role: 'Residual / noise evidence analysis',
      description: 'Provides residual-consistency and noise-statistical evidence to enhance interpretability.',
    },
  },
};

function metricWidth(value) {
  if (typeof value !== 'number') return 0;
  return Math.max(0, Math.min(value * 100, 100));
}

function MetricRow({ label, value, tone = 'bg-slate-700' }) {
  const hasValue = typeof value === 'number' && Number.isFinite(value);
  const percentText = hasValue ? `${(value * 100).toFixed(1)}%` : 'N/A';

  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3 text-sm">
        <span className="font-medium text-ink">{label}</span>
        <span className="font-mono text-muted">{percentText}</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-panel">
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${metricWidth(value)}%` }} />
      </div>
    </div>
  );
}

function BranchRegion({ branch, copy, evidenceScore, decisionWeight, note, compact = false }) {
  return (
    <div className={`flex h-full flex-col gap-5 ${compact ? 'px-0 py-4 xl:px-6' : 'px-0 py-5 xl:px-6 xl:py-4'}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="section-title mb-2">{branch.role}</div>
          <h4 className="text-xl font-semibold tracking-tight text-ink">{branch.title}</h4>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-[11px] font-medium text-slate-700">
          {copy.mainDecision}
        </span>
      </div>

      <p className="text-sm leading-7 text-muted">{branch.description}</p>
      {note && <p className="rounded-[16px] border border-line bg-panel px-4 py-3 text-xs leading-6 text-muted">{note}</p>}

      <div className={`grid gap-4 ${compact ? 'lg:grid-cols-[minmax(0,1fr)_280px]' : 'xl:grid-cols-2'}`}>
        <MetricRow label={copy.evidence} value={evidenceScore} tone="bg-slate-700" />
        <MetricRow label={copy.decisionWeight} value={decisionWeight} tone="bg-slate-400" />
      </div>
    </div>
  );
}

const BranchContribution = ({ result, language = 'zh' }) => {
  if (!result) return null;

  const copy = copyMap[language] || copyMap.zh;
  const branchScores = normalizeBranchScores(result);
  const decisionWeights = normalizeDecisionWeights(result);
  const noiseDisabled = result?.noise_enabled_for_decision === false;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.05 }}
      className="flex h-full min-w-0 flex-col rounded-[32px] border border-line bg-white p-5 shadow-card lg:p-6"
    >
      <div className="mb-5 flex flex-col gap-3 border-b border-line pb-5 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <p className="section-title mb-2">{copy.eyebrow}</p>
          <h3 className="text-2xl font-semibold tracking-tight text-ink">{copy.title}</h3>
        </div>
        <p className="text-sm leading-7 text-muted xl:max-w-none">{copy.overview}</p>
      </div>

      <div className="grid gap-0 divide-y divide-line xl:grid-cols-12 xl:divide-y-0">
        <div className="xl:col-span-6 xl:border-r xl:border-line">
          <BranchRegion
            branch={copy.semantic}
            copy={copy}
            evidenceScore={branchScores.semantic}
            decisionWeight={decisionWeights.semantic}
          />
        </div>

        <div className="xl:col-span-6">
          <BranchRegion
            branch={copy.frequency}
            copy={copy}
            evidenceScore={branchScores.frequency}
            decisionWeight={decisionWeights.frequency}
          />
        </div>

        <div className="xl:col-span-12 xl:border-t xl:border-line">
          <BranchRegion
            branch={copy.noise}
            copy={copy}
            evidenceScore={branchScores.noise}
            decisionWeight={decisionWeights.noise}
            note={noiseDisabled ? copy.noiseDisabled : null}
            compact
          />
        </div>
      </div>
    </motion.div>
  );
};

export default BranchContribution;
