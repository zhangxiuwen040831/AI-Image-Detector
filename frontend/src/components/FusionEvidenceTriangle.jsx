import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ZoomIn, GitBranch, Maximize } from 'lucide-react';
import { toRenderableImageSrc } from '../utils/imageSrc';

const copyMap = {
  zh: {
    titleEvidence: '分支证据三角图',
    titleUsage: '三分支证据分布',
    decisionTitle: '最终判定权重',
    semantic: '语义结构',
    noise: '噪声残差',
    frequency: '频域分布',
    noData: '未获取到真实分支权重',
    evidenceDesc: '三角图表示三类证据的相对分布，而非单一权重值。每个顶点对应一种证据来源，图中位置反映当前样本在语义结构、频域分布与噪声残差证据空间中的综合特征。',
    usageDesc: '该视图用于观察样本在三类证据空间中的整体分布关系，帮助理解不同取证线索之间的互补性与协同性。',
    architectureNote: '三角图优先使用三分支证据分布权重；该权重用于解释证据分布，不等同于最终判定权重。',
    weightTitle: '权重分布',
  },
  en: {
    titleEvidence: 'Branch Evidence Triangle',
    titleUsage: 'Three-Branch Evidence Distribution',
    decisionTitle: 'Final Decision Weights',
    semantic: 'Semantic Structure',
    noise: 'Noise Residual',
    frequency: 'Frequency Distribution',
    noData: 'No real branch weights available',
    evidenceDesc: 'The triangle reflects the relative distribution of three evidence types rather than a single weight value. Each vertex corresponds to one evidence source, and the sample position reflects its combined semantic structure, frequency distribution, and noise residual characteristics.',
    usageDesc: 'This view helps interpret how the sample is distributed across the three evidence spaces, highlighting complementarity and collaboration among forensic cues.',
    architectureNote: 'The triangle prioritizes three-branch evidence weights for explanation; these weights are evidence distribution cues, not necessarily final decision weights.',
    weightTitle: 'Weight distribution',
  },
};

function finiteNumber(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value);
  }
  if (typeof value === 'string' && value.trim() !== '') {
    return Number.isFinite(Number(value));
  }
  return false;
}

function toFiniteNumber(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function extractWeights(value) {
  if (!value) return { weights: null, valid: false };
  const raw = {
    semantic: value.semantic ?? value.rgb,
    frequency: value.frequency ?? value.freq,
    noise: value.noise,
  };
  const clean = Object.fromEntries(
    Object.entries(raw).map(([key, entry]) => {
      const parsed = toFiniteNumber(entry);
      return [key, parsed !== null ? Math.max(parsed, 0) : null];
    }),
  );
  const validValues = Object.values(clean).filter(finiteNumber);
  const total = validValues.reduce((sum, entry) => sum + entry, 0);
  if (!validValues.length || total <= 1e-8) {
    return { weights: null, valid: false };
  }
  return {
    valid: true,
    weights: {
      semantic: (clean.semantic ?? 0) / total,
      frequency: (clean.frequency ?? 0) / total,
      noise: (clean.noise ?? 0) / total,
    },
  };
}

const FusionEvidenceTriangle = ({
  imageBase64,
  branchContribution,
  analysisMode = 'support_weighted_usage',
  mode = 'tri_fusion',
  language = 'zh',
  embedded = false,
}) => {
  const [zoomed, setZoomed] = useState(false);
  const src = toRenderableImageSrc(imageBase64);
  const copy = copyMap[language] || copyMap.zh;
  const isDecisionMode = analysisMode === 'decision_weights';
  const isEvidenceMode = !isDecisionMode;
  const shellClass = embedded
    ? 'mx-auto flex h-full w-full max-w-[760px] flex-col'
    : 'glass-card flex h-full cursor-pointer flex-col p-5';
  const canvasClass = embedded
    ? 'relative mx-auto aspect-[300/340] w-full max-w-[320px] overflow-hidden rounded-[24px] border border-line bg-panel'
    : 'relative flex w-full aspect-square items-center justify-center overflow-hidden rounded-[22px] border border-line bg-panel md:aspect-[4/3] xl:aspect-[21/10]';

  const normalized = useMemo(() => {
    const { weights, valid } = extractWeights(branchContribution);
    return {
      valid,
      semantic: weights?.semantic ?? null,
      frequency: weights?.frequency ?? null,
      noise: weights?.noise ?? null,
    };
  }, [branchContribution]);

  const hasBranchData = normalized.valid;

  const vertices = {
    semantic: { x: 50, y: 8 },
    frequency: { x: 12, y: 88 },
    noise: { x: 88, y: 88 },
  };

  const calculatePointPosition = () => ({
    x:
      normalized.semantic * vertices.semantic.x
      + normalized.frequency * vertices.frequency.x
      + normalized.noise * vertices.noise.x,
    y:
      normalized.semantic * vertices.semantic.y
      + normalized.frequency * vertices.frequency.y
      + normalized.noise * vertices.noise.y,
  });

  const renderSVGTriangle = () => {
    const point = calculatePointPosition();

    return (
      <svg
        viewBox="0 0 100 100"
        className={`h-auto w-full transition-transform duration-500 ${zoomed ? 'scale-125' : 'scale-100 hover:scale-105'}`}
      >
        <polygon
          points={`${vertices.semantic.x},${vertices.semantic.y} ${vertices.frequency.x},${vertices.frequency.y} ${vertices.noise.x},${vertices.noise.y}`}
          fill="rgba(241, 245, 249, 0.8)"
          stroke="rgba(51, 65, 85, 0.9)"
          strokeWidth="1.2"
        />
        {[0.25, 0.5, 0.75].map((scale) => {
          const cx = 50;
          const cy = 61.3;
          const pts = Object.values(vertices).map((vertex) => `${cx + (vertex.x - cx) * scale},${cy + (vertex.y - cy) * scale}`).join(' ');
          return <polygon key={scale} points={pts} fill="none" stroke="rgba(148, 163, 184, 0.45)" strokeWidth="0.8" />;
        })}

        <text x="50" y="5" fill="#334155" fontSize="4.2" textAnchor="middle" fontWeight="600">
          {copy.semantic}
        </text>
        <text x="12" y="96" fill="#64748b" fontSize="4.2" textAnchor="middle" fontWeight="600">
          {copy.frequency}
        </text>
        <text x="88" y="96" fill="#94a3b8" fontSize="4.2" textAnchor="middle" fontWeight="600">
          {copy.noise}
        </text>

        <circle cx={point.x} cy={point.y} r="5.5" fill="rgba(51, 65, 85, 0.12)" />
        <circle cx={point.x} cy={point.y} r="3.2" fill="#334155" />
        <circle cx={point.x} cy={point.y} r="1.4" fill="white" />
      </svg>
    );
  };

  const weightRows = [
    [copy.semantic, normalized.semantic],
    [copy.frequency, normalized.frequency],
    [copy.noise, normalized.noise],
  ];

  const heading = !embedded && (
    <div className={`mb-4 ${embedded ? '' : 'border-b border-line pb-4'}`}>
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white shadow-soft">
          <GitBranch className="h-5 w-5 text-primary" />
        </div>
        <div>
          <p className="section-title">{language === 'zh' ? '融合视图' : 'Fusion view'}</p>
          <h3 className="text-xl font-semibold tracking-tight text-ink">
            {isDecisionMode ? copy.decisionTitle : copy.titleUsage}
          </h3>
        </div>
      </div>
    </div>
  );

  const body = (
    <div className={`grid gap-4 ${embedded ? 'xl:grid-cols-[minmax(0,0.56fr)_minmax(280px,0.44fr)] xl:items-center' : ''}`}>
      <div className="flex flex-col gap-3">
        <div>
          <p className="section-title mb-2">{language === 'zh' ? '融合视图' : 'Fusion view'}</p>
          <h3 className="text-xl font-semibold tracking-tight text-ink">
            {isDecisionMode ? copy.decisionTitle : copy.titleUsage}
          </h3>
        </div>

        <p className="rounded-[18px] border border-line bg-panel px-4 py-3 text-sm leading-7 text-muted">
          {isEvidenceMode ? copy.evidenceDesc : copy.usageDesc} {copy.architectureNote}
        </p>
        {hasBranchData && (
          <div className="rounded-[18px] border border-line bg-panel px-4 py-3">
            <div className="section-title mb-2">{copy.weightTitle}</div>
            <div className="grid gap-2 text-sm text-muted">
              {weightRows.map(([label, value]) => (
                <div key={label} className="flex items-center justify-between gap-3">
                  <span>{label}</span>
                  <span className="font-mono text-ink">{(value * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className={`${canvasClass} ${embedded ? 'xl:self-center' : ''}`}>
        {(!imageBase64 && !hasBranchData) ? (
          <div className="flex h-full w-full items-center justify-center text-sm text-muted">
            {copy.noData}
          </div>
        ) : hasBranchData ? (
          renderSVGTriangle()
        ) : (
          <img
            src={src}
            alt={isEvidenceMode ? copy.titleEvidence : copy.titleUsage}
            className={`h-full w-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`}
          />
        )}

        {(imageBase64 || hasBranchData) && (
          <div className="absolute right-3 top-3 rounded-full border border-line bg-white/90 p-2 text-muted shadow-soft">
            {zoomed ? <Maximize className="h-4 w-4" /> : <ZoomIn className="h-4 w-4" />}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.08 }}
      className={shellClass}
      onClick={() => (imageBase64 || hasBranchData) && setZoomed(!zoomed)}
    >
      {heading}
      {body}
    </motion.div>
  );
};

export default FusionEvidenceTriangle;
