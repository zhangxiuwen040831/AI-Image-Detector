import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ZoomIn, GitBranch, Maximize } from 'lucide-react';
import { toRenderableImageSrc } from '../utils/imageSrc';

const copyMap = {
  zh: {
    titleEvidence: '分支证据三角图',
    titleUsage: '融合权重三角图',
    semantic: '语义',
    noise: '噪声',
    frequency: '频域',
    noData: '当前未提供三角图分析数据',
    hover: '三角图可视化（点击可放大）',
    evidenceDesc: '顶点分别对应语义、噪声与频域证据。图中点展示当前样本的证据分布，而不是旧式纯 gate 权重。',
    usageDesc: '顶点分别对应语义、噪声与频域路径使用比例。该视图更接近 gate/fusion 权重。',
    baseOnlyNote: '在 base_only 模式下，noise 顶点通常接近 0，因为它不参与最终判定。',
  },
  en: {
    titleEvidence: 'Branch Evidence Triangle',
    titleUsage: 'Fusion Weight Triangle',
    semantic: 'Semantic',
    noise: 'Noise',
    frequency: 'Frequency',
    noData: 'No triangle analysis data available',
    hover: 'Triangle visualization (click to zoom)',
    evidenceDesc: 'Vertices correspond to semantic, noise, and frequency evidence. The point shows sample-level evidence distribution rather than legacy gate weights.',
    usageDesc: 'Vertices correspond to semantic, noise, and frequency path usage ratios. This is closer to gate/fusion weights.',
    baseOnlyNote: 'In base_only mode, the noise vertex is usually near zero because it does not take part in the final decision.',
  },
};

const FusionEvidenceTriangle = ({
  imageBase64,
  branchContribution,
  analysisMode = 'support_weighted_usage',
  mode = 'base_only',
  language = 'zh',
}) => {
  const [zoomed, setZoomed] = useState(false);
  const src = toRenderableImageSrc(imageBase64);
  const copy = copyMap[language] || copyMap.zh;
  const isEvidenceMode = analysisMode === 'support_weighted_usage';

  const normalized = useMemo(() => {
    const rgb = typeof branchContribution?.rgb === 'number' ? Math.max(branchContribution.rgb, 0) : 0;
    const noise = typeof branchContribution?.noise === 'number' ? Math.max(branchContribution.noise, 0) : 0;
    const frequency = typeof branchContribution?.frequency === 'number' ? Math.max(branchContribution.frequency, 0) : 0;
    const total = rgb + noise + frequency;
    if (total <= 1e-8) {
      return { rgb: 1 / 3, noise: 1 / 3, frequency: 1 / 3 };
    }
    return {
      rgb: rgb / total,
      noise: noise / total,
      frequency: frequency / total,
    };
  }, [branchContribution]);

  const hasBranchData = branchContribution && (
    branchContribution.rgb !== null ||
    branchContribution.noise !== null ||
    branchContribution.frequency !== null
  );

  const drawTriangle = (centerX, centroidY, size) => {
    const height = size * Math.sqrt(3) / 2;
    const topY = centroidY - height * 2 / 3;
    const bottomY = centroidY + height / 3;
    return [
      `${centerX},${topY}`,
      `${centerX - size / 2},${bottomY}`,
      `${centerX + size / 2},${bottomY}`,
    ].join(' ');
  };

  const calculatePointPosition = (rgb, noise, frequency) => {
    const centerX = 150;
    const centroidY = 175;
    const size = 240;
    const height = size * Math.sqrt(3) / 2;

    const x1 = centerX;
    const y1 = centroidY - height * 2 / 3;
    const x2 = centerX - size / 2;
    const y2 = centroidY + height / 3;
    const x3 = centerX + size / 2;
    const y3 = centroidY + height / 3;

    return {
      x: rgb * x1 + noise * x2 + frequency * x3,
      y: rgb * y1 + noise * y2 + frequency * y3,
    };
  };

  const renderSVGTriangle = () => {
    const centerX = 150;
    const centroidY = 175;
    const baseSize = 240;
    const height = baseSize * Math.sqrt(3) / 2;
    const topY = centroidY - height * 2 / 3;
    const bottomY = centroidY + height / 3;
    const point = calculatePointPosition(normalized.rgb, normalized.noise, normalized.frequency);

    return (
      <svg
        viewBox="0 0 300 340"
        className={`w-full h-full transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`}
      >
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {[0.25, 0.5, 0.75, 1].map((scale, index) => (
          <polygon
            key={`triangle-${scale}`}
            points={drawTriangle(centerX, centroidY, baseSize * scale)}
            fill="none"
            stroke={index === 3 ? 'rgba(218, 32, 90, 0.8)' : 'rgba(255, 255, 255, 0.2)'}
            strokeWidth={index === 3 ? 2 : 1}
          />
        ))}

        <text x={centerX} y={topY - 12} fill="#60A5FA" fontSize="12" textAnchor="middle" fontWeight="600">
          {copy.semantic}
        </text>
        <text x={centerX - baseSize / 2 - 15} y={bottomY + 5} fill="#A78BFA" fontSize="12" textAnchor="middle" fontWeight="600">
          {copy.noise}
        </text>
        <text x={centerX + baseSize / 2 + 15} y={bottomY + 5} fill="#34D399" fontSize="12" textAnchor="middle" fontWeight="600">
          {copy.frequency}
        </text>

        <circle cx={point.x} cy={point.y} r="14" fill="rgba(218, 32, 90, 0.18)" filter="url(#glow)" />
        <circle cx={point.x} cy={point.y} r="8" fill="#DA205A" filter="url(#glow)" />
        <circle cx={point.x} cy={point.y} r="3.5" fill="white" />
      </svg>
    );
  };

  if (!imageBase64 && !hasBranchData) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.7 }}
        className="glass-card rounded-2xl p-4 h-full flex flex-col justify-center relative overflow-hidden border border-white/10"
      >
        <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full flex items-center justify-center gap-2">
          <GitBranch className="w-6 h-6" style={{ color: 'var(--color-primary, #DA205A)' }} />
          {isEvidenceMode ? copy.titleEvidence : copy.titleUsage}
        </h3>
        <div className="w-full aspect-square flex items-center justify-center bg-black/30 rounded-xl border border-white/10 text-sm text-gray-300">
          {copy.noData}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.7 }}
      className="glass-card rounded-2xl p-4 h-full flex flex-col justify-center relative overflow-hidden group cursor-pointer border border-white/10"
      onClick={() => setZoomed(!zoomed)}
    >
      <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full flex items-center justify-center gap-2">
        <GitBranch className="w-6 h-6" style={{ color: 'var(--color-primary, #DA205A)' }} />
        {isEvidenceMode ? copy.titleEvidence : copy.titleUsage}
      </h3>
      <div className="w-full aspect-square relative flex items-center justify-center bg-black/40 rounded-xl border border-white/10 overflow-hidden">
        {hasBranchData ? (
          renderSVGTriangle()
        ) : (
          <img
            src={src}
            alt={isEvidenceMode ? copy.titleEvidence : copy.titleUsage}
            className={`w-full h-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`}
          />
        )}
        <div className="absolute top-2 right-2 bg-black/60 p-1.5 rounded-full text-white/80 opacity-0 group-hover:opacity-100 transition-opacity">
          {zoomed ? <Maximize className="w-4 h-4" /> : <ZoomIn className="w-4 h-4" />}
        </div>
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2 text-xs text-center text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
          {copy.hover}
        </div>
      </div>
      <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
        {isEvidenceMode ? copy.evidenceDesc : copy.usageDesc}
      </p>
      {mode === 'base_only' && (
        <p className="text-[11px] text-cyan-300/80 text-center mt-2 leading-relaxed">
          {copy.baseOnlyNote}
        </p>
      )}
    </motion.div>
  );
};

export default FusionEvidenceTriangle;
