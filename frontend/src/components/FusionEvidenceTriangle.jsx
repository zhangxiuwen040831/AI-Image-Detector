import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ZoomIn, GitBranch, Maximize } from 'lucide-react';
import { toRenderableImageSrc } from '../utils/imageSrc';

const FusionEvidenceTriangle = ({ imageBase64, branchContribution, language = 'zh' }) => {
  const [zoomed, setZoomed] = useState(false);
  const src = toRenderableImageSrc(imageBase64);

  const hasBranchData = branchContribution && 
    (branchContribution.rgb !== null || branchContribution.noise !== null || branchContribution.frequency !== null);

  const drawTriangle = (centerX, centroidY, size) => {
    const height = size * Math.sqrt(3) / 2;
    const topY = centroidY - height * 2 / 3;
    const bottomY = centroidY + height / 3;
    const points = [
      `${centerX},${topY}`,
      `${centerX - size / 2},${bottomY}`,
      `${centerX + size / 2},${bottomY}`
    ].join(' ');
    return points;
  };

  const calculatePointPosition = (rgb, noise, frequency) => {
    const total = (rgb || 0) + (noise || 0) + (frequency || 0);
    const w1 = total > 0 ? (rgb || 0) / total : 1/3;
    const w2 = total > 0 ? (noise || 0) / total : 1/3;
    const w3 = total > 0 ? (frequency || 0) / total : 1/3;

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

    const x = w1 * x1 + w2 * x2 + w3 * x3;
    const y = w1 * y1 + w2 * y2 + w3 * y3;

    return { x, y };
  };

  const renderSVGTriangle = () => {
    const centerX = 150;
    const centroidY = 175;
    const baseSize = 240;
    const height = baseSize * Math.sqrt(3) / 2;
    const topY = centroidY - height * 2 / 3;
    const bottomY = centroidY + height / 3;

    const point = calculatePointPosition(
      branchContribution?.rgb,
      branchContribution?.noise,
      branchContribution?.frequency
    );

    return (
      <svg 
        viewBox="0 0 300 340" 
        className={`w-full h-full transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`}
      >
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {[0.25, 0.5, 0.75, 1].map((scale, index) => (
          <polygon
            key={`triangle-${index}`}
            points={drawTriangle(centerX, centroidY, baseSize * scale)}
            fill="none"
            stroke={index === 3 ? 'rgba(218, 32, 90, 0.8)' : 'rgba(255, 255, 255, 0.2)'}
            strokeWidth={index === 3 ? 2 : 1}
          />
        ))}

        <text x={centerX} y={topY - 12} fill="#60A5FA" fontSize="12" textAnchor="middle" fontWeight="600">
          {language === 'zh' ? '语义' : 'Semantic'}
        </text>
        <text x={centerX - baseSize / 2 - 15} y={bottomY + 5} fill="#A78BFA" fontSize="12" textAnchor="middle" fontWeight="600">
          {language === 'zh' ? '噪声' : 'Noise'}
        </text>
        <text x={centerX + baseSize / 2 + 15} y={bottomY + 5} fill="#34D399" fontSize="12" textAnchor="middle" fontWeight="600">
          {language === 'zh' ? '频域' : 'Frequency'}
        </text>

        <circle cx={point.x} cy={point.y} r="12" fill="rgba(218, 32, 90, 0.3)" filter="url(#glow)" />
        <circle cx={point.x} cy={point.y} r="8" fill="#DA205A" filter="url(#glow)" />
        <circle cx={point.x} cy={point.y} r="4" fill="white" />
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
          {language === 'zh' ? '融合证据三角图' : 'Fusion Evidence Triangle'}
        </h3>
        <div className="w-full aspect-square flex items-center justify-center bg-black/30 rounded-xl border border-white/10 text-sm text-gray-300">
          {language === 'zh' ? '当前未提供融合证据图' : 'No fusion evidence image provided'}
        </div>
        <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
          {language === 'zh' ? '该图展示语义、频域与噪声证据在融合决策中的相对权重。' : 'This chart shows the relative weights of semantic, frequency, and noise evidence in fused decision-making.'}
        </p>
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
        {language === 'zh' ? '融合证据三角图' : 'Fusion Evidence Triangle'}
      </h3>
      <div className="w-full aspect-square relative flex items-center justify-center bg-black/40 rounded-xl border border-white/10 overflow-hidden">
        {hasBranchData ? (
          renderSVGTriangle()
        ) : (
          <img
            src={src}
            alt="Fusion Evidence Triangle"
            className={`w-full h-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`}
          />
        )}
        <div className="absolute top-2 right-2 bg-black/60 p-1.5 rounded-full text-white/80 opacity-0 group-hover:opacity-100 transition-opacity">
          {zoomed ? <Maximize className="w-4 h-4" /> : <ZoomIn className="w-4 h-4" />}
        </div>
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2 text-xs text-center text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
          {language === 'zh' ? '融合证据可视化（点击可放大）' : 'Fusion evidence visualization (click to zoom)'}
        </div>
      </div>
      <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
        {language === 'zh' ? '顶点分别对应语义、噪声、频域证据，图中点反映当前样本的融合权重。' : 'Vertices correspond to semantic, noise, and frequency evidence; the point reflects current sample fusion weights.'}
      </p>
    </motion.div>
  );
};

export default FusionEvidenceTriangle;
