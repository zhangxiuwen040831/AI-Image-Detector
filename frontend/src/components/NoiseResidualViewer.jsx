import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ZoomIn, Eye, Layers } from 'lucide-react';
import { toRenderableImageSrc } from '../utils/imageSrc';

const NoiseResidualViewer = ({ imageBase64, language = 'zh', embedded = false }) => {
  const [zoomed, setZoomed] = useState(false);
  const src = toRenderableImageSrc(imageBase64);
  const shellClass = embedded
    ? `flex h-full min-w-0 flex-col rounded-[26px] border border-line bg-panel p-4 lg:p-5 ${imageBase64 ? 'cursor-pointer' : ''}`
    : `glass-card h-full flex flex-col p-5 ${imageBase64 ? 'cursor-pointer' : ''}`;
  const imageClass = embedded
    ? 'relative w-full min-h-[200px] overflow-hidden rounded-[22px] border border-line bg-white lg:min-h-[220px] xl:min-h-[240px]'
    : 'relative w-full aspect-square overflow-hidden rounded-[22px] border border-line bg-panel md:aspect-[4/3] xl:aspect-[16/10]';

  const emptyState = (
    <div className={`${embedded ? 'min-h-[200px] lg:min-h-[220px] xl:min-h-[240px]' : 'aspect-square md:aspect-[4/3] xl:aspect-[16/10]'} flex w-full items-center justify-center rounded-[22px] border border-line bg-white text-sm text-muted`}>
      {language === 'zh' ? '当前未提供该证据图' : 'No evidence image provided'}
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className={shellClass}
      onClick={() => imageBase64 && setZoomed(!zoomed)}
    >
      <div className={`mb-4 ${embedded ? '' : 'border-b border-line pb-4'}`}>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white shadow-soft">
            <Layers className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="section-title">{language === 'zh' ? '诊断证据' : 'Diagnostic evidence'}</p>
            <h3 className="text-xl font-semibold tracking-tight text-ink">
              {language === 'zh' ? '噪声残差证据' : 'Noise residual evidence'}
            </h3>
          </div>
        </div>
      </div>

      {!imageBase64 ? emptyState : (
        <div className={imageClass}>
          <img
            src={src}
            alt="Noise Residual"
            className={`h-full w-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`}
            style={{ imageRendering: 'pixelated' }}
          />
          <div className="absolute right-3 top-3 rounded-full border border-line bg-white/90 p-2 text-muted shadow-soft">
            {zoomed ? <Eye className="h-4 w-4" /> : <ZoomIn className="h-4 w-4" />}
          </div>
        </div>
      )}

      <p className="mt-4 text-sm leading-7 text-muted">
        {language === 'zh'
          ? '该证据图用于辅助分析图像中的噪声结构是否符合自然拍摄特征。'
          : 'This evidence image helps analyze whether the image noise structure matches natural capture characteristics.'}
      </p>
    </motion.div>
  );
};

export default NoiseResidualViewer;
