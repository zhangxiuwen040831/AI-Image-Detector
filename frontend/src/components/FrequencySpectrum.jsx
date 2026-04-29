import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Maximize, Activity, ZoomIn } from 'lucide-react';
import { toRenderableImageSrc } from '../utils/imageSrc';

const FrequencySpectrum = ({ imageBase64, language = 'zh', embedded = false }) => {
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
      transition={{ duration: 0.45, delay: 0.05 }}
      className={shellClass}
      onClick={() => imageBase64 && setZoomed(!zoomed)}
    >
      <div className={`mb-4 ${embedded ? '' : 'border-b border-line pb-4'}`}>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white shadow-soft">
            <Activity className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="section-title">{language === 'zh' ? '频域证据' : 'Spectral evidence'}</p>
            <h3 className="text-xl font-semibold tracking-tight text-ink">
              {language === 'zh' ? '频谱证据' : 'Frequency spectrum evidence'}
            </h3>
          </div>
        </div>
      </div>

      {!imageBase64 ? emptyState : (
        <div className={imageClass}>
          <img
            src={src}
            alt="Frequency Spectrum"
            className={`h-full w-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`}
            style={{ imageRendering: 'pixelated' }}
          />
          <div className="absolute right-3 top-3 rounded-full border border-line bg-white/90 p-2 text-muted shadow-soft">
            {zoomed ? <Maximize className="h-4 w-4" /> : <ZoomIn className="h-4 w-4" />}
          </div>
        </div>
      )}

      <p className="mt-4 text-sm leading-7 text-muted">
        {language === 'zh'
          ? '该证据图用于辅助评估图像在频域上的自然性与规则性。'
          : 'This evidence image helps assess the naturalness and regularity of the image in the frequency domain.'}
      </p>
    </motion.div>
  );
};

export default FrequencySpectrum;
