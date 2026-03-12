
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Maximize, Activity, ZoomIn } from 'lucide-react';
import { toRenderableImageSrc } from '../utils/imageSrc';

const FrequencySpectrum = ({ imageBase64, language = 'zh' }) => {
  const [zoomed, setZoomed] = useState(false);
  const src = toRenderableImageSrc(imageBase64);

  if (!imageBase64) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.6 }}
        className="glass-card rounded-2xl p-4 h-full flex flex-col justify-center relative overflow-hidden border border-white/10"
      >
        <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full flex items-center justify-center gap-2">
            <Activity className="w-6 h-6" style={{ color: 'var(--color-primary, #DA205A)' }} />
            {language === 'zh' ? '频谱证据' : 'Frequency Spectrum Evidence'}
          </h3>
          <div className="w-full aspect-square flex items-center justify-center bg-black/30 rounded-xl border border-white/10 text-sm text-gray-300">
            {language === 'zh' ? '当前未提供该证据图' : 'No evidence image provided'}
          </div>
          <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
            {language === 'zh' ? '频谱图用于辅助观察频域能量分布与潜在异常模式。' : 'Frequency spectrum helps observe frequency domain energy distribution and potential abnormal patterns.'}
          </p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.6 }}
      className="glass-card rounded-2xl p-4 h-full flex flex-col justify-center relative overflow-hidden group cursor-pointer border border-white/10"
      onClick={() => setZoomed(!zoomed)}
    >
      <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full flex items-center justify-center gap-2">
        <Activity className="w-6 h-6" style={{ color: 'var(--color-primary, #DA205A)' }} />
        {language === 'zh' ? '频谱证据' : 'Frequency Spectrum Evidence'}
      </h3>
      
      <div className="w-full aspect-square relative flex items-center justify-center bg-black/40 rounded-xl border border-white/10 overflow-hidden">
        <img 
          src={src} 
          alt="Frequency Spectrum" 
          className={`w-full h-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`} 
          style={{ imageRendering: 'pixelated' }}
        />
        
        <div className="absolute top-2 right-2 bg-black/60 p-1.5 rounded-full text-white/80 opacity-0 group-hover:opacity-100 transition-opacity">
          {zoomed ? <Maximize className="w-4 h-4" /> : <ZoomIn className="w-4 h-4" />}
        </div>
        
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2 text-xs text-center text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
          {language === 'zh' ? '频谱可视化（点击可放大）' : 'Frequency spectrum visualization (click to zoom)'} 
        </div>
      </div>
      <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
        {language === 'zh' ? '该证据图用于辅助评估图像在频域上的自然性与规则性。' : 'This evidence image helps assess the naturalness and regularity of the image in the frequency domain.'}
      </p>
    </motion.div>
  );
};

export default FrequencySpectrum;
