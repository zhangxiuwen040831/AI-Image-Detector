import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, Maximize } from 'lucide-react';
import { toRenderableImageSrc } from '../utils/imageSrc';

const AttentionHeatmap = ({ gradCamBase64, gradCamOverlayBase64, language = 'zh' }) => {
  const [zoomed, setZoomed] = useState(false);
  const [imageStatus, setImageStatus] = useState('idle');
  const src = toRenderableImageSrc(gradCamOverlayBase64 || gradCamBase64);

  useEffect(() => {
    if (!src) {
      setImageStatus('idle');
    }
  }, [src]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className={src ? "glass-card rounded-2xl p-4 flex flex-col items-center justify-center relative overflow-hidden group cursor-pointer border border-white/10 h-full" : "glass-card rounded-2xl p-4 flex flex-col items-center justify-center relative overflow-hidden border border-white/10 h-full"}
      onClick={() => src && setZoomed(!zoomed)}
    >
      <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full flex items-center justify-center gap-2">
        <Eye className="w-6 h-6" style={{ color: 'var(--color-primary, #DA205A)' }} />
        <span>{language === 'zh' ? '注意力热力图' : 'Attention Heatmap'}</span>
      </h3>
      
      {src ? (
        <div className="w-full aspect-square relative flex items-center justify-center bg-black/40 rounded-xl border border-white/10 overflow-hidden">
          <img 
            src={src} 
            alt="Attention Heatmap" 
            className={`w-full h-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`} 
            style={{ imageRendering: 'pixelated' }}
            onLoad={() => {
              setImageStatus('loaded');
              console.log('[AttentionHeatmap] image loaded', { length: src.length });
            }}
            onError={(event) => {
              setImageStatus('error');
              console.error('[AttentionHeatmap] image load error', {
                hasGradCam: Boolean(gradCamBase64),
                hasGradCamOverlay: Boolean(gradCamOverlayBase64),
                currentSrc: event.currentTarget?.currentSrc || '',
              });
            }}
          />
          
          <div className="absolute top-2 right-2 bg-black/60 p-1.5 rounded-full text-white/80 opacity-0 group-hover:opacity-100 transition-opacity">
            {zoomed ? <Maximize className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </div>
          
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2 text-xs text-center text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
            {language === 'zh' ? '注意力热图（点击可放大）' : 'Attention heatmap (click to zoom)'} 
          </div>
        </div>
      ) : (
        <div className="w-full aspect-square flex items-center justify-center bg-black/30 rounded-xl border border-white/10 text-sm text-gray-300">
          {language === 'zh' ? '当前版本暂未生成注意力热图' : 'Attention heatmap not generated in current version'}
        </div>
      )}
      {src && imageStatus === 'error' && (
        <div className="w-full mt-3 p-2 text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg text-center">
          {language === 'zh' ? '热图加载失败，请检查后端返回的 base64 内容与浏览器控制台日志' : 'Failed to load heatmap, please check backend returned base64 content and browser console logs'}
        </div>
      )}
      <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
        {language === 'zh' ? '展示模型在推理过程中关注的图像区域，帮助理解模型决策依据。' : 'Shows areas the model focused on during inference, helping understand model decision basis.'}
      </p>
    </motion.div>
  );
};

export default AttentionHeatmap;
