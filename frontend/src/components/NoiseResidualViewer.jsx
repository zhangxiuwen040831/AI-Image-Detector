
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ZoomIn, Eye, Layers } from 'lucide-react';

const NoiseResidualViewer = ({ imageBase64 }) => {
  const [zoomed, setZoomed] = useState(false);
  const src = `data:image/png;base64,${imageBase64}`;

  if (!imageBase64) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.5 }}
      className="glass-card p-6 h-full flex flex-col justify-center relative overflow-hidden group cursor-pointer"
      onClick={() => setZoomed(!zoomed)}
    >
      <h3 className="text-lg font-bold text-white mb-4 text-center tracking-wider uppercase border-b border-white/10 pb-2 w-full flex items-center justify-center gap-2">
        <Layers className="w-4 h-4 text-primary" />
        Noise Residual (SRM)
      </h3>
      
      <div className="w-full h-64 relative flex items-center justify-center bg-black/40 rounded-lg border border-white/5 overflow-hidden">
        <img 
          src={src} 
          alt="Noise Residual" 
          className={`max-w-full max-h-full object-contain transition-transform duration-500 ${zoomed ? 'scale-150' : 'scale-100 hover:scale-105'}`} 
          style={{ imageRendering: 'pixelated' }}
        />
        
        <div className="absolute top-2 right-2 bg-black/60 p-1.5 rounded-full text-white/80 opacity-0 group-hover:opacity-100 transition-opacity">
          {zoomed ? <Eye className="w-4 h-4" /> : <ZoomIn className="w-4 h-4" />}
        </div>
        
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2 text-xs text-center text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity">
          High-pass filtered texture artifacts
        </div>
      </div>
    </motion.div>
  );
};

export default NoiseResidualViewer;
