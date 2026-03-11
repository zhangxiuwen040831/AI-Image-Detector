
import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { CheckCircle, AlertTriangle } from 'lucide-react';

const DetectionResult = ({ result }) => {
  if (!result) return null;

  const isAIGC = result.prediction === 'AIGC';
  const confidencePercent = Math.round(result.confidence * 100);
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="relative overflow-hidden rounded-2xl glass-card p-8 text-center border-l-4 border-l-primary neon-glow"
    >
      <div className="absolute top-0 right-0 w-32 h-32 bg-primary/20 rounded-full blur-3xl -z-10 animate-pulse" />
      
      <div className="flex flex-col items-center justify-center gap-4">
        <motion.div
          initial={{ rotate: -90, scale: 0 }}
          animate={{ rotate: 0, scale: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
          className={clsx(
            "p-4 rounded-full border-4 shadow-2xl",
            isAIGC ? "bg-red-500/10 border-red-500 text-red-500" : "bg-green-500/10 border-green-500 text-green-500"
          )}
        >
          {isAIGC ? <AlertTriangle className="w-12 h-12" /> : <CheckCircle className="w-12 h-12" />}
        </motion.div>
        
        <h2 className="text-4xl font-bold tracking-tight text-white neon-text">
          {isAIGC ? "AI GENERATED" : "REAL IMAGE"}
        </h2>
        
        <div className="flex items-center gap-4 mt-2">
          <div className="px-4 py-2 bg-white/5 rounded-lg border border-white/10">
            <span className="text-gray-400 text-xs uppercase tracking-wider block mb-1">Confidence</span>
            <span className="text-2xl font-mono font-bold text-primary">{confidencePercent}%</span>
          </div>
          <div className="px-4 py-2 bg-white/5 rounded-lg border border-white/10">
            <span className="text-gray-400 text-xs uppercase tracking-wider block mb-1">Probability</span>
            <span className="text-2xl font-mono font-bold text-secondary">{result.probability.toFixed(4)}</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default DetectionResult;
