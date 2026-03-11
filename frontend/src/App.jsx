
import React, { useState, useCallback, Suspense, lazy } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Search, Zap, Info, ExternalLink } from 'lucide-react';
import UploadPanel from './components/UploadPanel';

// Lazy load visualization components for instant initial render
const DetectionResult = lazy(() => import('./components/DetectionResult'));
const ProbabilityChart = lazy(() => import('./components/ProbabilityChart'));
const BranchContribution = lazy(() => import('./components/BranchContribution'));
const NoiseResidualViewer = lazy(() => import('./components/NoiseResidualViewer'));
const FrequencySpectrum = lazy(() => import('./components/FrequencySpectrum'));

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = useCallback(async (file) => {
    if (!file) {
      setResult(null);
      setError(null);
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  return (
    <div className="min-h-screen relative overflow-hidden bg-background">
      {/* Dynamic Background */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/10 rounded-full blur-[120px] -z-10 animate-pulse" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-secondary/10 rounded-full blur-[120px] -z-10 animate-pulse" />

      {/* Header */}
      <header className="glass sticky top-0 z-50 py-4 px-6 mb-8 border-b border-white/5">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center neon-glow">
              <Shield className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white leading-none">AI IMAGE DETECTOR</h1>
              <span className="text-[10px] text-gray-400 font-mono tracking-widest uppercase">Research Forensic System v1.0</span>
            </div>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <a href="#" className="text-sm text-gray-400 hover:text-white transition-colors">Documentation</a>
            <a href="#" className="text-sm text-gray-400 hover:text-white transition-colors">Architecture</a>
            <a href="#" className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm font-medium transition-all">GitHub <ExternalLink className="inline-block w-3 h-3 ml-1" /></a>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 pb-20">
        {/* Stage 1: Upload */}
        <div className="text-center mb-12">
          <motion.h2 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-6xl font-bold mb-4 tracking-tighter"
          >
            Digital <span className="text-gradient">Forensics</span> Lab
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-gray-400 max-w-2xl mx-auto text-lg"
          >
            Advanced multi-branch neural analysis for AIGC verification. Detect spectral artifacts and noise residuals in real-time.
          </motion.p>
        </div>

        <UploadPanel onUpload={handleUpload} isAnalyzing={isAnalyzing} />

        {/* Stage 2 & 3: Results & Analysis */}
        <AnimatePresence>
          {result && (
            <motion.div 
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              <div className="lg:col-span-2 space-y-6">
                <Suspense fallback={<div className="glass-card p-20 animate-pulse text-center text-gray-500">Loading Result...</div>}>
                  <DetectionResult result={result} />
                </Suspense>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                    <NoiseResidualViewer imageBase64={result.srm_image} />
                  </Suspense>
                  <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                    <FrequencySpectrum imageBase64={result.spectrum_image} />
                  </Suspense>
                </div>
              </div>

              <div className="space-y-6">
                <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                  <ProbabilityChart probability={result.probability} />
                </Suspense>
                <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                  <BranchContribution scores={result.branch_scores} />
                </Suspense>
                
                <div className="glass-card p-6 bg-primary/5 border-primary/20">
                  <div className="flex items-center gap-2 text-primary mb-3">
                    <Info className="w-4 h-4" />
                    <span className="text-xs font-bold uppercase tracking-widest">Research Note</span>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed font-mono">
                    Model weights (best.pth) optimized for ResNet18 Multi-Branch architecture. FFT and SRM filters used for artifact detection in high-frequency bands. Accuracy: 87.31%.
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <div className="mt-8 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-500 text-center glass">
            {error}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="py-10 border-t border-white/5 text-center">
        <div className="flex items-center justify-center gap-2 mb-4">
          <Zap className="w-4 h-4 text-secondary" />
          <span className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">Inference engine active - CUDA Optimized</span>
        </div>
        <p className="text-xs text-gray-600">© 2026 AI Forensic Research Lab. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
