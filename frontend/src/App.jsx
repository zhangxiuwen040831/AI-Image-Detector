
import React, { useState, useCallback, Suspense, lazy, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Info, ExternalLink } from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import Documentation from './components/Documentation';
import ShieldAnimation from './components/ShieldAnimation';
import IntroAnimation from './components/IntroAnimation';

// Lazy load visualization components for instant initial render
const DetectionResult = lazy(() => import('./components/DetectionResult'));
const ProbabilityChart = lazy(() => import('./components/ProbabilityChart'));
const BranchContribution = lazy(() => import('./components/BranchContribution'));
const NoiseResidualViewer = lazy(() => import('./components/NoiseResidualViewer'));
const FrequencySpectrum = lazy(() => import('./components/FrequencySpectrum'));
const ExplanationReport = lazy(() => import('./components/ExplanationReport'));
const FusionEvidenceTriangle = lazy(() => import('./components/FusionEvidenceTriangle'));

const API_URL = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '')).trim();
const API_BASE = API_URL ? API_URL.replace(/\/$/, '') : '';
const DETECT_ENDPOINT = API_BASE ? `${API_BASE}/detect` : '/detect';

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showDocumentation, setShowDocumentation] = useState(false);
  const [language, setLanguage] = useState('zh'); // zh for Chinese, en for English
  const [showIntro, setShowIntro] = useState(true);
  const branchVisualization = result?.branch_evidence || result?.branch_scores || result?.branch_contribution;
  const triangleVisualization = result?.branch_triangle || branchVisualization;

  const handleUpload = useCallback(async (file, preview) => {
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
      const response = await fetch(DETECT_ENDPOINT, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let detailMessage = 'Analysis failed. Please try again.';
        try {
          const errorData = await response.json();
          if (errorData?.detail) {
            detailMessage = String(errorData.detail);
          }
        } catch {
          detailMessage = 'Analysis failed. Please try again.';
        }
        throw new Error(detailMessage);
      }

      const data = await response.json();
      console.log('[Detect] endpoint:', DETECT_ENDPOINT);
      console.log('[Detect] result:', data);
      console.log('[Detect] artifacts:', data?.artifacts);
      console.log('[Detect] fusion_evidence length:', data?.artifacts?.fusion_evidence ? String(data.artifacts.fusion_evidence).length : 0);
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
      <AnimatePresence>
        {showIntro && (
          <IntroAnimation onComplete={() => setShowIntro(false)} />
        )}
      </AnimatePresence>

      {/* Dynamic Background */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/10 rounded-full blur-[120px] -z-10 animate-pulse" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-secondary/10 rounded-full blur-[120px] -z-10 animate-pulse" />

      {/* Header */}
      <header className="glass sticky top-0 z-50 py-4 px-6 mb-8 border-b border-white/5">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center neon-glow">
              <ShieldAnimation />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white leading-none">AI IMAGE DETECTOR</h1>
              <span className="text-[10px] text-gray-400 font-mono tracking-widest uppercase">Research Forensic System v1.0</span>
            </div>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <div className="flex items-center gap-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg p-1 transition-all duration-300">
              <motion.button 
                onClick={() => setLanguage('zh')} 
                className={`px-3 py-1 text-xs font-medium transition-all duration-300 ${language === 'zh' ? 'bg-primary text-white rounded' : 'text-gray-400 hover:text-white'}`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                中文
              </motion.button>
              <span className="text-gray-600">/</span>
              <motion.button 
                onClick={() => setLanguage('en')} 
                className={`px-3 py-1 text-xs font-medium transition-all duration-300 ${language === 'en' ? 'bg-primary text-white rounded' : 'text-gray-400 hover:text-white'}`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                English
              </motion.button>
            </div>
            <a href="#" onClick={(e) => { e.preventDefault(); setShowDocumentation(!showDocumentation); }} className="text-sm text-gray-400 hover:text-white transition-colors">{language === 'zh' ? '文档与架构' : 'Documentation'}</a>
            <a href="https://github.com/zhangxiuwen040831/AI-Image-Detector" target="_blank" rel="noopener noreferrer" className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm font-medium transition-all cursor-pointer hover:shadow-lg hover:shadow-primary/20">GitHub <ExternalLink className="inline-block w-3 h-3 ml-1" /></a>
          </nav>
        </div>
      </header>

      <main className="max-w-[1083px] mx-auto px-4 md:px-6 pb-12 w-full">
        {showDocumentation ? (
          <Documentation language={language} />
        ) : (
          <>
            {/* Stage 1: Upload */}
            <div className="text-center mb-12">
              <motion.h2 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1 }}
                className="text-4xl md:text-6xl font-bold mb-4 tracking-tighter"
              >
                Digital <span className="text-gradient">Forensics</span> Lab
              </motion.h2>
              <AnimatePresence mode="wait">
                <motion.p 
                  key={language}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="text-gray-400 max-w-2xl mx-auto text-lg min-h-[72px] flex items-center justify-center"
                >
                  {language === 'zh' ? '基于 V10 最终 base_only 模型的 AIGC 图像检测系统：以语义与频域为主判别路径，噪声证据仅保留为辅助诊断。' : 'AIGC image detection system powered by the final V10 base-only model: semantic and frequency evidence form the primary decision path, while noise evidence remains diagnostic only.'}
                </motion.p>
              </AnimatePresence>
            </div>

            <UploadPanel onUpload={handleUpload} isAnalyzing={isAnalyzing} language={language} />

            {/* Stage 2 & 3: Results & Analysis */}
            <AnimatePresence>
              {result && (
                <motion.div 
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className="space-y-6 lg:min-h-[1043px]"
                >
                  <div className="space-y-6 min-w-0">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 min-w-0">
                      <div className="min-w-0">
                        <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                          <ProbabilityChart probabilities={result.probabilities} probability={result.probability} language={language} />
                        </Suspense>
                      </div>
                      <div className="min-w-0">
                        <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                          <BranchContribution
                            scores={branchVisualization}
                            analysisMode={result.branch_analysis_mode}
                            mode={result.mode}
                            language={language}
                          />
                        </Suspense>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                      <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                        <NoiseResidualViewer imageBase64={result.artifacts?.noise_residual || result.srm_image} language={language} />
                      </Suspense>
                      <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                        <FrequencySpectrum imageBase64={result.artifacts?.frequency_spectrum || result.spectrum_image} language={language} />
                      </Suspense>
                      <Suspense fallback={<div className="glass-card h-64 animate-pulse" />}>
                        <FusionEvidenceTriangle
                          imageBase64={result.artifacts?.fusion_evidence || result.fusion_evidence_image}
                          branchContribution={triangleVisualization}
                          analysisMode={result.branch_analysis_mode}
                          mode={result.mode}
                          language={language}
                        />
                      </Suspense>
                    </div>

                    <div className="glass-card p-6 bg-primary/5 border-primary/20">
                      <div className="flex items-center gap-2 text-primary mb-3">
                        <Info className="w-4 h-4" />
                        <AnimatePresence mode="wait">
                          <motion.span 
                            key={`${language}-note`}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.3 }}
                            className="text-xs font-bold uppercase tracking-widest"
                          >
                            {language === 'zh' ? '研究说明' : 'Research Note'}
                          </motion.span>
                        </AnimatePresence>
                      </div>
                      <AnimatePresence mode="wait">
                        <motion.p 
                          key={`${language}-desc`}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.3 }}
                          className="text-xs text-gray-400 leading-relaxed font-mono min-h-[60px]"
                        >
                          {language === 'zh' ? '当前交付的 best.pth 对应 V10 最终模型，默认推理模式为 base_only：semantic + frequency 负责最终判定。当前分支图默认展示的是样本级证据强度，而不是旧式纯 gate 权重，因此更能反映每张图的真实差异。' : 'The delivered best.pth corresponds to the final V10 model with base_only as the default inference mode: semantic + frequency drive the final decision. The branch visuals now show sample-level evidence intensity rather than legacy gate weights, so they better reflect per-image differences.'}
                        </motion.p>
                      </AnimatePresence>
                    </div>

                    <Suspense fallback={<div className="glass-card p-20 animate-pulse text-center text-gray-500">Loading Report...</div>}>
                      <ExplanationReport explanation={result.explanation} language={language} />
                    </Suspense>
                  </div>

                  <div className="xl:hidden">
                    <Suspense fallback={<div className="glass-card p-20 animate-pulse text-center text-gray-500">Loading Result...</div>}>
                        <DetectionResult result={result} language={language} />
                      </Suspense>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {result && (
              <div
                className="hidden xl:block fixed top-1/2 -translate-y-1/2 w-[220px] 2xl:w-[240px] z-40 pointer-events-none"
                style={{ right: 'max(16px, calc((100vw - 1863px) / 2 + 16px))' }}
              >
                <div className="pointer-events-auto">
                  <Suspense fallback={<div className="glass-card p-20 animate-pulse text-center text-gray-500">Loading Result...</div>}>
                    <DetectionResult result={result} language={language} />
                  </Suspense>
                </div>
              </div>
            )}

            {error && (
              <div className="mt-8 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-500 text-center glass">
                {error}
              </div>
            )}
          </>
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
