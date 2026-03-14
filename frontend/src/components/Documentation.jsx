import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Book, Cpu, BarChart3, Code, GitBranch, Zap, Server, Layers, Sparkles, ArrowRight, Upload, Eye } from 'lucide-react';

const Documentation = ({ language = 'zh' }) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
      className="max-w-5xl mx-auto px-6 py-12"
    >
      <div className="glass-card p-8 rounded-2xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center">
            <Book className="text-primary w-6 h-6" />
          </div>
          <h1 className="text-3xl font-bold text-white">AI Image Detector</h1>
        </div>

        <div className="prose prose-invert max-w-none">
          <AnimatePresence mode="wait">
            <motion.p 
              key={`${language}-intro1`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="text-gray-300 leading-relaxed mb-6 min-h-[80px]"
            >
              {language === 'zh' ? 'AI Image Detector 是一个先进的 AI 生成图像取证分析系统，专为 NTIRE 2026 Robust AIGC Detection 挑战设计。该项目采用多分支架构，融合 RGB 视觉特征、频域频谱特征和噪声残差特征，支持完整的训练、评估、推理流程，并提供基于 React + FastAPI 的 Web 界面和 GradCAM 可解释性分析。' : 'AI Image Detector is an advanced AI-generated image forensic analysis system designed for the NTIRE 2026 Robust AIGC Detection challenge. This project adopts a multi-branch architecture, fusing RGB visual features, frequency spectrum features, and noise residual features, supporting complete training, evaluation, and inference pipelines, and providing a React + FastAPI-based web interface and GradCAM explainability analysis.'}
            </motion.p>
          </AnimatePresence>

          <AnimatePresence mode="wait">
            <motion.h2 
              key={`${language}-features`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.3 }}
              className="text-2xl font-bold text-white mb-4 flex items-center gap-2"
            >
              <Sparkles className="text-secondary w-5 h-5" />
              {language === 'zh' ? '核心功能' : 'Key Features'}
            </motion.h2>
          </AnimatePresence>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div className="glass p-4 rounded-xl border border-primary/20">
              <GitBranch className="text-primary w-5 h-5 mb-2" />
              <h3 className="text-white font-semibold mb-1">{language === 'zh' ? '多分支架构' : 'Multi-Branch Architecture'}</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '三路并行网络，融合语义、频谱和噪声残差特征' : 'Three parallel branches, fusing semantic, spectrum, and noise residual features'}</p>
            </div>
            <div className="glass p-4 rounded-xl border border-secondary/20">
              <Cpu className="text-secondary w-5 h-5 mb-2" />
              <h3 className="text-white font-semibold mb-1">{language === 'zh' ? 'NTIRE 2026 支持' : 'NTIRE 2026 Support'}</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '完整支持 NTIRE Robust AIGC Detection 数据集' : 'Full support for NTIRE Robust AIGC Detection dataset'}</p>
            </div>
            <div className="glass p-4 rounded-xl border border-accent/20">
              <Server className="text-accent w-5 h-5 mb-2" />
              <h3 className="text-white font-semibold mb-1">{language === 'zh' ? 'FastAPI 后端' : 'FastAPI Backend'}</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '高性能异步推理 API 服务' : 'High-performance asynchronous inference API service'}</p>
            </div>
            <div className="glass p-4 rounded-xl border border-primary/20">
              <Layers className="text-primary w-5 h-5 mb-2" />
              <h3 className="text-white font-semibold mb-1">{language === 'zh' ? 'GradCAM 可解释性' : 'GradCAM Explainability'}</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '类激活热力图，可视化模型决策依据' : 'Class activation heatmaps, visualizing model decision basis'}</p>
            </div>
          </div>

          <AnimatePresence mode="wait">
            <motion.h2 
              key={`${language}-architecture`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.3 }}
              className="text-2xl font-bold text-white mb-4 flex items-center gap-2"
            >
              <Cpu className="text-secondary w-5 h-5" />
              {language === 'zh' ? '算法架构' : 'Algorithm Architecture'}
            </motion.h2>
          </AnimatePresence>

          <AnimatePresence mode="wait">
            <motion.p 
              key={`${language}-architecture-desc`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="text-gray-300 leading-relaxed mb-6 min-h-[40px]"
            >
              {language === 'zh' ? '本项目的核心是一个多模态混合检测器，采用 ' : 'The core of this project is a multi-modal hybrid detector, adopting a '}
              <strong className="text-primary">{language === 'zh' ? '三路并行网络' : 'three-branch parallel network'}</strong>
              {language === 'zh' ? ' 架构：' : ' architecture:'}
            </motion.p>
          </AnimatePresence>

          <div className="space-y-6 mb-8">
            <div className="glass p-6 rounded-xl border border-primary/20">
              <AnimatePresence mode="wait">
                <motion.h3 
                  key={`${language}-rgb-branch`}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  transition={{ duration: 0.3 }}
                  className="text-xl font-semibold text-white mb-3 flex items-center gap-2"
                >
                  <GitBranch className="text-primary w-5 h-5" />
                  {language === 'zh' ? 'RGB 分支' : 'RGB Branch'}
                </motion.h3>
              </AnimatePresence>
              <ul className="space-y-2 text-gray-300">
                <AnimatePresence mode="wait">
                  <motion.li 
                    key={`${language}-rgb-backbone`}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.3 }}
                    className="flex items-start gap-2"
                  >
                    <span className="text-primary font-mono text-sm mt-1">•</span>
                    <span><strong>Backbone:</strong> CLIP ViT (Pre-trained)</span>
                  </motion.li>
                </AnimatePresence>
                <AnimatePresence mode="wait">
                  <motion.li 
                    key={`${language}-rgb-function`}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.3, delay: 0.1 }}
                    className="flex items-start gap-2"
                  >
                    <span className="text-primary font-mono text-sm mt-1">•</span>
                    <span><strong>{language === 'zh' ? '作用:' : 'Function:'}</strong> {language === 'zh' ? '使用预训练的 CLIP ViT 提取高层语义特征' : 'Use pre-trained CLIP ViT to extract high-level semantic features'}</span>
                  </motion.li>
                </AnimatePresence>
              </ul>
            </div>

            <div className="glass p-6 rounded-xl border border-secondary/20">
              <AnimatePresence mode="wait">
                <motion.h3 
                  key={`${language}-frequency-branch`}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  transition={{ duration: 0.3 }}
                  className="text-xl font-semibold text-white mb-3 flex items-center gap-2"
                >
                  <Code className="text-secondary w-5 h-5" />
                  {language === 'zh' ? '频域分支' : 'Frequency Branch'}
                </motion.h3>
              </AnimatePresence>
              <ul className="space-y-2 text-gray-300">
                <AnimatePresence mode="wait">
                  <motion.li 
                    key={`${language}-freq-function`}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.3 }}
                    className="flex items-start gap-2"
                  >
                    <span className="text-secondary font-mono text-sm mt-1">•</span>
                    <span><strong>{language === 'zh' ? '方法:' : 'Method:'}</strong> FFT + CNN</span>
                  </motion.li>
                </AnimatePresence>
                <AnimatePresence mode="wait">
                  <motion.li 
                    key={`${language}-freq-function2`}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.3, delay: 0.1 }}
                    className="flex items-start gap-2"
                  >
                    <span className="text-secondary font-mono text-sm mt-1">•</span>
                    <span><strong>{language === 'zh' ? '作用:' : 'Function:'}</strong> {language === 'zh' ? '通过 FFT 变换提取图像的频域特征' : 'Extract frequency domain features of images through FFT transform'}</span>
                  </motion.li>
                </AnimatePresence>
              </ul>
            </div>

            <div className="glass p-6 rounded-xl border border-accent/20">
              <AnimatePresence mode="wait">
                <motion.h3 
                  key={`${language}-noise-branch`}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  transition={{ duration: 0.3 }}
                  className="text-xl font-semibold text-white mb-3 flex items-center gap-2"
                >
                  <Zap className="text-accent w-5 h-5" />
                  {language === 'zh' ? '噪声分支' : 'Noise Branch'}
                </motion.h3>
              </AnimatePresence>
              <ul className="space-y-2 text-gray-300">
                <AnimatePresence mode="wait">
                  <motion.li 
                    key={`${language}-noise-method`}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.3 }}
                    className="flex items-start gap-2"
                  >
                    <span className="text-accent font-mono text-sm mt-1">•</span>
                    <span><strong>{language === 'zh' ? '方法:' : 'Method:'}</strong> SRM + CNN</span>
                  </motion.li>
                </AnimatePresence>
                <AnimatePresence mode="wait">
                  <motion.li 
                    key={`${language}-noise-function`}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.3, delay: 0.1 }}
                    className="flex items-start gap-2"
                  >
                    <span className="text-accent font-mono text-sm mt-1">•</span>
                    <span><strong>{language === 'zh' ? '作用:' : 'Function:'}</strong> {language === 'zh' ? '利用 SRM 滤波器提取噪声残差' : 'Use SRM filters to extract noise residuals'}</span>
                  </motion.li>
                </AnimatePresence>
              </ul>
            </div>
          </div>

          <div className="glass p-6 rounded-xl mb-8">
            <AnimatePresence mode="wait">
              <motion.h3 
                key={`${language}-fusion`}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                transition={{ duration: 0.3 }}
                className="text-xl font-semibold text-white mb-3"
              >
                {language === 'zh' ? '融合与分类' : 'Fusion and Classification'}
              </motion.h3>
            </AnimatePresence>
            <AnimatePresence mode="wait">
              <motion.p 
                key={`${language}-fusion-desc`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="text-gray-300 min-h-[60px]"
              >
                {language === 'zh' ? '各分支特征经过提取后，通过 ' : 'After feature extraction from each branch, they are fused through '}
                <strong className="text-primary">Gated Fusion</strong>
                {language === 'zh' ? '（特征拼接）进行融合，然后通过二分类器进行最终预测（REAL vs AIGC）。' : ' (feature concatenation), and then final prediction is made through a binary classifier (REAL vs AIGC).'}
              </motion.p>
            </AnimatePresence>
          </div>

          <AnimatePresence mode="wait">
            <motion.h2 
              key={`${language}-system-arch`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.3 }}
              className="text-2xl font-bold text-white mb-4 flex items-center gap-2"
            >
              <Server className="text-secondary w-5 h-5" />
              {language === 'zh' ? '系统架构' : 'System Architecture'}
            </motion.h2>
          </AnimatePresence>

          <div className="glass p-6 rounded-xl mb-8">
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-primary font-mono text-sm mt-1">→</span>
                <span><strong>React Frontend:</strong> {language === 'zh' ? '现代化 Web 界面，支持图像上传和结果可视化' : 'Modern web interface, supporting image upload and result visualization'}</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-secondary font-mono text-sm mt-1">→</span>
                <span><strong>FastAPI Backend:</strong> {language === 'zh' ? '高性能异步推理 API' : 'High-performance asynchronous inference API'}</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent font-mono text-sm mt-1">→</span>
                <span><strong>Inference Engine:</strong> {language === 'zh' ? '集成 ForensicDetector 封装的多分支模型推理' : 'Integrated multi-branch model inference encapsulated by ForensicDetector'}</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary font-mono text-sm mt-1">→</span>
                <span><strong>GradCAM Explainability:</strong> {language === 'zh' ? '生成类激活热力图，可视化模型决策依据' : 'Generate class activation heatmaps, visualizing model decision basis'}</span>
              </li>
            </ul>
          </div>

          <AnimatePresence mode="wait">
            <motion.h2 
              key={`${language}-explainability`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.3 }}
              className="text-2xl font-bold text-white mb-4 flex items-center gap-2"
            >
              <Sparkles className="text-secondary w-5 h-5" />
              {language === 'zh' ? '可解释性功能' : 'Explainability Features'}
            </motion.h2>
          </AnimatePresence>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div className="glass p-4 rounded-xl border border-primary/20">
              <h3 className="text-white font-semibold mb-1">Grad-CAM</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '生成类激活热力图，高亮显示模型认为"伪造"的图像区域' : 'Generate class activation heatmaps, highlighting image regions the model considers "fake"'}</p>
            </div>
            <div className="glass p-4 rounded-xl border border-secondary/20">
              <h3 className="text-white font-semibold mb-1">{language === 'zh' ? '分支贡献' : 'Branch Contribution'}</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '量化 RGB、频域和噪声三个分支对最终决策的贡献比例' : 'Quantify the contribution ratio of RGB, frequency, and noise branches to the final decision'}</p>
            </div>
            <div className="glass p-4 rounded-xl border border-accent/20">
              <h3 className="text-white font-semibold mb-1">{language === 'zh' ? '频谱分析' : 'Spectrum Analysis'}</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '可视化图像的频域分布' : 'Visualize the frequency domain distribution of images'}</p>
            </div>
            <div className="glass p-4 rounded-xl border border-primary/20">
              <h3 className="text-white font-semibold mb-1">{language === 'zh' ? '噪声残差' : 'Noise Residuals'}</h3>
              <p className="text-gray-400 text-sm">{language === 'zh' ? '展示经过 SRM 滤波后的噪声残差图' : 'Display noise residual maps after SRM filtering'}</p>
            </div>
          </div>

          <AnimatePresence mode="wait">
            <motion.h2 
              key={`${language}-detection-flow`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.3 }}
              className="text-2xl font-bold text-white mb-4 flex items-center gap-2"
            >
              <ArrowRight className="text-secondary w-5 h-5" />
              {language === 'zh' ? '检测流程' : 'Detection Flow'}
            </motion.h2>
          </AnimatePresence>

          <div className="glass p-6 rounded-xl mb-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex-1 text-center">
                <div className="w-16 h-16 bg-primary/20 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Upload className="text-primary w-8 h-8" />
                </div>
                <h3 className="text-white font-semibold mb-1">{language === 'zh' ? '1. 上传图像' : '1. Upload Image'}</h3>
                <p className="text-gray-400 text-sm">{language === 'zh' ? '选择本地图片或拖拽上传' : 'Select local image or drag & drop'}</p>
              </div>
              <ArrowRight className="text-gray-500 w-6 h-6 rotate-90 md:rotate-0" />
              <div className="flex-1 text-center">
                <div className="w-16 h-16 bg-secondary/20 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Cpu className="text-secondary w-8 h-8" />
                </div>
                <h3 className="text-white font-semibold mb-1">{language === 'zh' ? '2. AI 分析' : '2. AI Analysis'}</h3>
                <p className="text-gray-400 text-sm">{language === 'zh' ? '多分支特征提取与融合' : 'Multi-branch feature extraction & fusion'}</p>
              </div>
              <ArrowRight className="text-gray-500 w-6 h-6 rotate-90 md:rotate-0" />
              <div className="flex-1 text-center">
                <div className="w-16 h-16 bg-accent/20 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Eye className="text-accent w-8 h-8" />
                </div>
                <h3 className="text-white font-semibold mb-1">{language === 'zh' ? '3. 查看结果' : '3. View Results'}</h3>
                <p className="text-gray-400 text-sm">{language === 'zh' ? '检测结果 + 可解释性分析' : 'Detection result + explainability'}</p>
              </div>
            </div>
          </div>

          <div className="border-t border-white/10 pt-8 mt-8">
            <p className="text-gray-500 text-sm text-center">
              © 2026 AI Forensic Research Lab. All rights reserved.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Documentation;
