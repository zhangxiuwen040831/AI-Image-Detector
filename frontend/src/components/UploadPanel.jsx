
import React, { useCallback, useState } from 'react';
import { Upload, X, FileImage, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';

const UploadPanel = ({ onUpload, isAnalyzing, language = 'zh' }) => {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const validateFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError(language === 'zh' ? '仅允许上传图像文件' : 'Only image files are allowed');
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError(language === 'zh' ? '文件大小必须小于10MB' : 'File size must be less than 10MB');
      return false;
    }
    return true;
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError(null);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        handleFile(file);
      }
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    setError(null);
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        handleFile(file);
      }
    }
  };

  const handleFile = (file) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
      onUpload(file, reader.result);
    };
    reader.readAsDataURL(file);
  };

  const clearFile = () => {
    setPreview(null);
    setError(null);
    onUpload(null, null);
  };

  return (
    <div className="w-full max-w-2xl mx-auto mb-8">
      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1 }}
            className={clsx(
              "relative border-2 border-dashed rounded-xl p-10 text-center transition-all duration-300 glass-card cursor-pointer group",
              dragActive ? "border-primary bg-primary/5 scale-[1.01]" : "border-white/20 hover:border-primary/50",
              error ? "border-red-500/50" : ""
            )}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload').click()}
          >
            <input
              id="file-upload"
              type="file"
              className="hidden"
              accept="image/*"
              onChange={handleChange}
              disabled={isAnalyzing}
            />
            
            <div className="flex flex-col items-center justify-center gap-4">
              <div className="p-4 rounded-full bg-white/5 group-hover:bg-primary/20 transition-colors duration-300">
                <Upload className="w-8 h-8 text-gray-400 group-hover:text-primary transition-colors" />
              </div>
              <div>
                <AnimatePresence mode="wait">
                  <motion.h3 
                    key={`${language}-upload-title`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="text-lg font-medium text-white mb-1 min-h-[24px]"
                  >
                    {language === 'zh' ? '上传图像进行分析' : 'Upload image for analysis'}
                  </motion.h3>
                </AnimatePresence>
                <AnimatePresence mode="wait">
                  <motion.p 
                    key={`${language}-upload-desc`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="text-sm text-gray-400 min-h-[20px]"
                  >
                    {language === 'zh' ? '拖拽文件或点击浏览' : 'Drag file or click to browse'}
                  </motion.p>
                </AnimatePresence>
              </div>
              <AnimatePresence mode="wait">
                <motion.p 
                  key={`${language}-upload-supported`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="text-xs text-gray-500 mt-2 min-h-[16px]"
                >
                  {language === 'zh' ? '支持：JPG, PNG, WEBP (最大10MB)' : 'Supported: JPG, PNG, WEBP (max 10MB)'}
                </motion.p>
              </AnimatePresence>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="relative rounded-xl overflow-hidden glass-card border-primary/30 neon-glow"
          >
            <div className="absolute top-4 right-4 z-10">
              <button
                onClick={(e) => { e.stopPropagation(); clearFile(); }}
                className="p-2 bg-black/50 hover:bg-red-500/80 rounded-full text-white transition-all backdrop-blur-sm"
                disabled={isAnalyzing}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="flex flex-col md:flex-row h-64">
              <div className="w-full md:w-1/2 bg-black/40 flex items-center justify-center p-4">
                <img 
                  src={preview} 
                  alt="Preview" 
                  className="max-h-full max-w-full object-contain rounded-lg shadow-lg" 
                />
              </div>
              <div className="w-full md:w-1/2 p-6 flex flex-col justify-center items-start border-l border-white/10">
                <div className="flex items-center gap-2 text-primary mb-2">
                  <FileImage className="w-5 h-5" />
                  <AnimatePresence mode="wait">
                    <motion.span 
                      key={`${language}-loaded`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="text-sm font-mono"
                    >
                      {language === 'zh' ? '图像已加载' : 'Image loaded'}
                    </motion.span>
                  </AnimatePresence>
                </div>
                <AnimatePresence mode="wait">
                  <motion.h3 
                    key={`${language}-ready`}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    transition={{ duration: 0.3 }}
                    className="text-xl font-bold text-white mb-4 min-h-[32px]"
                  >
                    {language === 'zh' ? '准备进行分析' : 'Ready for analysis'}
                  </motion.h3>
                </AnimatePresence>
                <AnimatePresence mode="wait">
                  <motion.p 
                    key={`${language}-analyze`}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    transition={{ duration: 0.3 }}
                    className="text-sm text-gray-400 mb-6 min-h-[40px]"
                  >
                    {language === 'zh' ? '系统将分析全局语义、噪声伪迹与频域伪迹特征，并进行融合判定。' : 'The system will analyze global semantics, noise artifacts, and frequency artifacts, then fuse them for the final decision.'}
                  </motion.p>
                </AnimatePresence>
                
                {isAnalyzing && (
                  <div className="flex items-center gap-3 w-full">
                    <div className="h-1 flex-1 bg-white/10 rounded-full overflow-hidden">
                      <motion.div 
                        className="h-full bg-primary"
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      />
                    </div>
                    <span className="text-xs font-mono text-primary animate-pulse">{language === 'zh' ? '分析中...' : 'Analyzing...'}</span>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {error && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400 text-sm"
        >
          <AlertCircle className="w-4 h-4" />
          {error}
        </motion.div>
      )}
    </div>
  );
};

export default UploadPanel;
