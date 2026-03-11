
import React, { useCallback, useState } from 'react';
import { Upload, X, FileImage, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';

const UploadPanel = ({ onUpload, isAnalyzing }) => {
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
      setError('Only image files are allowed');
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
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
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
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
                <h3 className="text-lg font-medium text-white mb-1">
                  Upload Image for Analysis
                </h3>
                <p className="text-sm text-gray-400">
                  Drag & drop or click to browse
                </p>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Supported: JPG, PNG, WEBP (Max 10MB)
              </p>
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
                  <span className="text-sm font-mono">IMAGE LOADED</span>
                </div>
                <h3 className="text-xl font-bold text-white mb-4">Ready for Inspection</h3>
                <p className="text-sm text-gray-400 mb-6">
                  System will analyze RGB, Noise Residuals, and Frequency Spectrum.
                </p>
                
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
                    <span className="text-xs font-mono text-primary animate-pulse">ANALYZING...</span>
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
