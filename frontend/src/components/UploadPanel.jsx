import React, { useCallback, useState } from 'react';
import { Upload, X, FileImage, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';

const UploadPanel = ({
  onUpload,
  isAnalyzing,
  language = 'zh',
  hasResult = false,
  layout = 'default',
  thresholdMode = 'standard',
  thresholdOptions = [],
  onThresholdModeChange,
  thresholdNote = '',
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const isRail = layout === 'rail';
  const isWorkspace = layout === 'workspace';

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
    <div className={`space-y-4 ${isWorkspace ? 'h-full' : ''}`}>
      <div className={`rounded-[28px] border border-line bg-white shadow-card ${isRail ? 'p-4 lg:p-5' : 'p-5 md:p-6'} ${isWorkspace ? 'flex h-full flex-col' : ''}`}>
        <div className={`mb-5 flex flex-col gap-3 ${isRail ? '' : 'md:flex-row md:items-end md:justify-between md:gap-6'}`}>
          <div>
            <p className="section-title mb-2">{language === 'zh' ? '主要操作' : 'Primary action'}</p>
            <h3 className="whitespace-nowrap text-2xl font-semibold tracking-tight text-ink">
              {language === 'zh' ? '上传待检测图像' : 'Upload image for inspection'}
            </h3>
          </div>
          <p className={`${isRail || isWorkspace ? 'max-w-none' : 'max-w-md'} text-sm leading-7 text-muted`}>
            {language === 'zh'
              ? '支持拖拽或点击选择。上传后会返回最终判定、三分支分析概览以及噪声残差与频谱等取证证据。'
              : 'Drag and drop or click to browse. Each upload returns the final decision, a three-branch analysis overview, and forensic evidence such as noise residuals and spectrum maps.'}
          </p>
        </div>

        <div className="mb-5 rounded-[24px] border border-line bg-panel p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="section-title mb-2">{language === 'zh' ? '阈值模式' : 'Threshold mode'}</p>
              <h4 className="text-lg font-semibold tracking-tight text-ink">
                {thresholdOptions.find((option) => option.key === thresholdMode)?.label || ''}
              </h4>
            </div>
            <div className="flex flex-wrap gap-2">
              {thresholdOptions.map((option) => {
                const active = option.key === thresholdMode;
                return (
                  <button
                    key={option.key}
                    type="button"
                    onClick={() => onThresholdModeChange?.(option.key)}
                    className={clsx(
                      'rounded-full border px-4 py-2 text-sm font-medium transition-colors',
                      active
                        ? 'border-slate-700 bg-slate-700 text-white'
                        : 'border-line bg-white text-ink hover:border-lineStrong hover:bg-slate-50'
                    )}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>
          <p className="mt-3 text-sm leading-7 text-muted">{thresholdNote}</p>
        </div>

        <AnimatePresence mode="wait">
          {!preview ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className={clsx(
                `group relative overflow-hidden rounded-[26px] border-2 border-dashed bg-panel text-center transition-all duration-300 ${isRail ? 'px-5 py-8' : 'px-6 py-12'}`,
                dragActive ? 'border-slate-500 bg-slate-100' : 'border-line hover:border-lineStrong hover:bg-slate-50',
                error ? 'border-red-300' : '',
                isAnalyzing ? 'cursor-progress' : 'cursor-pointer'
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

              <div className={`mx-auto flex flex-col items-center ${isRail || isWorkspace ? 'max-w-none' : 'max-w-xl'}`}>
                <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-[20px] border border-line bg-white shadow-soft">
                  <Upload className="h-7 w-7 text-primary" />
                </div>
                <h4 className="text-xl font-semibold tracking-tight text-ink">
                  {language === 'zh' ? '拖拽图像到此处，或点击选择文件' : 'Drop an image here or click to browse'}
                </h4>
                <p className="mt-3 text-sm leading-7 text-muted">
                  {language === 'zh'
                    ? '支持 JPG、PNG、WEBP，单文件大小上限 10MB。'
                    : 'Supports JPG, PNG, and WEBP, with a maximum size of 10MB per file.'}
                </p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="preview"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="overflow-hidden rounded-[26px] border border-line bg-panel"
            >
              <div className="flex items-center justify-between border-b border-line px-5 py-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white shadow-soft">
                    <FileImage className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="section-title">{language === 'zh' ? '已载入图像' : 'Loaded image'}</p>
                    <h4 className="text-base font-semibold text-ink">
                      {language === 'zh' ? '准备执行检测' : 'Ready for analysis'}
                    </h4>
                  </div>
                </div>

                <button
                  onClick={(e) => { e.stopPropagation(); clearFile(); }}
                  className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-3 py-2 text-sm font-medium text-muted transition-colors hover:text-ink"
                  disabled={isAnalyzing}
                >
                  <X className="h-4 w-4" />
                  <span>{language === 'zh' ? '清除' : 'Clear'}</span>
                </button>
              </div>

              <div className={`grid gap-0 ${isRail ? 'grid-cols-1' : 'md:grid-cols-[minmax(0,1.05fr)_minmax(280px,0.95fr)]'}`}>
                <div className={`flex items-center justify-center bg-white p-5 ${isRail ? 'min-h-[220px]' : 'min-h-[300px]'}`}>
                  <img
                    src={preview}
                    alt="Preview"
                    className={`${isRail ? 'max-h-[220px]' : 'max-h-[360px]'} w-full rounded-[20px] border border-line bg-panel object-contain shadow-soft`}
                  />
                </div>
                <div className={`flex flex-col justify-center border-t border-line p-6 ${isRail ? '' : 'md:border-l md:border-t-0'}`}>
                  <p className="section-title">{language === 'zh' ? '推理状态' : 'Inference status'}</p>
                  <h4 className="mt-3 text-2xl font-semibold tracking-tight text-ink">
                    {language === 'zh' ? '图像已载入工作区' : 'Image loaded into workspace'}
                  </h4>
                  <p className="mt-4 text-sm leading-7 text-muted">
                    {language === 'zh'
                      ? '系统会同时计算语义结构、频域分布与噪声残差线索，并以稳定推理路径给出最终判定。'
                      : 'The system computes semantic structure, frequency distribution, and noise residual cues, then uses the stable inference path for the final decision.'}
                  </p>

                  <div className="mt-6 rounded-[20px] border border-line bg-white p-4">
                    <div className="mb-3 flex items-center justify-between text-sm">
                      <span className="font-medium text-ink">{language === 'zh' ? '检测进度' : 'Analysis progress'}</span>
                      <span className="text-muted">
                        {hasResult
                          ? (language === 'zh' ? '已完成' : 'Completed')
                          : isAnalyzing
                            ? (language === 'zh' ? '进行中' : 'Running')
                            : (language === 'zh' ? '等待结果' : 'Waiting for result')}
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-panelMuted">
                      <motion.div
                        className="h-full rounded-full bg-slate-700"
                        initial={{ width: '8%' }}
                        animate={{ width: hasResult ? '100%' : isAnalyzing ? '92%' : '20%' }}
                        transition={
                          hasResult
                            ? { duration: 0.3, ease: 'easeOut' }
                            : isAnalyzing
                              ? { duration: 1.8, repeat: Infinity, repeatType: 'reverse', ease: 'easeInOut' }
                              : { duration: 0.35 }
                        }
                      />
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 rounded-[20px] border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          <AlertCircle className="h-4 w-4" />
          {error}
        </motion.div>
      )}
    </div>
  );
};

export default UploadPanel;
