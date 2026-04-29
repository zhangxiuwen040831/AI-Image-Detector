import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

const copy = {
  zh: {
    eyebrow: '',
    subtitle: 'Base-Only Industrial Candidate',
    description: '一个面向图像真实性证据审阅的单入口取证工作台。',
    cta: '点击进入',
  },
  en: {
    eyebrow: '',
    subtitle: 'Base-Only Industrial Candidate',
    description: 'A single-entry forensic workspace designed for calm, structured review of image authenticity evidence.',
    cta: 'Click To Enter',
  },
};

const IntroAnimation = ({ onComplete, language = 'zh' }) => {
  const [ready, setReady] = useState(false);
  const text = copy[language] || copy.zh;

  useEffect(() => {
    const timer = window.setTimeout(() => setReady(true), 120);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.35, ease: 'easeOut' } }}
      className="fixed inset-0 z-[100] cursor-pointer overflow-hidden bg-[#f7f8fa]"
      onClick={onComplete}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.98),rgba(247,248,250,0.96)_42%,rgba(240,244,248,0.92)_100%)]" />
      <div className="absolute inset-x-0 top-0 h-px bg-slate-200/80" />
      <div className="absolute inset-x-0 bottom-0 h-px bg-slate-200/80" />
      <div className="absolute left-[10%] top-[14%] h-40 w-40 rounded-full bg-white blur-3xl" />
      <div className="absolute right-[8%] bottom-[12%] h-52 w-52 rounded-full bg-slate-100 blur-3xl" />

      <div className="absolute inset-0 flex items-center justify-center px-6 py-10">
        <div className="flex w-full max-w-4xl flex-col items-center text-center">
          {text.eyebrow ? (
            <motion.p
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: ready ? 1 : 0, y: ready ? 0 : 12 }}
              transition={{ duration: 0.55 }}
              className="mb-6 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500"
            >
              {text.eyebrow}
            </motion.p>
          ) : null}

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: ready ? 1 : 0, y: ready ? 0 : 18 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
            className="mx-auto flex max-w-full flex-col items-center"
          >
            <h1
              className="whitespace-nowrap text-center text-[clamp(2.45rem,8.2vw,5.45rem)] font-medium uppercase leading-[0.94] tracking-[0.16em] text-slate-900"
              style={{ fontFamily: '"Helvetica Neue", "Segoe UI", Arial, sans-serif' }}
            >
              AI Image
            </h1>
            <h2
              className="mt-1 whitespace-nowrap text-center text-[clamp(2.75rem,9vw,5.95rem)] font-semibold uppercase leading-[0.92] tracking-[0.125em] text-slate-800"
              style={{ fontFamily: '"Helvetica Neue", "Segoe UI", Arial, sans-serif' }}
            >
              Detector
            </h2>
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: ready ? 1 : 0, y: ready ? 0 : 14 }}
            transition={{ duration: 0.55, delay: 0.08 }}
            className="mt-7 max-w-xl text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500 sm:mt-8"
          >
            {text.subtitle}
          </motion.p>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: ready ? 1 : 0, y: ready ? 0 : 16 }}
            transition={{ duration: 0.55, delay: 0.16 }}
            className="mt-5 max-w-lg text-sm leading-7 text-slate-500"
          >
            {text.description}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: ready ? 1 : 0, y: ready ? 0 : 18 }}
            transition={{ duration: 0.55, delay: 0.24 }}
            className="mt-10 inline-flex items-center gap-3 rounded-full border border-slate-300 bg-white px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-700 shadow-soft"
          >
            <span>{text.cta}</span>
            <ArrowRight className="h-4 w-4" />
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};

export default IntroAnimation;
