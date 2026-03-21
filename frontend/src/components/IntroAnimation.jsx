import React, { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';

const PARTICLE_LAYOUT = [
  { left: '21%', top: '25%', size: 3, delay: 0.15, duration: 12 },
  { left: '76%', top: '22%', size: 3, delay: 0.85, duration: 14 },
  { left: '17%', top: '69%', size: 2, delay: 0.4, duration: 15 },
  { left: '82%', top: '66%', size: 2, delay: 1.0, duration: 16 },
];

const IntroAnimation = ({ onComplete }) => {
  const [ready, setReady] = useState(false);
  const particles = useMemo(() => PARTICLE_LAYOUT, []);

  useEffect(() => {
    const timer = window.setTimeout(() => setReady(true), 260);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.01, filter: 'blur(5px)' }}
      transition={{ duration: 0.7, ease: 'easeOut' }}
      className="fixed inset-0 z-[100] cursor-pointer overflow-hidden bg-[#050608]"
      onClick={onComplete}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_41%,rgba(140,210,235,0.075),rgba(5,6,8,0)_32%),linear-gradient(180deg,#07090c_0%,#050608_52%,#040507_100%)]" />

      <motion.div
        className="absolute inset-x-[16%] top-[18.5%] h-px bg-gradient-to-r from-transparent via-white/16 to-transparent"
        initial={{ opacity: 0, scaleX: 0.72 }}
        animate={{ opacity: ready ? 1 : 0, scaleX: ready ? 1 : 0.72 }}
        transition={{ duration: 1.2, ease: 'easeOut' }}
      />

      <motion.div
        className="absolute inset-x-[25%] bottom-[20.5%] h-px bg-gradient-to-r from-transparent via-cyan-200/14 to-transparent"
        initial={{ opacity: 0, scaleX: 0.78 }}
        animate={{ opacity: ready ? 1 : 0, scaleX: ready ? 1 : 0.78 }}
        transition={{ duration: 1.25, ease: 'easeOut', delay: 0.1 }}
      />

      <motion.div
        className="absolute left-1/2 top-[46%] h-[32rem] w-[32rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-200/5 blur-[128px]"
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{
          opacity: ready ? [0.16, 0.24, 0.16] : 0,
          scale: ready ? [1, 1.03, 1] : 0.92,
        }}
        transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }}
      />

      {particles.map((particle, index) => (
        <motion.div
          key={`${particle.left}-${particle.top}-${index}`}
          className="absolute rounded-full bg-cyan-100/60"
          style={{
            left: particle.left,
            top: particle.top,
            width: particle.size,
            height: particle.size,
            boxShadow: '0 0 12px rgba(186, 230, 253, 0.18)',
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{
            opacity: ready ? [0.04, 0.14, 0.04] : 0,
            y: ready ? [0, -6, 0] : 0,
            x: ready ? [0, 2, 0] : 0,
            scale: ready ? [1, 1.12, 1] : 0.8,
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: particle.delay,
          }}
        />
      ))}

      <div className="absolute inset-0 flex items-center justify-center px-8 pointer-events-none">
        <div className="w-full max-w-6xl text-center">
          <motion.p
            initial={{ opacity: 0, y: 14, filter: 'blur(8px)' }}
            animate={{ opacity: ready ? 0.52 : 0, y: ready ? 0 : 14, filter: ready ? 'blur(0px)' : 'blur(8px)' }}
            transition={{ duration: 0.9, ease: 'easeOut' }}
            className="mb-10 text-[10px] uppercase tracking-[0.46em] text-white/46 md:mb-12 md:text-xs"
          >
            Final Delivery
          </motion.p>

          <div className="relative inline-flex flex-col items-center">
            <motion.h1
              initial={{ opacity: 0, y: 24, filter: 'blur(14px)', letterSpacing: '0.42em' }}
              animate={{
                opacity: ready ? 0.96 : 0,
                y: ready ? 0 : 24,
                filter: ready ? 'blur(0px)' : 'blur(14px)',
                letterSpacing: ready ? '0.24em' : '0.42em',
              }}
              transition={{ duration: 1.15, ease: 'easeOut' }}
              className="text-[30px] font-light uppercase leading-none text-white md:text-[70px]"
              style={{ fontFamily: '"Helvetica Neue", "SF Pro Display", "Segoe UI", Arial, sans-serif' }}
            >
              AI IMAGE
            </motion.h1>

            <motion.div
              initial={{ opacity: 0, scaleX: 0.82 }}
              animate={{ opacity: ready ? 0.72 : 0, scaleX: ready ? 1 : 0.82 }}
              transition={{ duration: 1, ease: 'easeOut', delay: 0.12 }}
              className="mx-auto my-5 h-px w-[min(32vw,248px)] bg-gradient-to-r from-transparent via-cyan-100/78 to-transparent md:my-6 md:w-[min(30vw,286px)]"
            />

            <motion.h2
              initial={{ opacity: 0, y: 24, filter: 'blur(14px)', letterSpacing: '0.56em' }}
              animate={{
                opacity: ready ? 0.98 : 0,
                y: ready ? 0 : 24,
                filter: ready ? 'blur(0px)' : 'blur(14px)',
                letterSpacing: ready ? '0.34em' : '0.56em',
              }}
              transition={{ duration: 1.18, ease: 'easeOut', delay: 0.08 }}
              className="text-[36px] font-medium uppercase leading-none text-cyan-50 md:text-[84px]"
              style={{ fontFamily: '"Helvetica Neue", "SF Pro Display", "Segoe UI", Arial, sans-serif' }}
            >
              DETECTOR
            </motion.h2>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: ready ? 1 : 0 }}
              transition={{ duration: 1.2, delay: 0.28 }}
              className="pointer-events-none absolute left-1/2 top-[calc(100%+10px)] w-[min(44vw,420px)] -translate-x-1/2"
              style={{
                transformOrigin: 'top center',
                transform: 'translateX(-50%) scaleY(-1)',
                WebkitMaskImage: 'linear-gradient(to bottom, rgba(255,255,255,0.22), rgba(255,255,255,0.03) 46%, transparent 82%)',
                maskImage: 'linear-gradient(to bottom, rgba(255,255,255,0.22), rgba(255,255,255,0.03) 46%, transparent 82%)',
              }}
            >
              <div className="text-white/10 text-[24px] font-light uppercase leading-none tracking-[0.24em] md:text-[56px]">
                AI IMAGE
              </div>
              <div className="mx-auto my-3 h-px w-[58%] bg-gradient-to-r from-transparent via-cyan-100/18 to-transparent md:my-4" />
              <div className="text-cyan-50/10 text-[28px] font-medium uppercase leading-none tracking-[0.34em] md:text-[68px]">
                DETECTOR
              </div>
            </motion.div>
          </div>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: ready ? 0.46 : 0, y: ready ? 0 : 12 }}
            transition={{ duration: 0.95, ease: 'easeOut', delay: 0.28 }}
            className="mt-10 text-[10px] uppercase tracking-[0.38em] text-slate-300/56 md:mt-14 md:text-xs"
          >
            Base-Only Industrial Candidate
          </motion.p>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: ready ? [0.24, 0.72, 0.24] : 0, y: ready ? 0 : 12 }}
        transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut', delay: 0.8 }}
        className="absolute bottom-14 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-white/[0.035] px-5 py-2 text-[10px] uppercase tracking-[0.34em] text-white/48 backdrop-blur-md md:text-xs"
      >
        Click To Enter
      </motion.div>
    </motion.div>
  );
};

export default IntroAnimation;
