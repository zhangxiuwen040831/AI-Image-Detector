import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Shield } from 'lucide-react';
import { motion } from 'framer-motion';

/**
 * ShieldAnimation Component
 * 
 * Implements a particle convergence and dispersion animation for the Shield logo.
 * 
 * Features:
 * - Converge: Particles fly in from OUTSIDE the container to the center.
 * - Static: Shield icon remains visible for a set duration.
 * - Disperse: Particles fly out from the center to OUTSIDE the container.
 * - Clipping: Particles are only visible strictly INSIDE the rounded container.
 * - Seamless loop.
 * - High performance (Canvas 2D).
 */

const ShieldAnimation = () => {
  const canvasRef = useRef(null);
  const [showIcon, setShowIcon] = useState(false);
  const animationRef = useRef(null);
  
  // Animation State Management
  const phaseRef = useRef('converge'); 
  const startTimeRef = useRef(Date.now());
  const particlesRef = useRef([]);

  // --- 2) Configuration Parameters ---
  const config = useMemo(() => ({
    // Visual Style
    particleCount: 60, // Slightly increased for better density
    colors: ['#ffffff', '#00D1FF'], // White and Cyan
    canvasSize: 40, // 40x40 logical pixels
    borderRadius: 8, // Container rounded corner radius
    
    // Animation Timing (ms) - Total 7 seconds cycle
    convergeDuration: 1500, // 1.5s
    staticDuration: 4000,   // 4.0s
    disperseDuration: 1500, // 1.5s
    
    // Motion Parameters
    center: 20, // 20px is center of 40px canvas
    
    // Particles start/end at this distance from center (OUTSIDE the 20px half-width)
    radiusMin: 25, 
    radiusMax: 45,
    
    // Easing
    easeOutCubic: (t) => 1 - Math.pow(1 - t, 3),
    easeInCubic: (t) => t * t * t,
  }), []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Handle High-DPI screens
    canvas.width = config.canvasSize * dpr;
    canvas.height = config.canvasSize * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = `${config.canvasSize}px`;
    canvas.style.height = `${config.canvasSize}px`;

    // --- 1) Animation Core Logic ---
    
    // Helper: Create Clipping Path for Rounded Rectangle
    const clipRoundedRect = (ctx, w, h, r) => {
        ctx.beginPath();
        ctx.moveTo(r, 0);
        ctx.lineTo(w - r, 0);
        ctx.quadraticCurveTo(w, 0, w, r);
        ctx.lineTo(w, h - r);
        ctx.quadraticCurveTo(w, h, w - r, h);
        ctx.lineTo(r, h);
        ctx.quadraticCurveTo(0, h, 0, h - r);
        ctx.lineTo(0, r);
        ctx.quadraticCurveTo(0, 0, r, 0);
        ctx.closePath();
        ctx.clip();
    };

    // Initialize Particles
    // Origins are now OUTSIDE the box to simulate flying in/out
    const initParticles = () => {
      const newParticles = [];
      for (let i = 0; i < config.particleCount; i++) {
        const angle = Math.random() * Math.PI * 2;
        const dist = config.radiusMin + Math.random() * (config.radiusMax - config.radiusMin);
        
        newParticles.push({
          // Origin: Outside the container
          originX: config.center + Math.cos(angle) * dist,
          originY: config.center + Math.sin(angle) * dist,
          
          x: 0, 
          y: 0,
          
          size: Math.random() * 1.5 + 0.5,
          color: config.colors[Math.floor(Math.random() * config.colors.length)],
          delayOffset: Math.random() * 0.3 
        });
      }
      return newParticles;
    };

    particlesRef.current = initParticles();
    
    const animate = () => {
      const now = Date.now();
      const elapsed = now - startTimeRef.current;
      
      // Clear entire canvas
      ctx.clearRect(0, 0, config.canvasSize, config.canvasSize);
      
      // Save context state before clipping
      ctx.save();
      
      // Apply Clipping Path (Rounded Rect)
      // This ensures particles outside the container are invisible
      clipRoundedRect(ctx, config.canvasSize, config.canvasSize, config.borderRadius);
      
      let progress = 0;
      let globalAlpha = 1;
      
      // State Machine
      if (phaseRef.current === 'converge') {
        progress = elapsed / config.convergeDuration;
        
        if (progress >= 1.0) {
           phaseRef.current = 'static';
           startTimeRef.current = now;
           setShowIcon(true);
           progress = 1.0;
        } else {
           if (showIcon) setShowIcon(false);
        }
      } 
      else if (phaseRef.current === 'static') {
        if (elapsed >= config.staticDuration) {
           phaseRef.current = 'disperse';
           startTimeRef.current = now;
           setShowIcon(false); 
        }
      } 
      else if (phaseRef.current === 'disperse') {
        progress = elapsed / config.disperseDuration;
        
        if (progress >= 1.0) {
           phaseRef.current = 'converge';
           startTimeRef.current = now;
           progress = 1.0;
        }
      }

      // Draw Particles
      if (phaseRef.current === 'converge' || phaseRef.current === 'disperse') {
        particlesRef.current.forEach(p => {
          let t = 0;
          
          if (phaseRef.current === 'converge') {
            const effectiveProgress = Math.max(0, (progress - p.delayOffset) / (1 - p.delayOffset));
            t = config.easeOutCubic(effectiveProgress); // 0 (outside) -> 1 (center)
            
            p.x = p.originX + (config.center - p.originX) * t;
            p.y = p.originY + (config.center - p.originY) * t;
            
            globalAlpha = 1; // Keep fully opaque, let clip handle visibility
            
          } else { // Disperse
            const effectiveProgress = Math.max(0, (progress - p.delayOffset) / (1 - p.delayOffset));
            t = config.easeOutCubic(effectiveProgress); // 0 (center) -> 1 (outside)
            
            p.x = config.center + (p.originX - config.center) * t;
            p.y = config.center + (p.originY - config.center) * t;
            
            globalAlpha = 1;
          }
          
          if (globalAlpha > 0.01) {
            ctx.globalAlpha = globalAlpha;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }
      
      // Restore context to remove clipping for next frame (though clearRect handles it, good practice)
      ctx.restore();
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => cancelAnimationFrame(animationRef.current);
  }, [config]); 

  return (
    <div className="relative w-6 h-6 flex items-center justify-center">
      <canvas 
        ref={canvasRef}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
        style={{ width: '40px', height: '40px' }} 
      />
      
      <motion.div
        animate={{ 
          opacity: showIcon ? 1 : 0, 
          scale: showIcon ? 1 : 0.2,
          filter: showIcon ? 'blur(0px)' : 'blur(4px)'
        }}
        transition={{ duration: 0.4, ease: "easeInOut" }} 
      >
        <Shield className="text-white w-6 h-6" />
      </motion.div>
    </div>
  );
};

export default ShieldAnimation;
