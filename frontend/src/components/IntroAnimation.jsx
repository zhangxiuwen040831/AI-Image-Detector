import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

const IntroAnimation = ({ onComplete }) => {
  const canvasRef = useRef(null);
  const requestRef = useRef();
  
  // Animation configuration
  const config = {
    textLine1: "AI IMAGE",
    textLine2: "DETECTOR",
    particleCount: 2000,
    particleSize: 1.5,
    mouseRadius: 100,
    forceMultiplier: 0.5,
    returnSpeed: 0.08,
    color: '255, 255, 255', // Base white
    accentColor: '0, 255, 255', // Cyan accent
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let width = window.innerWidth;
    let height = window.innerHeight;
    
    const handleResize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
      initParticles();
    };
    
    window.addEventListener('resize', handleResize);
    canvas.width = width;
    canvas.height = height;

    let particles = [];
    let animationPhase = 0; // 0: Intro (random), 1: Forming, 2: Hold (Waiting for click)
    let frameCount = 0;

    class Particle {
      constructor(x, y) {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.targetX = x;
        this.targetY = y;
        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
        this.size = Math.random() * config.particleSize + 0.5;
        this.color = Math.random() > 0.9 
          ? `rgba(${config.accentColor}, ${Math.random() * 0.8 + 0.2})` 
          : `rgba(${config.color}, ${Math.random() * 0.6 + 0.1})`;
      }

      update(mouse) {
        // Phase 0: Chaotic movement
        if (animationPhase === 0) {
          this.x += this.vx * 2;
          this.y += this.vy * 2;
          
          if (this.x < 0 || this.x > width) this.vx *= -1;
          if (this.y < 0 || this.y > height) this.vy *= -1;
        } 
        // Phase 1 & 2: Form text
        else {
          let dx = this.targetX - this.x;
          let dy = this.targetY - this.y;
          let distance = Math.sqrt(dx * dx + dy * dy);
          let forceDirectionX = dx / distance;
          let forceDirectionY = dy / distance;
          let force = distance * 0.05;

          if (mouse.x) {
            let dxMouse = mouse.x - this.x;
            let dyMouse = mouse.y - this.y;
            let distanceMouse = Math.sqrt(dxMouse * dxMouse + dyMouse * dyMouse);
            if (distanceMouse < config.mouseRadius) {
              const repelForce = (config.mouseRadius - distanceMouse) / config.mouseRadius;
              const angle = Math.atan2(dyMouse, dxMouse);
              this.x -= Math.cos(angle) * repelForce * 5;
              this.y -= Math.sin(angle) * repelForce * 5;
            }
          }

          if (distance < 1) {
             this.x = this.targetX;
             this.y = this.targetY;
          } else {
            this.vx = forceDirectionX * force * config.forceMultiplier;
            this.vy = forceDirectionY * force * config.forceMultiplier;
            this.x += this.vx;
            this.y += this.vy;
          }
        }
      }

      draw() {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const initParticles = () => {
      particles = [];
      
      const virtualCanvas = document.createElement('canvas');
      const vCtx = virtualCanvas.getContext('2d');
      virtualCanvas.width = width;
      virtualCanvas.height = height;
      
      // Font settings
      const fontSize = Math.min(width / 8, 120); // Smaller font to fit two lines
      vCtx.font = `900 ${fontSize}px "Inter", sans-serif`;
      vCtx.fillStyle = 'white';
      vCtx.textAlign = 'center';
      vCtx.textBaseline = 'middle';
      
      // Calculate spacing to match widths
      const line1 = config.textLine1;
      const line2 = config.textLine2;
      
      const width1 = vCtx.measureText(line1).width;
      const width2 = vCtx.measureText(line2).width;
      const maxWidth = Math.max(width1, width2);
      
      // Draw Line 1
      // We manually space out the shorter line to match maxWidth
      // Simple approach: adjust letter spacing via canvas (not supported directly in standard canvas API easily without manual positioning)
      // Alternatively: Draw normally centered. The user asked for "guarantee equal width".
      // Since we are sampling pixels, we can just draw them. 
      // To force equal width, we can calculate a scale factor or tracking.
      // Let's use a simpler approach: Draw them, and if one is shorter, increase spacing manually? 
      // Or just center them. "Guaranteed equal width" usually implies justified text.
      // Canvas `letterSpacing` is available in modern browsers but risky.
      // Let's just draw them centered. "DETECTOR" and "AI IMAGE" are very close in width.
      // Let's try to manually position characters? No, that's complex for this tool.
      // Let's just rely on the font. If strict width is needed, I'd need to stretch the shorter one.
      // Let's assume standard centering is close enough, or use tracking if supported.
      
      // Vertical positioning
      const centerY = height / 2;
      const lineHeight = fontSize * 1.2;
      
      // Draw Line 1 (AI IMAGE)
      // To match width, we can use `letterSpacing` property of context if environment supports it.
      // Or we can just draw.
      
      // Hack for "Equal Width":
      // We will draw line 1 and line 2. 
      // If we want them to align perfectly, let's use a monospaced font or accept slight difference?
      // No, user said "Attention to letter spacing, guarantee two lines width consistent".
      
      // Let's try to implement a simple justification.
      const drawJustified = (text, y, targetWidth) => {
          if (!targetWidth) {
              vCtx.fillText(text, width / 2, y);
              return vCtx.measureText(text).width;
          }
          
          const chars = text.split('');
          const totalCharWidth = chars.reduce((acc, char) => acc + vCtx.measureText(char).width, 0);
          const spaceAvailable = targetWidth - totalCharWidth;
          const spacing = spaceAvailable / (chars.length - 1);
          
          let startX = (width - targetWidth) / 2;
          
          chars.forEach((char, i) => {
              const charWidth = vCtx.measureText(char).width;
              // Center the char in its slot? No, just left align + spacing
              // Actually for 'AI IMAGE', the space is a character too.
              // Let's simply change the letterSpacing property on the context string.
              vCtx.fillText(char, startX + charWidth/2, y); // This is wrong.
              startX += charWidth + spacing;
          });
      };
      
      // Standard drawing first to measure
      const m1 = vCtx.measureText(line1).width;
      const m2 = vCtx.measureText(line2).width;
      const targetW = Math.max(m1, m2);
      
      // Draw Line 1
      if (m1 < targetW) {
          // Manually distribute
          const chars = line1.split('');
          const totalW = chars.reduce((sum, c) => sum + vCtx.measureText(c).width, 0);
          const gap = (targetW - totalW) / (chars.length - 1);
          let x = (width - targetW) / 2;
          chars.forEach(c => {
              vCtx.fillText(c, x + vCtx.measureText(c).width/2, centerY - lineHeight/2);
              x += vCtx.measureText(c).width + gap;
          });
      } else {
          vCtx.fillText(line1, width/2, centerY - lineHeight/2);
      }
      
      // Draw Line 2
      if (m2 < targetW) {
           const chars = line2.split('');
           const totalW = chars.reduce((sum, c) => sum + vCtx.measureText(c).width, 0);
           const gap = (targetW - totalW) / (chars.length - 1);
           let x = (width - targetW) / 2;
           chars.forEach(c => {
               vCtx.fillText(c, x + vCtx.measureText(c).width/2, centerY + lineHeight/2);
               x += vCtx.measureText(c).width + gap;
           });
      } else {
          vCtx.fillText(line2, width/2, centerY + lineHeight/2);
      }

      const imageData = vCtx.getImageData(0, 0, width, height).data;
      
      const step = 4; 
      for (let y = 0; y < height; y += step) {
        for (let x = 0; x < width; x += step) {
          const index = (y * width + x) * 4;
          if (imageData[index + 3] > 128) {
            if (Math.random() > 0.5) {
                particles.push(new Particle(x, y));
            }
          }
        }
      }
      
      // Extra background particles
      while (particles.length < config.particleCount / 2) {
          particles.push(new Particle(Math.random() * width, Math.random() * height));
      }
    };

    let mouse = { x: null, y: null };
    window.addEventListener('mousemove', (e) => {
      mouse.x = e.x;
      mouse.y = e.y;
    });

    initParticles();

    const animate = () => {
      frameCount++;
      
      // Clear with trail
      ctx.fillStyle = `rgba(0, 0, 0, 0.2)`;
      ctx.fillRect(0, 0, width, height);

      // Phase Control
      if (frameCount > 60 && animationPhase === 0) animationPhase = 1; // Start forming
      if (frameCount > 200 && animationPhase === 1) animationPhase = 2; // Hold indefinitely
      
      // Draw particles
      if (animationPhase === 1 || animationPhase === 2) {
         // Connect particles near mouse
         connectParticles();
      }

      particles.forEach(particle => {
        particle.update(mouse);
        particle.draw();
      });

      // Scanline
      if (animationPhase === 2) {
        const scanY = (frameCount * 5) % height;
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, scanY);
        ctx.lineTo(width, scanY);
        ctx.stroke();
      }

      requestRef.current = requestAnimationFrame(animate);
    };

    const connectParticles = () => {
        if (!mouse.x) return;
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i < particles.length; i++) {
            const dx = particles[i].x - mouse.x;
            const dy = particles[i].y - mouse.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < 150) {
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(mouse.x, mouse.y);
                ctx.stroke();
            }
        }
    };

    requestRef.current = requestAnimationFrame(animate);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(requestRef.current);
    };
  }, []);

  return (
    <motion.div 
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.1, filter: "blur(10px)" }}
      transition={{ duration: 0.8 }}
      className="fixed inset-0 z-[100] bg-black flex flex-col items-center justify-center overflow-hidden cursor-pointer"
      onClick={onComplete}
    >
      <canvas ref={canvasRef} className="block absolute inset-0" />
      
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 1, 0] }}
        transition={{ duration: 2, repeat: Infinity, delay: 3 }}
        className="absolute bottom-20 text-white/50 text-sm font-mono tracking-[0.2em] pointer-events-none"
      >
        [ CLICK TO ENTER ]
      </motion.div>
    </motion.div>
  );
};

export default IntroAnimation;
