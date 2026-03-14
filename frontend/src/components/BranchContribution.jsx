
import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { motion } from 'framer-motion';

const BranchContribution = ({ scores, language = 'zh' }) => {
  const rgb = typeof scores?.rgb === 'number' ? scores.rgb : 0;
  const noise = typeof scores?.noise === 'number' ? scores.noise : 0;
  const frequency = typeof scores?.frequency === 'number' ? scores.frequency : 0;
  const data = useMemo(() => [
    { name: language === 'zh' ? '语义' : 'Semantic', score: rgb * 100 },
    { name: language === 'zh' ? '噪声' : 'Noise', score: noise * 100 },
    { name: language === 'zh' ? '频域' : 'Frequency', score: frequency * 100 },
  ], [rgb, noise, frequency, language]);

  const COLORS = ['#DA205A', '#00D1FF', '#7C3AED'];

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="glass-card rounded-2xl p-4 border border-white/10 flex flex-col justify-center relative overflow-hidden h-full min-w-0"
    >
      <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full">
        {language === 'zh' ? '分支贡献分析' : 'Branch Contribution Analysis'}
      </h3>
      
      <div className="w-full h-48 relative">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(255,255,255,0.05)" />
            <XAxis type="number" hide />
            <YAxis 
              dataKey="name" 
              type="category" 
              tick={{ fill: '#aaa', fontSize: 12, fontWeight: 600 }} 
              axisLine={false}
              tickLine={false}
              width={50}
            />
            <Tooltip 
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
              itemStyle={{ color: '#fff', fontSize: '12px' }}
              formatter={(value) => `${value.toFixed(1)}%`}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={20}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
        {language === 'zh' ? '该图展示语义、噪声与频域分支对最终融合判定的相对贡献，用于定位本次推理的关键证据来源。' : 'This chart shows the relative contributions of the semantic, noise, and frequency branches to the fused decision, helping identify key evidence sources for this inference.'}
      </p>
    </motion.div>
  );
};

export default BranchContribution;
