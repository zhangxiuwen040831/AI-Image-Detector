
import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { motion } from 'framer-motion';

const BranchContribution = ({ scores }) => {
  const data = useMemo(() => [
    { name: 'RGB', score: scores.rgb * 100 },
    { name: 'Noise', score: scores.noise * 100 },
    { name: 'Freq', score: scores.frequency * 100 },
  ], [scores]);

  const COLORS = ['#DA205A', '#00D1FF', '#7C3AED'];

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="glass-card p-6 h-full flex flex-col justify-center relative overflow-hidden"
    >
      <h3 className="text-lg font-bold text-white mb-4 text-center tracking-wider uppercase border-b border-white/10 pb-2 w-full">
        Branch Contribution
      </h3>
      
      <div className="w-full h-64 relative">
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
      
      <div className="absolute bottom-4 right-4 text-xs text-gray-500 italic">
        * Feature importance based on normalized activation
      </div>
    </motion.div>
  );
};

export default BranchContribution;
