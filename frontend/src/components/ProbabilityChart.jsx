
import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { motion } from 'framer-motion';

const ProbabilityChart = ({ probability }) => {
  const data = useMemo(() => [
    { name: 'AI Generated', value: probability * 100 },
    { name: 'Real', value: (1 - probability) * 100 },
  ], [probability]);

  const COLORS = ['#DA205A', '#00D1FF'];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="glass-card p-6 h-full flex flex-col items-center justify-center relative overflow-hidden"
    >
      <h3 className="text-lg font-bold text-white mb-4 text-center tracking-wider uppercase border-b border-white/10 pb-2 w-full">
        Detection Probability
      </h3>
      
      <div className="w-full h-64 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              fill="#8884d8"
              paddingAngle={5}
              dataKey="value"
              startAngle={90}
              endAngle={-270}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(255,255,255,0.1)" strokeWidth={2} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
              itemStyle={{ color: '#fff', fontSize: '12px' }}
              formatter={(value) => `${value.toFixed(1)}%`}
            />
            <Legend verticalAlign="bottom" height={36} iconType="circle" />
          </PieChart>
        </ResponsiveContainer>
        
        {/* Center Text */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
          <span className="text-3xl font-bold text-white block drop-shadow-lg">
            {(probability * 100).toFixed(0)}%
          </span>
          <span className="text-xs text-gray-400 uppercase tracking-widest">AIGC</span>
        </div>
      </div>
    </motion.div>
  );
};

export default ProbabilityChart;
